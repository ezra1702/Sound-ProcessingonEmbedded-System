/* ===========================================================================================
 *  Lampu Kendali Suara — Keyword Spotting Deep Learning di ESP32 / ESP32-S3
 *  Kelompok: Christama Ezra Yudianto (245150307111009), Abdillah Fatah Suryanto (245150300111008),
 *            Ivang Arfandia (245150307111003), Jahzeel Erason Nicodemus (245150300111002)
 *
 *  Pipeline di board (semua lokal, tanpa internet):
 *    INMP441 (I2S, 16 kHz) -> endpoint energi adaptif -> jendela 1,0 s -> log-Mel 98x40 -> DS-CNN INT8
 *    -> kata (nyala/mati/naik/turun/diam/lain) + penutur (embedding, prototipe) -> state machine -> LED PWM
 *  Web: board menyajikan halaman demo (http://<ip>/) + WebSocket /ws (protokol: demo_web/PROTOKOL.md).
 *  Fitur: kunci suara (hanya penutur dikenal), daftar pengguna baru (few-shot, tersimpan di NVS),
 *         peta perhatian (CAM) + spectrogram dikirim ke web, watchdog mic -> fail-safe OFF.
 *  Board: Arduino-ESP32 core 3.x. Pilih "ESP32S3 Dev Module" (disarankan, PSRAM: OPI/QSPI bila ada) atau "ESP32 Dev Module".
 * =========================================================================================== */
#include <WiFi.h>
#include <Preferences.h>
#include <ESP_I2S.h>
#include "esp_http_server.h"
#include "mbedtls/base64.h"
#include "config.h"
#include "web_page.h"
#include "kws_engine.h"
#if USE_OLED
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
Adafruit_SSD1306 oled(128, 64, &Wire, -1);
bool oledOk = false;
#endif

// ------------------------------------------------------------------------------ audio & endpoint
#define BLOCK        1600                       // 100 ms
#define RING_N       24000                      // 1,5 s riwayat audio
#define CLIP_WEB_N   19200                      // panjang audio dari mic HP (1,2 s)
I2SClass I2S;
static int16_t *ring;                           // ring buffer audio (heap / PSRAM)
static volatile uint32_t ringPos = 0;           // total sampel yang pernah ditulis
static volatile uint32_t lastAudioMs = 0;
static volatile float levelDb = -90.f;
static volatile bool vadOpen = false;

struct Job { int16_t *pcm; uint8_t source; };   // source: 0 = mic board, 1 = mic HP
static QueueHandle_t jobQ;
static int16_t *jobPcm[2];                      // buffer jendela 1,0 s per sumber
static volatile bool jobBusy[2] = {false, false};

// ------------------------------------------------------------------------------ state sistem
static kws_user_t users[KWS_MAX_USERS];
static int nUsers = 0;
static bool lockOn = false;                     // kunci suara: hanya penutur dikenal
static bool powerOn = false; static int level = LEVEL_ON;
enum { ST_IDLE, ST_COOLDOWN, ST_FAULT }; static int fsmState = ST_IDLE;
static uint32_t coolUntil = 0;
static const char *ST_NAME[] = {"IDLE", "COOLDOWN", "FAULT"};
// pendaftaran pengguna baru
static bool enrolling = false; static int enrollIdx = 0; static char enrollName[24];
static float enrollF[4][KWS_WIDTH]; static float enrollE[KWS_EMB]; static int enrollCnt[4];
static const int ENROLL_TOTAL = 12;             // 3 x 4 kata
static Preferences prefs;
static SemaphoreHandle_t wsLock;
static httpd_handle_t server = NULL;
static int clientFds[8]; static int nClients = 0;
static volatile bool helloPending = false;
static uint32_t nInfer = 0;

static inline uint32_t tms() { return (uint32_t)millis(); }

// ------------------------------------------------------------------------------ WebSocket: kirim ke semua klien
static void wsBroadcast(const char *msg, size_t len) {
  if (!server) return;
  xSemaphoreTake(wsLock, portMAX_DELAY);
  httpd_ws_frame_t f = {}; f.type = HTTPD_WS_TYPE_TEXT; f.payload = (uint8_t *)msg; f.len = len;
  for (int i = 0; i < nClients;) {
    int fd = clientFds[i];
    if (httpd_ws_get_fd_info(server, fd) != HTTPD_WS_CLIENT_WEBSOCKET || httpd_ws_send_frame_async(server, fd, &f) != ESP_OK) {
      clientFds[i] = clientFds[--nClients]; continue;           // klien putus -> buang
    }
    ++i;
  }
  xSemaphoreGive(wsLock);
}
static void wsSendStr(const String &s) { wsBroadcast(s.c_str(), s.length()); }

static String b64(const uint8_t *src, size_t n) {
  size_t olen = 0; mbedtls_base64_encode(NULL, 0, &olen, src, n);
  String out; out.reserve(olen + 1);
  uint8_t *buf = (uint8_t *)malloc(olen + 1); if (!buf) return out;
  mbedtls_base64_encode(buf, olen + 1, &olen, src, n); buf[olen] = 0; out = (const char *)buf; free(buf);
  return out;
}

static String stateJson() {
  return String("\"power\":") + (powerOn ? "true" : "false") + ",\"level\":" + level + ",\"state\":\"" + ST_NAME[fsmState] + "\"";
}

static String usersJson() {
  String s = "[";
  for (int i = 0; i < nUsers; ++i) { if (i) s += ","; s += "{\"name\":\"" + String(users[i].name) + "\",\"enrolled\":" + (i >= 4 ? "true" : "false") + "}"; }
  return s + "]";
}

static void sendHello() {
  String s = "{\"type\":\"hello\",\"device\":\"" + String(ESP.getChipModel()) + " @" + ESP.getCpuFreqMHz() + " MHz\"";
  s += ",\"model\":\"ds_cnn_int8_w" + String(KWS_WIDTH) + "\",\"payload_bytes\":" + String(KWS_MODEL_BYTES);
  s += ",\"macs\":" + String(KWS_MACS) + ",\"fs\":16000,\"classes\":[\"nyala\",\"mati\",\"naik\",\"turun\",\"diam\",\"lain\"]";
  s += ",\"features\":[\"probs\",\"speaker\",\"cam\",\"spec\",\"enroll\",\"lock\"],\"lock\":" + String(lockOn ? "true" : "false");
  s += ",\"users\":" + usersJson() + ",\"free_heap\":" + String(ESP.getFreeHeap()) + ",\"psram\":" + String(ESP.getPsramSize());
  s += ",\"spk_threshold\":" + String(KWS_SPK_THRESHOLD, 3) + ",\"kw_min_prob\":" + String(KW_MIN_PROB, 2) + "," + stateJson() + "}";
  wsSendStr(s);
}

// ------------------------------------------------------------------------------ lampu & state machine
static void applyLamp() { ledcWrite(PIN_LAMP, powerOn ? (uint32_t)(255.0f * powf(level / 10.0f, 2.2f)) : 0); }

static const char *fsmApply(int kw, bool bypassCooldown) {
  if (fsmState == ST_FAULT) return "NONE";
  if (!bypassCooldown && tms() < coolUntil) return "NONE";
  const char *a = "NONE";
  if (kw == 0) { powerOn = true; a = "ON"; }
  else if (kw == 1) { powerOn = false; a = "OFF"; }
  else if ((kw == 2 || kw == 3) && powerOn) {
    int nx = level + (kw == 2 ? 1 : -1);
    if (nx < 0 || nx > 10) a = "IGNORED_BOUND"; else { level = nx; a = kw == 2 ? "LEVEL_UP" : "LEVEL_DOWN"; }
  }
  if (strcmp(a, "NONE") && strcmp(a, "IGNORED_BOUND")) { coolUntil = tms() + COOLDOWN_MS; fsmState = ST_COOLDOWN; }
  applyLamp();
  return a;
}

#if USE_OLED
static void oledShow(const char *l1, const char *l2) {
  if (!oledOk) return;
  oled.clearDisplay(); oled.setTextColor(SSD1306_WHITE);
  oled.setTextSize(1); oled.setCursor(0, 0); oled.print(WiFi.getMode() == WIFI_AP ? WiFi.softAPIP() : WiFi.localIP());
  oled.setCursor(92, 0); oled.print(powerOn ? "ON " : "OFF"); oled.print(level);
  oled.drawRect(0, 12, 128, 6, SSD1306_WHITE); if (powerOn) oled.fillRect(1, 13, 126 * level / 10, 4, SSD1306_WHITE);
  oled.setTextSize(2); oled.setCursor(0, 24); oled.print(l1);
  oled.setTextSize(1); oled.setCursor(0, 50); oled.print(l2);
  oled.display();
}
#else
static void oledShow(const char *, const char *) {}
#endif

// ------------------------------------------------------------------------------ pengguna (NVS)
static void saveUsers() {
  prefs.begin("kws", false);
  prefs.putUChar("n", (uint8_t)(nUsers - 4));
  for (int i = 4; i < nUsers; ++i) prefs.putBytes(("u" + String(i - 4)).c_str(), &users[i], sizeof(kws_user_t));
  prefs.end();
}
static void loadUsers() {
  nUsers = kws_load_builtin_users(users);
  prefs.begin("kws", true);
  int n = prefs.getUChar("n", 0);
  for (int i = 0; i < n && nUsers < KWS_MAX_USERS; ++i)
    if (prefs.getBytes(("u" + String(i)).c_str(), &users[nUsers], sizeof(kws_user_t)) == sizeof(kws_user_t)) ++nUsers;
  prefs.end();
}

static const char *const ENROLL_WORDS[4] = {"nyala", "mati", "naik", "turun"};
static void enrollPrompt() {
  String s = "{\"type\":\"enroll\",\"state\":\"prompt\",\"name\":\"" + String(enrollName) + "\",\"index\":" + enrollIdx +
             ",\"total\":" + ENROLL_TOTAL + ",\"word\":\"" + ENROLL_WORDS[enrollIdx % 4] + "\"}";
  wsSendStr(s);
  char l2[32]; snprintf(l2, sizeof l2, "daftar %d/%d", enrollIdx + 1, ENROLL_TOTAL); oledShow(ENROLL_WORDS[enrollIdx % 4], l2);
}

static void enrollConsume(const kws_result_t &r) {
  if (r.kw == 4 && r.prob[4] > 0.8f) {           // terdeteksi diam -> minta ulang
    wsSendStr("{\"type\":\"enroll\",\"state\":\"retry\",\"reason\":\"suara terlalu pelan / tidak terdengar\"}"); enrollPrompt(); return;
  }
  int w = enrollIdx % 4; float n2 = 0.f;
  for (int c = 0; c < KWS_WIDTH; ++c) n2 += r.f[c] * r.f[c];
  n2 = sqrtf(n2) + 1e-9f;
  for (int c = 0; c < KWS_WIDTH; ++c) enrollF[w][c] += r.f[c] / n2;
  for (int j = 0; j < KWS_EMB; ++j) enrollE[j] += r.e[j];
  enrollCnt[w]++; enrollIdx++;
  if (enrollIdx < ENROLL_TOTAL) { enrollPrompt(); return; }
  kws_user_t u = {}; strncpy(u.name, enrollName, sizeof(u.name) - 1); u.has_kw = 1;
  for (int k = 0; k < 4; ++k) {
    float s = 0.f; for (int c = 0; c < KWS_WIDTH; ++c) s += enrollF[k][c] * enrollF[k][c];
    s = sqrtf(s) + 1e-9f; for (int c = 0; c < KWS_WIDTH; ++c) u.kw[k][c] = enrollF[k][c] / s;
  }
  float s = 0.f; for (int j = 0; j < KWS_EMB; ++j) s += enrollE[j] * enrollE[j];
  s = sqrtf(s) + 1e-9f; for (int j = 0; j < KWS_EMB; ++j) u.spk[j] = enrollE[j] / s;
  int slot = -1;
  for (int i = 4; i < nUsers; ++i) if (!strcmp(users[i].name, u.name)) slot = i;   // nama sama -> timpa
  if (slot < 0) slot = nUsers < KWS_MAX_USERS ? nUsers++ : KWS_MAX_USERS - 1;
  users[slot] = u; saveUsers(); enrolling = false;
  wsSendStr("{\"type\":\"enroll\",\"state\":\"done\",\"name\":\"" + String(u.name) + "\"}");
  wsSendStr("{\"type\":\"users\",\"list\":" + usersJson() + "}");
  oledShow("Terdaftar", u.name);
}

// ------------------------------------------------------------------------------ inferensi (task terpisah)
static int8_t xq[KWS_FRAMES * KWS_MELS];
static uint8_t specU8[KWS_FRAMES * KWS_MELS];
static uint8_t camU8[KWS_CAM_H * KWS_CAM_W];

static void inferTask(void *) {
  kws_result_t r; Job job;
  for (;;) {
    if (xQueueReceive(jobQ, &job, portMAX_DELAY) != pdTRUE) continue;
    uint32_t t0 = micros(); kws_frontend(job.pcm, xq, NULL);
    uint32_t t1 = micros(); kws_infer(xq, &r); kws_decide(&r, users, nUsers, KWS_SPK_THRESHOLD);
    uint32_t t2 = micros();
    jobBusy[job.source] = false; ++nInfer;
    int camK = r.kw;                                  // CAM kelas terpilih; bila ditolak pakai keyword terkuat
    if (r.kw >= 4) { camK = 0; for (int k = 1; k < 4; ++k) if (r.prob[k] > r.prob[camK]) camK = k; }
    kws_cam(&r, camK, camU8);
    for (int i = 0; i < KWS_FRAMES * KWS_MELS; ++i) specU8[i] = (uint8_t)((int)xq[i] + 128);
    const char *src = job.source ? "web_mic" : "board_mic";
    if (enrolling) { enrollConsume(r); continue; }
    bool isKw = r.kw < 4 && r.kw_prob >= KW_MIN_PROB;
    const char *kind, *action; String label = "null";
    if (!isKw) { kind = "reject"; action = "REJECT"; }
    else if (lockOn && r.speaker < 0) { kind = "reject"; action = "REJECT_SPEAKER"; label = String("\"") + KWS_CLASS_NAMES[r.kw] + "\""; }
    else { kind = "keyword"; label = String("\"") + KWS_CLASS_NAMES[r.kw] + "\""; action = fsmApply(r.kw, false); }
    String s; s.reserve(7200);
    s = "{\"type\":\"event\",\"kind\":\"" + String(kind) + "\",\"label\":" + label + ",\"action\":\"" + action + "\"," + stateJson();
    s += ",\"t_ms\":" + String(tms()) + ",\"source\":\"" + src + "\",\"score\":" + String(r.kw_prob, 3);
    s += ",\"latency_ms\":" + String((t2 - t0) / 1000.0f, 2) + ",\"lat_fe_ms\":" + String((t1 - t0) / 1000.0f, 2) + ",\"lat_nn_ms\":" + String((t2 - t1) / 1000.0f, 2);
    s += ",\"probs\":["; for (int k = 0; k < KWS_N_CLASSES; ++k) { if (k) s += ","; s += String(r.prob[k], 3); } s += "]";
    s += ",\"speaker\":" + (r.speaker >= 0 ? "\"" + String(users[r.speaker].name) + "\"" : String("null")) + ",\"speaker_cos\":" + String(r.speaker_cos, 3);
    s += ",\"speaker_scores\":{"; for (int u = 0; u < nUsers; ++u) { if (u) s += ","; s += "\"" + String(users[u].name) + "\":" + String(r.speaker_cos_all[u], 3); } s += "}";
    s += ",\"cam_h\":" + String(KWS_CAM_H) + ",\"cam_w\":" + String(KWS_CAM_W) + ",\"cam_b64\":\"" + b64(camU8, sizeof camU8) + "\"";
    s += ",\"spec_t\":" + String(KWS_FRAMES) + ",\"spec_f\":" + String(KWS_MELS) + ",\"spec_b64\":\"" + b64(specU8, sizeof specU8) + "\"}";
    wsSendStr(s);
    Serial.printf("[%8.2fs] %-9s %-7s %-6s p=%.2f spk=%s(%.2f) -> %s | %s lvl %d | fe %.1f ms nn %.1f ms\n", tms() / 1000.0f, src, kind,
                  KWS_CLASS_NAMES[r.kw], r.kw_prob, r.speaker >= 0 ? users[r.speaker].name : "?", r.speaker_cos, action,
                  powerOn ? "ON" : "OFF", level, (t1 - t0) / 1000.0f, (t2 - t1) / 1000.0f);
    char l2[40]; snprintf(l2, sizeof l2, "%s %.0f%% %s", r.speaker >= 0 ? users[r.speaker].name : "?", 100 * r.kw_prob, action);
    oledShow(isKw ? KWS_CLASS_NAMES[r.kw] : "(tolak)", l2);
  }
}

// jendela 1,0 s berenergi terbesar dari buffer linear
static void centerWindow(const int16_t *src, int n, int16_t *dst) {
  if (n <= KWS_WIN) { memcpy(dst, src, n * 2); memset(dst + n, 0, (KWS_WIN - n) * 2); return; }
  int64_t e = 0, best = -1; int bi = 0;
  for (int i = 0; i < KWS_WIN; ++i) e += (int32_t)src[i] * src[i];
  for (int s = 0;; s += 160) {
    if (e > best) { best = e; bi = s; }
    if (s + 160 + KWS_WIN > n) break;
    for (int i = 0; i < 160; ++i) { e -= (int32_t)src[s + i] * src[s + i]; e += (int32_t)src[s + KWS_WIN + i] * src[s + KWS_WIN + i]; }
  }
  memcpy(dst, src + bi, KWS_WIN * 2);
}

static void submitFromRing(uint32_t endPos) {
  if (jobBusy[0]) return;
  static int16_t lin[RING_N];                    // linearisasi 1,5 s terakhir (statis, 48 KB)
  uint32_t n = endPos < RING_N ? endPos : RING_N;
  for (uint32_t i = 0; i < n; ++i) lin[i] = ring[(endPos - n + i) % RING_N];
  centerWindow(lin, n, jobPcm[0]);
  jobBusy[0] = true; Job j = {jobPcm[0], 0};
  if (xQueueSend(jobQ, &j, 0) != pdTRUE) jobBusy[0] = false;
}

static void audioTask(void *) {
  static int32_t raw[BLOCK];
  float floorDb = -70.f, peakDb = -120.f; int warm = 0, quiet = 0, sinceOpen = 0; bool active = false;
  float warmBuf[10]; uint32_t lastLevelSent = 0;
  for (;;) {
    size_t got = I2S.readBytes((char *)raw, sizeof raw);
    int ns = got / 4; if (ns <= 0) continue;
    double se = 0;
    for (int i = 0; i < ns; ++i) {
      int32_t v = raw[i] >> MIC_SHIFT; v = v > 32767 ? 32767 : (v < -32768 ? -32768 : v);
      ring[(ringPos + i) % RING_N] = (int16_t)v; se += (double)v * v;
    }
    ringPos += ns; lastAudioMs = tms();
    float db = 10.f * log10f((float)(se / ns) / (32768.f * 32768.f) + 1e-12f); levelDb = db;
    if (warm < 10) {                                              // 1 s pertama: kalibrasi lantai derau (median)
      warmBuf[warm++] = db; float t[10]; memcpy(t, warmBuf, sizeof(float) * warm);
      for (int a = 0; a < warm; ++a) for (int b = a + 1; b < warm; ++b) if (t[b] < t[a]) { float x = t[a]; t[a] = t[b]; t[b] = x; }
      floorDb = t[warm / 2]; continue;
    }
    if (!active) floorDb = db < floorDb ? 0.8f * floorDb + 0.2f * db : fminf(floorDb + 0.05f, db);
    if (!active && db > floorDb + GATE_DB) { active = true; peakDb = db; quiet = 0; sinceOpen = 0; }
    else if (active) {
      ++sinceOpen; peakDb = fmaxf(peakDb, db);
      quiet = (db < floorDb + 6.f || db < peakDb - 20.f) ? quiet + 1 : 0;
      if (quiet >= 2 || sinceOpen >= 12) { active = false; submitFromRing(ringPos); }
    }
    vadOpen = active;
    if (tms() - lastLevelSent >= 100 && nClients) {
      lastLevelSent = tms(); char m[80];
      int n = snprintf(m, sizeof m, "{\"type\":\"level\",\"rms_db\":%.1f,\"vad\":%s}", db, active ? "true" : "false");
      wsBroadcast(m, n);
    }
  }
}

// ------------------------------------------------------------------------------ HTTP + WebSocket
static esp_err_t pageHandler(httpd_req_t *req) {
  httpd_resp_set_type(req, "text/html; charset=utf-8");
  httpd_resp_set_hdr(req, "Content-Encoding", "gzip");
  httpd_resp_set_hdr(req, "Cache-Control", "no-store");
  return httpd_resp_send(req, (const char *)WEB_PAGE_GZ, WEB_PAGE_GZ_LEN);
}

static String jsonStr(const char *js, const char *key) {           // ambil nilai string sederhana "key":"..."
  String pat = String("\"") + key + "\""; const char *p = strstr(js, pat.c_str()); if (!p) return "";
  p = strchr(p + pat.length(), ':'); if (!p) return ""; p = strchr(p, '"'); if (!p) return "";
  const char *q = strchr(p + 1, '"'); if (!q) return ""; return String(p + 1).substring(0, q - p - 1);
}

static void handleText(const char *js) {
  String type = jsonStr(js, "type");
  if (type == "cmd") {
    String a = jsonStr(js, "action"); int kw = a == "ON" ? 0 : a == "OFF" ? 1 : a == "LEVEL_UP" ? 2 : a == "LEVEL_DOWN" ? 3 : -1;
    if (kw < 0) return;
    const char *act = fsmApply(kw, true);
    wsSendStr("{\"type\":\"event\",\"kind\":\"manual\",\"label\":null,\"action\":\"" + String(act) + "\",\"source\":\"web\",\"t_ms\":" + String(tms()) + "," + stateJson() + "}");
  } else if (type == "set") {
    if (strstr(js, "\"lock\":true")) lockOn = true; else if (strstr(js, "\"lock\":false")) lockOn = false;
    wsSendStr(String("{\"type\":\"settings\",\"lock\":") + (lockOn ? "true" : "false") + "}");
  } else if (type == "enroll_start") {
    String nm = jsonStr(js, "name"); nm.trim(); if (!nm.length()) return;
    strncpy(enrollName, nm.c_str(), sizeof(enrollName) - 1); enrollName[sizeof(enrollName) - 1] = 0;
    memset(enrollF, 0, sizeof enrollF); memset(enrollE, 0, sizeof enrollE); memset(enrollCnt, 0, sizeof enrollCnt);
    enrollIdx = 0; enrolling = true; enrollPrompt();
  } else if (type == "enroll_cancel") {
    enrolling = false; wsSendStr("{\"type\":\"enroll\",\"state\":\"cancelled\"}");
  } else if (type == "user_delete") {
    String nm = jsonStr(js, "name");
    for (int i = 4; i < nUsers; ++i) if (nm == users[i].name) { for (int k = i; k < nUsers - 1; ++k) users[k] = users[k + 1]; --nUsers; break; }
    saveUsers(); wsSendStr("{\"type\":\"users\",\"list\":" + usersJson() + "}");
  } else if (type == "hello") {
    helloPending = true;
  }
}

static esp_err_t wsHandler(httpd_req_t *req) {
  if (req->method == HTTP_GET) {                                    // handshake: catat klien
    xSemaphoreTake(wsLock, portMAX_DELAY);
    if (nClients < 8) clientFds[nClients++] = httpd_req_to_sockfd(req);
    xSemaphoreGive(wsLock);
    helloPending = true; return ESP_OK;
  }
  httpd_ws_frame_t f = {}; f.type = HTTPD_WS_TYPE_TEXT;
  if (httpd_ws_recv_frame(req, &f, 0) != ESP_OK || f.len > 2 * CLIP_WEB_N + 64) return ESP_FAIL;
  uint8_t *buf = (uint8_t *)malloc(f.len + 1); if (!buf) return ESP_ERR_NO_MEM;
  f.payload = buf;
  if (httpd_ws_recv_frame(req, &f, f.len) != ESP_OK) { free(buf); return ESP_FAIL; }
  buf[f.len] = 0;
  if (f.type == HTTPD_WS_TYPE_BINARY && f.len >= 2 * 4000) {        // audio mic HP: PCM16 LE mono 16 kHz
    if (!jobBusy[1]) {
      centerWindow((const int16_t *)buf, f.len / 2, jobPcm[1]);
      jobBusy[1] = true; Job j = {jobPcm[1], 1};
      if (xQueueSend(jobQ, &j, 0) != pdTRUE) jobBusy[1] = false;
    }
  } else if (f.type == HTTPD_WS_TYPE_TEXT) handleText((const char *)buf);
  free(buf); return ESP_OK;
}

static void startServer() {
  httpd_config_t cfg = HTTPD_DEFAULT_CONFIG(); cfg.stack_size = 8192; cfg.max_open_sockets = 7; cfg.lru_purge_enable = true;
  if (httpd_start(&server, &cfg) != ESP_OK) { Serial.println("HTTP server gagal"); return; }
  httpd_uri_t page = {}; page.uri = "/"; page.method = HTTP_GET; page.handler = pageHandler; httpd_register_uri_handler(server, &page);
  httpd_uri_t ws = {}; ws.uri = "/ws"; ws.method = HTTP_GET; ws.handler = wsHandler; ws.is_websocket = true; httpd_register_uri_handler(server, &ws);
}

// ------------------------------------------------------------------------------ setup & loop
void setup() {
  Serial.begin(115200); delay(300);
  Serial.println("\n=== Lampu Kendali Suara — KWS DS-CNN INT8 ===");
  ledcAttach(PIN_LAMP, 5000, 8); applyLamp();
#if USE_OLED
  Wire.begin(PIN_OLED_SDA, PIN_OLED_SCL);
  oledOk = oled.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  if (oledOk) { oled.clearDisplay(); oled.setTextColor(SSD1306_WHITE); oled.setCursor(0, 0); oled.print("KWS: memulai..."); oled.display(); }
#endif
  wsLock = xSemaphoreCreateMutex(); jobQ = xQueueCreate(2, sizeof(Job));
  ring = (int16_t *)(psramFound() ? ps_malloc(RING_N * 2) : malloc(RING_N * 2));
  jobPcm[0] = (int16_t *)malloc(KWS_WIN * 2); jobPcm[1] = (int16_t *)malloc(KWS_WIN * 2);
  if (!ring || !jobPcm[0] || !jobPcm[1] || !kws_init()) { Serial.println("MEMORI TIDAK CUKUP"); while (1) delay(1000); }
  loadUsers();
  Serial.printf("Model ds_cnn_int8 lebar %d, %d MAC, %d B | pengguna %d | heap bebas %u B | PSRAM %u B\n",
                KWS_WIDTH, KWS_MACS, KWS_MODEL_BYTES, nUsers, ESP.getFreeHeap(), ESP.getPsramSize());

  WiFi.mode(WIFI_STA);
  if (strlen(WIFI_SSID)) {
    WiFi.begin(WIFI_SSID, WIFI_PASS); Serial.print("WiFi tersambung ke "); Serial.print(WIFI_SSID);
    for (int i = 0; i < 30 && WiFi.status() != WL_CONNECTED; ++i) { delay(500); Serial.print('.'); }
    Serial.println();
  }
  if (WiFi.status() == WL_CONNECTED) Serial.printf("Buka di browser: http://%s/\n", WiFi.localIP().toString().c_str());
  else { WiFi.mode(WIFI_AP); WiFi.softAP(AP_SSID, AP_PASS); Serial.printf("Access Point \"%s\" (sandi %s) -> buka http://%s/\n", AP_SSID, AP_PASS, WiFi.softAPIP().toString().c_str()); }
  startServer();

  I2S.setPins(PIN_I2S_SCK, PIN_I2S_WS, -1, PIN_I2S_SD);
  if (!I2S.begin(I2S_MODE_STD, 16000, I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO, I2S_STD_SLOT_LEFT)) Serial.println("I2S gagal dimulai — cek kabel mic");
  xTaskCreatePinnedToCore(audioTask, "audio", 6144, NULL, 5, NULL, 0);
  xTaskCreatePinnedToCore(inferTask, "infer", 12288, NULL, 3, NULL, 1);
  oledShow("Siap", "ucapkan: nyala");
}

void loop() {
  static uint32_t lastState = 0;
  if (helloPending) { helloPending = false; delay(50); sendHello(); }
  uint32_t now = tms();
  if (fsmState == ST_COOLDOWN && now >= coolUntil) { fsmState = ST_IDLE; wsSendStr("{\"type\":\"state\"," + stateJson() + "}"); }
  bool stalled = now - lastAudioMs > WATCHDOG_MS;                  // watchdog mic
  if (stalled && fsmState != ST_FAULT) {
    fsmState = ST_FAULT; powerOn = false; applyLamp();
    wsSendStr("{\"type\":\"event\",\"kind\":\"failsafe\",\"action\":\"FAILSAFE_OFF\",\"source\":\"board_mic\",\"t_ms\":" + String(now) + "," + stateJson() + "}");
    Serial.println("WATCHDOG: audio berhenti -> fail-safe OFF"); oledShow("MIC ERR", "fail-safe OFF");
  } else if (!stalled && fsmState == ST_FAULT) {
    fsmState = ST_IDLE;
    wsSendStr("{\"type\":\"event\",\"kind\":\"recover\",\"action\":\"RECOVER\",\"source\":\"board_mic\",\"t_ms\":" + String(now) + "," + stateJson() + "}");
  }
  if (now - lastState > 5000) { lastState = now; }
  delay(20);
}
