// MP16 FITUR TAMBAHAN - baseline porting ESP32 (Arduino-ESP32 core 2.x, I2S MEMS mic mis. INMP441).
// Header beku disalin ke folder sketch: mp14_realtime_constants.h, mp14_ring_buffer.h, mp15_classical_model_int8.h,
// mp16_project_config.h, mp16_command_fsm.h. Front-end 39-D (filter MP6 -> preprocessing MP7 -> feature MP8) harus diport
// dari features/kws_frontend.py ke fungsi mp16_extract39() (CMSIS/ESP-DSP untuk FFT). Angka latency/RAM WAJIB diukur di perangkat.
#include <Arduino.h>
#include <driver/i2s.h>
#define CRIT_ENTER() portENTER_CRITICAL_ISR(&ring_mux)
#define CRIT_EXIT()  portEXIT_CRITICAL_ISR(&ring_mux)
static portMUX_TYPE ring_mux = portMUX_INITIALIZER_UNLOCKED;
#include "mp14_realtime_constants.h"
#include "mp14_ring_buffer.h"
#include "mp15_classical_model_int8.h"
#include "mp16_project_config.h"
#include "mp16_command_fsm.h"

#define PIN_LED 2
#define PIN_PWM 4
static block_ring_t ring;
static cmd_fsm_t fsm;
static int16_t window_pcm[(MP16_SAMPLE_RATE * 12) / 10];   // 1,2 s = jendela endpoint desain final v2
extern bool mp16_extract39(const int16_t *pcm, int n, float *x39, bool *endpoint);  // TODO: port kws_frontend.py

static void i2s_task(void *) {           // produser: baca 1 blok DMA (10 ms) -> ring buffer
  int16_t blk[DMA_BLOCK_SAMPLES]; size_t got; uint32_t seq = 0;
  for (;;) { i2s_read(I2S_NUM_0, blk, sizeof(blk), &got, portMAX_DELAY); ring_push(&ring, blk, seq++); }
}

void setup() {
  Serial.begin(115200); pinMode(PIN_LED, OUTPUT); ledcSetup(0, 5000, 8); ledcAttachPin(PIN_PWM, 0);
  i2s_config_t cfg = { (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX), AUDIO_FS_HZ, I2S_BITS_PER_SAMPLE_16BIT, I2S_CHANNEL_FMT_ONLY_LEFT,
                       I2S_COMM_FORMAT_STAND_I2S, 0, DMA_BUFFER_COUNT * 2, DMA_BLOCK_SAMPLES, false, false, 0 };
  i2s_pin_config_t pins = { .bck_io_num = 26, .ws_io_num = 25, .data_out_num = -1, .data_in_num = 33 };
  i2s_driver_install(I2S_NUM_0, &cfg, 0, NULL); i2s_set_pin(I2S_NUM_0, &pins);
  ring_init(&ring); fsm_init(&fsm);
  xTaskCreatePinnedToCore(i2s_task, "i2s", 4096, NULL, 5, NULL, 0);
}

void loop() {                            // konsumen: DSP + inferensi tiap INFERENCE_STRIDE_FRAMES blok
  static int hop = 0, blocks_since_audio = 0; int16_t blk[DMA_BLOCK_SAMPLES]; uint32_t seq;
  if (!ring_pop(&ring, blk, &seq)) { vTaskDelay(1); if (++blocks_since_audio > 50) { fsm_step(&fsm, 0, 0, 0, 0); blocks_since_audio = 0; } return; }
  blocks_since_audio = 0;
  memmove(window_pcm, window_pcm + DMA_BLOCK_SAMPLES, sizeof(window_pcm) - sizeof(blk));
  memcpy(window_pcm + (sizeof(window_pcm)/2 - DMA_BLOCK_SAMPLES), blk, sizeof(blk));
  if (++hop < INFERENCE_STRIDE_FRAMES) return; hop = 0;
  uint32_t t0 = micros(); float x39[39]; bool endpoint = false;
  int cls = 0, accepted = 0;
  if (mp16_extract39(window_pcm, sizeof(window_pcm)/2, x39, &endpoint) && endpoint) { cls = mp15_classical_predict(x39); accepted = 1; /* + gate OOD */ }
  int act = fsm_step(&fsm, 1, endpoint, cls, accepted);
  digitalWrite(PIN_LED, fsm.power); ledcWrite(0, fsm.power ? fsm.level * 25 : 0);
  if (act != ACT_NONE) Serial.printf("{\"t_ms\":%lu,\"cls\":%d,\"act\":%d,\"power\":%d,\"level\":%d,\"lat_us\":%lu,\"ovf\":%lu}\n",
                                     millis(), cls, act, fsm.power, fsm.level, micros() - t0, (unsigned long)ring.overflow_count);
}
