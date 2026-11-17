// ====================== KONFIGURASI — sesuaikan dengan rangkaian kalian ======================
#pragma once

// --- WiFi: isi SSID/password hotspot. Bila kosong / gagal tersambung 15 s, board membuat Access Point sendiri.
#define WIFI_SSID   ""
#define WIFI_PASS   ""
#define AP_SSID     "KWS-Lampu"
#define AP_PASS     "kelompok16"      // minimal 8 karakter

// --- Pin (default berbeda untuk ESP32 klasik dan ESP32-S3)
#if CONFIG_IDF_TARGET_ESP32S3
  #define PIN_I2S_SCK   4             // INMP441 SCK
  #define PIN_I2S_WS    5             // INMP441 WS
  #define PIN_I2S_SD    6             // INMP441 SD
  #define PIN_LAMP      2             // LED lampu (lewat resistor 220 ohm) / gate MOSFET
  #define PIN_OLED_SDA  8
  #define PIN_OLED_SCL  9
#else
  #define PIN_I2S_SCK   26
  #define PIN_I2S_WS    25
  #define PIN_I2S_SD    33
  #define PIN_LAMP      18
  #define PIN_OLED_SDA  21
  #define PIN_OLED_SCL  22
#endif
// INMP441: pin L/R ke GND (kanal kiri), VDD ke 3V3.

#define USE_OLED      1               // 1 = pakai OLED SSD1306 128x64 I2C (alamat 0x3C), 0 = tanpa OLED
#define MIC_SHIFT     14              // sampel I2S 32-bit >> MIC_SHIFT -> int16 (kecilkan = lebih keras)
#define GATE_DB       12.0f           // endpoint terbuka bila energi > lantai derau + GATE_DB
#define KW_MIN_PROB   0.60f           // perintah diterima bila probabilitas kata >= nilai ini
#define LEVEL_ON      5               // level terang saat "nyala"
#define COOLDOWN_MS   800
#define WATCHDOG_MS   500
