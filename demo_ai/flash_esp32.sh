#!/usr/bin/env bash
# Compile + upload firmware KWS AI ke ESP32 / ESP32-S3 (macOS).
#   ./flash_esp32.sh                 -> compile saja (cek build)
#   ./flash_esp32.sh /dev/cu.usbmodem1101          -> compile + upload ESP32-S3
#   BOARD=esp32 ./flash_esp32.sh /dev/cu.usbserial-0001   -> ESP32 klasik
#   CDC=cdc ./flash_esp32.sh /dev/cu.usbmodem1101  -> S3 yang hanya punya port USB native (Serial tak muncul tanpa ini)
# Setelah upload: buka Serial Monitor 115200 -> board mencetak alamat web dashboard.
set -euo pipefail
cd "$(dirname "$0")"
AC="${ARDUINO_CLI:-/Applications/Arduino IDE.app/Contents/Resources/app/lib/backend/resources/arduino-cli}"
[ -x "$AC" ] || AC="$(command -v arduino-cli)"
BOARD="${BOARD:-esp32s3}"
if [ "$BOARD" = "esp32s3" ]; then FQBN="esp32:esp32:esp32s3:PartitionScheme=no_ota,CDCOnBoot=${CDC:-default}"; else FQBN="esp32:esp32:esp32:PartitionScheme=no_ota"; fi
[ -f esp32_kws_ai/kws_model_int8.h ] || { echo "kws_model_int8.h belum ada: jalankan 'python export_c.py' dulu"; exit 1; }
echo ">> compile $FQBN"
"$AC" compile --fqbn "$FQBN" esp32_kws_ai
if [ $# -ge 1 ]; then
  echo ">> upload ke $1"
  "$AC" upload --fqbn "$FQBN" -p "$1" esp32_kws_ai
  echo ">> selesai. Monitor serial (Ctrl+C keluar):"
  "$AC" monitor -p "$1" -c baudrate=115200
else
  echo ">> build OK. Port yang terdeteksi:"; "$AC" board list
fi
