# Jetson Nano Speaking OCR

**Offline, screenless visual assistant for Jetson Nano 4 GB: it describes objects and reads Polish printed text aloud.**

[Polski](README.pl.md) · [Clean SD card setup](docs/SD_CARD_SETUP.md)

## What it does

The assistant has two event-driven modes:

```text
Describe: camera → YOLOv5n → TensorRT → Polish description → Piper → speaker
Read:     camera → grayscale/Canny → contours → perspective correction
          → adaptive threshold → Tesseract (pol) → Piper → speaker
```

- Fully local after installation; no cloud inference or speech service.
- Operated over SSH/keyboard or with two GPIO buttons.
- Designed around the constraints of JetPack 4.6.1, Ubuntu 18.04 and Python 3.6.
- Keeps the camera stream and Piper voice loaded for the session to avoid repeated USB wake-ups and model reloads.
- Runs TensorRT YOLO in an isolated process: consecutive descriptions reuse it, while selecting OCR stops it and releases CUDA memory.

OCR finds the largest sufficiently large convex quadrilateral, orders its
corners and rectifies the page at full resolution. If the whole page is not
visible or perspective correction fails, it safely processes the full frame
and still invokes Tesseract.

> This is an assistive **demonstrator**, not a certified mobility or safety device. Detection and OCR can be wrong; do not rely on it for navigation, traffic, medication or other safety-critical decisions.

## Hardware

- Jetson Nano 4 GB Developer Kit with active cooling;
- 64–128 GB A1/A2 microSD card;
- autofocus UVC USB camera (CSI is possible after adapting the capture pipeline);
- USB speaker or USB sound card;
- stable 5 V / 4 A supply for development;
- optionally, two normally-open momentary buttons;
- for mobile use: a sufficiently rated power bank and a properly engineered, fused 5 V supply path.

Do not feed 9 V or 12 V into the Nano's 5 V input. A PD trigger must be followed by a correctly adjusted step-down converter. Verify voltage and polarity with a meter before connecting the board, and follow the carrier-board guide for barrel-power selection (commonly the J48 jumper).

## Quick start on a prepared Jetson

Start with JetPack 4.6.1. The complete from-zero procedure is in [Clean SD card setup](docs/SD_CARD_SETUP.md).

```bash
git clone https://github.com/daniel-w-p/Jetson_Nano_Speaking_OCR.git ~/nano-speaker
cd ~/nano-speaker
chmod +x scripts/*.sh
./scripts/create_swap.sh
./scripts/bootstrap_jetson.sh
python3 src/say.py
python3 src/main_demo.py --action read
```

Object detection additionally requires the device-specific TensorRT engine:

```bash
# Install the JetPack 4-compatible PyTorch wheel first; see the SD setup guide.
./scripts/build_yolo.sh
python3 src/main_demo.py --action describe
```

Run the interactive two-mode demo:

```bash
python3 src/main_demo.py --mode keyboard
```

## Configuration

The bootstrap copies `config/config.example.json` to the ignored, device-local `config/config.json`. Edit it to select the camera index, UVC pixel format/FPS, ALSA device, OCR parameters, YOLO confidence, model paths and GPIO pins. The default camera session explicitly negotiates `MJPG`, 1280×720 at 15 FPS, discards 30 warm-up frames and continuously retains the newest frame.

Existing local `config.json` files do not require migration. Configuration is
deep-merged with the defaults, so missing OCR keys are filled automatically.
The complete set is available in
[`config/config.example.json`](config/config.example.json):

```json
"ocr": {
  "language": "pol",
  "page_segmentation_mode": 6,
  "max_characters": 700,
  "scale_factor": 1.6,
  "threshold_block_size": 31,
  "threshold_c": 11,
  "page_detection": {
    "enabled": true,
    "max_width": 800,
    "blur_kernel": 5,
    "canny_low": 50,
    "canny_high": 150,
    "contour_candidates": 10,
    "approx_epsilon_ratio": 0.02,
    "min_page_area_ratio": 0.20,
    "clipped_page_fallback": true,
    "frame_border_thickness": 3,
    "debug_console": false,
    "debug_images": false
  }
}
```

- `max_width` limits only the copy used for Canny and contours; perspective
  correction uses the full-resolution frame.
- `min_page_area_ratio` controls how much of the image a page must occupy.
- `approx_epsilon_ratio` controls contour-to-quadrilateral approximation.
- `clipped_page_fallback` closes visible page edges against the frame when
  part of the page extends outside the image.
- `frame_border_thickness` controls the helper border on the Canny map; the
  default `3` bridges edges ending a few pixels before the frame.
- `debug_console` enables the console status without writing diagnostic
  images. Enabling any `debug_*` flag, including `debug_images`, also prints
  this status.
- `scale_factor`, `threshold_block_size` and `threshold_c` prepare the
  rectified image directly for Tesseract.
- Setting `page_detection.enabled` to `false` forces full-frame OCR.

For a USB speaker, find the ALSA identifier with `aplay -l`, then set for example:

```json
"aplay_device": "plughw:CARD=Device,DEV=0"
```

Defaults use physical header pins 11 and 13 (`GPIO.BOARD`). Wire each normally-open button between its configured signal pin and a GND pin; the code enables internal pull-ups. Never connect 5 V to a GPIO pin.

## Headless service

First verify camera, OCR, speech and YOLO separately. Then enable GPIO in `config/config.json` and install the service:

```bash
# bootstrap_jetson.sh has already installed Jetson.GPIO, its udev rule and group.
sudo reboot
cd ~/nano-speaker
./scripts/install_service.sh gpio
journalctl -u nano-speaker.service -f
```

The installer derives the current user and absolute checkout path; nothing is hard-coded to `/home/daniel`.

## Repository layout

```text
config/     device configuration template
docs/       clean-install guides (English and Polish)
scripts/    bootstrap, diagnostics, TensorRT build, swap and systemd helpers
src/        camera, OCR, speech, detection and controller modules
systemd/    service template
tests/      hardware-independent unit tests
```

Generated models, TensorRT engines, local configuration and temporary captures are intentionally ignored by Git.

## Test and troubleshoot

```bash
python3 -m unittest discover -s tests -v
./scripts/diagnose.sh
fswebcam -r 1280x720 --no-banner tmp/camera.jpg
speaker-test -t wav -c 2
tesseract tmp/camera.jpg stdout -l pol --psm 6
```

For the first tests on the Jetson, temporarily set:

```json
"page_detection": {
  "debug_images": true
}
```

Every read action continues to overwrite `tmp/page_raw.jpg` and
`tmp/page_prepared.png`. With debugging enabled it also writes:

- `tmp/page_edges.png` — the Canny edge map;
- `tmp/page_detected.jpg` — the raw frame with the selected page outlined in
  green for a complete contour or orange for corners inferred from the frame;
- `tmp/page_warped.png` — the rectified grayscale page before thresholding;
  a stale file is removed when full-frame fallback is used.

When any `debug_*` flag is `true`, the console—or `journalctl` in service
mode—reports which path was selected, for example:

```text
[OCR debug] Fallback: brak; wykryto pełny czworokąt kartki.
[OCR debug] Fallback: narożniki domknięte granicą kadru.
[OCR debug] Fallback: OCR całej klatki; nie znaleziono wiarygodnego czworokąta kartki.
```

Inspect them in this order: `page_raw` → `page_edges` → `page_detected` →
`page_warped` → `page_prepared`. This separates camera, edge detection,
contour selection, perspective and thresholding problems. Disable
`debug_images` after tuning to reduce SD-card writes.

Common issues:

- `No frame received`: change `camera.device` after checking `v4l2-ctl --list-devices`.
- Silence or wrong output: set `speech.aplay_device`; check mute and gain in `alsamixer`.
- Poor OCR: enable diagnostics, use even light and avoid glare; if the outline
  is wrong, tune the Canny thresholds or `min_page_area_ratio`. An orange
  outline is expected when the page extends beyond the frame.
- Build killed: confirm swap is active and rerun `BUILD_JOBS=1 ./scripts/build_yolo.sh`.
- Sudden resets or throttling: use `tegrastats`; improve the 5 V supply and cooling.

## Technical baseline

The reproducible target is JetPack 4.6.1 (L4T 32.7.1, CUDA 10.2, TensorRT 8.2.1), Python 3.6, Tesseract Polish data, archived Piper `2023.11.14-2`, and the `mailrocketsystems/JetsonYolov5` TensorRT wrapper. TensorRT engines are hardware/software-build artifacts and must be generated on the target Nano; they are not committed.

Sources: [NVIDIA JetPack 4.6.1](https://developer.nvidia.com/embedded/jetpack-sdk-461), [NVIDIA Nano setup](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit), [Piper release](https://github.com/rhasspy/piper/releases/tag/2023.11.14-2), [JetsonYolov5](https://github.com/mailrocketsystems/JetsonYolov5).

## License

Project code is licensed under the [MIT License](LICENSE). Downloaded models and third-party repositories retain their own licenses.
