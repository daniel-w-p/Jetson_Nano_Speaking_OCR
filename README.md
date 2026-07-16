# Jetson Nano Speaking OCR

**Offline, screenless visual assistant for Jetson Nano 4 GB: it describes objects and reads Polish printed text aloud.**

[Polski](README.pl.md) · [Clean SD card setup](docs/SD_CARD_SETUP.md)

## What it does

The assistant has two event-driven modes:

```text
Describe: camera → YOLOv5n → TensorRT → Polish description → Piper → speaker
Read:     camera → OpenCV → Tesseract (pol) → Piper → speaker
```

- Fully local after installation; no cloud inference or speech service.
- Operated over SSH/keyboard or with two GPIO buttons.
- Designed around the constraints of JetPack 4.6.1, Ubuntu 18.04 and Python 3.6.
- Uses one workload at a time to fit within the Nano's 4 GB RAM.

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

The bootstrap copies `config/config.example.json` to the ignored, device-local `config/config.json`. Edit it to select the camera index, ALSA device, thresholds, model paths and GPIO pins.

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

Common issues:

- `No frame received`: change `camera.device` after checking `v4l2-ctl --list-devices`.
- Silence or wrong output: set `speech.aplay_device`; check mute and gain in `alsamixer`.
- Poor OCR: fill the frame, avoid glare, use even light and keep the page parallel to the camera.
- Build killed: confirm swap is active and rerun `BUILD_JOBS=1 ./scripts/build_yolo.sh`.
- Sudden resets or throttling: use `tegrastats`; improve the 5 V supply and cooling.

## Technical baseline

The reproducible target is JetPack 4.6.1 (L4T 32.7.1, CUDA 10.2, TensorRT 8.2.1), Python 3.6, Tesseract Polish data, archived Piper `2023.11.14-2`, and the `mailrocketsystems/JetsonYolov5` TensorRT wrapper. TensorRT engines are hardware/software-build artifacts and must be generated on the target Nano; they are not committed.

Sources: [NVIDIA JetPack 4.6.1](https://developer.nvidia.com/embedded/jetpack-sdk-461), [NVIDIA Nano setup](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit), [Piper release](https://github.com/rhasspy/piper/releases/tag/2023.11.14-2), [JetsonYolov5](https://github.com/mailrocketsystems/JetsonYolov5).

## License

Project code is licensed under the [MIT License](LICENSE). Downloaded models and third-party repositories retain their own licenses.
