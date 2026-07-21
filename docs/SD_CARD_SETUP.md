# From a blank microSD card to a working assistant

[Polski](SD_CARD_SETUP.pl.md) · [README](../README.md)

This runbook targets the **Jetson Nano 4 GB Developer Kit** and JetPack **4.6.1**. Read commands before running them. Flashing erases the selected card, and power wiring mistakes can destroy the board.

## 1. Prepare the card on another computer

You need a 64–128 GB A1/A2 microSD card, reader, internet connection, keyboard, display for the first boot (or serial headless setup), camera, speaker, active cooling and a reliable Nano power supply.

1. Download the **Jetson Nano Developer Kit** image from the [official JetPack 4.6.1 page](https://developer.nvidia.com/embedded/jetpack-sdk-461). Do not select the Nano 2 GB image and do not use JetPack 5/6.
2. Install [balenaEtcher](https://etcher.balena.io/) or follow [NVIDIA's OS-specific flashing instructions](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit).
3. Select the downloaded ZIP as the source, carefully select the microSD card as the target, then flash and validate it.
4. Eject the card safely and insert it into the unpowered Nano.

## 2. First boot

Attach active cooling, display, keyboard, network and a known-good supply. Complete the Ubuntu wizard: accept the license, choose language/time zone, create a user and password, and let the filesystem use the card. Then open a terminal:

```bash
sudo apt-get update
sudo apt-get install -y git curl wget htop nano v4l-utils alsa-utils
sudo reboot
```

Keep the JetPack/L4T baseline fixed while reproducing the build; do not run a blind distribution upgrade. Apply NVIDIA security updates later as a separately tested maintenance step and rebuild the TensorRT engine after stack changes.

After reboot, verify the baseline:

```bash
cat /etc/nv_tegra_release       # expected release R32.7.1 for JetPack 4.6.1
python3 --version               # expected Python 3.6.x
/usr/local/cuda/bin/nvcc --version  # expected CUDA 10.2
dpkg -l | grep -E 'tensorrt|nvinfer'
```

For remote administration, obtain the address with `hostname -I`, then connect from another computer using `ssh USER@ADDRESS`.

## 3. Clone and prepare memory

Clone this repository:

```bash
git clone https://github.com/daniel-w-p/Jetson_Nano_Speaking_OCR.git ~/nano-speaker
cd ~/nano-speaker
chmod +x scripts/*.sh
./scripts/create_swap.sh 4G
sudo nvpmodel -m 0
sudo jetson_clocks
```

`nvpmodel -m 0` is the 10 W profile on Nano. Use the full-performance profile during installation and validation with adequate cooling. A 5 W profile can be evaluated later.

## 4. Install the base stack and models

```bash
./scripts/bootstrap_jetson.sh
./scripts/diagnose.sh
```

Run the bootstrap as the regular login user, without putting `sudo` before the script name. It requests `sudo` only for package installation and system configuration and can safely be run again.

The bootstrap does not install `python3-pycuda`, because that package has no installation candidate on some JetPack 4.6.1 images. Instead, it:

- installs `python3-pip`, Python headers, the compiler toolchain and Boost libraries;
- adds `/usr/local/cuda/bin` and `/usr/local/cuda/lib64` to `~/.bashrc` without duplicating entries;
- installs Python 3.6-compatible build tools, including Cython 0.29.36 and NumPy;
- builds PyCUDA 2022.1 from source against the CUDA 10.2 headers and libraries;
- verifies that PyCUDA imports and reports the CUDA version.

It then installs OpenCV, Tesseract with Polish data, ALSA and the remaining tools; downloads the archived ARM64 Piper binary and Polish `gosia-medium` voice; and creates the local configuration. After the first successful run, load the saved variables in the current terminal (new terminals do this automatically):

```bash
source ~/.bashrc
python3 -c "import pycuda.driver as cuda; print('CUDA version:', cuda.get_version())"
```

If an earlier bootstrap stopped only at PyCUDA, rerunning it is the simplest recovery. The following block is the manual equivalent of its PyCUDA build stage and can be used for diagnosis:

```bash
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_INC_DIR=/usr/local/cuda/include
export CUDA_NDARRAY_CUDA_H=1

python3 -m pip install --user --upgrade "pip<22" "setuptools<60" "wheel<0.38"
python3 -m pip install --user "Cython==0.29.36" numpy
python3 -m pip install --user --no-cache-dir \
  --global-option=build_ext \
  --global-option="-I/usr/local/cuda/include" \
  --global-option="-L/usr/local/cuda/lib64" \
  "pycuda==2022.1"
python3 -c "import pycuda.driver as cuda; print('CUDA version:', cuda.get_version())"
```

Do not run `sudo pip3`; Python packages are installed for the same user that will run the service. Inspect or edit the generated configuration:

```bash
nano config/config.json
```

## 5. Validate each peripheral

### Camera

```bash
v4l2-ctl --list-devices
ls -l /dev/video*
fswebcam -r 1280x720 --no-banner tmp/camera.jpg
```

Open `tmp/camera.jpg` over SFTP or temporarily on the desktop. If the camera is not `/dev/video0`, change `camera.device` in `config/config.json`. The included capture code is for UVC/V4L2; a CSI camera needs an appropriate GStreamer pipeline.

### Audio

```bash
aplay -l
speaker-test -t wav -c 2
aplay /usr/share/sounds/alsa/Front_Center.wav
```

Use `alsamixer` to unmute and set gain. If the wrong output is selected, put the stable identifier from `aplay -L` into `speech.aplay_device`, for example `plughw:CARD=Device,DEV=0`.

### Piper

```bash
python3 src/say.py
```

### Polish OCR

Place a well-lit printed page parallel to the camera:

```bash
python3 src/main_demo.py --action read
```

The processed frame is kept as `tmp/page_prepared.png` for diagnosis.

## 6. Build YOLOv5n TensorRT

The conversion script in the selected YOLO wrapper imports PyTorch. For JetPack 4.x/Python 3.6, install the final compatible NVIDIA community wheel, PyTorch 1.10. The source announcement is the [NVIDIA Jetson forum post](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048/1276); the linked wheel is:

```bash
sudo apt-get install -y libopenblas-dev libopenmpi-dev
python3 -m pip install --user --upgrade "pip<22"
wget -O /tmp/torch-1.10.0-cp36-cp36m-linux_aarch64.whl \
  https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl
python3 -m pip install --user /tmp/torch-1.10.0-cp36-cp36m-linux_aarch64.whl
python3 -c "import torch; print(torch.__version__)"
```

Build the engine on the target Nano:

```bash
cd ~/nano-speaker
./scripts/build_yolo.sh
python3 src/main_demo.py --action describe
```

The script clones `mailrocketsystems/JetsonYolov5`, converts the included YOLOv5n weights, compiles its TensorRT plugin and serializes `yolov5n.engine`. This can take time. If compilation is killed, confirm `free -h` shows swap and run:

```bash
BUILD_JOBS=1 ./scripts/build_yolo.sh
```

Do not copy an engine built for a different TensorRT/CUDA/GPU stack. Rebuild it after relevant JetPack/TensorRT changes. The external wrapper is GPL-3.0 and remains a separate ignored checkout; review its license for redistribution.

## 7. Test the integrated controller

```bash
python3 -m unittest discover -s tests -v
python3 src/main_demo.py --mode keyboard
```

Press `1` to describe one frame, `2` to read a page and `q` to exit. Do not proceed to GPIO until both actions reliably return to the prompt.

## 8. Add two GPIO buttons

Power off and disconnect the supply. The defaults use physical header numbering:

- describe button: physical pin 11 ↔ GND (for example pin 9);
- read button: physical pin 13 ↔ GND (for example pin 14).

Use one normally-open button per signal. Internal pull-ups are enabled; no external voltage is required. Jetson GPIO is 3.3 V only and is not 5 V tolerant.

The bootstrap already installed Jetson.GPIO, copied its udev permissions rule and added the user to the `gpio` group. Reboot so the new membership applies (the following commands are only needed if that step was skipped):

```bash
python3 -m pip install --user "Jetson.GPIO==2.1.6"
sudo groupadd -f -r gpio
sudo usermod -a -G gpio "$USER"
GPIO_RULE="$(python3 -c 'import Jetson.GPIO, os; print(os.path.join(os.path.dirname(Jetson.GPIO.__file__), "99-gpio.rules"))')"
sudo cp "$GPIO_RULE" /etc/udev/rules.d/99-gpio.rules
sudo reboot
```

Enable GPIO in `config/config.json` and test in the foreground:

```json
"gpio": {
  "enabled": true,
  "describe_pin": 11,
  "read_pin": 13,
  "board_numbering": true,
  "bounce_time_ms": 500
}
```

```bash
cd ~/nano-speaker
python3 src/main_demo.py --mode gpio
```

## 9. Enable headless autostart

```bash
cd ~/nano-speaker
./scripts/install_service.sh gpio
systemctl status nano-speaker.service
journalctl -u nano-speaker.service -f
```

Reboot and test both buttons. Only after this succeeds, optionally disable the graphical desktop:

```bash
sudo systemctl set-default multi-user.target
sudo reboot
```

Restore it later with `sudo systemctl set-default graphical.target`. Stop or remove the assistant with:

```bash
sudo systemctl disable --now nano-speaker.service
```

## 10. Mobile power acceptance test

Finish all software work using a proven mains supply. For mobile use, the Nano must receive a stable, correctly polarized 5 V rail with enough transient current. A possible architecture is USB-C PD power bank → 9/12 V trigger → fused 5 V step-down rated for the load → Nano barrel input, but component ratings, cable loss, converter cooling and board revision all matter.

Before connecting the Nano, measure the converter output under load and configure the carrier board for barrel power as documented for its revision (commonly the J48 jumper). Never connect raw PD voltage to the 5 V input and never adjust the converter while attached. Monitor the final system:

```bash
tegrastats
sudo dmesg -w
```

Acceptance means repeated OCR and YOLO actions do not trigger undervoltage symptoms, resets, thermal throttling or USB disconnects, and the enclosure does not obstruct airflow.

## Final acceptance checklist

- [ ] Correct JetPack/L4T, Python, CUDA and TensorRT versions.
- [ ] Swap active; adequate free storage.
- [ ] Sharp camera capture and stable device index.
- [ ] Clear audio after reboot with a stable ALSA identifier.
- [ ] Polish Piper prompt and Polish Tesseract result.
- [ ] TensorRT engine built and one-shot description works.
- [ ] Unit tests pass and both controller modes recover after actions.
- [ ] GPIO buttons work without false repeats.
- [ ] Service starts after a cold boot; logs contain no restart loop.
- [ ] Power and thermal test passes under repeated maximum load.
