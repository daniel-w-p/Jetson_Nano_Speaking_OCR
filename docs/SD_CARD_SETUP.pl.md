# Od czystej karty microSD do działającego asystenta

[English](SD_CARD_SETUP.md) · [README](../README.pl.md)

Instrukcja dotyczy **Jetson Nano 4 GB Developer Kit** i JetPack **4.6.1**. Przeczytaj polecenia przed wykonaniem. Flashowanie usuwa zawartość wybranej karty, a błąd w zasilaniu może uszkodzić płytkę.

## 1. Przygotuj kartę na innym komputerze

Potrzebujesz karty microSD 64–128 GB A1/A2, czytnika, internetu, klawiatury, monitora do pierwszego uruchomienia (alternatywnie konfiguracji szeregowej), kamery, głośnika, aktywnego chłodzenia i pewnego zasilacza Nano.

1. Pobierz obraz **Jetson Nano Developer Kit** z [oficjalnej strony JetPack 4.6.1](https://developer.nvidia.com/embedded/jetpack-sdk-461). Nie wybieraj obrazu Nano 2 GB ani JetPack 5/6.
2. Zainstaluj [balenaEtcher](https://etcher.balena.io/) lub użyj [instrukcji NVIDIA dla swojego systemu](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit).
3. Jako źródło wskaż pobrany ZIP, bardzo uważnie wybierz kartę microSD, wykonaj zapis i walidację.
4. Bezpiecznie wysuń kartę i włóż ją do wyłączonego Nano.

## 2. Pierwsze uruchomienie

Podłącz chłodzenie, monitor, klawiaturę, sieć i sprawdzony zasilacz. Przejdź kreator Ubuntu: zaakceptuj licencję, wybierz język i strefę, utwórz użytkownika i hasło oraz rozszerz system plików na kartę. Otwórz terminal:

```bash
sudo apt-get update
sudo apt-get install -y git curl wget htop nano v4l-utils alsa-utils
sudo reboot
```

Podczas odtwarzania projektu zachowaj stałą bazę JetPack/L4T; nie wykonuj w ciemno pełnej aktualizacji dystrybucji. Aktualizacje bezpieczeństwa NVIDIA wdrażaj później jako osobno przetestowaną zmianę, a po zmianie stosu ponownie zbuduj silnik TensorRT.

Po restarcie:

```bash
cat /etc/nv_tegra_release       # oczekiwane R32.7.1 dla JetPack 4.6.1
python3 --version               # oczekiwany Python 3.6.x
/usr/local/cuda/bin/nvcc --version  # oczekiwana CUDA 10.2
dpkg -l | grep -E 'tensorrt|nvinfer'
```

Do pracy zdalnej sprawdź adres przez `hostname -I` i połącz się poleceniem `ssh UZYTKOWNIK@ADRES`.

## 3. Sklonuj projekt i przygotuj pamięć

```bash
git clone https://github.com/daniel-w-p/Jetson_Nano_Speaking_OCR.git ~/nano-speaker
cd ~/nano-speaker
chmod +x scripts/*.sh
./scripts/create_swap.sh 4G
sudo nvpmodel -m 0
sudo jetson_clocks
```

Profil `nvpmodel -m 0` to 10 W na Nano. Użyj go podczas instalacji i testów z dobrym chłodzeniem. Tryb 5 W oceniaj dopiero później.

## 4. Zainstaluj bazę i modele

```bash
./scripts/bootstrap_jetson.sh
./scripts/diagnose.sh
nano config/config.json
```

Uruchom bootstrap jako zwykły użytkownik, bez `sudo` przed nazwą skryptu. Skrypt sam używa `sudo` tylko do instalacji pakietów i konfiguracji systemowej. Można go bezpiecznie uruchamiać ponownie.

Bootstrap nie instaluje `python3-pycuda`, ponieważ ten pakiet nie ma kandydata w części obrazów JetPack 4.6.1. Zamiast tego:

- instaluje `python3-pip`, nagłówki Pythona, kompilator i biblioteki Boost;
- dopisuje do `~/.bashrc` ścieżki `/usr/local/cuda/bin` i `/usr/local/cuda/lib64` (bez tworzenia duplikatów);
- instaluje wersje narzędzi zgodne z Pythonem 3.6, w tym Cython 0.29.36 i NumPy;
- buduje PyCUDA 2022.1 ze źródeł z nagłówkami i bibliotekami CUDA 10.2;
- sprawdza import PyCUDA i zgłaszaną wersję CUDA.

Następnie instaluje OpenCV, Tesseract z językiem polskim, ALSA i pozostałe narzędzia; pobiera archiwalny Piper ARM64 oraz głos `gosia-medium`; tworzy lokalną konfigurację. Po pierwszym poprawnym wykonaniu wczytaj zapisane zmienne w bieżącym terminalu (nowy terminal zrobi to automatycznie):

```bash
source ~/.bashrc
python3 -c "import pycuda.driver as cuda; print('Wersja CUDA:', cuda.get_version())"
```

Jeśli bootstrap zatrzymał się wcześniej wyłącznie na PyCUDA, najprościej uruchomić go ponownie. Poniższy blok jest ręcznym odpowiednikiem samego etapu kompilacji i może służyć do diagnostyki:

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
python3 -c "import pycuda.driver as cuda; print('Wersja CUDA:', cuda.get_version())"
```

Nie uruchamiaj `sudo pip3`: pakiety Pythona są instalowane dla użytkownika, pod którym działa później usługa.

## 5. Sprawdź każde urządzenie

### Kamera

```bash
v4l2-ctl --list-devices
ls -l /dev/video*
fswebcam -r 1280x720 --no-banner tmp/camera.jpg
```

Obejrzyj zdjęcie przez SFTP lub tymczasowo w środowisku graficznym. Jeśli kamera nie jest `/dev/video0`, zmień `camera.device`. Kod obsługuje USB UVC/V4L2; kamera CSI wymaga odpowiedniego pipeline GStreamer.

### Dźwięk

```bash
aplay -l
speaker-test -t wav -c 2
aplay /usr/share/sounds/alsa/Front_Center.wav
```

W `alsamixer` wyłącz wyciszenie i ustaw poziom. Jeżeli wybrano złe wyjście, wpisz stabilny identyfikator z `aplay -L` do `speech.aplay_device`, np. `plughw:CARD=Device,DEV=0`.

### Piper i polski OCR

```bash
python3 src/say.py
# Umieść równomiernie oświetloną kartkę równolegle do kamery:
python3 src/main_demo.py --action read
```

Obraz po obróbce zostaje w `tmp/page_prepared.png`.

## 6. Zbuduj YOLOv5n TensorRT

Konwersja wag wymaga PyTorch. Dla JetPack 4.x i Pythona 3.6 zainstaluj końcową zgodną wersję 1.10 z [ogłoszenia na forum NVIDIA](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048/1276):

```bash
sudo apt-get install -y libopenblas-dev libopenmpi-dev
python3 -m pip install --user --upgrade "pip<22"
wget -O /tmp/torch-1.10.0-cp36-cp36m-linux_aarch64.whl \
  https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl
python3 -m pip install --user /tmp/torch-1.10.0-cp36-cp36m-linux_aarch64.whl
python3 -c "import torch; print(torch.__version__)"
```

Zbuduj silnik na docelowym Nano:

```bash
cd ~/nano-speaker
./scripts/build_yolo.sh
python3 src/main_demo.py --action describe
```

Skrypt pobiera `mailrocketsystems/JetsonYolov5`, konwertuje dołączone wagi, kompiluje plugin TensorRT i tworzy `yolov5n.engine`. Gdy kompilator zostanie ubity, sprawdź `free -h` i użyj:

```bash
BUILD_JOBS=1 ./scripts/build_yolo.sh
```

Nie kopiuj silnika z innej wersji CUDA/TensorRT/GPU; po zmianie stosu zbuduj go ponownie. Zewnętrzny wrapper ma licencję GPL-3.0 i pozostaje osobnym, ignorowanym checkoutem — sprawdź warunki przed redystrybucją.

## 7. Test integracji

```bash
python3 -m unittest discover -s tests -v
python3 src/main_demo.py --mode keyboard
```

`1` opisuje pojedynczą klatkę, `2` czyta kartkę, `q` kończy. Nie przechodź do GPIO, dopóki oba działania nie wracają niezawodnie do promptu.

## 8. Dwa przyciski GPIO

Wyłącz zasilanie. Domyślna numeracja dotyczy fizycznych pinów złącza:

- „opisz”: pin 11 ↔ GND, np. pin 9;
- „czytaj”: pin 13 ↔ GND, np. pin 14.

Użyj dwóch przycisków chwilowych NO. Program włącza wewnętrzne podciąganie, więc nie podawaj napięcia. GPIO Jetsona pracuje z 3,3 V i nie toleruje 5 V.

```bash
python3 -m pip install --user "Jetson.GPIO==2.1.6"
sudo groupadd -f -r gpio
sudo usermod -a -G gpio "$USER"
GPIO_RULE="$(python3 -c 'import Jetson.GPIO, os; print(os.path.join(os.path.dirname(Jetson.GPIO.__file__), "99-gpio.rules"))')"
sudo cp "$GPIO_RULE" /etc/udev/rules.d/99-gpio.rules
sudo reboot
```

Skrypt `bootstrap_jetson.sh` wykonał już instalację biblioteki, reguły udev i grupy; powyższe polecenia są potrzebne tylko wtedy, gdy ten krok pominięto. Restart uaktywnia nowe członkostwo w grupie.

W `config/config.json` ustaw `gpio.enabled` na `true`, zweryfikuj piny i uruchom test:

```bash
cd ~/nano-speaker
python3 src/main_demo.py --mode gpio
```

## 9. Autostart bez ekranu

```bash
cd ~/nano-speaker
./scripts/install_service.sh gpio
systemctl status nano-speaker.service
journalctl -u nano-speaker.service -f
```

Uruchom Nano ponownie i sprawdź przyciski. Dopiero po sukcesie możesz wyłączyć pulpit:

```bash
sudo systemctl set-default multi-user.target
sudo reboot
```

Pulpit przywrócisz przez `sudo systemctl set-default graphical.target`, a usługę wyłączysz poleceniem `sudo systemctl disable --now nano-speaker.service`.

## 10. Odbiór zasilania mobilnego

Oprogramowanie przygotuj na pewnym zasilaczu sieciowym. Mobilnie Nano musi otrzymać stabilne, poprawnie spolaryzowane 5 V z zapasem prądu chwilowego. Możliwy tor to powerbank USB-C PD → trigger 9/12 V → zabezpieczona przetwornica step-down 5 V → barrel jack, ale znaczenie mają parametry elementów, spadki na kablach, chłodzenie przetwornicy i rewizja płytki.

Przed podłączeniem Nano zmierz napięcie pod obciążeniem i zgodnie z instrukcją rewizji carrier board wybierz zasilanie barrel (zwykle zworką J48). Nie podawaj surowego napięcia PD na wejście 5 V i nie reguluj przetwornicy podłączonej do płytki. Obserwuj:

```bash
tegrastats
sudo dmesg -w
```

Test jest zaliczony, gdy wielokrotne OCR i YOLO nie powodują resetów, odłączania USB ani throttlingu, a obudowa nie blokuje chłodzenia.

## Lista odbiorcza

- [ ] Poprawne wersje JetPack/L4T, Python, CUDA i TensorRT.
- [ ] Aktywny swap i wystarczająco dużo miejsca.
- [ ] Ostry obraz i stabilny indeks kamery.
- [ ] Dźwięk po restarcie przez stabilny identyfikator ALSA.
- [ ] Polski komunikat Piper i wynik Tesseract PL.
- [ ] Zbudowany silnik TensorRT i działający pojedynczy opis.
- [ ] Testy przechodzą, oba tryby wracają do gotowości.
- [ ] Przyciski działają bez fałszywych powtórzeń.
- [ ] Usługa startuje po zimnym rozruchu bez pętli restartów.
- [ ] Zasilanie i temperatury są stabilne przy powtarzanym pełnym obciążeniu.
