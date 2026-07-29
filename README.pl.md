# Jetson Nano Speaking OCR

**Lokalny, bezekranowy asystent wzrokowy dla Jetson Nano 4 GB: opisuje obiekty i czyta na głos polski tekst drukowany.**

[English (domyślny)](README.md) · [Instalacja od czystej karty SD](docs/SD_CARD_SETUP.pl.md)

## Działanie

```text
Opis:    kamera → YOLOv5n → TensorRT → polski komunikat → Piper → głośnik
Czytanie: kamera → szarość/Canny → kontury → korekcja perspektywy
          → próg adaptacyjny → Tesseract (pol) → Piper → głośnik
```

- Po instalacji działa całkowicie lokalnie.
- Obsługa przez SSH/klawiaturę albo dwa przyciski GPIO.
- Zgodność ze starym stosem JetPack 4.6.1, Ubuntu 18.04 i Python 3.6.
- Kamera i model Pipera pozostają aktywne przez całą sesję, co eliminuje ciągłe wybudzanie USB i przeładowywanie głosu.
- TensorRT YOLO działa w osobnym procesie: kolejne opisy używają go ponownie, a wybór OCR kończy proces i zwalnia pamięć CUDA.

OCR szuka największego, odpowiednio dużego i wypukłego czworokąta, porządkuje
jego narożniki i prostuje kartkę w pełnej rozdzielczości. Jeśli nie widzi całej
kartki albo transformacja się nie powiedzie, bezpiecznie przetwarza całą klatkę
i nadal uruchamia Tesseract.

> To demonstrator technologii wspomagającej, a nie certyfikowane urządzenie nawigacyjne lub bezpieczeństwa. Wyniki mogą być błędne; nie należy polegać na nich w ruchu drogowym, przy lekach ani w innych sytuacjach krytycznych.

## Sprzęt

- Jetson Nano 4 GB Developer Kit i aktywne chłodzenie;
- karta microSD 64–128 GB A1/A2;
- kamera USB UVC z autofocusem;
- głośnik USB lub karta dźwiękowa USB;
- do uruchomienia stabilny zasilacz 5 V / 4 A;
- opcjonalnie dwa przyciski chwilowe NO;
- mobilnie: odpowiedni powerbank oraz poprawnie zaprojektowany i zabezpieczony tor 5 V.

Nie podawaj 9 V ani 12 V na wejście 5 V Nano. Za wyzwalaczem PD musi znaleźć się poprawnie ustawiona przetwornica step-down. Przed podłączeniem płytki zmierz napięcie i polaryzację oraz sprawdź w instrukcji carrier board wybór zasilania barrel (zwykle zworka J48).

## Szybki start

Pełna procedura od pustej karty znajduje się w dokumencie [Instalacja od czystej karty SD](docs/SD_CARD_SETUP.pl.md).

```bash
git clone https://github.com/daniel-w-p/Jetson_Nano_Speaking_OCR.git ~/nano-speaker
cd ~/nano-speaker
chmod +x scripts/*.sh
./scripts/create_swap.sh
./scripts/bootstrap_jetson.sh
python3 src/say.py
python3 src/main_demo.py --action read
```

Detekcja obiektów wymaga dodatkowo zbudowania silnika TensorRT na docelowym Nano:

```bash
# Najpierw zainstaluj wheel PyTorch zgodny z JetPack 4 — szczegóły w instrukcji SD.
./scripts/build_yolo.sh
python3 src/main_demo.py --action describe
python3 src/main_demo.py --mode keyboard
```

## Konfiguracja

Skrypt startowy kopiuje `config/config.example.json` do ignorowanego przez Git pliku `config/config.json`. Można w nim zmienić indeks kamery, format UVC/FPS, urządzenie ALSA, parametry OCR, próg YOLO, ścieżki modeli i piny GPIO. Domyślna sesja kamery wymusza `MJPG`, 1280×720 przy 15 FPS, odrzuca 30 klatek rozgrzewkowych i stale przechowuje najnowszą klatkę.

Istniejący lokalny `config.json` nie wymaga migracji. Program głęboko łączy go
z wartościami domyślnymi, więc brak nowych kluczy OCR jest automatycznie
uzupełniany. Pełny zestaw znajduje się w
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
    "debug_images": false
  }
}
```

- `max_width` ogranicza tylko kopię używaną do Canny i konturów; właściwa
  transformacja korzysta z pełnej klatki.
- `min_page_area_ratio` określa minimalną część obrazu zajmowaną przez kartkę.
- `approx_epsilon_ratio` steruje przybliżaniem konturu czworokątem.
- `clipped_page_fallback` pozwala domknąć widoczne krawędzie kartki granicami
  kadru, gdy część strony wychodzi poza obraz.
- `frame_border_thickness` określa grubość pomocniczej ramki na mapie Canny;
  wartość `3` łączy krawędzie kończące się kilka pikseli od brzegu.
- `scale_factor`, `threshold_block_size` i `threshold_c` przygotowują
  wyprostowany obraz bezpośrednio dla Tesseract.
- Ustawienie `page_detection.enabled` na `false` wymusza OCR całej klatki.

Identyfikator głośnika sprawdzisz poleceniem `aplay -l`. Przykład:

```json
"aplay_device": "plughw:CARD=Device,DEV=0"
```

Domyślne piny fizyczne to 11 i 13 (`GPIO.BOARD`). Każdy przycisk NO podłącz między osobny pin sygnałowy i GND; program włącza wewnętrzne podciąganie. Nigdy nie podłączaj 5 V do GPIO.

## Autostart bez ekranu

Najpierw przetestuj oddzielnie kamerę, mowę, OCR i YOLO. Następnie włącz GPIO w konfiguracji:

```bash
# bootstrap_jetson.sh zainstalował już Jetson.GPIO, regułę udev i grupę.
sudo reboot
cd ~/nano-speaker
./scripts/install_service.sh gpio
journalctl -u nano-speaker.service -f
```

Instalator automatycznie podstawia bieżącego użytkownika i katalog repozytorium.

## Testy i diagnostyka

```bash
python3 -m unittest discover -s tests -v
./scripts/diagnose.sh
fswebcam -r 1280x720 --no-banner tmp/camera.jpg
speaker-test -t wav -c 2
tesseract tmp/camera.jpg stdout -l pol --psm 6
```

Do pierwszych testów na Jetsonie ustaw tymczasowo:

```json
"page_detection": {
  "debug_images": true
}
```

Po każdej akcji odczytu program nadal nadpisuje `tmp/page_raw.jpg` i
`tmp/page_prepared.png`. Przy włączonym debugowaniu powstają dodatkowo:

- `tmp/page_edges.png` — krawędzie wykryte przez Canny;
- `tmp/page_detected.jpg` — surowa klatka z obrysem kartki: zielonym dla
  pełnego konturu albo pomarańczowym dla narożników dopowiedzianych z granic
  kadru;
- `tmp/page_warped.png` — wyprostowana kartka przed progowaniem; przy
  fallbacku nieaktualny plik jest usuwany.

Sprawdzaj je w tej kolejności: `page_raw` → `page_edges` → `page_detected` →
`page_warped` → `page_prepared`. Pozwala to ustalić, czy problem powstał przy
rejestracji obrazu, detekcji krawędzi, wyborze konturu, transformacji czy
binaryzacji. Po zakończeniu strojenia wyłącz `debug_images`, aby ograniczyć
zapisy na karcie SD.

Najczęstsze problemy:

- brak klatki: ustaw właściwe `camera.device` po `v4l2-ctl --list-devices`;
- brak dźwięku: ustaw `speech.aplay_device` i sprawdź `alsamixer`;
- słaby OCR: włącz diagnostykę, zapewnij równomierne światło i usuń odblaski;
  jeśli obrys jest błędny, dostrój progi Canny lub `min_page_area_ratio`.
  Pomarańczowy obrys jest oczekiwany, gdy kartka wychodzi poza kadr;
- przerwana kompilacja: sprawdź swap i użyj `BUILD_JOBS=1 ./scripts/build_yolo.sh`;
- resety lub throttling: uruchom `tegrastats`, popraw zasilanie i chłodzenie.

Docelowy stos to JetPack 4.6.1 (L4T 32.7.1, CUDA 10.2, TensorRT 8.2.1), Python 3.6, Tesseract PL, Piper `2023.11.14-2` i wrapper `mailrocketsystems/JetsonYolov5`. Silnik TensorRT jest artefaktem zależnym od platformy i nie jest przechowywany w repozytorium.

Źródła: [NVIDIA JetPack 4.6.1](https://developer.nvidia.com/embedded/jetpack-sdk-461), [uruchomienie Nano](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit), [Piper](https://github.com/rhasspy/piper/releases/tag/2023.11.14-2), [JetsonYolov5](https://github.com/mailrocketsystems/JetsonYolov5).

Kod projektu jest dostępny na licencji [MIT](LICENSE). Pobrane modele i zależności zachowują własne licencje.
