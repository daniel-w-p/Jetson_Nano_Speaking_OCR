"""Headless/interactive event loop for the two assistant modes."""

import argparse
import signal
import sys
import time
import traceback

from describe_once import describe_once
from read_page import read_once
from say import say


def run_action(action):
    try:
        if action == "describe":
            return describe_once()
        if action == "read":
            return read_once()
        raise ValueError("Unknown action: {0}".format(action))
    except Exception as error:
        traceback.print_exc()
        try:
            say("Wystąpił błąd. " + str(error))
        except Exception:
            traceback.print_exc()
        return None


def keyboard_loop():
    say("Asystent gotowy. Naciśnij jeden, aby opisać obraz, albo dwa, aby przeczytać kartkę.")
    while True:
        command = input("1=describe, 2=read, q=quit: ").strip().lower()
        if command == "1":
            run_action("describe")
        elif command == "2":
            run_action("read")
        elif command == "q":
            say("Kończę działanie.")
            return


def gpio_loop():
    from config import load_config
    try:
        import Jetson.GPIO as GPIO
    except ImportError:
        raise RuntimeError("Jetson.GPIO is not installed")
    settings = load_config()["gpio"]
    if not settings.get("enabled", False):
        raise RuntimeError("GPIO mode is disabled in config/config.json")
    GPIO.setmode(GPIO.BOARD if settings["board_numbering"] else GPIO.BCM)
    pins = {int(settings["describe_pin"]): "describe", int(settings["read_pin"]): "read"}
    for pin in pins:
        GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    busy = [False]

    def pressed(channel):
        if busy[0]:
            return
        busy[0] = True
        try:
            run_action(pins[channel])
        finally:
            busy[0] = False

    try:
        for pin in pins:
            GPIO.add_event_detect(
                pin, GPIO.FALLING, callback=pressed,
                bouncetime=int(settings["bounce_time_ms"])
            )
        say("Asystent gotowy.")
        signal.pause()
    finally:
        GPIO.cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("keyboard", "gpio"), default="keyboard")
    parser.add_argument("--action", choices=("describe", "read"))
    args = parser.parse_args()
    if args.action:
        run_action(args.action)
    elif args.mode == "gpio":
        gpio_loop()
    else:
        keyboard_loop()


if __name__ == "__main__":
    main()
