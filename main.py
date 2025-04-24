import serial
import time
import os
import json
from datetime import datetime
from typing import Union
from pathlib import Path
import numpy as np
import subprocess
import sys
from analyzer import set_ceiling_color, analyze_image

# --- Serial Configuration ---
SERIAL_PORT = '/dev/cu.usbmodem64E8335D24DC2'  # Update with your macOS serial port
BAUD_RATE = 921600                             # Must match the rate in your Arduino sketch
TIMEOUT = 1                                    # Serial timeout in seconds

ZOOM_LINK   = "https://zoom.us/j/1234567890"         # put your real link here


# --- File Saving Configuration ---
SAVE_FOLDER = os.path.join(".", "images")
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)
    print("Created folder:", SAVE_FOLDER)

# --- Drink Tracking Configuration ---
previous_drink = None
coffee_count = 0
water_count = 0
fruit_punch_count = 0
MAX_LEDS = 4

def update_drink_counts(current_drink):
    global previous_drink, coffee_count, water_count, fruit_punch_count
    
    if previous_drink is not None and previous_drink != current_drink:
        if previous_drink == "coffee":
            coffee_count = min(coffee_count + 1, MAX_LEDS)
        elif previous_drink == "water":
            water_count = min(water_count + 1, MAX_LEDS)
        elif previous_drink == "fruit_punch":
            fruit_punch_count = min(fruit_punch_count + 1, MAX_LEDS)

def send_drink_to_arduino(drink_type, count):
    command = f"{drink_type},{count}\n"
    ser.write(command.encode('utf-8'))
    print(f"Sent to Arduino: {command.strip()}")

def update_led(drink_type):
    """
    Sends the current drink type and its count to Arduino to update LED indicators.
    """
    if drink_type == "coffee":
        send_drink_to_arduino("coffee", coffee_count)
    elif drink_type == "water":
        send_drink_to_arduino("water", water_count)
    elif drink_type == "fruit_punch":
        send_drink_to_arduino("fruit punch", fruit_punch_count)

# --- Open the Serial Port ---
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
    print("Serial port opened on", SERIAL_PORT)
    print("\n--------------------------------------------------\n\n")
except serial.SerialException as e:
    print("Error opening serial port:", e)
    exit(1)

# Allow the Arduino a moment to reset and start transmitting
time.sleep(2)

def capture_jpeg():
    """
    Reads from the serial port until a complete JPEG image is received.
    A valid JPEG image begins with 0xFF 0xD8 and ends with 0xFF 0xD9.
    Returns the image data as bytes or None on timeout.
    """
    data = bytearray()
    start_found = False
    start_time = time.time()
    
    while True:
        byte = ser.read(1)
        if not byte:
            if time.time() - start_time > TIMEOUT:
                return None
            continue

        if not start_found:
            if byte[0] == 0xFF:
                next_byte = ser.read(1)
                if next_byte and next_byte[0] == 0xD8:
                    data.extend([0xFF, 0xD8])
                    start_found = True
        else:
            data.append(byte[0])
            if len(data) >= 2 and data[-2] == 0xFF and data[-1] == 0xD9:
                break
    return bytes(data)

def save_image(data, filename):
    """Saves binary data to a file."""
    with open(filename, "wb") as f:
        f.write(data)
    print(f"INFO: Image saved as {filename} (length: {len(data)} bytes)\n")

def initialize_ceiling_image():
    """
    Captures the first image from the camera, saves it as the ceiling image in ./data,
    sets it as the ceiling color reference, and sends a "TEST" command to Arduino.
    """

    ser.write("TEST\n".encode('utf-8'))
    print("Sent to Arduino: TEST")

    ceiling_image_filename = os.path.join(".", "data", "empty.jpg")

    print("Capturing ceiling image...")
    while not os.path.exists(ceiling_image_filename):
        ceiling_image_data = capture_jpeg()
        if ceiling_image_data:
            save_image(ceiling_image_data, ceiling_image_filename)

    set_ceiling_color(ceiling_image_filename)
    print("Ceiling color set.")

def zoom():
    line = ser.readline().decode("utf-8", errors="ignore").strip()

    if line:
        #print(f"DEBUG: Received line: '{line}'")  # Uncommented for debugging
        if "BUTTON_PRESSED" in line.upper():
            print("Button press detected. Launching Zoom...")
            try:
                if sys.platform.startswith('darwin'):
                    subprocess.run(['open', ZOOM_LINK], check=True)
                elif sys.platform.startswith('win'):
                    subprocess.run(['start', ZOOM_LINK], shell=True, check=True)
                elif sys.platform.startswith('linux'):
                    subprocess.run(['xdg-open', ZOOM_LINK], check=True)
                else:
                    print("ERROR: Unsupported OS for launching Zoom.")
            except subprocess.CalledProcessError as e:
                print(f"ERROR: Failed to launch Zoom: {e}")


if __name__ == "__main__":
    # Initialize ceiling image and notify Arduino
    initialize_ceiling_image()

    while True:
        zoom()

        jpeg_data = capture_jpeg()
        if jpeg_data is None:
            continue

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_image_filename = os.path.join(SAVE_FOLDER, f"capture_{timestamp}.jpg")
        try:
            save_image(jpeg_data, raw_image_filename)
        except Exception as e:
            print("ERROR: Failed to save image:", e)
            continue

        # saving final image into folder
        final_folder = "final"
        if not os.path.exists(final_folder):
            os.makedirs(final_folder)

        final_image_path = os.path.join(final_folder, f"capture_{timestamp}.jpg")
        drink_type = analyze_image(raw_image_filename, final_image_path)
        print(f"Analyzed drink type: {drink_type}")

        if drink_type == "nothing":
            print("No drink detected, skipping LED update.")
        else:
            # Added code after saving the final image
            current_drink = drink_type

            update_drink_counts(current_drink)

            if previous_drink and previous_drink != current_drink:
                update_led(previous_drink)

            previous_drink = current_drink

        response_json = {
            "timestamp": timestamp,
            "image_path": raw_image_filename,
            "drink_counts": {
                "coffee": coffee_count,
                "water": water_count,
                "fruit_punch": fruit_punch_count
            }
        }
        print(json.dumps(response_json, indent=2))
        print("\n--------------------------------------------------\n")
