import os
from PIL import Image
import pyautogui
import mss
import pygetwindow as gw
import time

path = "camera_calibration\images"
arma3_window = gw.getWindowsWithTitle('Arma 3 "')[0]
screen_width, screen_height = pyautogui.size()
monitor = {
    "top": 0,
    "left": screen_width // 4,
    "width": screen_width // 2,
    "height": screen_height // 4,
}


def simulate_upper_half_screenshot():
    with mss.mss() as sct:
        sct_img = sct.grab(monitor)
        img = Image.frombytes("RGB", sct_img.size, sct_img.rgb)
        return img


def save_screenshot_with_unique_name(screenshot):
    if not os.path.exists(path):
        os.makedirs(path)

    base_name = path + "\{}.png"
    index = 1
    while os.path.exists(base_name.format(index)):
        index += 1
    screenshot.save(base_name.format(index))


arma3_window.activate()
time.sleep(0.1)
screenshot = simulate_upper_half_screenshot()
save_screenshot_with_unique_name(screenshot)
