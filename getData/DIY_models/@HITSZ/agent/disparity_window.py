from camera_parameters import StereoCameraParameters
from pynput.keyboard import Controller, KeyCode
from PyQt5 import QtCore, QtWidgets
from PIL import Image, ImageTk
from vispy import app, scene
from tkinter import Toplevel
import pygetwindow as gw
import tkinter as tk
import numpy as np
import threading
import pyautogui
import pickle
import time
import zmq
import mss
import cv2
import os


class DisparityWindow:
    def __init__(self):
        self.running = True
        self.pid = os.getpid()
        self.arma3_window = gw.getWindowsWithTitle('Arma 3 "')[0]
        self.screen_width, self.screen_height = pyautogui.size()
        self.window_width = self.screen_width // 4
        self.window_height = self.screen_height // 4

        self.flag_focus = False
        self.flag_capture = False
        self.disparity_pointcloud = np.zeros(
            (self.window_height, self.window_width), dtype=np.uint8
        )
        self.disparity_image = np.zeros(
            (self.window_height, self.window_width), dtype=np.uint8
        )
        self.lock = threading.Lock()

        self.init_disparity_window()
        self.init_zmq()
        self.init_stereo_params()
        self.start_threads()

    def init_disparity_window(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.disparity_window = self.create_window(
            "disparity",
            self.window_width,
            self.window_height,
            self.screen_width - self.window_width,
            0,
        )
        self.disparity_label = tk.Label(self.disparity_window)
        self.disparity_label.pack()

    def create_window(self, title, window_width, window_height, x, y):
        window = Toplevel(self.root)
        window.title(title)
        window.overrideredirect(True)
        window.attributes("-topmost", True)
        window.bind("<Escape>", self.close_window)
        window.geometry(f"{window_width}x{window_height}+{x}+{y}")
        return window

    def init_zmq(self):
        context = zmq.Context()
        self.arma3_sub_socket = context.socket(zmq.SUB)
        self.arma3_sub_socket.connect("tcp://localhost:114")
        self.arma3_sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")

        self.arma3_pub_socket = context.socket(zmq.PUB)
        self.arma3_pub_socket.bind("tcp://*:514")

        self.points_pub_socket = context.socket(zmq.PUB)
        self.points_pub_socket.bind("tcp://*:515")

    def init_stereo_params(self):
        blockSize = 11
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=16 * 5,
            blockSize=blockSize,
            P1=8 * 3 * blockSize**2,
            P2=32 * 3 * blockSize**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=100,
            preFilterCap=64,
            mode=cv2.STEREO_SGBM_MODE_SGBM,
        )
        self.stereoR = cv2.ximgproc.createRightMatcher(self.stereo)
        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(
            matcher_left=self.stereo
        )
        self.wls_filter.setLambda(8000)
        self.wls_filter.setSigmaColor(1.5)

        stereo_params = StereoCameraParameters()
        self.left_map1 = stereo_params.left_map1
        self.left_map2 = stereo_params.left_map2
        self.right_map1 = stereo_params.right_map1
        self.right_map2 = stereo_params.right_map2

        self.monitor_L = {
            "top": 0,
            "left": self.screen_width // 4,
            "width": self.screen_width // 4,
            "height": self.screen_height // 4,
        }
        self.monitor_R = {
            "top": 0,
            "left": self.screen_width // 2,
            "width": self.screen_width // 4,
            "height": self.screen_height // 4,
        }

    def start_threads(self):
        time.sleep(0.5)
        self.threads = []
        self.threads.append(
            threading.Thread(target=self.zmq_communication, daemon=True)
        )
        self.threads.append(threading.Thread(target=self.update_disparity, daemon=True))

        for thread in self.threads:
            thread.start()
        message = ("pid", self.pid)
        self.points_pub_socket.send(pickle.dumps(message))
        self.root.after(10, self.mainloop)
        self.root.mainloop()

    def mainloop(self):
        if not self.flag_focus:
            self.flag_focus = True
            self.arma3_window.activate()
            time.sleep(0.1)
            self.press_key(KeyCode.from_vk(0x6A), 0.1)
        if self.flag_capture:
            self.capture_and_process_images()
            threading.Thread(
                target=self.synthesize_disparity_maps,
                args=(
                    self.imgL,
                    self.imgR,
                ),
                daemon=True,
            ).start()
            self.arma3_pub_socket.send_string("DONE")
            self.flag_capture = False

        self.root.after(10, self.mainloop)

    def capture_and_process_images(self):
        with mss.mss() as sct:
            time.sleep(0.1)
            self.imgL = self.capture_and_remap(
                sct, self.monitor_L, self.left_map1, self.left_map2
            )
            self.imgR = self.capture_and_remap(
                sct, self.monitor_R, self.right_map1, self.right_map2
            )

    def capture_and_remap(self, sct, monitor, map1, map2):
        img = sct.grab(monitor)
        img = np.array(img)
        return cv2.remap(img, map1, map2, cv2.INTER_LINEAR)

    def zmq_communication(self):
        self.arma3_pub_socket.send_string("DONE")
        while self.running:
            try:
                message = self.arma3_sub_socket.recv_string(zmq.NOBLOCK)
                if message.startswith("Disparity"):
                    self.pos = message.split(";")[1]
                    self.flag_capture = True
            except zmq.Again:
                pass
            time.sleep(0.01)

    def update_disparity(self):
        while self.running:
            with self.lock:
                image = Image.fromarray(self.disparity_image.copy())
            photo = ImageTk.PhotoImage(image)
            self.disparity_label.configure(image=photo)
            self.disparity_label.image = photo
            time.sleep(0.1)

    def synthesize_disparity_maps(self, imgL, imgR):
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        grayL = cv2.equalizeHist(grayL)
        grayR = cv2.equalizeHist(grayR)
        disparity = self.stereo.compute(grayL, grayR)
        disparityR = self.stereoR.compute(grayR, grayL)
        disparity = self.wls_filter.filter(
            disparity, grayL, disparity_map_right=disparityR, right_view=grayR
        )

        disparity[(disparity >= 255) | (disparity <= 0)] = -1

        self.disparity_image = cv2.applyColorMap(
            np.uint8(254 - disparity), cv2.COLORMAP_JET
        )
        message = (
            "disparity",
            disparity,
            np.array(self.imgL),
            self.pos,
        )
        self.points_pub_socket.send(pickle.dumps(message))

    def close_window(self, event):
        message = ("flag", "close")
        self.points_pub_socket.send(pickle.dumps(message))
        self.running = False
        for thread in self.threads:
            if thread.is_alive():
                thread.join()
        self.root.quit()
        self.root.destroy()

    def press_key(self, key, duration):
        Controller().press(key)
        time.sleep(duration)
        Controller().release(key)
        time.sleep(duration)


if __name__ == "__main__":
    DisparityWindow()
