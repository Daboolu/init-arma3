from camera_parameters import StereoCameraParameters
from pynput.keyboard import Controller, KeyCode
from zz_unused_.threading_utils import call_slow_function
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import pygetwindow as gw
import tkinter as tk
import numpy as np
import ctypes.wintypes
import threading
import pyautogui
import pygame
import shutil
import ctypes
import time
import cv2
import zmq
import mss
import sys
import os


class BinocularCamera:
    def __init__(self):
        stereo_params = StereoCameraParameters()
        self.left_map1, self.left_map2, self.right_map1, self.right_map2, self.Q = (
            stereo_params.get_rectification_parameters()
        )
        self.running = True
        self.screen_width, self.screen_height = pyautogui.size()
        self.window_height = self.screen_height // 4
        self.window_width = self.screen_width // 4
        self.arma3_window = gw.getWindowsWithTitle('Arma 3 "')[0]
        self.keyboard = Controller()
        context = zmq.Context()
        self.cam_sub_socket = context.socket(zmq.SUB)
        self.cam_sub_socket.connect("tcp://localhost:114")
        self.cam_sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.cam_pub_socket = context.socket(zmq.PUB)
        self.cam_pub_socket.bind("tcp://*:514")
        self.disparity_image = np.zeros(
            (self.window_height, self.window_width), dtype=np.uint8
        )
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
        blockSize = 7
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
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.geometry(
            f"{self.screen_width // 4}x{self.screen_height // 4}+{self.screen_width - self.screen_width // 4}+{0}"
        )
        self.placeholder_image = np.zeros(
            (self.screen_height // 4, self.screen_width // 4), dtype=np.uint8
        )
        self.image_tk = ImageTk.PhotoImage(
            image=Image.fromarray(self.placeholder_image)
        )
        self.label = tk.Label(self.root, image=self.image_tk)
        self.label.pack()
        self.fetch_images()

    def fetch_images(self):
        self.multithreading(self.disparity_display)
        time.sleep(0.5)
        self.arma3_window.activate()
        time.sleep(0.1)
        self.press_key(KeyCode.from_vk(0x6A), 0.1)
        self.cam_pub_socket.send_string("DONE")
        while self.running:
            try:
                message = self.cam_sub_socket.recv_string(zmq.NOBLOCK)
                split_message = message.split(":")
                if split_message[0] == "Depth":
                    with mss.mss() as sct:
                        self.imgL = sct.grab(self.monitor_L)
                        self.imgL = np.array(self.imgL)
                        self.imgL = cv2.cvtColor(self.imgL, cv2.COLOR_BGR2GRAY)
                        self.imgL = cv2.remap(
                            self.imgL, self.left_map1, self.left_map2, cv2.INTER_LINEAR
                        )
                    with mss.mss() as sct:
                        self.imgR = sct.grab(self.monitor_R)
                        self.imgR = np.array(self.imgR)
                        self.imgR = cv2.cvtColor(self.imgR, cv2.COLOR_BGR2GRAY)
                        self.imgR = cv2.remap(
                            self.imgR,
                            self.right_map1,
                            self.right_map2,
                            cv2.INTER_LINEAR,
                        )
                        call_slow_function(self.synthesize_depth_maps, ())
                        self.cam_pub_socket.send_string("DONE")
                if split_message[0] == "Move":
                    self.goal = split_message[1]
            except zmq.Again:
                pass

    def synthesize_depth_maps(self):
        local_imgL = self.imgL
        local_imgR = self.imgR
        disparity = self.stereo.compute(local_imgL, local_imgR)
        disparityR = self.stereoR.compute(local_imgR, local_imgL)
        disparity = self.wls_filter.filter(
            disparity, local_imgL, disparity_map_right=disparityR, right_view=local_imgR
        )
        self.disparity_image = np.uint8(255 - disparity)

    def disparity_display(self):
        pygame.init()
        window_width = self.screen_width // 4
        window_height = self.screen_height // 4
        screen = pygame.display.set_mode((window_width, window_height), pygame.NOFRAME)

        x = self.screen_width - window_width
        y = 0

        hwnd = pygame.display.get_wm_info()["window"]
        ctypes.windll.user32.SetWindowPos(
            hwnd, ctypes.wintypes.HWND(-1), x, y, 0, 0, 0x0001
        )

        disparity = np.zeros((self.window_height, self.window_width), dtype=np.uint8)
        disp_rgb = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)
        disp_surface = pygame.surfarray.make_surface(disp_rgb.transpose((1, 0, 2)))

        while self.running:
            disparity = self.disparity_image
            disp_rgb = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)
            disp_surface = pygame.surfarray.make_surface(disp_rgb.transpose((1, 0, 2)))
            screen.blit(disp_surface, (0, 0))
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    pygame.quit()
                    self.running = False
                    sys.exit()

    def press_key(self, key, duration):
        self.keyboard.press(key)
        time.sleep(duration)
        self.keyboard.release(key)
        time.sleep(duration)

    def multithreading(self, func):
        thread = threading.Thread(target=func, daemon=True)
        thread.start()


def main():
    BinocularCamera()


if __name__ == "__main__":
    main()
