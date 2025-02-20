import os
import numpy as np
from camera_parameters import StereoCameraParameters
from vispy import app, scene
import pyautogui
from PyQt5 import QtCore, QtWidgets
import zmq
import time
import threading
import psutil
import cv2
import pickle
import logging
import open3d as o3d


class PointCloudWindow:
    def __init__(self):
        self.running = True
        self.disparity_process = None
        self.counter = 0
        self.flag = False
        self.init_logging()
        self.init_data()
        self.init_zmq()
        self.start_threads()
        self.init_window()

    def init_logging(self):
        log_dir = os.path.join(os.path.dirname(__file__), "logging")
        os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(
                    os.path.join(log_dir, "PointCloudWindow.log"), mode="w"
                ),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def init_window(self):
        screen_width, screen_height = pyautogui.size()
        self.window_width = screen_width // 4
        self.window_height = screen_height // 4

        self.canvas = scene.SceneCanvas(keys=None, show=True, bgcolor="gray")
        self.canvas.native.setWindowFlags(
            QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint
        )
        self.canvas.native.move(self.window_width, screen_height - self.window_height)
        self.canvas.native.resize(screen_width // 2, self.window_height)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(
            fov=150, azimuth=0, elevation=0, center=(0, 0, 0)
        )

        self.scatter = scene.visuals.Markers()
        self.scatter.set_data(
            np.vstack(
                (self.points_all[:, 0], self.points_all[:, 1], self.points_all[:, 2])
            ).T,
            face_color=self.colors_all,
            edge_width=0,
            size=5,
        )
        self.view.add(self.scatter)
        self.axis = scene.visuals.XYZAxis(parent=self.view.scene)
        self.canvas.events.close.connect(self.on_close)
        self.canvas.show()
        self.canvas.app.run()

    def init_data(self):
        stereo_params = StereoCameraParameters()
        self.Q = stereo_params.Q

        self.pcd = o3d.geometry.PointCloud()
        self.points_all = np.array([[0, 0, 0]])
        self.colors_all = np.array([[1, 1, 1]])

    def init_zmq(self):
        context = zmq.Context()
        self.disparity_sub_socket = context.socket(zmq.SUB)
        self.disparity_sub_socket.connect("tcp://localhost:515")
        self.disparity_sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")

    def zmq_communication(self):
        while self.running:
            try:
                message = self.disparity_sub_socket.recv(zmq.NOBLOCK)
                message = pickle.loads(message)
                msg_type = message[0]
                # self.logger.info(f"Received message: {message}")
                if msg_type == "pid":
                    pid = message[1]
                    self.disparity_process = psutil.Process(pid)
                elif msg_type == "flag":
                    if message[1] == "close":
                        self.running = False
                        self.canvas.close()
                elif msg_type == "disparity":
                    disparity = message[1]
                    image_left = message[2]
                    pos_string = message[3]
                    threading.Thread(
                        target=self.update_points,
                        args=(
                            disparity,
                            image_left,
                            pos_string,
                        ),
                        daemon=True,
                    ).start()
            except zmq.Again:
                pass
            except Exception as e:
                self.logger.error(f"An error occurred: {e}")
            time.sleep(0.01)

    def start_threads(self):
        threading.Thread(target=self.zmq_communication, daemon=True).start()
        while self.disparity_process == None:
            time.sleep(0.1)

    def update_points(self, disparity, image_left, pos_string):
        pos_num = [float(num) for num in pos_string.split(",")]
        (R, T) = self.getRotationMatrix(
            pos_num[0], pos_num[1], pos_num[2], pos_num[3], 0, 0
        )

        disparity[disparity <= 50] = -1
        points = cv2.reprojectImageTo3D(
            disparity, self.Q, handleMissingValues=True
        ).reshape(-1, 3)
        points = points[::10]
        colors = image_left.reshape(-1, 4)[::10, :3]
        colors = colors[:, [2, 1, 0]]

        z_max = np.max(points[:, 2]) * 0.9
        mask = points[:, 2] < z_max
        points = points[mask] * 0.16
        colors = colors[mask]

        points = (R @ points.T).T + T
        points = np.round(points / 0.5) * 0.5
        points, indices = np.unique(points, axis=0, return_index=True)
        colors = colors[indices]

        self.pcd.points = o3d.utility.Vector3dVector(points)
        self.pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
        cl, ind = self.pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.5)
        filtered_points = self.pcd.select_by_index(ind)
        points = filtered_points.points  # 点坐标
        colors = filtered_points.colors  # 点rgb
        time = 0
        self.points_all = np.vstack((self.points_all, points))
        self.colors_all = np.vstack((self.colors_all, colors))

        self.points_all, unique_indices = np.unique(
            self.points_all, axis=0, return_index=True
        )
        self.colors_all = self.colors_all[unique_indices]

        x = self.points_all[:, 0]
        y = self.points_all[:, 1]
        z = self.points_all[:, 2]

        self.scatter.set_data(
            np.vstack((x, y, z)).T, face_color=self.colors_all, edge_width=0, size=5
        )
        self.view.camera.azimuth = -pos_num[3]
        self.view.camera.center = (pos_num[0], pos_num[1], pos_num[2])
        self.canvas.update()
        self.flag = True

    def getRotationMatrix(self, x, y, z, yaw, pitch, roll):
        yaw, pitch, roll = (
            np.radians(yaw),
            np.radians(pitch - 90.0),
            np.radians(roll),
        )
        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(pitch), -np.sin(pitch)],
                [0, np.sin(pitch), np.cos(pitch)],
            ]
        )

        Ry = np.array(
            [
                [np.cos(roll), 0, np.sin(roll)],
                [0, 1, 0],
                [-np.sin(roll), 0, np.cos(roll)],
            ]
        )

        Rz = np.array(
            [
                [np.cos(-yaw), -np.sin(-yaw), 0],
                [np.sin(-yaw), np.cos(-yaw), 0],
                [0, 0, 1],
            ]
        )

        R = Rz @ Ry @ Rx
        T = np.array([x, y, z])
        return (R, T)

    def on_close(self, event):
        output_pcd = o3d.geometry.PointCloud()
        output_pcd.points = o3d.utility.Vector3dVector(self.points_all)
        output_pcd.colors = o3d.utility.Vector3dVector(self.colors_all)
        o3d.io.write_point_cloud(
            r"D:\steam\steamapps\common\Arma 3\DIY_models\@HITSZ\agent\output.ply",
            output_pcd,
        )
        os._exit(0)


if __name__ == "__main__":
    PointCloudWindow()
