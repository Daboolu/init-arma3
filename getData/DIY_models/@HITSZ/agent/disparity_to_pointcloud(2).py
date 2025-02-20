import cv2
import os
import math
import numpy as np
import torch
import open3d as o3d
import zmq
from sklearn.cluster import DBSCAN
from pytorch3d.structures import Pointclouds
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


def camera_calibration(save_path=None, override=False, image_size=None, left_intrinsics=None, left_distortion=None,
                       right_intrinsics=None, right_distortion=None, R=None, T=None):
    if save_path is None:
        save_path = "stereo_params.npz"

    if os.path.exists(save_path) and override is False:
        ret = np.load(save_path)
        return ret

    if image_size is None:
        image_size = (960, 540)

    if left_intrinsics is None:
        left_intrinsics = np.array(
            [
                [480.848082806618, 0, 479.349790865097],
                [0, 480.317300981934, 271.489960306777],
                [0, 0, 1]
            ]
        )
    if left_distortion is None:
        left_distortion = np.array([0.00260081621252845, -0.00139921398092175, 0, 0, 0])
    if right_intrinsics is None:
        right_intrinsics = np.array(
            [
                [480.828732625387, 0, 479.279109375706],
                [0, 480.312446180564, 271.357524349737],
                [0, 0, 1]
            ]
        )
    if right_distortion is None:
        right_distortion = np.array([0.00356112246793776, -0.00272917897030244, 0, 0, 0])

    if R is None:
        R = np.array(
            [
                [0.999999995878492, -5.46494494328387e-06, 9.06264286308695e-05],
                [5.43763246338141e-06, 0.999999954572720, 0.000301371845712365],
                [-9.06280714945010e-05, -0.000301371351677048, 0.999999950480929],
            ]
        )

    if T is None:
        T = np.array([-9.88485423343530, -0.0643088837682704, 0.147124435288212])

    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_intrinsics, left_distortion,
                                                                      right_intrinsics, right_distortion, image_size, R,
                                                                      T)

    left_map1, left_map2 = cv2.initUndistortRectifyMap(left_intrinsics, left_distortion, R1, P1, image_size,
                                                       cv2.CV_16SC2)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(right_intrinsics, right_distortion, R2, P2, image_size,
                                                         cv2.CV_16SC2)

    np.savez(save_path, image_size=image_size, left_map1=left_map1, left_map2=left_map2, right_map1=right_map1,
             right_map2=right_map2, Q=Q)

    return {"image_size": image_size,
            "left_map1": left_map1,
            "left_map2": left_map2,
            "right_map1": right_map1,
            "right_map2": right_map2,
            "Q": Q,
            }


def list_png_files(directory):
    png_files = {}
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            index = int(os.path.splitext(filename)[0].split('depth')[-1])
            png_files[index] = os.path.join(directory, filename)
    return png_files


def read_camera_positions(file_path=None):
    if file_path is None:
        file_path = "images\output.txt"
    camera_positions = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            index, data = line.strip().split(':')
            position, angle = data.strip().split('],')
            position = list(map(float, position.strip('[').split(',')))
            angle = float(angle.strip())
            camera_positions[str(index)] = (position, angle)
    return camera_positions


def rotation_matrix(yaw, pitch=0, roll=0):
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
    return R


def rotation_matrix_with_tensor(yaw, pitch=0, roll=0):
    yaw, pitch, roll = (
        torch.tensor(np.radians(yaw), dtype=torch.float32),
        torch.tensor(np.radians(pitch - 90.0), dtype=torch.float32),
        torch.tensor(np.radians(roll), dtype=torch.float32),
    )
    Rx = torch.tensor(
        [
            [1, 0, 0],
            [0, torch.cos(pitch), -torch.sin(pitch)],
            [0, torch.sin(pitch), torch.cos(pitch)],
        ],
        dtype=torch.float32
    )

    Ry = torch.tensor(
        [
            [torch.cos(roll), 0, torch.sin(roll)],
            [0, 1, 0],
            [-torch.sin(roll), 0, torch.cos(roll)],
        ],
        dtype=torch.float32
    )

    Rz = torch.tensor(
        [
            [torch.cos(-yaw), -torch.sin(-yaw), 0],
            [torch.sin(-yaw), torch.cos(-yaw), 0],
            [0, 0, 1],
        ],
        dtype=torch.float32
    )

    R = Rz @ Ry @ Rx
    return R


def transform_to_world(point, cam_position, yaw_angle):
    R = rotation_matrix(yaw_angle)
    point_world = (R @ point.T).T + np.array(cam_position)
    # R_inv = np.linalg.inv(R)
    # point_world = R_inv @ (point - np.array(cam_position))
    return point_world


def transform_to_world_with_tensor(point, cam_position, yaw_angle, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    R = rotation_matrix_with_tensor(yaw_angle).to(device)
    point_world = (R @ point.T).T + torch.tensor(cam_position, dtype=torch.float32).to(device)
    # R_inv = np.linalg.inv(R)
    # point_world = R_inv @ (point - np.array(cam_position))
    return point_world


class IncrementalPointCloud:
    def __init__(self, points=None, voxel_size=1, block_size=100, ema_beta=0.8, device=None):
        self.block = {}
        self.points = points
        if points is None:
            self.point_size = 0
        else:
            self.point_size = points.shape[0]
        self.voxel_size = voxel_size
        self.block_size = block_size
        self.ema_beta = ema_beta
        self.get_coord = self.get_centroid_coord
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

    def voxel_downsample(self, voxel_size=None, build_blocks=True, block_size=None):
        if self.points is None:
            return

        if voxel_size is None:
            voxel_size = self.voxel_size

        coords = torch.floor(self.points / voxel_size).int().to(self.device)
        unique_coords = torch.unique(coords, dim=0)
        downsampled_points = []
        for i in unique_coords:
            p = self.get_coord(points, coords, i)
            downsampled_points.append(p)
        self.points = torch.stack(downsampled_points, dim=0)
        self.point_size = self.points.shape[0]
        if build_blocks:
            if block_size is None:
                block_size = self.block_size
            for i in range(self.point_size):
                coord_ind = str(unique_coords[i])
                blk_ind = str(unique_coords[i] // block_size)
                if blk_ind in self.block.keys():
                    self.block[blk_ind][coord_ind] = i
                else:
                    self.block[blk_ind] = {}
                    self.block[blk_ind][coord_ind] = i
        else:
            self.block = {}

    def incremental_voxel_downsample(self, points, voxel_size=None):
        if voxel_size is None:
            voxel_size = self.voxel_size

        if not torch.is_tensor(points):
            points = torch.tensor(points, dtype=torch.float32).to(self.device)
        coords = torch.floor(points / voxel_size).int()
        unique_coords = torch.unique(coords, dim=0)
        for i in unique_coords:
            p = self.get_coord(points, coords, i)
            coord_ind = str(i)
            blk_ind = str(i // self.block_size)
            if blk_ind in self.block.keys():
                if coord_ind in self.block[blk_ind].keys():
                    self.points[self.block[blk_ind][coord_ind]] = self.ema_beta * self.points[self.block[blk_ind][str(i)]] + (1 - self.ema_beta) * p
                else:
                    self.point_size = self.point_size + 1
                    self.block[blk_ind][coord_ind] = self.point_size
                    self.points = torch.cat([self.points, p], dim=0)
            else:
                self.block[blk_ind] = {}
                self.point_size = self.point_size + 1
                self.block[blk_ind][coord_ind] = self.point_size
                if self.points is not None:
                    self.points = torch.cat([self.points, p], dim=0)
                else:
                    self.points = p

    def remove_outliers_dbscan(self, points, coords=None, eps=5, min_samples=10):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        is_tensor = False

        if coords is not None:
            if torch.is_tensor(coords):
                is_tensor = True
                coords = coords.cpu().numpy()
            labels = dbscan.fit_predict(coords)
            assert (coords[(labels != -1).nonzero()[0]] == coords[labels != -1]).all()
            inlier_coords = coords[labels != -1]
            inlier_points = points[labels != -1]
            if is_tensor:
                inlier_coords = torch.IntTensor(inlier_coords).to(self.device)
            return inlier_points, inlier_coords
        else:
            if torch.is_tensor(points):
                is_tensor = True
                points = points.cpu().numpy()
            labels = dbscan.fit_predict(points)
            inliers = points[labels != -1]
            if is_tensor:
                inliers = torch.FloatTensor(inliers).to(self.device)
            return inliers

    def incremental_voxel_downsample_with_dbscan(self, points, voxel_size=None, eps=5, min_samples=10):
        if voxel_size is None:
            voxel_size = self.voxel_size

        if not torch.is_tensor(points):
            points = torch.tensor(points, dtype=torch.float32).to(self.device)
        print(points.shape)
        coords = torch.floor(points / voxel_size).int()
        unique_coords = torch.unique(coords, dim=0)
        print(unique_coords.shape)
        downsampled_points = []
        for i in unique_coords:
            p = self.get_coord(points, coords, i)
            downsampled_points.append(p)
        downsampled_points = torch.stack(downsampled_points, dim=0)
        downsampled_inlier_points, downsampled_inlier_coords = self.remove_outliers_dbscan(downsampled_points, unique_coords, eps=eps, min_samples=min_samples)
        print(downsampled_inlier_points.shape, downsampled_inlier_points[0])
        for i, p in enumerate(downsampled_inlier_points):
            coord_ind = str(downsampled_inlier_coords[i])
            blk_ind = str(downsampled_inlier_coords[i] // self.block_size)
            if blk_ind in self.block.keys():
                if coord_ind in self.block[blk_ind].keys():
                    self.points[self.block[blk_ind][coord_ind]] = self.ema_beta * self.points[self.block[blk_ind][coord_ind]] + (1 - self.ema_beta) * p.squeeze(0)
                else:
                    self.point_size = self.point_size + 1
                    self.block[blk_ind][coord_ind] = self.point_size
                    self.points = torch.cat([self.points, p])
            else:
                self.block[blk_ind] = {}
                self.point_size = self.point_size + 1
                self.block[blk_ind][coord_ind] = self.point_size
                if self.points is not None:
                    self.points = torch.cat([self.points, p])
                else:
                    self.points = p

    def get_centroid_coord(self, points, coords, i):
        assert (points[(coords == i).all(dim=1).nonzero(as_tuple=True)[0]] == points[(coords == i).all(dim=1)]).all()
        return points[(coords == i).all(dim=1)].mean(dim=0, keepdim=True)

    def get_random_coord(self, points, coords, i):
        cur_points = points[coords == i]
        return cur_points[torch.randperm(cur_points.shape[0])[0]]


def update_point_cloud_with_open3d(pcd, disparity, Q, cam_position, yaw_angle):
    points_3D = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)
    points_3D = points_3D.reshape(-1, points_3D.shape[2])

    mask = np.isfinite(points_3D).all(axis=1)
    points_3D = points_3D[mask]
    z_threshold = np.max(points_3D[:, 2])
    points_3D = points_3D[points_3D[:, 2] < z_threshold]
    points_3D = transform_to_world(points_3D, cam_position, yaw_angle)
    new_points = np.array(points_3D)

    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(new_points)

    cl, ind = new_pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
    new_pcd = new_pcd.select_by_index(ind)

    # pcd.points.extend(o3d.utility.Vector3dVector(new_points))
    pcd.points.extend(new_pcd.points)

    voxel_size = 1
    pcd = pcd.voxel_down_sample(voxel_size)


def update_point_cloud_with_pytorch3d(ipc, disparity, Q, cam_position, yaw_angle, device=None):
    points_3D = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    points_3D = torch.tensor(points_3D, dtype=torch.float32).to(device)
    points_3D = points_3D.reshape(-1, points_3D.shape[2])

    mask = torch.isfinite(points_3D).all(dim=1)
    points_3D = points_3D[mask]
    z_threshold = torch.max(points_3D[:, 2])
    points_3D = points_3D[points_3D[:, 2] < z_threshold]
    points_3D = transform_to_world_with_tensor(points_3D, cam_position, yaw_angle)

    ipc.incremental_voxel_downsample(points_3D)
    print(ipc.points.shape)


def restore_pcd_with_open3d(pcd, images_dir="images", file_path="output.txt", save_path="point_cloud.ply"):
    Q = camera_calibration()["Q"]
    positions = read_camera_positions(file_path)
    for index in positions.keys():
        cam_pos, yaw_angle = positions[index]
        cur_path = os.path.join(images_dir, 'depth' + str(index) + '.png')
        try:
            assert os.path.exists(cur_path), "cur_path: {} not exist!".format(cur_path)
            disparity = cv2.imread(cur_path, cv2.IMREAD_GRAYSCALE)
            print("reading {}".format(cur_path))
        except AssertionError:
            print("pass {}".format(cur_path))
            continue
        update_point_cloud_with_open3d(pcd, disparity, Q, cam_pos, yaw_angle)

        o3d.io.write_point_cloud(save_path, pcd, write_ascii=False, compressed=True)

    return pcd


def restore_pcd_with_pytorch3d(ipc, images_dir="images", file_path="output.txt", save_path="point_cloud.ply"):
    Q = camera_calibration()["Q"]
    positions = read_camera_positions(file_path)
    for index in positions.keys():
        cam_pos, yaw_angle = positions[index]
        cur_path = os.path.join(images_dir, 'depth' + str(index) + '.png')
        try:
            assert os.path.exists(cur_path), "cur_path: {} not exist!".format(cur_path)
            disparity = cv2.imread(cur_path, cv2.IMREAD_GRAYSCALE)
            print("reading {}".format(cur_path))
        except AssertionError:
            print("pass {}".format(cur_path))
            continue
        update_point_cloud_with_pytorch3d(ipc, disparity, Q, cam_pos, yaw_angle)

    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(ipc.points.cpu().numpy())
    o3d.io.write_point_cloud(save_path, o3d_pcd, write_ascii=False, compressed=True)

    return o3d_pcd


class FileHandler(FileSystemEventHandler):
    def __init__(self, pcd, pos_filename='output.txt', save_path="point_cloud.ply", index=0):
        self.data = {}
        self.pcd = pcd
        self.Q = camera_calibration()["Q"]
        self.index = index
        self.pos_filename = pos_filename
        self.save_path = save_path
        self.trigger_exception = 0

    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith('.png'):
            image_path = event.src_path
            index = int(os.path.splitext(os.path.basename(image_path))[0].split('depth')[-1])
            if str(index) in self.data.keys():
                self.data[str(index)]["image"] = image_path
            else:
                self.data[str(index)] = {"image": image_path}
            if self.index < index:
                for i in range(self.index, index):
                    cur_path = 'depth' + str(i) + '.png'
                    assert os.path.exists(cur_path), "cur_path: {} not exist!".format(cur_path)
                    if str(i) in self.data.keys():
                        self.data[str(i)]["image"] = cur_path
                    else:
                        self.data[str(i)] = {"image": cur_path}
            self.trigger()

    def on_modified(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith(self.pos_filename):
            with open(self.pos_filename, 'r') as file:
                lines = file.readlines()
                index, data = lines[-1].strip().split(':')
                position, angle = data.strip().split('],')
                position = list(map(float, position.strip('[').split(',')))
                angle = float(angle.strip())
                if str(index) in self.data.keys():
                    self.data[str(index)]["cam_pos"] = (position, angle)
                else:
                    self.data[str(index)] = {"cam_pos": (position, angle)}
            if self.index < index:
                new_positions = read_camera_positions(event.src_path)
                cur_index = self.index
                while cur_index in new_positions.keys():
                    if str(cur_index) in self.data.keys():
                        self.data[str(cur_index)]["cam_pos"] = new_positions[str(cur_index)]
                    else:
                        self.data[str(cur_index)] = {"cam_pos": new_positions[str(cur_index)]}
            self.trigger()

    def trigger(self):
        data = self.data.setdefault(str(self.index), default={})
        if "image" not in data or "cam_pos" not in data:
            self.trigger_exception = self.trigger_exception + 1
            if self.trigger_exception > 5:
                print("pass index {}".format(self.index))
                self.index = self.index + 1
                self.trigger_exception = 0
            return

        disparity = cv2.imread(data["image"], cv2.IMREAD_GRAYSCALE)
        cam_pos, yaw_angle = data["cam_pos"]
        update_point_cloud(self.pcd, disparity, self.Q, cam_pos, yaw_angle)
        o3d.io.write_point_cloud(self.save_path, self.pcd, write_ascii=False, compressed=True)
        o3d.visualization.draw_geometries([self.pcd])


def zmq_receiver(pcd, Q, port, save_path="point_cloud.ply"):
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.connect(f'tcp://127.0.0.1:{port}')

    try:
        while True:
            """
            context = zmq.Context()
            socket = context.socket(zmq.PUSH)
            socket.bind(f'tcp://127.0.0.1:{port}')
            
            _, img_bytes = cv2.imencode('depth98.png', image)
            text = f"98: [2905.55, 5952.47, 4.49183], 308.38"
            socket.send_multipart([img_bytes.tobytes(), text.encode('utf-8')])
            """
            img_bytes, text_bytes = socket.recv_multipart()
            nparr = np.frombuffer(img_bytes, np.uint8)
            disparity = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

            text = text_bytes.decode('utf-8')
            index, data = text[-1].strip().split(':')
            position, angle = data.strip().split('],')
            cam_pos = list(map(float, position.strip('[').split(',')))
            yaw_angle = float(angle.strip())

            update_point_cloud(pcd, disparity, Q, cam_pos, yaw_angle)
            o3d.io.write_point_cloud(save_path, pcd, write_ascii=False, compressed=True)
            o3d.visualization.draw_geometries([pcd])

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Receiver terminated.")


def main():
    images_dir = "images"
    file_path = "images\output.txt"
    save_path = "images\point_cloud.ply"

    # x = torch.randn(4, 3)
    # print(x)
    # print(torch.unique(x, sorted=True, dim=0))

    # pcd = o3d.geometry.PointCloud()
    ipc = IncrementalPointCloud()

    pcd = restore_pcd_with_pytorch3d(ipc, images_dir, file_path, save_path)
    o3d.visualization.draw_geometries([pcd])

    # event_handler = FileHandler(pcd, file_path, save_path)
    # observer = Observer()
    # observer.schedule(event_handler, path=images_dir, recursive=False)
    # observer.start()

    # try:
    #     while True:
    #         pass
    # except KeyboardInterrupt:
    #     observer.stop()
    # observer.join()


if __name__ == "__main__":
    main()
