import numpy as np
import cv2


class StereoCameraParameters:
    def __init__(self):
        self.image_size = (960, 540)
        self.left_intrinsics = np.array(
            [
                [480.848082806618, 0, 479.349790865097],
                [0, 480.317300981934, 271.489960306777],
                [0, 0, 1],
            ]
        )
        self.left_distortion = np.array(
            [
                0.00260081621252845,
                -0.00139921398092175,
                0,
                0,
                0,
            ]
        )
        self.right_intrinsics = np.array(
            [
                [480.828732625387, 0, 479.279109375706],
                [0, 480.312446180564, 271.357524349737],
                [0, 0, 1],
            ]
        )
        self.right_distortion = np.array(
            [
                0.00356112246793776,
                -0.00272917897030244,
                0,
                0,
                0,
            ]
        )
        self.R = np.array(
            [
                [0.999999995878492, -5.46494494328387e-06, 9.06264286308695e-05],
                [5.43763246338141e-06, 0.999999954572720, 0.000301371845712365],
                [-9.06280714945010e-05, -0.000301371351677048, 0.999999950480929],
            ]
        )
        self.T = np.array([-9.88485423343530, -0.0643088837682704, 0.147124435288212])
        self.rectification_parameters()

    def rectification_parameters(self):
        R1, R2, P1, P2, self.Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
            self.left_intrinsics,
            self.left_distortion,
            self.right_intrinsics,
            self.right_distortion,
            self.image_size,
            self.R,
            self.T,
        )
        self.left_map1, self.left_map2 = cv2.initUndistortRectifyMap(
            self.left_intrinsics,
            self.left_distortion,
            R1,
            P1,
            self.image_size,
            cv2.CV_16SC2,
        )
        self.right_map1, self.right_map2 = cv2.initUndistortRectifyMap(
            self.right_intrinsics,
            self.right_distortion,
            R2,
            P2,
            self.image_size,
            cv2.CV_16SC2,
        )
