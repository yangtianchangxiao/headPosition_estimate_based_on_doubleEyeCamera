#!usr/bin/env python3
# __author__ = caox
import cv2
import numpy as np


class headpose:
    def __init__(self, image, camera_mark, detector, predictor, points_num_landmark):
        self.image = image
        self.mark = camera_mark
        self.detector = detector
        self.predictor = predictor
        self.points_num_landmark = points_num_landmark

    def get_point(self):
        dets = self.detector(self.image, 0)

        if len(dets) == 0:
            print('find no face')
            return None
        face_areas = [(det.right() - det.left()) * (det.bottom() - det.top()) for det in dets]
        face_rectangle = dets[int(np.argmax(np.asarray(face_areas)))]
        landmark_shape = self.predictor(self.image, face_rectangle)
        if landmark_shape.num_parts != self.points_num_landmark:
            print('ERROR: landmark_shape.num_parts-{}'.format(landmark_shape.num_parts))
            return None
            # 2D image points. If you change the image, you need to change vector
        return np.array([
            (landmark_shape.part(30).x, landmark_shape.part(30).y),  # Nose tip
            (landmark_shape.part(8).x, landmark_shape.part(8).y),  # Chin
            (landmark_shape.part(36).x, landmark_shape.part(36).y),  # Left eye left corner
            (landmark_shape.part(45).x, landmark_shape.part(45).y),  # Right eye right corner
            (landmark_shape.part(48).x, landmark_shape.part(48).y),  # Left Mouth corner
            (landmark_shape.part(54).x, landmark_shape.part(54).y)  # Right mouth corner
        ], dtype="double")

    # Get rotation vector and translation vector
    def get_pose_estimation(self, image_points):
        # 3D model points.
        object_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corner
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner
        ])

        # Camera internals
        focal_length = self.image.shape[1]
        center = (self.image.shape[1] / 2, self.image.shape[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        dist_coeffs = np.array([[0.1070, -0.1482, -0.0030, -0.0022]])  # Assuming no lens distortion
        success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        return success, rvec, tvec, camera_matrix, dist_coeffs

    # Convert rotation vector to euler angle

    def get_euler_angle(self, rvec):
        # Calculate rotation angles
        theta = cv2.norm(rvec, cv2.NORM_L2)

        # Transformed to quaterniond
        w = np.cos(theta / 2)
        x = np.sin(theta / 2) * rvec[0][0] / theta
        y = np.sin(theta / 2) * rvec[1][0] / theta
        z = np.sin(theta / 2) * rvec[2][0] / theta

        ysqr = y * y
        # Pitch (x-axis rotation)
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + ysqr)
        pitch = int((np.arctan2(t0, t1) / np.pi) * 180)

        # Yaw (y-axis rotation)
        t2 = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
        yaw = int((np.arcsin(t2) / np.pi) * 180)

        # Roll (z-axis rotation)
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (ysqr + z * z)
        roll = int((np.arctan2(t3, t4) / np.pi) * 180)

        return pitch, yaw, roll
