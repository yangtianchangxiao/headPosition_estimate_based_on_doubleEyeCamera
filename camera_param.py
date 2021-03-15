#!usr/bin/env python3
# __author__ = caox
import cv2
import numpy as np
import dlib
import math

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C://Users//a//Desktop//shape_predictor_68_face_landmarks.dat")
POINTS_NUM_LANDMARK = 68
# record the highest location of eye corner
height_record = 1000
# temporary record of last rectangular
p3_last = 0
p4_last = 0
p5_last = 0
p6_last = 0
class headpose:
    def __init__(self, image,camera_mark,detector,predictor,points_num_landmark):
        self.image=image
        self.mark=camera_mark
        self.detector=detector
        self.predictor=predictor
        self.points_num_landmark=points_num_landmark


    def get_point(self):
        dets = self.detector(self.image, 0)

        if len(dets) == 0:
            print('find no face')
            return None
        face_areas = [(det.right() - det.left()) * (det.bottom() - det.top()) for det in dets]
        face_rectangle = dets[int(np.argmax(np.asarray(face_areas)))]
        landmark_shape = self.predictor(self.image, face_rectangle)
        if landmark_shape.num_parts != POINTS_NUM_LANDMARK:
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

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        return success, rvec, tvec, camera_matrix, dist_coeffs


    # Convert rotation vector to euler angle
    def get_euler_angle(self,rvec):
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
if __name__ == '__main__':
    # set camera paramgments
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # image,camera_mark,detector,predictor,points_num_landmark

    while cap.isOpened():
        _, frame = cap.read()
        leftpicture = headpose(frame, 'left', detector, predictor, POINTS_NUM_LANDMARK)
        left_image_points = leftpicture.get_point()
        if left_image_points is None:
            print('get_image_points failed')
            continue
        ret, rotation_vector, translation_vector, camera_matrix, dist_coeffs = leftpicture.get_pose_estimation(left_image_points)
        if not ret:
            print('get_pose_estimation failed')
            continue
        # Draw Landmarks
        for p in left_image_points:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), 10)

        # Show Angels
        pitch, yaw, roll = leftpicture.get_euler_angle(rotation_vector)
       #  cv2.putText(frame, 'Pitch:{}, Yaw:{}, Roll:{}'.format(pitch, yaw, roll), (0, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1)
        # Project a 3D point (0, 0, 1000.0) onto the image plane.
        # We use this to draw a line sticking out of the nose
        nose_end_point2D, jacobian = cv2.projectPoints(np.array([(1000.0, 1000.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        p1 = (int(left_image_points[0][0]), int(left_image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        cv2.line(frame, p1, p2, (255, 0, 0), 2)

        p3 = (int(left_image_points[2][0])-40, int(left_image_points[2][1])-40)
        p4 = (int(left_image_points[3][0])+40, int(left_image_points[3][1])-40)
        length = p4[0] - p3[0]
        p5 = (int(left_image_points[3][0])+40, int(left_image_points[1][1]))
        p6 = (int(left_image_points[2][0])-40, int(left_image_points[1][1]))
        height_record = min(int(left_image_points[2][1]), int(left_image_points[3][1]), height_record)
        print(height_record)
        print(int(left_image_points[2][1]))
        # update the highest position of eye corners
        if height_record < min(int(left_image_points[2][1]), int(left_image_points[3][1]))-40:
            cv2.putText(frame, "don't tuo bei", (50, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 1)
            height_record = min(int(left_image_points[2][1]), int(left_image_points[3][1]), height_record)
            cv2.line(frame, (int(640-length/2), height_record), (int(640+length/2), height_record), (0, 0, 255), 2)
            cv2.line(frame, (int(640+length/2), height_record), (int(640+length/2), p6[1]), (0, 0, 255), 2)
            cv2.line(frame, (int(640+length/2), p6[1]), (int(640-length/2), p6[1]), (0, 0, 255), 2)
            cv2.line(frame, (int(640-length/2), p6[1]), (int(640-length/2), height_record),(0, 0, 255), 2)
            cv2.imshow('', frame)
            cv2.waitKey(1)
            continue
        # print rectangular on the frame
        cv2.line(frame, p3, p4, (0, 0, 255), 2)
        cv2.line(frame, p4, p5, (0, 0, 255), 2)
        cv2.line(frame, p5, p6, (0, 0, 255), 2)
        cv2.line(frame, p6, p3, (0, 0, 255), 2)



        cv2.imshow('', frame)
        cv2.waitKey(1)
