import cv2
import headpose_estimate
import points_position
import dlib
import numpy as np
import camera_calibrator
from Stabilizer import Stabilizer

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C://Users//a//Desktop//shape_predictor_68_face_landmarks.dat")
POINTS_NUM_LANDMARK = 68
eyePointsLeft = []
eyePointsRight = []
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
useStabilizer = True
means_filter_times = 0
means_leftEyeCoordinate = np.array([0,0,0,0])
leftEyeCoordinate_sum = np.array([0,0,0,0])





def headposeAngle(Image,Image2):
    HeadPose = headpose_estimate.headpose(Image, 'left', detector, predictor, POINTS_NUM_LANDMARK)
    ImagePoints = HeadPose.get_point()
    HeadPose2 = headpose_estimate.headpose(Image2, 'right', detector, predictor, POINTS_NUM_LANDMARK)
    ImagePoints2 = HeadPose2.get_point()
    cv2.imshow('', Image)
    cv2.waitKey(1)
    num = 0
    #不知道为啥要求我声明下ret和rotation_vector
    while True:
        if ImagePoints is None:
            return False, 0, 0, 0, [], []
        if ImagePoints2 is None:
            return False, 0, 0, 0, [], []
        ret, rotation_vector, translation_vector, camera_matrix, dist_coeffs = HeadPose.get_pose_estimation(ImagePoints)
        print(ret)
        if not ret:
        #    print('get_pose_estimation failed')
            return False, 0, 0, 0, [], []
        # Draw Landmarks
        for p in ImagePoints:
            cv2.circle(Image, (int(p[0]), int(p[1])), 3, (0, 0, 255), 10)
        eyeLeft = np.array([
            [int(ImagePoints[3][0]), int(ImagePoints[3][1])], [int(ImagePoints[4][0]), int(ImagePoints[4][1])]
        ])
        eyeRight = np.array([
            [int(ImagePoints2[3][0]), int(ImagePoints2[3][1])], [int(ImagePoints2[4][0]), int(ImagePoints2[4][1])]
        ])
        # Show Angels
        pitch, yaw, roll = HeadPose.get_euler_angle(rotation_vector)
        return ret, pitch, yaw, roll, eyeLeft, eyeRight




while True:
    # 从摄像头读取图片
    success, img = cap.read()
    if success:
        # 获取左右摄像头的图像
        size = img.shape
        imageHeight = size[0]
        imageWidth = size[1]
        ImageL = img[:, 0:imageWidth//2, :]
        ImageR = img[:, imageWidth//2:imageWidth, :]
        # Acquire pitch yaw roll
        ret, pitch, yaw, roll, eyePointsLeft, eyePointsRight = headposeAngle(ImageL, ImageR)
        # If failed, try again
        if not ret:
            continue
        #print('lefEyeCoordinate')
        # get eyes 3D coordinate
        leftEyeCoordinate = points_position.points_coordinate(eyePointsLeft[0], eyePointsRight[0])
        rightEyeCoordinate = points_position.points_coordinate(eyePointsLeft[1], eyePointsRight[1])
        # Stabilize
        left_list = leftEyeCoordinate.flatten()
        right_list = rightEyeCoordinate.flatten()

        if useStabilizer is True:
            # initiate Stabilizer (2d and 1d)
            posStabilizers = [Stabilizer(
                state_num=4,
                measure_num=2,
                cov_process=0.06,
                cov_measure=0.1) for _ in range(6)]
            posStabilizers_one = [Stabilizer(
                state_num=4,
                measure_num=2,
                cov_process=0.06,
                cov_measure=0.1) for _ in range(6)]
            # stabilize x and y
            allFramePoints = np.vstack((left_list, right_list))
           # print(allFramePoints)
            steadyPoints = []
            for pt, stb in zip(allFramePoints, posStabilizers):
                p = np.array([[np.float32(pt[0])], [np.float32(pt[1])]])
                stb.update(p)
                steadyPoint = np.array([stb.state[0][0], stb.state[1][0]], stb.state[2][0])
                steadyPoints.append(steadyPoint)
            steadyPoints = np.array(steadyPoints)
            # get x and y
            # type(leftEyeCoordinate) is np.ndarry, so convert it to list
            leftEyeCoordinate_xy = np.ndarray.tolist(steadyPoints[0])
            rightEyeCoordinate_xy = np.ndarray.tolist(steadyPoints[1])

            # stabilize y, z
            steadyPoints = []
            left_z = left_list[2:4]
            right_z = right_list[2:4]
            framePoints = np.vstack((left_z, right_z))
            for pt, stb in zip(framePoints, posStabilizers_one):
                p = np.array([[np.float32(pt[0])], [np.float32(pt[1])]])
                stb.update(p)
                steadyPoint = np.array([stb.state[0][0], stb.state[1][0]], stb.state[2][0])
                steadyPoints.append(steadyPoint)
            steadyPoints = np.array(steadyPoints)
            # type(leftEyeCoordinate) is np.ndarry, so convert it to list
            leftEyeCoordinate_z = np.ndarray.tolist(steadyPoints[0])
            rightEyeCoordinate_z = np.ndarray.tolist(steadyPoints[1])
            leftEyeCoordinate = np.array(leftEyeCoordinate_xy + leftEyeCoordinate_z)

            leftEyeCoordinate_sum = leftEyeCoordinate_sum+leftEyeCoordinate

            #print(leftEyeCoordinate)

            means_filter_times = means_filter_times + 1
            if means_filter_times > 10:
                means_leftEyeCoordinate = leftEyeCoordinate_sum/means_filter_times
               # print(means_leftEyeCoordinate)
                means_filter_times = 0
                leftEyeCoordinate_sum = np.array([0,0,0,0])

        a = np.ndarray.tolist(means_leftEyeCoordinate)
        cv2.putText(ImageR, 'leftEye:{}'.format(int(a[2])), (50, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 1)
        if roll > 5 or roll < -5:
            cv2.putText(ImageR, 'wai tou:{}'.format(int(roll)), (50, 150), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)
        cv2.imshow('', ImageR)
        cv2.waitKey(1)