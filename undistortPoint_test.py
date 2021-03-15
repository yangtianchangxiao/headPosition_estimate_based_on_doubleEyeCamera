import camera_calibrator
import cv2
import numpy as np
left_camera_matrix = np.array([[726.9847, 1.5998, 613.1270],
                               [0, 729.2007, 363.1410],
                               [0, 0, 1.0000]])
left_distortion = np.array([[0.1070, -0.1482, -0.0030, -0.0022, 0]])
right_camera_matrix = np.array([[724.4213, 1.0275, 615.8249],
                                [0, 727.1497, 345.0370],
                                [0, 0, 1.0000]])
right_distortion = np.array([0.1056, -0.1084, -0.0023, -0.0024, 0])

def points_coordinate(leftPoints, rightPoints):
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = camera_calibrator.calculate()
    leftPoints = tuple(leftPoints)
    rightPoints  = tuple(rightPoints)
    pt1 = cv2.undistortPoints(leftPoints, left_camera_matrix, left_distortion, None, R1, P1)
    pt2 = cv2.undistortPoints(rightPoints, right_camera_matrix, right_distortion, None, R2, P2)
    points4D = cv2.triangulatePoints(P1, P2, pt1, pt2)
    points3D = points4D / points4D[3]
    print(points3D)
    print(points3D[0:3])
    return points3D

points_coordinate([155,300],[253,157])



