import cv2
import numpy as np

def calculate():
    left_camera_matrix = np.array([[726.9847, 1.5998, 613.1270],
                                   [0, 729.2007, 363.1410],
                                   [0, 0, 1.0000]])
    left_distortion = np.array([[0.1070, -0.1482, -0.0030, -0.0022, 0]])
    right_camera_matrix = np.array([[724.4213, 1.0275, 615.8249],
                                    [0, 727.1497, 345.0370],
                                    [0, 0, 1.0000]])
    right_distortion = np.array([0.1056, -0.1084, -0.0023, -0.0024, 0])
    R = np.matrix([[0.9999, 0.0028, -0.0109],
                  [-0.0028, 1.0000, -0.0010],
                  [0.0109, 0.0010, 0.9999]])
    T = np.array([-52.0290, 0.1025, 0.0263])
    size = (1280,720)
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                      right_camera_matrix, right_distortion, size, R, T)
    return R1, R2, P1, P2, Q, validPixROI1, validPixROI2
if __name__ == '__main__' :
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = calculate()


