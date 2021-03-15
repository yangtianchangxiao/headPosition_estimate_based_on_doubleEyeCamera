import cv2

AUTO = True  # 自动拍照，或手动按s键拍照


cv2.namedWindow("left")
cv2.namedWindow("right")
cv2.moveWindow("left", 0, 0)
cv2.moveWindow("right", 700, 0)
camera = cv2.VideoCapture(1)
# 设置分辨率
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
folder = "./screenshot/"  # 拍照文件目录


def shot(pos, frame):
    global counter
    path = folder + pos + "_" + str(counter) + ".jpg"

    cv2.imwrite(path, frame)
    print("snapshot saved into: " + path)


while True:
    ret, frame = camera.read()
    # 裁剪坐标为[y0:y1, x0:x1]
    left_frame = frame[0:480, 0:640]
    right_frame = frame[0:480, 640:1280]

    cv2.imshow("left", left_frame)
    cv2.imshow("right", right_frame)

    key = cv2.waitKey(0)
    if key == ord("q"):
        break
    elif key == ord("s"):
        shot("left", left_frame)
        shot("right", right_frame)
        counter += 1
camera.release()
cv2.destroyWindow("left")
cv2.destroyWindow("right")