import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
import Ui_Q3


def Aug_Reality():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((8*11,3), np.float32)
    objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    for i in range(1, 6, 1):
        img = cv2.imread(f'./{i}.bmp')

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (11,8), None)
        # If found, add object points, image points (after refining them)

        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            #cv2.drawChessboardCorners(img, (11,8), corners2, ret)
            #img = cv2.resize(img, (640, 480))
            #cv2.imshow(f'img', img)
            #cv2.waitKey(500)
        else:
            print(f'#{i} Not Found')
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


    Corners = np.float32([[3, 3, -3], [1, 1, 0], [3, 5, 0], [5, 1, 0]])
    for i in range(1, 6, 1):
        idx = i-1
        img = cv2.imread(f'./{i}.bmp')
        imgpts, jac = cv2.projectPoints(Corners, rvecs[idx], tvecs[idx], mtx, dist)

        imgpts = np.int32(imgpts).reshape(-1,2)
        for idx_0 in range(imgpts.shape[0]):
            for idx_1 in range(imgpts.shape[0]):
                cv2.line(img, tuple(imgpts[idx_0]), tuple(imgpts[idx_1]),[0,0,255],4)  #BGR
        img = cv2.resize(img, (640, 480))
        cv2.imshow('img',img)
        cv2.waitKey(500)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_Q3.Ui_Q3()
    ui.setupUi(MainWindow)
    ui.Button.clicked.connect(Aug_Reality)
    MainWindow.show()
    sys.exit(app.exec_())