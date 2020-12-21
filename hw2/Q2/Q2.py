import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
import Ui_Q2

global mtx, dist, rvecs, tvecs
global calibrate_flag
global current_img_id
calibrate_flag = False
current_img_id = 1

def FindCorner_and_Calibrate():
    global mtx, dist, rvecs, tvecs
    global calibrate_flag
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((8*11,3), np.float32)
    objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    for i in range(1, 16, 1):
        img = cv2.imread(f'./{i}.bmp')

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (11,8), None)
        # If found, add object points, image points (after refining them)

        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (11,8), corners2, ret)
            img = cv2.resize(img, (640, 480))
            cv2.imshow(f'img', img)
            cv2.waitKey(300)
        else:
            print(f'#{i} Not Found')
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    calibrate_flag = True
    print('Calibrate DONE !')

def Show_Intrinsic():
    global mtx
    if calibrate_flag:
        print(mtx)
    else:
        print('Please do 2.1 Find Corner first !')
def ChangeImage(ui):
    global current_img_id
    current_img_id = int(ui.comboBox.currentText()) - 1

def Show_Extrinsic():
    global rvecs, tvecs
    global current_img_id
    if calibrate_flag:
        rot, _ = cv2.Rodrigues(rvecs[current_img_id])
        extrinsic = np.zeros(shape=[3, 4])
        extrinsic[0:3, 0:3] = rot
        for idx in range(3):
            extrinsic[idx, 3] = tvecs[current_img_id][idx, 0]
        """
        print(rvecs[current_img_id])
        print(tvecs[current_img_id])
        print(rot)
        """
        print(extrinsic)
    else:
        print('Please do 2.1 Find Corner first !')

def Show_Distortion():
    global dist
    if calibrate_flag:
        print(dist[0])
    else:
        print('Please do 2.1 Find Corner first !')

if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_Q2.Ui_Q2()
    ui.setupUi(MainWindow)
    choice_list = []
    for i in range(1, 16, 1):
        choice_list.append(f'{i}')
    ui.comboBox.addItems(choice_list)
    ui.comboBox.currentIndexChanged.connect(lambda:ChangeImage(ui))
    ui.Button_2_1.clicked.connect(FindCorner_and_Calibrate)
    ui.Button_2_2.clicked.connect(Show_Intrinsic)
    ui.Button_2_3.clicked.connect(Show_Extrinsic)
    ui.Button_2_4.clicked.connect(Show_Distortion)
    MainWindow.show()
    sys.exit(app.exec_())
