import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
import Ui_Q4
import copy
focal = 2826
baseline = 178

def draw_disparity(event,x,y,flags,param):
    global disparity, disparity_show, disparity_show_copy
    if event == cv2.EVENT_LBUTTONDOWN :
        disparity_show = copy.copy(disparity_show_copy)
        x_ = int(x*(disparity.shape[1]/640.0))
        y_ = int(y*(disparity.shape[0]/480.0))
        disparity_value = disparity[y_, x_]
        distance_value = int(focal*baseline/disparity_value)
        if disparity_value<0:
            disparity_value=0
            distance_value=0
        #disparity_show
        disparity_show[480-50:480, 640-200:640, :] = 255
        cv2.circle(disparity_show, (x, y), 2, (255, 0, 0), -1)
        cv2.putText(disparity_show, f'Disparity: {disparity_value} pixels', (640-200, 480-30), cv2.FONT_HERSHEY_DUPLEX,
  0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(disparity_show, f'Depth: {distance_value} mm', (640-200, 480-10), cv2.FONT_HERSHEY_DUPLEX,
  0.5, (0, 0, 0), 1, cv2.LINE_AA)


def show_disparity():
    global disparity, disparity_show, disparity_show_copy
    imgL = cv2.imread('./imgL.png', 0)
    imgR = cv2.imread('./imgR.png', 0)
    Max_Disparity = 256
    stereo = cv2.StereoBM_create(numDisparities=Max_Disparity, blockSize=25)
    
    disparity = stereo.compute(imgL, imgR)
    #plt.imshow(disparity, 'gray') 
    #plt.show() 
    
    #print(disparity)
    min = disparity.min()
    max = disparity.max()

    disparity_show = np.uint8(255.0 * (disparity - min) / (max - min))
    disparity_show = cv2.resize(disparity_show, (640, 480))
    disparity_show = np.tile(np.reshape(disparity_show, [disparity_show.shape[0], disparity_show.shape[1], 1]), (1, 1, 3))
    disparity_show_copy = copy.copy(disparity_show)
    #cv2.imshow('disparity', disparity)
    #cv2.waitKey()
    cv2.namedWindow('disparity')
    cv2.setMouseCallback('disparity',draw_disparity)
    while(1):
        cv2.imshow('disparity',disparity_show)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()  
if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_Q4.Ui_Q4()
    ui.setupUi(MainWindow)
    ui.pushButton.clicked.connect(show_disparity)
    MainWindow.show()
    sys.exit(app.exec_())
