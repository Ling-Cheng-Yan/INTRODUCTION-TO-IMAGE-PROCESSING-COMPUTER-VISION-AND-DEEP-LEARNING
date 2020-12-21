import cv2
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtCore

import Ui_Q1


def Draw_Contour():
    
    for img_name in ['coin01.jpg', 'coin02.jpg']:
        img = cv2.imread(img_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
        gray = cv2.GaussianBlur(gray, (13, 13), 0)
        gray = cv2.GaussianBlur(gray, (15, 15), 0)
        binary = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,13,2)
        canny = cv2.Canny(binary, 30, 150)
    
        contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
        cv2.drawContours(img,contours,-1,(0,0,255),3)  
    
        cv2.imshow(img_name, img)


def Get_CoinNumbers(img):
    
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    gray = cv2.GaussianBlur(gray, (13, 13), 0)
    gray = cv2.GaussianBlur(gray, (15, 15), 0)
    binary = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,13,2)
    canny = cv2.Canny(binary, 30, 150)
  
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
    return len(contours)

def Show_CoinNumbers(ui):
    img_01 = cv2.imread('./coin01.jpg')
    img_02 = cv2.imread('./coin02.jpg')
    num_01 = Get_CoinNumbers(img_01)
    num_02 = Get_CoinNumbers(img_02)
    _translate = QtCore.QCoreApplication.translate
    ui.label.setText(_translate("Q1", f"There are {num_01}  coins in coin01.jpg"))
    ui.label_2.setText(_translate("Q1", f"There are {num_02} coins in coin02.jpg"))


if __name__=='__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_Q1.Ui_Q1()
    ui.setupUi(MainWindow)
    ui.Button_1_1.clicked.connect(Draw_Contour)
    ui.Button_1_2.clicked.connect(lambda:Show_CoinNumbers(ui))
    MainWindow.show()
    sys.exit(app.exec_())



    img_01 = cv2.imread('./coin01.jpg')
    img_02 = cv2.imread('./coin02.jpg')
    Draw_Contour(img_01)
    Draw_Contour(img_02)
    Get_CoinNumbers(img_01)
    Get_CoinNumbers(img_02)
