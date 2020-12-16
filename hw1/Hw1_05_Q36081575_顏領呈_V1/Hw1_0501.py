import sys
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import random

'''
Q5.1
'''
print("--------Q5.1's Answer--------")
print("Load Cifar10 training dataset and randomly show 10 images and labels respectively.")

label_dict={0:"airplain",1:"automobile",2:"bird",3:"cat",4:"deer",5:"dog",
            6:"frog",7:"horse",8:"ship",9:"truck"}

def plot_images_labels_prediction(images,labels,prediction,idx,num=10):
    fig=plt.gcf()                                           #取得 pyplot 物件參考
    fig.set_size_inches(12, 14)                    #設定畫布大小為 12 吋*14吋
    if num > 25: 
        num=25                       #限制最多顯示 25 個子圖
    for i in range(0, num):                            #依序顯示 num 個子圖
        ax=plt.subplot(5, 5, i+1)                     #建立 5*5 個子圖中的第 i+1 個
        ax.imshow(images[idx], cmap='binary')      #顯示子圖
        title=str(idx) + "." + label_dict[labels[idx][0]] + str(labels[idx]) 
        if len(prediction) > 0:                    #有預測值就加入標題中
            title += ",predict=" + str(prediction[idx])
        ax.set_title(title, fontsize=10)            #設定標題
        ax.set_xticks([]);                                #不顯示 x 軸刻度
        ax.set_yticks([]);                                #不顯示 y 軸刻度
        idx += 1                                              #樣本序號增量 1
    plt.show()

a = random.randint(0, 10)
b = random.randint(11, 25)
c = b - a
while c != 10:
    a = random.randint(0, 10)
    b = random.randint(11, 25)
    c = b - a

(x_train_image, y_train_label), (x_test_image, y_test_label) = cifar10.load_data() #載入 Cifar-10 資料集
plot_images_labels_prediction(x_train_image, y_train_label,[], a, b)
