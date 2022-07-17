from PyQt5.QtWidgets import QApplication, QPushButton, QRadioButton, QMainWindow, QAction
from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtCore, QtWidgets
import cv2
import easygui
from pylab import *
import os
import tkinter
import sys
from tkinter import messagebox
import matplotlib.pyplot as plt
import numpy as np

#os.chdir('E:/Univ/Info 3/Algoritmi de calcul stiintific main/untitled/Poze/')
imS = 1


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'Image processing'
        self.left = 150
        self.top = 100
        self.width = 500
        self.height = 600
        self.initui()

    def initui(self):
        # Fereastra principala
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setStyleSheet("background-color: rgb(132, 169, 217);")
        self.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        # groupbox1
        self.groupBox = QtWidgets.QGroupBox(self)
        self.groupBox.setGeometry(QtCore.QRect(40, 50, 300, 510))
        self.groupBox.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        # radiobuttons
        self.fedge = QRadioButton("Detecție de margini", self.groupBox)
        self.fedge.setGeometry(QtCore.QRect(20, 30, 200, 20))
        self.fedge.setObjectName("r1")
        self.fGaussian = QRadioButton("Filtrul Gaussian", self.groupBox)
        self.fGaussian.setGeometry(QtCore.QRect(20, 70, 200, 20))
        self.fGaussian.setObjectName("r2")
        self.fThresholding = QRadioButton("Thresholding", self.groupBox)
        self.fThresholding.setGeometry(QtCore.QRect(20, 110, 200, 20))
        self.fThresholding.setObjectName("r3")
        self.fGradient = QRadioButton("Gradient", self.groupBox)
        self.fGradient.setGeometry(QtCore.QRect(20, 150, 200, 20))
        self.fGradient.setObjectName("r4")
        self.fErosion = QRadioButton("Eroziune", self.groupBox)
        self.fErosion.setGeometry(QtCore.QRect(20, 190, 200, 20))
        self.fErosion.setObjectName("r5")
        self.fDilation = QRadioButton("Dilatare", self.groupBox)
        self.fDilation.setGeometry(QtCore.QRect(20, 230, 200, 20))
        self.fDilation.setObjectName("r6")
        self.fCorner = QRadioButton("Detecție de colțuri", self.groupBox)
        self.fCorner.setGeometry(QtCore.QRect(20, 270, 200, 20))
        self.fCorner.setObjectName("r7")
        self.fContour = QRadioButton("Detecție de contur", self.groupBox)
        self.fContour.setGeometry(QtCore.QRect(20, 310, 200, 20))
        self.fContour.setObjectName("r8")
        self.fContour2 = QRadioButton("Detecție de contur pe imagine", self.groupBox)
        self.fContour2.setGeometry(QtCore.QRect(20, 350, 200, 20))
        self.fContour2.setObjectName("r9")
        self.fContour3 = QRadioButton("Contur convex", self.groupBox)
        self.fContour3.setGeometry(QtCore.QRect(20, 390, 200, 20))
        self.fContour3.setObjectName("r10")
        # pushbuttons
        self.btnApply = QPushButton('Aplică', self.groupBox)
        self.btnApply.setGeometry(QtCore.QRect(100, 470, 75, 23))
        self.btnApply.setStyleSheet("background-color: rgb(132, 169, 217);")
        self.btnApply.setObjectName("btnApply")

        # Menu
        mainMenu = self.menuBar()
        loadMenu = mainMenu.addMenu('Încarcă')
        # button.clicked.connect(self.on_click)
        loadButton = QAction('Selectează poza din folder', self)
        loadButton.triggered.connect(self.load_click)
        loadMenu.addAction(loadButton)
        self.btnApply.setEnabled(False)
        self.btnApply.clicked.connect(self.modify)
        self.show()

    @pyqtSlot()
    def load_click(self):
        imgS = easygui.fileopenbox()
        img = cv2.imread(imgS)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.show()
        global imS
        imS = img_rgb
        self.btnApply.setEnabled(True)

    @staticmethod
    def edg():
        edges = cv2.Canny(imS, 100, 200) #cv2.Canny(imagine, threshold1, threshold2) aplica mai intai filtrul gaussian, apoi sobel x, sbel y...
        plt.imshow(edges)
        plt.axis('off')
        plt.show()

    @staticmethod
    def gaussian():
        gaussian_kernel_x = cv2.getGaussianKernel(5, 1) #(kernelSize, sigma)
        gaussian_kernel_y = cv2.getGaussianKernel(5, 1)
        gaussian_kernel = gaussian_kernel_x * gaussian_kernel_y.T
        filtered_image = cv2.filter2D(imS, -1, gaussian_kernel) #(imagine, destinatia/adancimea , kernel)
        plt.imshow(filtered_image)
        plt.axis('off')
        plt.show()

    @staticmethod
    def thresholding():
        imgn = cv2.cvtColor(imS, cv2.COLOR_RGB2GRAY)
        _, thresh_binary = cv2.threshold(imgn, thresh=127, maxval=255, type=cv2.THRESH_BINARY) # daca valoarea e mai mare de 127 =>light => =1 else =0
        plt.imshow(thresh_binary)
        plt.axis('off')
        plt.show()

    @staticmethod
    def gradient():
        img = cv2.cvtColor(imS, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(img, cv2.CV_64F, dx=1, dy=0, ksize=5)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, dx=0, dy=1, ksize=5)
        blended = cv2.addWeighted(src1=sobel_x, alpha=0.5, src2=sobel_y, beta=0.5, gamma=0)
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        images = [sobel_x, sobel_y, blended, laplacian]
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
        titleg = ['Sobel X', 'Sobel Y', 'Sobel X+Y', 'Laplacian']
        for i, p in enumerate(images):
            ax = axs[i % 2, i // 2]
            ax.imshow(p, cmap='gray')
            axs[i % 2, i // 2].set_title(titleg[i])
            ax.axis('off')
        plt.show()

    @staticmethod
    def erosion():
        img = imS
        kernel = np.ones((9, 9), np.uint8)
        img_copy = img.copy()
        img_copy = cv2.erode(img_copy, kernel, iterations=3)
        plt.imshow(img_copy)
        plt.axis('off')
        plt.show()

    @staticmethod
    def dilation():
        img = imS
        kernel = np.ones((9, 9), np.uint8)
        img_dilate = cv2.dilate(img, kernel, iterations=3)
        plt.imshow(img_dilate, cmap="gray")
        plt.axis('off')
        plt.show()

    @staticmethod
    def cornerD():
        img_gray = cv2.cvtColor(imS, cv2.COLOR_RGB2GRAY)
        # Apply Harris corner detection
        dst = cv2.cornerHarris(img_gray, blockSize=2, ksize=3, k=.04)
        img_2 = imS.copy()
        img_2[dst > 0.01 * dst.max()] = [255, 0, 0]
        # Plot the image
        plt.figure(figsize=(20, 20))
        plt.imshow(img_2)
        plt.axis('off')
        plt.show()

    @staticmethod
    def contour():
        img = imS
        img_blur = cv2.bilateralFilter(img, d=7, sigmaSpace=75, sigmaColor=75)
        # Convert to grayscale
        img_gray = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
        # Apply the thresholding
        a = img_gray.max()
        _, thresh = cv2.threshold(img_gray, a / 2 + 60, a, cv2.THRESH_BINARY_INV)
        plt.imshow(thresh, cmap='gray')
        plt.axis('off')
        plt.show()

    @staticmethod
    def contour2():
        img_blur = cv2.bilateralFilter(imS, d=7, sigmaSpace=75, sigmaColor=75)
        img_gray = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
        # Apply the thresholding
        a = img_gray.max()
        _, thresh = cv2.threshold(img_gray, a / 2 + 60, a, cv2.THRESH_BINARY_INV)
        #plt.imshow(thresh, cmap='gray')
        contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # Draw the contour
        img_copy = imS.copy()
        final = cv2.drawContours(img_copy, contours, contourIdx=-1, color=(0, 0, 255), thickness=2)
        plt.imshow(img_copy)
        plt.axis('off')
        plt.show()

    @staticmethod
    def contour3():
        img = imS
        img_blur = cv2.bilateralFilter(img, d=7, sigmaSpace=75, sigmaColor=75)
        img_gray = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
        a = img_gray.max()
        _, thresh = cv2.threshold(img_gray, a / 2 + 60, a, cv2.THRESH_BINARY_INV)
        plt.imshow(thresh, cmap='gray')
        contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        c_0 = contours[0]
        hull = cv2.convexHull(c_0)
        img_copy = img.copy()
        img_hull = cv2.drawContours(img_copy, contours=[hull], contourIdx=0, color=(255, 0, 0), thickness=2)
        plt.imshow(img_hull)
        plt.axis('off')
        plt.show()

    def modify(self):
        if self.fedge.isChecked():
            self.edg()
        elif self.fGaussian.isChecked():
            self.gaussian()
        elif self.fThresholding.isChecked():
            self.thresholding()
        elif self.fGradient.isChecked():
            self.gradient()
        elif self.fErosion.isChecked():
            self.erosion()
        elif self.fDilation.isChecked():
            self.dilation()
        elif self.fCorner.isChecked():
            self.cornerD()
        elif self.fContour.isChecked():
            self.contour()
        elif self.fContour2.isChecked():
            self.contour2()
        elif self.fContour3.isChecked():
            self.contour3()
        else:
            root = tkinter.Tk()
            root.withdraw()
            messagebox.showwarning("Atenție", "Selectați un filtru")
            return


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
