#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

import cv2
import sys
import numpy as np

class My_App(QtWidgets.QMainWindow):
    def __init__(self):
        super(My_App, self).__init__()
        loadUi("./SIFT_app.ui", self)

        # Hardcoded video path
        self._video_path = "/home/fizzer/SIFT_app/video.mp4"
        
        self._cam_fps = 10
        self._is_cam_enabled = False
        self._is_template_loaded = False

        self.browse_button.clicked.connect(self.SLOT_browse_button)
        self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)

        # Timer setup
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.SLOT_query_camera)
        self._timer.setInterval(int(1000 / self._cam_fps) )

         # SIFT and FLANN setup
        self.sift = cv2.SIFT_create()
        FLANN_INDEX_KDTREE = 1 #0 vs 1?
        self.index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees=5)
        self.search_params = dict(checks = 50) #empty vs checks 50
        self.flann = cv2.FlannBasedMatcher(self.index_params, self.search_params)

    def SLOT_browse_button(self):
        dlg = QtWidgets.QFileDialog()
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)

        if dlg.exec_():
            self.template_path = dlg.selectedFiles()[0]
            self.template_img = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)#load template as grayscale

            self._is_template_loaded = True #sets boolean to true

            pixmap = QtGui.QPixmap(self.template_path)
            self.template_label.setPixmap(pixmap)
            print("Loaded template image file: " + self.template_path)

    def convert_cv_to_pixmap(self, cv_img):
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, channel = cv_img.shape
        bytesPerLine = channel * width
        q_img = QtGui.QImage(cv_img.data, width, height, 
                    bytesPerLine, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(q_img)

    
    def SLOT_query_camera(self):#called repeatedly by thge timer
        if not hasattr(self, '_camera_device') or not self._camera_device.isOpened():
            return

        ret, frame = self._camera_device.read()
        
        if not ret:  # if reach end of video
            self._timer.stop()
            self._camera_device.release()
            self._is_cam_enabled = False
            self.toggle_cam_button.setText("&Enable camera")
            print("Video playback finished")
            return

        

        if hasattr(self, 'template_img') and self._is_template_loaded:
              # Convert frame to grayscale
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#decode into BGR, then converts to grayscale, obtain original colour data

            # Detect SIFT features in template and frame
            kp1, des1 = self.sift.detectAndCompute(self.template_img, None)
            kp2, des2 = self.sift.detectAndCompute(frame_gray, None)

            # Match features using FLANN
            matches = self.flann.knnMatch(des1, des2, k=2)

            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

            # Compute homography if enough good matches are found
            if len(good_matches) > 10:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Compute homography
                H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                # Get template corners
                h, w = self.template_img.shape
                corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)

                # Transform template corners to frame using homography
                transformed_corners = cv2.perspectiveTransform(corners, H)

                # Draw bounding box on the frame
                frame = cv2.polylines(frame, [np.int32(transformed_corners)], True, (0, 255, 0), 3)
            
        #DISPLAY FRAME IN GUI
        pixmap = self.convert_cv_to_pixmap(frame)
        self.live_image_label.setPixmap(pixmap)


    def SLOT_toggle_camera(self):
        if self._is_cam_enabled:
            self._timer.stop()
            if hasattr(self, '_camera_device'):
                self._camera_device.release()
            self._is_cam_enabled = False
            self.toggle_cam_button.setText("&Enable camera")
        else:                
            # Initialize video capture with hardcoded file path
            self._camera_device = cv2.VideoCapture(self._video_path)
            
            if not self._camera_device.isOpened():
                print(f"Error opening video file: {self._video_path}")
                return
                
            self._timer.start()
            self._is_cam_enabled = True
            self.toggle_cam_button.setText("&Disable camera")
    

    


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myApp = My_App()
    myApp.show()
    sys.exit(app.exec_())