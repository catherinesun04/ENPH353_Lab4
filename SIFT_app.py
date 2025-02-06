#!/usr/bin/env python3

## @file sift_app.py
#  @brief A PyQt5-based GUI application for SIFT-based object detection in video streams.
#  @details This application loads a video file and allows users to load a template image.
#  It then detects the template in the video using the SIFT feature detector and FLANN-based matcher.

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi
import cv2
import sys
import numpy as np

## @class My_App
#  @brief Main application class handling UI and SIFT-based image processing.
#  @details This class initializes the GUI, sets up event listeners, and manages video processing.
class My_App(QtWidgets.QMainWindow):
    ## @brief Constructor method to initialize the application.
    #  @details Sets up UI components, initializes OpenCV SIFT detector, and sets up video capture.
    def __init__(self):
        super(My_App, self).__init__()
        loadUi("./SIFT_app.ui", self)

        ## @brief Hardcoded video file path.
        self._video_path = "/home/fizzer/SIFT_app/video.mp4"
        
        ## @brief Frame rate for the video feed.
        self._cam_fps = 10
        
        ## @brief Boolean flag indicating if the camera feed is enabled.
        self._is_cam_enabled = False
        
        ## @brief Boolean flag indicating if a template image is loaded.
        self._is_template_loaded = False

        # Connecting UI buttons to their respective functions
        self.browse_button.clicked.connect(self.SLOT_browse_button)
        self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)

        ## @brief Timer to periodically fetch frames from the video feed.
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.SLOT_query_camera)
        self._timer.setInterval(int(1000 / self._cam_fps))

        # SIFT and FLANN initialization
        ## @brief SIFT feature detector instance.
        self.sift = cv2.SIFT_create()
        
        ## @brief FLANN index parameter.
        FLANN_INDEX_KDTREE = 1  # KD-tree algorithm
        
        ## @brief Parameters for FLANN-based matcher.
        self.index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        self.search_params = dict(checks=50)  # Number of times the algorithm iterates for matches
        
        ## @brief FLANN-based matcher instance.
        self.flann = cv2.FlannBasedMatcher(self.index_params, self.search_params)

    ## @brief Opens a file dialog to allow the user to select a template image.
    #  @details The selected image is converted to grayscale and displayed in the UI.
    def SLOT_browse_button(self):
        dlg = QtWidgets.QFileDialog()
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)

        if dlg.exec_():
            self.template_path = dlg.selectedFiles()[0]
            self.template_img = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)  # Load template as grayscale

            self._is_template_loaded = True  # Set flag to indicate template is loaded

            pixmap = QtGui.QPixmap(self.template_path)
            self.template_label.setPixmap(pixmap)
            print("Loaded template image file: " + self.template_path)

    ## @brief Converts an OpenCV image to a QPixmap for GUI display.
    #  @param cv_img OpenCV image in BGR format.
    #  @return QPixmap object for display in the UI.
    def convert_cv_to_pixmap(self, cv_img):
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, channel = cv_img.shape
        bytesPerLine = channel * width
        q_img = QtGui.QImage(cv_img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(q_img)

    ## @brief Reads frames from the video feed and detects the template using SIFT.
    #  @details This function is called periodically by a QTimer.
    def SLOT_query_camera(self):
        if not hasattr(self, '_camera_device') or not self._camera_device.isOpened():
            return

        ret, frame = self._camera_device.read()
        
        if not ret:  # If end of video reached
            self._timer.stop()
            self._camera_device.release()
            self._is_cam_enabled = False
            self.toggle_cam_button.setText("&Enable camera")
            print("Video playback finished")
            return

        # If a template image is loaded, perform SIFT-based detection
        if hasattr(self, 'template_img') and self._is_template_loaded:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale

            kp1, des1 = self.sift.detectAndCompute(self.template_img, None)
            kp2, des2 = self.sift.detectAndCompute(frame_gray, None)

            matches = self.flann.knnMatch(des1, des2, k=2)

            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

            # Compute homography if enough good matches are found
            if len(good_matches) > 10:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                h, w = self.template_img.shape
                corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)

                transformed_corners = cv2.perspectiveTransform(corners, H)
                frame = cv2.polylines(frame, [np.int32(transformed_corners)], True, (0, 255, 0), 3)
        
        # Display frame in GUI
        pixmap = self.convert_cv_to_pixmap(frame)
        self.live_image_label.setPixmap(pixmap)

    ## @brief Toggles the video feed on and off.
    #  @details This function initializes the video capture and starts/stops the QTimer accordingly.
    def SLOT_toggle_camera(self):
        if self._is_cam_enabled:
            self._timer.stop()
            if hasattr(self, '_camera_device'):
                self._camera_device.release()
            self._is_cam_enabled = False
            self.toggle_cam_button.setText("&Enable camera")
        else:                
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
