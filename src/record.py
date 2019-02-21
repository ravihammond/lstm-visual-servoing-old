#!/usr/bin/env python2
from __future__ import print_function

import rospy
import std_msgs.msg
import sensor_msgs.msg
import geometry_msgs.msg
import lstm_visual_servoing.msg

import cv2
import cv_bridge
bridge = cv_bridge.CvBridge()
import tf

from PIL import ImageFont, ImageDraw, Image
import numpy as np
import argparse
import sys
import os
import shutil
import json
import datetime
import time
import csv

from utils import query_yes_no

class Recorder():
    def __init__(self, training_dir, max_frames, prefix):
        rospy.init_node('recorder', anonymous=False)
        rospy.Subscriber("/camera/color/image_raw", sensor_msgs.msg.Image, self.image_callback)
        rospy.Subscriber("visual_control",lstm_visual_servoing.msg.Control,self.control_callback)
        rospy.Subscriber("record_enabled",lstm_visual_servoing.msg.Recorder,self.recorder_callback)
        self._tf_listener = tf.TransformListener()

        # Saves path of directory to save training data to 
        self._training_dir = training_dir
        # Number of frames to reach before recording is stopped
        self._max_frames = max_frames
        # Name of directory to save temporary images to
        self._temp_dir = self._training_dir + "temp/"
        # frame rate of robot and recorder (Hz)
        self._frame_rate = 20

        # Colors used for control panel text (B G R)
        self._color_white = (255,255,255)
        self._color_black = (40,40,40)
        
        # Colors used for control panel background (R G B)
        self._color_red = (200,50,50)
        self._color_blue = (50,50,200)

        # Time of start recording, used for the filenames
        self._recording_start_time = ""
        # Current state of GUI: { 0: NORMAL, 1: RECORDING, 2: SAVING }
        self._recording_state = 0

        # Paths of each saved image .jpg and velocities .json files for the current sequence
        self._recorded_velocities = []
        # Name of file to save robot control messages to
        self._vel_filename = "velocities.csv"
        # Current user input
        self._control_message = []

        # Name of meta data file
        self._total_meta_path = os.path.join(self._training_dir, "meta.json")
        # Meta data for current sequence
        self._seq_meta = { "sequences" : 1, "frames" : 0, "time" : 0 }
        # Meta data for current session 
        self._session_meta = { "sequences" : 0, "frames" : 0, "time" : 0 }
        # Meta data for all data in file
        self._total_meta = self.get_total_meta()

        # All fonts for recording panel text
        font_regular = "/usr/share/fonts/truetype/roboto/hinted/Roboto-Regular.ttf"
        font_bold = "/usr/share/fonts/truetype/roboto/hinted/Roboto-Bold.ttf"
        self._font_title = ImageFont.truetype(font_bold, 40)
        self._font_title_med = ImageFont.truetype(font_bold, 30)
        self._font_title_small = ImageFont.truetype(font_bold, 27)
        self._font_regular = ImageFont.truetype(font_regular, 25)

        self._sequence_save_prefix = prefix

    # Extract total meta information
    def get_total_meta(self):
        sequences = 0;
        frames = 0;
        time = 0;

        # Loop through each sequence
        for seq_dir in os.listdir(self._training_dir):
            seq_path = os.path.join(self._training_dir, seq_dir)
            vel_file_path = os.path.join(seq_path, self._vel_filename)

            # Ensure directory contains valid sequence
            if not os.path.isdir(seq_path) or not os.path.exists(vel_file_path):
                continue

            sequences += 1
                
            # Count number of frames in sequence
            with open(vel_file_path, "r") as f:
                reader = csv.reader(f, delimiter = ",")
                data = list(reader)
                frames += len(data) - 1

        return { "sequences" : sequences, "frames" : frames, "time" : frames / self._frame_rate}

    def record(self):
        # cv2.namedWindow("Recorder", cv2.WINDOW_OPENGL)
        cv2.namedWindow("Recorder")
        r = rospy.Rate(self._frame_rate)

        # Loop while ros is not shutdown, and the cv2 window is open
        while not rospy.is_shutdown() and cv2.getWindowProperty("Recorder", cv2.WND_PROP_VISIBLE) == 1.0:
            if self._img is not None:
                img_crop = self.get_cropped_image()
                img_show = cv2.resize(img_crop.copy(), (0,0), fx=1.4, fy=1.4)

                # Normal state
                if self._recording_state is 0:
                    img_show = self.normal_overlay(img_show)

                # Recording state
                elif self._recording_state is 1:
                    # Stop recording when maximum number of frames is reached
                    self._seq_meta['frames'] += 1
                    self._seq_meta['time'] = time.time() - self._record_start_time
                    img_show = self.recording_overlay(img_crop, img_show)

                    # Stop recording when maximum number of frames is reached
                    if self._seq_meta['frames'] >= self._max_frames:
                        self._recording_state = 2

                    # Filenames for current image and velocities being saved
                    frame_filename = "%06d.png" % self._seq_meta['frames']
                    frame_path = self._temp_dir + frame_filename

                    # Resize images to resnet size
                    img_resized = cv2.resize(img_crop, (224, 224))

                    # Add the x y z coordinates of the robot to the save file
                    camera_t, camera_r = self._tf_listener.lookupTransform(
                        'base','camera_color_optical_frame', rospy.Time())
                    csv_save_list = self._control_message + camera_t

                    # Save current frame and velocities
                    cv2.imwrite(frame_path, img_resized)
                    self._recorded_velocities.append(csv_save_list)

                # Saving state
                elif self._recording_state is 2:
                    img_show = self.saving_overlay(img_show)

                #show the image to the user
                h,w,c = img_show.shape
                cv2.resizeWindow("Recorder", w, h)
                cv2.imshow("Recorder", img_show)
                cv2.waitKey(1)

            r.sleep()

        # Delete directory if path exists
        if os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)

    # Crop image and return it
    def get_cropped_image(self):
        # Crop a square image out the center of the rectangular one
        h,w,c = self._img.shape
        s = min(h,w)
        x1 = (w-s)/2
        x2 = x1 + s
        y1 = (h-s)/2
        y2 = y1 + s
        img_crop= self._img[y1:y2,x1:x2,:]

        # Create a copy of the image to show the user with drawings
        return img_crop

    # Regular interface when not recording
    def normal_overlay(self, img_show):
        panel = self.create_panel(img_show, self._color_white)
        draw = ImageDraw.Draw(panel)

        self.draw_meta(draw, "Total", self._total_meta, 80)
        self.draw_meta(draw, "Current", self._session_meta, 400)

        return np.concatenate((img_show, np.array(panel)), axis=0)

    def draw_meta(self, draw, title, meta, pos):
        draw.text((pos, 8), title, self._color_black, font=self._font_title_small)

        label_str = "Sequences: %d" % meta["sequences"]
        draw.text((pos, 35), label_str, self._color_black, font=self._font_regular)

        label_str = "Time: %s" % self.get_time_str(meta["time"])
        draw.text((pos, 60), label_str, self._color_black, font=self._font_regular)

        label_str = "Frames: %s" % meta["frames"]
        draw.text((pos, 85), label_str, self._color_black, font=self._font_regular)

    # Interface when not recording
    def recording_overlay(self, img_crop, img_show):
        panel = self.create_panel(img_show, self._color_blue)
        draw = ImageDraw.Draw(panel)
        draw.text((30, 35),"Recording", self._color_white, font=self._font_title)
        self.draw_record_save_text(draw)

        return np.concatenate((img_show, np.array(panel)), axis=0)

    # Regular interface when not recording
    def saving_overlay(self, img_show):
        panel = self.create_panel(img_show, self._color_red)
        draw = ImageDraw.Draw(panel)
        draw.text((30, 40),"Save Recording? (Y/X)", self._color_white, font=self._font_title_med)
        self.draw_record_save_text(draw)

        return np.concatenate((img_show, np.array(panel)), axis=0)

    # Creates a panel menu for recording
    def create_panel(self, img_show, color):
        h,w,c = img_show.shape
        return Image.new("RGB", (w,130), color)

    # Draw background for control panel in SAVING state
    def draw_record_save_text(self, draw):
        label_str = "Time elapsed: %s" % self.get_time_str(self._seq_meta['time'])
        draw.text((368, 24), label_str, self._color_white, font=self._font_regular)

        label_str = "Frames: %d" % self._seq_meta['frames']
        draw.text((368, 64), label_str, self._color_white, font=self._font_regular)

    def get_time_str(self, t):
        return '%0.2d:%0.2d:%0.2d' % (t // 3600, (t % 3600) // 60, ((t % 3600) % 60))

    # Reset directory to store temporary image files
    def refresh_temp_dir(self):
        # Delete directory if path exists
        if os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)

        # Create temporary image directory
        os.makedirs(self._temp_dir)

    # Start recording sequence
    def start_recording(self):
        self._record_start_time = time.time()
        self._recording_state = 1
        self.refresh_temp_dir()
        self._time = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        self._recorded_velocities.append(['vx','vy','vz','rx','ry','rz','claw', 'px', 'py', 'pz'])

    # Adds the information of the previous recording to the total recording information
    def update_meta_data(self):
        keys = ["sequences", "time", "frames"]

        for key in keys:
            self._session_meta[key] += self._seq_meta[key]
            self._total_meta[key] += self._seq_meta[key]

        with open(self._total_meta_path, 'w') as fp:
            json.dump(self._total_meta, fp)

    # Saves the metadata for the current sequence, and load it to the JSON file
    def save_current_sequence(self):
        # Save recording information
        self.update_meta_data()

        # Rename temp directory
        if os.path.exists(self._temp_dir):
            os.rename(self._temp_dir, self._training_dir + self._time + self._sequence_save_prefix)

        # Write recorded velocities to file
        vel_path = os.path.join(self._training_dir, self._time + self._sequence_save_prefix, self._vel_filename)
        with open(vel_path, 'w') as csv_file:  
            csv.writer(csv_file, delimiter=',').writerows(self._recorded_velocities)

    # Delete the temp file holding the last recorded sequence
    def clear_current_sequence(self):
        # Delete temporary directory holding recorded frames
        if os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)

    # Saved option has been pressed, either delete or save the recorded sequence
    def save_option_pressed(self, is_saving):
        # SAVE option chosen
        if is_saving:
            # Save the recorded sequence metadata
            self.save_current_sequence()
        else:
            self.clear_current_sequence()

        # Reset all information saved for sequence
        self._seq_meta['frames'] = 0
        self._seq_meta['time'] = 0
        self._record_start_time = 0
        self._recorded_velocities = []

        # Move state of program to 0: NORMAL
        self._recording_state = 0

    # Handle Incoming video frame from realsense camera
    def image_callback(self, msg):
        rgb_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        bgr_img = rgb_img[...,::-1]
        self._img = bgr_img

    # Handle recorder control messages 
    def recorder_callback(self, msg):
        if self._recording_state is 0 and msg.record:
            self.start_recording()
        elif self._recording_state is 2:
            if msg.save:
                self.save_option_pressed(True)
            if msg.clear:
                self.save_option_pressed(False)

    # Handle robot control messages
    def control_callback(self, msg):
        self._control_message = [
                msg.vx,
                msg.vy,
                msg.vz,
                msg.rx,
                msg.ry,
                msg.rz,
                msg.claw
            ]

# Ensures the training data directory exists
def check_training_directory(directory):
    directory = directory.strip('/') + '/'

    # Ensures training data directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
        return directory

    # ask if you want to add to the contents of directory if not empty
    if len(os.listdir(directory)) > 0:
        query = "Directory: " + directory + " already contains training data, would you like to add to it?"
        if not query_yes_no(query):
            sys.exit()

    return directory


# Main function
if __name__ == "__main__" :
    # Argument parsing gets image directory, training data directory, and window size
    parser = argparse.ArgumentParser()
    parser.add_argument("training", help="name of directory to save recorded training data to")
    parser.add_argument("-f", "--frames", dest="frames", type=int, default=1000, help="automatically stops recording after given number of frames")
    parser.add_argument("-p", "--prefix", dest="prefix", type=str, default="", help="string to add to end of sequence directory")
    args = parser.parse_args()

    training_dir = check_training_directory(args.training)

    recorder = Recorder(training_dir, args.frames, args.prefix)
    recorder.record()

