#!/usr/bin/env python2
import torch
import rospy
import tf

import cv2
import numpy as np
import os
import sys

import sensor_msgs.msg
import lstm_visual_servoing.msg
import cv_bridge
bridge = cv_bridge.CvBridge()
from torchvision import transforms

class VisualServo():
    def __init__(self, model_path):
        rospy.init_node('visual_servo', anonymous=False)

        self.ctrl_pub = rospy.Publisher("visual_control",lstm_visual_servoing.msg.Control,queue_size=10)
        rospy.Subscriber("/camera/color/image_raw", sensor_msgs.msg.Image, self.image_callback)
        self._tf_listener = tf.TransformListener()

        self._img_rgb = None

        self._model = torch.load(model_path)

        self._model.eval()
        self._model.cuda()

        cv2.namedWindow("Camera",0)

    def run(self):
        #create blank message
        msg = lstm_visual_servoing.msg.Control()

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])

        out_smooth = np.zeros((6))
        smooth_a = 0.2

        #publish the latest control message at 30Hz
        r = rospy.Rate(20)

        i = 0
        while not rospy.is_shutdown():
            if not self._img_rgb is None:
                i += 1
                #crop the center of the image
                h,w,c = self._img_rgb.shape
                s = min(h,w)
                x1 = (w-s)/2
                x2 = x1 + s
                y1 = (h-s)/2
                y2 = y1 + s

                img_crop= self._img_rgb[y1:y2,x1:x2,:]

                img_resized = cv2.resize(img_crop,(224,224))
                cv2.imshow("Camera",img_resized[:,:,::-1])
                cv2.waitKey(1)
                img_tensor = transform(img_resized)
                img_tensor.unsqueeze_(0)

                X_img = torch.autograd.Variable(img_tensor, volatile=True).cuda()
                camera_t, camera_r = self._tf_listener.lookupTransform(
                    'base','camera_color_optical_frame', rospy.Time())
                np_coords = np.array(camera_t).reshape((1,3))
                # print("np_coords")
                # print(np_coords.shape)
                # print(np_coords)
                tens_coords = torch.FloatTensor(np_coords)
                # print("tens_coords")
                # print(tens_coords.shape)
                # print(tens_coords)
                X_coords = torch.autograd.Variable(tens_coords, volatile=True).cuda()
                # print("X_coords")
                # print(X_coords.shape)
                # print(X_coords)

                y_vel, y_claw = self._model(X_img, X_coords)
                np_vel = np.squeeze(np.array(y_vel.data))
                np_claw = np.squeeze(np.array(y_claw.data))

                vel_str = np.array2string(np_vel, precision=3, floatmode='fixed')
                claw_str = np.array2string(np_claw, precision=0, floatmode='fixed')

                msg_vel = np_vel * 3.8 * 1.5
                msg_vel = smooth_a * (msg_vel - out_smooth)

                msg_claw = np_claw
                # output_str = "%s %s:%d i: %d" % (vel_str, claw_str, i)
                # print(output_str)
                # print(msg_vel)
                # print(msg_open)
                # print(msg_close)

                msg = lstm_visual_servoing.msg.Control()
                msg.vx,msg.vy,msg.vz,msg.rx,msg.ry,msg.rz = msg_vel
                msg.claw = msg_claw

            self.ctrl_pub.publish(msg)
            r.sleep()

    def image_callback(self, msg):
        self._img_rgb = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

