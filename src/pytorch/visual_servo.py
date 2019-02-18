#!/usr/bin/env python2
import torch
import rospy

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
                # img_resized = img_resized[:,:,::-1].copy()
                img_tensor = transform(img_resized)
                # print(img_tensor)
                img_tensor.unsqueeze_(0)

                X = torch.autograd.Variable(img_tensor, volatile=True).cuda()

                y_vel, y_open, y_close = self._model(X)
                np_vel = np.squeeze(np.array(y_vel.data))
                np_open = np.squeeze(np.array(y_open.data))
                np_close = np.squeeze(np.array(y_close.data))

                # np_open = np.array([5,4], dtype=np.float)
                # if i % 100 == 0:
                    # np_open = np.array([3,6], dtype=np.float)

                # np_close = np.array([5,4], dtype=np.float)
                # if (i + 50) % 100 == 0:
                    # np_close = np.array([3,6], dtype=np.float)

                vel_str = np.array2string(np_vel, precision=3, floatmode='fixed')
                open_str = np.array2string(np_open, precision=0, floatmode='fixed')
                close_str = np.array2string(np_close, precision=0, floatmode='fixed')

                msg_vel = np_vel * 1.8
                msg_vel = smooth_a * (msg_vel - out_smooth)

                y_open.cpu()
                msg_open = np.where(np_open == np.max(np_open))[0]
                msg_close = np.where(np_close == np.max(np_close))[0]
                output_str = "%s %s:%d %s:%d i: %d" % (vel_str, open_str, msg_open, close_str, msg_close, i)
                print(output_str)
                # print(msg_vel)
                # print(msg_open)
                # print(msg_close)

                msg = lstm_visual_servoing.msg.Control()
                msg.vx,msg.vy,msg.vz,msg.rx,msg.ry,msg.rz = msg_vel
                msg.open = msg_open
                msg.close = msg_close

            self.ctrl_pub.publish(msg)
            r.sleep()

    def image_callback(self, msg):
        self._img_rgb = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

