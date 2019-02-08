#!/usr/bin/env python2
import rospy
import std_msgs.msg
import sensor_msgs.msg
import geometry_msgs.msg
import lstm_visual_servoing.msg
import tf
import pyquaternion
import math
import numpy as np
import time

class StateMachine:
    def __init__(self):
        rospy.init_node('state_maching', anonymous=False)
        rospy.Subscriber("visual_control", lstm_visual_servoing.msg.Control, self.visual_control_callback)
        rospy.Subscriber("gripper_pos", std_msgs.msg.Float32, self.gripper_pos_callback)

        self.pub = rospy.Publisher("move_at_speed",geometry_msgs.msg.Twist,queue_size=10)
        self.pub_grip = rospy.Publisher("move_gripper_to_pos",std_msgs.msg.Float32,queue_size=10)
        self.tf_listener = tf.TransformListener()
        self.tf_broadcaster = tf.TransformBroadcaster()

        self.last_control_msg_time = time.time()
        print("State Machine Spinning")
        self.spin()
        self.gripper_pos = 0

    def spin(self):
        #create blank message
        self.visual_control_msg = lstm_visual_servoing.msg.Control()

        claw_control = 0

        #process the latest control message at 30Hz
        r = rospy.Rate(20)
        while not rospy.is_shutdown():
            try:
                camera_t,camera_r = self.tf_listener.lookupTransform( 'base','camera_rgb_optical_frame', rospy.Time())
                q_camera = pyquaternion.Quaternion(camera_r[3],camera_r[0],camera_r[1],camera_r[2])
                ctrl = self.visual_control_msg
                tran_v = q_camera.rotate((ctrl.vx,ctrl.vy,ctrl.vz))
                rot_v = q_camera.rotate((ctrl.rx,ctrl.ry,ctrl.rz))
                claw_open = ctrl.open
                claw_close = ctrl.close

                safe_trans_v, safe_rot_v = self.get_safety_return_speeds(camera_t,camera_r)

                msg = geometry_msgs.msg.Twist()
                msg.linear.x, msg.linear.y, msg.linear.z = tran_v
                msg.angular.x, msg.angular.y, msg.angular.z = rot_v

                dist = math.sqrt(camera_t[0]**2 + camera_t[1]**2)
                msg.linear.x += safe_trans_v[0]
                msg.linear.y += safe_trans_v[1]
                msg.linear.z += safe_trans_v[2] 
                if camera_t[2] < 0.25 and msg.linear.z < 0:
                    msg.linear.z = 0.00
                percent = 1 if dist < 0.92 else max(0,1 - ((dist - 0.92) / (0.95 - 0.92)))
                if camera_t[2] < 0.23 and msg.linear.z <= 0:
                    msg.linear.z = 0.2 * percent

                msg.angular.x += safe_rot_v[0]
                msg.angular.y += safe_rot_v[1]
                msg.angular.z += safe_rot_v[2]

                v_speed = 0.15 #m/s
                r_speed = math.radians(20) #deg/s
                msg.linear.x *= v_speed
                msg.linear.y *= v_speed
                msg.linear.z *= v_speed

                msg.angular.x *= r_speed
                msg.angular.y *= r_speed
                msg.angular.z *= r_speed

                if claw_open == 1.0:
                    claw_control = 1
                if claw_close == 1.0:
                    claw_control = 90

                # print(msg)
                if time.time() - self.last_control_msg_time < 1.0:
                    if claw_open is not claw_close:
                        self.pub_grip.publish(claw_control)
                    self.pub.publish(msg)
                # self.tf_broadcaster.sendTransform(camera_t,(safe_q[1],safe_q[2],safe_q[3],safe_q[0] ),rospy.Time.now(),"safe","base")

            except (tf.LookupException,tf.ExtrapolationException):
                pass

            r.sleep()

    def get_safety_return_speeds(self,camera_t, camera_r):

        safe_trans_v = [0.0,0.0,0.0]
        safe_rot_v = np.array([0.0,0.0,0.0])

        q_camera = pyquaternion.Quaternion(camera_r[3],camera_r[0],camera_r[1],camera_r[2])


        cam_z = q_camera.rotate(np.array([0.0,0.0,1.0]))
        cam_y = q_camera.rotate(np.array([0.0,1.0,0.0]))


        #PAN TILT LIMITS
        full_rot_speed = 2.2 #degrees per second
        full_rot_speed_dist = math.radians(5) #degrees per second

        #get the normal to the plane that runs throught the robot z axis and the camera position
        v_plane_norm = np.array([-camera_t[1],camera_t[0],0.0])
        v_plane_norm /= np.linalg.norm(v_plane_norm)
        #project camera z axis onto plane
        z_proj = cam_z - np.dot(cam_z,v_plane_norm)*v_plane_norm
        #normalise z projection
        z_proj /= np.linalg.norm(z_proj)

        #PAN
        pan_angle_limit = math.radians(1) # degrees
        #get the pan rotation axis between camera z and plane
        pan_axis = -np.cross(z_proj,cam_z)
        #get the pan angle
        pan_angle = np.linalg.norm(pan_axis) #TODO This not correct way to get angle from cross product. It needs a sin
        #normalise pan rotation axis
        pan_axis /= np.linalg.norm(pan_axis)

        pan_return_speed = min(full_rot_speed,max(0,(pan_angle-pan_angle_limit)/full_rot_speed_dist*full_rot_speed))

        safe_rot_v += pan_axis * pan_return_speed

        #TILT LIMIT
        out_norm = np.array([camera_t[0],camera_t[1],0.0])
        out_norm /= np.linalg.norm(out_norm)
        tilt_axis = np.cross(z_proj,out_norm)
        tilt_sign_y = np.sign(np.dot(tilt_axis,v_plane_norm))
        tilt_sign_x = np.sign(np.dot(z_proj,out_norm))

        tilt_angle = math.asin(np.linalg.norm(tilt_axis))
        if tilt_sign_x < 0:
            tilt_angle = math.pi - tilt_angle
        tilt_angle *= tilt_sign_y


        tilt_angle_min = math.radians(-40)
        tilt_raw = -80
        tilt_angle_max = math.radians(tilt_raw)
        dist = math.sqrt(camera_t[0]**2 + camera_t[1]**2)
        tilt_lock = 0.75
        tilt_lock_max = 0.92
        if dist > tilt_lock:
            percent = (dist - tilt_lock) / (tilt_lock_max - tilt_lock)
            tilt_angle_max = math.radians(tilt_raw + ((-62-tilt_raw) * percent))

        tilt_return_speed = -min(full_rot_speed,max(0,
            (tilt_angle-tilt_angle_max)/full_rot_speed_dist*full_rot_speed))
        tilt_return_speed += min(full_rot_speed,max(0,
            (tilt_angle_min-tilt_angle)/full_rot_speed_dist*full_rot_speed))

        # tilt_return_speed
        safe_rot_v -= v_plane_norm * tilt_return_speed

        #LOOK UP
        if tilt_angle > -math.pi/4 and tilt_angle < math.pi/4:
            up_dir = np.array([0.0,0.0,1.0])
        else:
            up_dir = out_norm
        #project up direction onto camera xy plane
        up_proj = up_dir - np.dot(up_dir,cam_z)*cam_z
        #normalise project up direction
        up_proj /= np.linalg.norm(up_proj)
        #If up direction is withing +- 90deg  cam_y
        if np.dot(up_proj,-cam_y) > 0:
            #Get rotation speed from cross product
            safe_rot_v += -np.cross(up_proj,-cam_y)*5

        full_speed_dist = 0.04
        full_speed = 2.0

        #Inner Cylinder
        cylinder_radius = 0.28
        dist = math.sqrt( camera_t[0]**2 + camera_t[1]**2)
        cylinder_return_speed = max(0,(cylinder_radius-dist)/full_speed_dist*full_speed)
        safe_trans_v[0] += camera_t[0]/dist * cylinder_return_speed
        safe_trans_v[1] += camera_t[1]/dist * cylinder_return_speed

        #Outer Sphere
        sphere_radius = 0.72
        dist = math.sqrt( (camera_t[0] * 0.73)**2 + (camera_t[1] * 0.73)**2 + (camera_t[2] * 1)**2)
        cylinder_return_speed = min(0,(sphere_radius-dist)/full_speed_dist*full_speed)
        safe_trans_v[0] += camera_t[0]/dist * cylinder_return_speed
        safe_trans_v[1] += camera_t[1]/dist * cylinder_return_speed
        safe_trans_v[2] += camera_t[2]/dist * cylinder_return_speed

        #back Wall
        wall_unit_norm = [0.7071,-0.7071]
        dist = camera_t[0] * wall_unit_norm[0] + camera_t[1] * wall_unit_norm[1] + 0.13
        wall_return_speed = -min(0,dist/full_speed_dist*full_speed)
        safe_trans_v[0] += wall_unit_norm[0] * wall_return_speed
        safe_trans_v[1] += wall_unit_norm[1] * wall_return_speed

        return (safe_trans_v, safe_rot_v.tolist())

    def gripper_pos_callback(self, data):
        self.gripper_pos = data

    def visual_control_callback(self, data):
        self.visual_control_msg = data
        self.last_control_msg_time = time.time()

if __name__ == "__main__":
    StateMachine()

