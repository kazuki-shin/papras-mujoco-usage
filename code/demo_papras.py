#!/usr/bin/self.env python
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import JointState

import random
import numpy as np
import matplotlib.pyplot as plt
import time
from rotation import mat2euler, euler2mat
from mujoco_parser import MuJoCoParserClass
np.set_printoptions(precision=2)

print("Done.")

class PAPRAS_MuJoCo:
    def __init__(self):
        # ros init
        sub_joint_state = rospy.Subscriber('/joint_states', JointState, self.joint_state_callback)

        # mujoco init
        self.env = MuJoCoParserClass(
            name='PAPRAS', rel_xml_path='../asset/papras/mjmodel.xml', VERBOSE=False)
        self.env.init_viewer(TERMINATE_GLFW=True, INITIALIZE_GLFW=True,
                        window_width=0.5, window_height=0.5)
        self.env.set_max_tick(max_tick=2000)
        self.init_ik()

    def rand_val(self, minimum, maximum):
        return minimum + (maximum - minimum) * random.random()

    def joint_state_callback(self, msg):

        self.current_position = msg.position
        self.current_velocity = msg.velocity
        self.current_effort = msg.effort

    def init_ik(self):
        # Prepare for IK
        # panda_eef / panda_link_4 / panda_link_5 / panda_link_6
        self.body_name = 'robot1/end_link'
        self.q = self.env.get_q_pos(q_pos_idxs=self.env.rev_joint_idxs)
        self.p_EE = self.env.get_p_body(body_name=self.body_name)
        R_EE = self.env.get_R_body(body_name=self.body_name)
        self.p_trgt = self.p_EE + np.array([0.2, 0.0, -0.2])
        self.R_trgt = R_EE

        self.start_time = time.time()
        self.next_target = False

    def mirror_joint_state(self):
        self.q[:6] = self.current_position[2:]
        self.q[-2:] = self.current_position[:2]
        self.env.forward(q_pos=self.q, q_pos_idxs=self.env.rev_joint_idxs)

    def random_ik(self):
        if self.next_target:
            self.next_target = False
            self.start_time = time.time()
            self.p_trgt_x = self.rand_val(-0.2, 0.05)
            self.p_trgt_y = self.rand_val(-0.2, 0.2)
            self.p_trgt_z = self.rand_val(-0.2, 0.05)
            self.p_trgt = self.p_EE + np.array([self.p_trgt_x, self.p_trgt_y, self.p_trgt_z])

            trgt_roll = self.rand_val(-0.785398, 0.785398)
            trgt_pitch = self.rand_val(-0.785398, 0.785398)
            trgt_yaw = self.rand_val(-0.785398, 0.785398)
            self.R_trgt_euler = [trgt_roll, trgt_pitch, trgt_yaw]
            self.R_trgt = euler2mat(self.R_trgt_euler)

            # print("target XYZ: ", self.p_trgt, "   RPY: ", self.R_trgt_euler)

        # Numerical IK
        dq, err = self.env.one_step_ik(
            body_name=self.body_name, p_trgt=self.p_trgt, R_trgt=self.R_trgt, th=3.0*np.pi/180.0)
        self.q = self.q + dq

        if (err < 1e-4).all():
            end_time = time.time()
            total_time = end_time - self.start_time
            # print(total_time)
            sel
            f.next_target = True

        # FK
        self.env.forward(q_pos=self.q, q_pos_idxs=self.env.rev_joint_idxs)

    def render(self):
        self.env.add_marker(self.env.get_p_body(self.body_name), radius=0.1,
                    color=np.array([1, 0, 0, 0.5]))
        self.env.add_marker(self.p_trgt, radius=0.1, color=np.array([0, 0, 1, 0.5]))
        self.env.render(RENDER_ALWAYS=True)

if __name__ == '__main__':
    rospy.init_node('mujoco', anonymous=True)
    rate = rospy.Rate(125) # 10hz

    papras_mujoco = PAPRAS_MuJoCo()

    while not rospy.is_shutdown():

        papras_mujoco.random_ik()
        papras_mujoco.render()
        
        rate.sleep()