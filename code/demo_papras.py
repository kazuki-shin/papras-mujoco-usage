#!/usr/bin/self.env python
import rospy
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TwistStamped, PoseStamped
import random
import numpy as np
import matplotlib.pyplot as plt
import time
from rotation import mat2euler, euler2mat
from mujoco_parser import MuJoCoParserClass
import pdb

np.set_printoptions(precision=2)

print("Done.")

class PAPRAS_MuJoCo:
    def __init__(self):
        # ros init

        sub_delta_x = rospy.Subscriber('/servo_server/left_pos_delta',PoseStamped, self.get_delta_x_message)
        sub_twisted_joint_state = rospy.Subscriber('/servo_server/delta_twist_cmds',TwistStamped, self.twisted_stamp_callback)
        self.execute_pub = rospy.Publisher("/arm1_controller/command", JointTrajectory, queue_size=100)
        self.use_incoming_delta_x = True
        self.publish_period = 0.008
        self.planning_frame = "robot1/link1"
        self.joint_state = JointState()
        self.latest_twist_msg = None
        self.new_input = False
        self.get_new_joint_state = True
        self.delta_x = [0,0,0,0,0,0]
        self.body_name = 'robot1/end_link'
        # mujoco init
        self.env = MuJoCoParserClass(
            name='PAPRAS', rel_xml_path='../asset/papras/mjmodel.xml', VERBOSE=True)
        self.env.init_viewer(TERMINATE_GLFW=True, INITIALIZE_GLFW=True,
                        window_width=0.5, window_height=0.5)
        
        # Prepare for IK
        self.q = self.env.get_q_pos(q_pos_idxs=self.env.rev_joint_idxs)
        self.p_EE = self.env.get_p_body(body_name=self.body_name)
        R_EE = self.env.get_R_body(body_name=self.body_name)
        self.p_trgt = self.p_EE + np.array([0.2, 0.0, -0.2])
        self.R_trgt = R_EE

        self.start_time = time.time()
        self.next_target = False

    def get_delta_x_message(self,msg):
        delta_x_pos = np.array([ msg.pose.position.x, 
                        msg.pose.position.y,
                        msg.pose.position.z 
                        ])
        delta_x_ori = np.array([ msg.pose.orientation.x,
                        msg.pose.orientation.y,
                        msg.pose.orientation.z])

        base_frame = "robot1/link2"
        eef_frame = "robot1/end_link"

        base_frame_r = self.env.get_R_body(body_name=base_frame)
        base_frame_r = np.linalg.inv(base_frame_r)

        eef_frame_r = self.env.get_R_body(body_name=eef_frame)

        base_to_eef_r = base_frame_r @ eef_frame_r 

        cur_linear_delta = (base_to_eef_r @ delta_x_pos.T)
        cur_angular_delta = (base_to_eef_r @ delta_x_ori.T) 
        self.delta_x = np.concatenate((cur_linear_delta,cur_angular_delta),axis=None)


    def rand_val(self, minimum, maximum):
        return minimum + (maximum - minimum) * random.random()

    def twisted_stamp_callback(self, msg):
        # update to most recent twist_msg
        self.latest_twist_msg = msg
        self.new_input = True
    
    def calculate_cartesian_cmd(self, twist_msg):
        res = np.empty(6)

        res[0] = twist_msg.twist.linear.x * self.publish_period
        res[1] = twist_msg.twist.linear.y * self.publish_period
        res[2] = twist_msg.twist.linear.z * self.publish_period
        res[3] = twist_msg.twist.angular.x * self.publish_period
        res[4] = twist_msg.twist.angular.y * self.publish_period
        res[5] = twist_msg.twist.angular.z * self.publish_period
        
        return res
    
    def create_transformation_matrix(self,R,p):
        p.shape = (p.shape[0],1)
        tfm = np.hstack((R,p))
        tfm = np.vstack((tfm,[0,0,0,1]))
        return tfm
        

    def calculate_single_iteration(self):
        if self.new_input != False:
            self.new_input = False
            current_twist_msg = self.latest_twist_msg
            # transform into base frame
            cur_linear_vel = np.array([current_twist_msg.twist.linear.x, current_twist_msg.twist.linear.y, current_twist_msg.twist.linear.z])
            cur_angular_vel = np.array([current_twist_msg.twist.angular.x, current_twist_msg.twist.angular.y, current_twist_msg.twist.angular.z])
            #
            base_frame = "robot1/link2"
            eef_frame = "robot1/end_link"


            base_frame_r = self.env.get_R_body(body_name=base_frame)
            base_frame_r = np.linalg.inv(base_frame_r)

            eef_frame_r = self.env.get_R_body(body_name=eef_frame)

            base_to_eef_r = base_frame_r @ eef_frame_r 

            cur_linear_vel = (base_to_eef_r @ cur_linear_vel.T)
            cur_angular_vel = (base_to_eef_r @ cur_angular_vel.T) 

            updated_twist_msg = TwistStamped()
            updated_twist_msg.twist.linear.x = cur_linear_vel[0]
            updated_twist_msg.twist.linear.y = cur_linear_vel[1]
            updated_twist_msg.twist.linear.z = cur_linear_vel[2]
            updated_twist_msg.twist.angular.x = cur_angular_vel[0]
            updated_twist_msg.twist.angular.y = cur_angular_vel[1]
            updated_twist_msg.twist.angular.z = cur_angular_vel[2]


            self.delta_x = self.calculate_cartesian_cmd(updated_twist_msg)

    def servo_calc(self):
                
        eef_frame = "robot1/end_link"
        delta_theta, _ = self.env.one_step_ik(body_name = eef_frame, delta_x = self.delta_x)
        self.delta_x = np.zeros(6)

        if (abs(delta_theta) < 1e-4).all():
            return 

        new_joint_state = JointState()
        self.q = self.q + delta_theta
        temp_pos = self.q[:6]
        new_joint_state.position = temp_pos

        # update internal robot state in mujoco scene
        self.env.forward(q_pos=self.q, q_pos_idxs=self.env.rev_joint_idxs)        

        # Stop if new configuration in collision
        # n_contacts = self.env.sim.data.ncon
        # if n_contacts > 1:
        #     new_joint_state.position = curr_pos[2:]
        #     print("Collision Detected. Halting robot.")
        #     # Visualize collisions
        #     for c_idx in range(n_contacts):
        #         contact    = self.env.sim.data.contact[c_idx]
        #         p_contact  = contact.pos
        #         bodyname1  = self.env.body_idx2name(self.env.sim.model.geom_bodyid[contact.geom1])
        #         bodyname2  = self.env.body_idx2name(self.env.sim.model.geom_bodyid[contact.geom2])
        #         label      = '%s-%s'%(bodyname1,bodyname2)
        #         self.env.add_marker(pos=p_contact,radius=0.1,color=np.array([1,0,0,0.5]),label=label)
        
        
        # calc joint velocities
        # TODO: temp removing sync with Gazebo
        new_joint_state.velocity = delta_theta[:6] / self.publish_period            

        # Compose joint trajectory message
        traj_msg = JointTrajectory()
        traj_msg.header.stamp = rospy.Time()
        traj_msg.header.frame_id = self.planning_frame
        traj_msg.joint_names = ['robot1/joint1',
                                'robot1/joint2',
                                'robot1/joint3',
                                'robot1/joint4',
                                'robot1/joint5',
                                'robot1/joint6']

        # Append traj point 
        point = JointTrajectoryPoint()                  
        point.positions = new_joint_state.position
        point.velocities = new_joint_state.velocity
        point.time_from_start = rospy.Duration(self.publish_period)
        traj_msg.points.append(point)
        # enforce position bound limits on joint_state
        self.execute_pub.publish(traj_msg)

    def render(self):
        # self.env.add_marker(self.env.get_p_body(self.body_name), radius=0.1,
        #             color=np.array([1, 0, 0, 0.5]))
        # self.env.add_marker(self.p_trgt, radius=0.1, color=np.array([0, 0, 1, 0.5]))
        self.env.render(RENDER_ALWAYS=False)

if __name__ == '__main__':
    rospy.init_node('mujoco', anonymous=True)
    rate = rospy.Rate(125) # 10hz

    papras_mujoco = PAPRAS_MuJoCo()

    while not rospy.is_shutdown():
        if not papras_mujoco.get_delta_x_message:
            papras_mujoco.calculate_single_iteration()
        papras_mujoco.servo_calc()
        papras_mujoco.render()
        
        rate.sleep()