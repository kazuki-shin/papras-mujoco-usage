#!/usr/bin/self.env python
import rospy
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TwistStamped
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
        sub_joint_state = rospy.Subscriber('/joint_states', JointState, self.joint_state_callback)
        sub_twisted_joint_state = rospy.Subscriber('/servo_server/delta_twist_cmds',TwistStamped, self.twisted_stamp_callback)
        self.execute_pub = rospy.Publisher("/arm1_controller/command", JointTrajectory, queue_size=100)

        self.publish_period = 0.008
        self.planning_frame = "robot1/link1"
        self.joint_state = JointState()
        self.latest_twist_msg = None
        self.new_input = False
        self.delta_x = [0,0,0,0,0,0]
        # mujoco init
        self.env = MuJoCoParserClass(
            name='PAPRAS', rel_xml_path='../asset/papras/mjmodel.xml', VERBOSE=True)
        self.env.init_viewer(TERMINATE_GLFW=True, INITIALIZE_GLFW=True,
                        window_width=0.5, window_height=0.5)
        self.init_ik()

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
            cur_linear_vel = np.array([current_twist_msg.twist.linear.x, current_twist_msg.twist.linear.y, current_twist_msg.twist.linear.z, 1])
            cur_angular_vel = np.array([current_twist_msg.twist.angular.x, current_twist_msg.twist.angular.y, current_twist_msg.twist.angular.z, 1])
            #
            base_frame = "robot1/link2"
            eef_frame = "robot1/end_link"

            mujoco_joint_state = self.joint_state.position
            self.q = mujoco_joint_state
            self.env.forward(q_pos=self.q, q_pos_idxs=self.env.rev_joint_idxs)


            base_frame_r = self.env.get_R_body(body_name=base_frame)
            base_frame_point = self.env.get_p_body(body_name=base_frame)
            eef_frame_r = self.env.get_R_body(body_name=eef_frame)
            eef_frame_point = self.env.get_p_body(body_name=eef_frame)

            base_frame_t = np.linalg.inv(self.create_transformation_matrix(base_frame_r,base_frame_point))
            eef_frame_t = self.create_transformation_matrix(eef_frame_r,eef_frame_point)

            transformation_matrix = base_frame_t @ eef_frame_t 

            cur_linear_vel = (transformation_matrix @ cur_linear_vel.T).T 
            cur_angular_vel = (transformation_matrix @ cur_angular_vel.T).T 

            



            updated_twist_msg = TwistStamped()
            updated_twist_msg.twist.linear.x = cur_linear_vel[0]
            updated_twist_msg.twist.linear.y = cur_linear_vel[1]
            updated_twist_msg.twist.linear.z = cur_linear_vel[2]
            updated_twist_msg.twist.angular.x = cur_angular_vel[0]
            updated_twist_msg.twist.angular.y = cur_angular_vel[1]
            updated_twist_msg.twist.angular.z = cur_angular_vel[2]
            print(current_twist_msg)
            print(updated_twist_msg)

            self.delta_x = self.calculate_cartesian_cmd(updated_twist_msg)
            print(self.delta_x)




    # def main_loop(self):
    #     pass         
    # def update_joints(self):
    #     pass

    def joint_state_callback(self, msg):
        self.joint_state.position = msg.position 
        self.joint_state.velocity = msg.velocity 
        self.joint_state.effort = msg.effort
        
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

    def servo_calc(self):
        mujoco_joint_state = self.joint_state.position
        # update mujoco scene with updated joint states
        self.q = mujoco_joint_state
        self.env.forward(q_pos=self.q, q_pos_idxs=self.env.rev_joint_idxs)

        # get jacobian w.r.t eef link
        jacobian_link = "robot1/link2"
        J_p, J_R, J_full = self.env.get_J_body(body_name=jacobian_link)
        # print("J_full", J_full)

        # compute J_† = V * D_† * U_T and dT = J_† * dX
        pseudo_inverse = np.linalg.pinv(J_full)
        delta_theta = pseudo_inverse @ self.delta_x
        # print("pseudoinverse: ", pseudo_inverse)
        # print("delta_theta", delta_theta)

        # singularity check 
        # U, s, VT = np.linalg.svd(delta_theta)
        # S  =  np.diag(s)
        # print("singular matrix", S)

        # enforce velocity limit on delta_theta

        # Add the deltas to each joint
        new_joint_state = self.joint_state
        # for i in range(len(mujoco_joint_state)):
        new_joint_state.position = mujoco_joint_state + delta_theta

        # update internal robot state in mujoco scene
        self.q = mujoco_joint_state
        self.env.forward(q_pos=self.q, q_pos_idxs=self.env.rev_joint_idxs)        

        # Stop if new configuration in collision
        # n_contacts = self.env.sim.data.ncon
        # if n_contacts > 0:
        #     new_joint_state.position = self.joint_state.position
        #     print("Collision Detected. Halting robot.")
        
        # Visualize collisions
        # for c_idx in range(n_contacts):
        #     contact    = self.env.sim.data.contact[c_idx]
        #     p_contact  = contact.pos
        #     bodyname1  = self.env.body_idx2name(env.sim.model.geom_bodyid[contact.geom1])
        #     bodyname2  = self.env.body_idx2name(env.sim.model.geom_bodyid[contact.geom2])
        #     label      = '%s-%s'%(bodyname1,bodyname2)
        #     self.env.add_marker(pos=p_contact,radius=0.1,color=np.array([1,0,0,0.5]),label=label)
        
        # calc joint velocities
        new_joint_state.velocity = delta_theta / self.publish_period            

        # Compose joint trajectory message
        traj_msg = JointTrajectory()
        traj_msg.header.stamp = rospy.Time.now()
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
        print(traj_msg)
        self.execute_pub.publish(traj_msg)


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
            self.next_target = True

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
        papras_mujoco.calculate_single_iteration()
        papras_mujoco.servo_calc()
        papras_mujoco.render()
        
        rate.sleep()