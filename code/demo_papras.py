import random
import numpy as np
import matplotlib.pyplot as plt
import time
from rotation import mat2euler, euler2mat
from mujoco_parser import MuJoCoParserClass
np.set_printoptions(precision=2)
print("Done.")

env = MuJoCoParserClass(
    name='PAPRAS', rel_xml_path='../asset/papras/mjmodel.xml', VERBOSE=False)
env.init_viewer(TERMINATE_GLFW=True, INITIALIZE_GLFW=True,
                window_width=0.5, window_height=0.5)
env.set_max_tick(max_tick=2000)

# Prepare for IK
# panda_eef / panda_link_4 / panda_link_5 / panda_link_6
body_name = 'robot1/end_link'
q = env.get_q_pos(q_pos_idxs=env.rev_joint_idxs)
p_EE = env.get_p_body(body_name=body_name)
R_EE = env.get_R_body(body_name=body_name)
p_trgt = p_EE + np.array([0.2, 0.0, -0.2])
R_trgt = R_EE


def rand_val(minimum, maximum):
    return minimum + (maximum - minimum) * random.random()


start_time = time.time()
next_target = False

# Buffers
err_list = np.zeros(env.max_tick)
q_list = np.zeros((env.max_tick, env.n_rev_joint))
while env.IS_ALIVE():
    if next_target:
        next_target = False
        start_time = time.time()
        p_trgt_x = rand_val(-0.2, 0.05)
        p_trgt_y = rand_val(-0.2, 0.2)
        p_trgt_z = rand_val(-0.2, 0.05)
        p_trgt = p_EE + np.array([p_trgt_x, p_trgt_y, p_trgt_z])

        trgt_roll = rand_val(-0.785398, 0.785398)
        trgt_pitch = rand_val(-0.785398, 0.785398)
        trgt_yaw = rand_val(-0.785398, 0.785398)
        R_trgt_euler = [trgt_roll, trgt_pitch, trgt_yaw]
        R_trgt = euler2mat(R_trgt_euler)

        print("target XYZ: ", p_trgt, "   RPY: ", R_trgt_euler)

    # Numerical IK
    dq, err = env.one_step_ik(
        body_name=body_name, p_trgt=p_trgt, R_trgt=R_trgt, th=3.0*np.pi/180.0)
    q = q + dq

    if (err < 1e-4).all():
        end_time = time.time()
        total_time = end_time - start_time
        print(total_time)
        next_target = True

    # FK
    env.forward(q_pos=q, q_pos_idxs=env.rev_joint_idxs)
    env.add_marker(env.get_p_body(body_name), radius=0.1,
                   color=np.array([1, 0, 0, 0.5]))
    env.add_marker(p_trgt, radius=0.1, color=np.array([0, 0, 1, 0.5]))
    env.render(RENDER_ALWAYS=True)
    # Append
    q_list[env.tick-1, :] = env.get_q_pos(q_pos_idxs=env.rev_joint_idxs)
print("Done.")
