## Simple `MuJoCo` usage

What can we get out of `MuJoco`?

```
env = gym.make('Reacher-v2')
obs,info = env.reset()
for tick in range(1000):
    env.render()
    action = policy(obs)
    obs,reward,done,_ = env.step(action)
```
For those who have run the code above, you are already running `MuJoCo` under the hood. However, `MuJoCo` is not just some physics engines that simulates some robots. In this repository, we focus on the core functionalities of `MuJoCo` (or any other proper simulators) and how we can leverage such information in robot learning tasks through the lens of a Roboticist. 

In particular, we will distinguish `kinematic` and `dynamic` simulations (e.g., forward/inverse kinematics/dynamcis).

Contact: sungjoon-choi@korea.ac.kr 

# Mujoco_py Ubuntu 20.04 Installation
Install mujoco binaries
```
mkdir ~/.mujoco && cd ~/.mujoco
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
tar -xf mujoco210-linux-x86_64.tar.gz
cd ~/.mujoco/mujoco210/bin && ./simulate ../model/humanoid.xml
```

Install mujoco_py and dependencies
```
pip3 install -U 'mujoco-py<2.2,>=2.1'
sudo apt-get install patchelf
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
pip3 install screeninfo scipy
```

### update ~/.bashrc
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/papras/.mujoco/mujoco210/bin
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia # if using nvidia gpu
```

### patchelf issue
https://github.com/openai/mujoco-py/issues/652
sudo apt-get install patchelf

### GLEW initalization error: Missing GL version
https://github.com/openai/mujoco-py/issues/268
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so

### MjViewer key bindings are as follows:
- TAB: Switch between MuJoCo cameras.
- H: Toggle hiding all GUI components.
- SPACE: Pause/unpause the simulation.
- RIGHT: Advance simulation by one step.
- V: Start/stop video recording.
- T: Capture screenshot.
- I: Drop into ``ipdb`` debugger.
- S/F: Decrease/Increase simulation playback speed.
- C: Toggle visualization of contact forces (off by default).
- D: Enable/disable frame skipping when rendering lags behind real time.
- R: Toggle transparency of geoms.
- M: Toggle display of mocap bodies.
- E: Toggle visualization of reference frames
