import os
root_dir = os.path.dirname(os.path.abspath(__file__))
import cv2
import pybullet as p
import threading
import argparse
import time
from env.kuka_ir_env import KukaIrEnv
from keybord_control import Keyboard
from game_pad import Gamepad


class Kuka_sim:
    def __init__(self, env):
        self.env = env
        self.kuka_id = self.env._kuka.kukaUid
        self.init_pos()
    
    def init_pos(self):
        self.joint_ids = self.env._kuka.kukaGetJointIndex     # 0, 1, 2, 3, 4, 5, 6, 7, 8, 11
        self.param_ids = []
        joint_name_lst = []
        i_pos = [
            0.006411874501842649, 0.41318442787173143, -0.01140244401433773, 
            -1.5893163205429706, 0.005379, 1.1376840457008266, -0.006534958891813817, 
            5.800820781903633e-05, -self.env.finger_angle, self.env.finger_angle
            ]
        
        # set joints
        for i in range(p.getNumJoints(self.kuka_id)):
            info = p.getJointInfo(self.kuka_id, i)
            # print(info)
            joint_name = info[1]
            if i in self.joint_ids:
                joint_name_lst.append(joint_name)
                
        for i in range(len(self.joint_ids)):
            self.param_ids.append(p.addUserDebugParameter(joint_name_lst[i].decode("utf-8"), -4, 4, i_pos[i]))
        return

    def get_joint(self):
        joint_pos = {"qpos": [], "qvel": [], "torque": [], "effort": []}
        keys = [i for i in joint_pos.keys()]
        for i in range(len(self.joint_ids)):
            joint_state = p.getJointState(self.kuka_id, self.joint_ids[i])
            for j in range(len(joint_state)):
                joint_pos[keys[j]].append(joint_state[j])
        joint_pos["qpos"][-2] = self.env.finger_angle * -1
        joint_pos["qpos"][-1] = self.env.finger_angle
        return joint_pos
    
    def control_pos(self):
        count = 0
        while count <= 100:
            # print(count)
            for i in range(len(self.param_ids)):
                target_joint = p.readUserDebugParameter(self.param_ids[i])
                p.setJointMotorControl2(self.kuka_id, self.joint_ids[i], p.POSITION_CONTROL, target_joint, force=5 * 240.)
                current_joint = self.get_joint()
                # top_img = self.get_top_img()
                hand_img = self.get_hand_img()
            time.sleep(0.01)
            count += 1
        return
    
    
    def get_top_img(self):
        screen = self.env._get_observation()[0]
        top_img = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
        # cv2.imwrite("images/test1.jpg", top_img)
        return top_img
    
    def get_hand_img(self):
        screen = self.env._get_hand_cam()[0]
        hand_img = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
        # cv2.imwrite("images/test2.jpg", hand_img)
        return hand_img
    

def main(args):
    env = KukaIrEnv(renders=True, isDiscrete=True, numObjects=1)
    env.reset()
    kuka_sim = Kuka_sim(env)
    if args.control == "keyboard":
        keyboard = Keyboard()
    elif args.control == "gamepad":
        gamepad = Gamepad()
    
    p.setRealTimeSimulation(1)
    # kuka_sim.control_pos()
    while True:
        env.reset()
        for i in range(1000):
            # input on keyboard or gamepad
            if args.control == "keyboard":
                dx, dy, dz, da, grip = keyboard.control()
                keyboard.update([dx, dy, dz, da, grip])
            elif args.control == "gamepad":
                dx, dy, dz, da, grip = gamepad.control()
            env.arm_control(dx, dy, dz, da, grip)
            
            # kuka_sim.get_top_img()
            # kuka_sim.get_hand_img()
            joint_pos = kuka_sim.get_joint()
            # print(joint_pos["qpos"])
        

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--control', action='store', type=str, help='control with keyboard or gamepad', default='keyboard', required=False)
    args = parser.parse_args()
    main(args)