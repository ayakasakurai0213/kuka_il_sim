import os
root_dir = os.path.dirname(os.path.abspath(__file__))
import cv2
import pybullet as p
import time
from env.kuka_ir_env import KukaIrEnv

from keybord_control import Keyboard


class Kuka_sim:
    def __init__(self, env):
        self.env = env
        self.kuka_id = self.env._kuka.kukaUid
        self.init_pos()
     
   
    def init_pos(self):
        self.joint_ids = []     # 0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13
        self.param_ids = []
        joint_name_lst = []
        i_pos = [
            0.006411874501842649, 0.41318442787173143, -0.01140244401433773, 
            -1.5893163205429706, 0.005379, 1.1376840457008266, -0.006534958891813817, 
            5.800820781903633e-05, -0.29991772759079405, -4.1277527065243654e-05, 
            0.299948297597285, -0.0002196091555209944
            ]
        
        # set joints
        for i in range(p.getNumJoints(self.kuka_id)):
            info = p.getJointInfo(self.kuka_id, i)
            # print(info)
            joint_name = info[1]
            joint_type = info[2]
            if joint_type == p.JOINT_PRISMATIC or joint_type == p.JOINT_REVOLUTE:
                self.joint_ids.append(i) 
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
        return joint_pos
    
    
    def control_pos(self):
        count = 0
        while count <= 100:
            # print(count)
            for i in range(len(self.param_ids)):
                target_joint = p.readUserDebugParameter(self.param_ids[i])
                p.setJointMotorControl2(self.kuka_id, self.joint_ids[i], p.POSITION_CONTROL, target_joint, force=5 * 240.)
                current_joint = self.get_joint()
                # top_img = self.get_screen()
                hand_img = self.get_hand_img()
            time.sleep(0.01)
            count += 1
        return
    
    def get_screen(self):
        screen = self.env._get_observation()[0]  
        img = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
        cv2.imwrite("images/test.jpg", img)      
        return img
    
    def get_hand_img(self):
        screen = self.env._get_hand_cam()[0]
        img = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
        cv2.imwrite("images/test2.jpg", img)
        return img
    

def main():
    env = KukaIrEnv(renders=True, isDiscrete=True)
    env.reset()
    kuka_sim = Kuka_sim(env)
    keyboard = Keyboard()
    
    p.setRealTimeSimulation(1)
    while True:
        # kuka_sim.control_pos()
        env.reset()
        # kuka_sim.get_screen()
        # kuka_sim.get_hand_img()
        for i in range(1000):
            # キーボードまたはゲームパッド入力取得
            action, text = keyboard.get_pressed_key()
            keyboard.update(text)
            env.arm_control(action)
        

if __name__ == "__main__": 
    main()