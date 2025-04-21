import os
import sys
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(base_dir)
sys.path.append(base_dir)
import time
import threading
import numpy as np
import h5py
import argparse
import dm_env
import collections
import cv2
import pybullet as p
from env.kuka_il_env import KukaIlEnv
from keybord_control import Keyboard
from game_pad import Gamepad


def save_data(args, timesteps, actions, dataset_path):
    data_size = len(actions)
    data_dict = {
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/observations/effort': [],
        '/action': [],
        '/base_action': [],
        # '/base_action_t265': [],
    }

    # camera image
    for cam_name in args.camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []
        if args.use_depth_image:
            data_dict[f'/observations/images_depth/{cam_name}'] = []

    # len(action): max_timesteps, len(time_steps): max_timesteps + 1
    while actions:
        action = actions.pop(0)   # current action
        ts = timesteps.pop(0)     # previous frame

        data_dict['/observations/qpos'].append(ts.observation['qpos'])
        data_dict['/observations/qvel'].append(ts.observation['qvel'])
        data_dict['/observations/effort'].append(ts.observation['effort'])
        data_dict['/action'].append(action)
        data_dict['/base_action'].append(ts.observation['base_vel'])

        # data_dict['/base_action_t265'].append(ts.observation['base_vel_t265'])
        for cam_name in args.camera_names:
            data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])
            if args.use_depth_image:
                data_dict[f'/observations/images_depth/{cam_name}'].append(ts.observation['images_depth'][cam_name])

    t0 = time.time()
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
        # properties of the text
        root.attrs['sim'] = False   # emulation
        root.attrs['compress'] = False  # compressed images

        # create groups of observations and images
        obs = root.create_group('observations')
        image = obs.create_group('images')
        for cam_name in args.camera_names:
            _ = image.create_dataset(cam_name, (data_size, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )
        if args.use_depth_image:
            image_depth = obs.create_group('images_depth')
            for cam_name in args.camera_names:
                _ = image_depth.create_dataset(cam_name, (data_size, 480, 640), dtype='uint16',
                                             chunks=(1, 480, 640), )

        _ = obs.create_dataset('qpos', (data_size, 10))
        _ = obs.create_dataset('qvel', (data_size, 10))
        _ = obs.create_dataset('effort', (data_size, 10))
        _ = root.create_dataset('action', (data_size, 10))
        _ = root.create_dataset('base_action', (data_size, 2))

        # data_dict write into h5py.File
        for name, array in data_dict.items():  
            root[name][...] = array
    print(f'\033[32m\nSaving: {time.time() - t0:.1f} secs. %s \033[0m\n'%dataset_path)


class KukaOperator:
    def __init__(self, env, args):
        self.env = env
        self.kuka_id = self.env._kuka.kukaUid
        self.args = args
        if self.args.control == "keyboard":
            self.controller = Keyboard()
        else:
            if args.control in ["joycon", "dualshock"]:
                self.controller = Gamepad(args.control)
            else:
                raise Exception("Error! Please input 'joycon' or 'dualshock' in argument: control.")
        self.init()
        

    def init(self):
        self.joint_ids = self.env._kuka.kukaGetJointIndex
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
    

    def get_joint(self):
        joint_pos = {"qpos": [], "qvel": [], "torque": [], "effort": [], "base_action": None}
        keys = [i for i in joint_pos.keys()]
        for i in range(len(self.joint_ids)):
            joint_state = p.getJointState(self.kuka_id, self.joint_ids[i])
            for j in range(len(joint_state)):
                joint_pos[keys[j]].append(joint_state[j])
        joint_pos["qpos"][-2] = self.env.finger_angle * -1
        joint_pos["qpos"][-1] = self.env.finger_angle
        return joint_pos
      
    
    def get_top_img(self):
        rgb_img, depth_img = self.env._get_observation()
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        return rgb_img, depth_img
    
    def get_hand_img(self):
        rgb_img, depth_img = self.env._get_hand_cam()
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        return rgb_img, depth_img
    
    def get_img_thread(self):
        while True:
            img_top, img_top_depth = self.get_top_img()
            img_hand, img_hand_depth = self.get_hand_img()

            if self.args.use_depth_image == False:
                img_top_depth = None
                img_hand_depth = None
                
            self.img_data = (img_top, img_hand, img_top_depth, img_hand_depth)


    def process(self):
        timesteps = []
        actions = []
        # image data
        image = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
        image_dict = dict()
        for cam_name in self.args.camera_names:
            image_dict[cam_name] = image
        count = 0
        
        thread = threading.Thread(target=self.get_img_thread, name="get_img_thread", daemon=True)
        thread.start()
        print("loading thread...")
        time.sleep(3)
        print("start.")
        while count < self.args.max_timesteps + 1:
            dx, dy, dz, da, grip = self.controller.control()
            if self.args.control == "keyboard":
                self.controller.update([dx, dy, dz, da, grip])
            self.env.arm_control(dx, dy, dz, da, grip)
            
            # collecting image and joint data
            count += 1
            (img_top, img_hand, img_top_depth, img_hand_depth) = self.img_data
            arm_joint = self.get_joint()
            robot_base = None
            if self.args.use_robot_base:
                robot_base = arm_joint["base_action"]
                
            # image info
            image_dict = dict()
            image_dict[self.args.camera_names[0]] = img_top
            image_dict[self.args.camera_names[1]] = img_hand

            # collecting joint data
            obs = collections.OrderedDict()  # ordered dict
            obs['images'] = image_dict
            if self.args.use_depth_image:
                image_dict_depth = dict()
                image_dict_depth[self.args.camera_names[0]] = img_top_depth
                image_dict_depth[self.args.camera_names[1]] = img_hand_depth
                obs['images_depth'] = image_dict_depth
                
            obs['qpos'] = np.array(arm_joint["qpos"])
            actions.append(obs['qpos'])
            obs['qvel'] = np.array(arm_joint["qvel"])
            obs['effort'] = np.array(arm_joint["effort"])
            
            if self.args.use_robot_base:
                obs['base_vel'] = robot_base
            else:
                obs['base_vel'] = [0.0, 0.0]

            if count == 1:
                ts = dm_env.TimeStep(
                    step_type=dm_env.StepType.FIRST,
                    reward=None,
                    discount=None,
                    observation=obs)
                timesteps.append(ts)
                continue

            # time step
            ts = dm_env.TimeStep(
                step_type=dm_env.StepType.MID,
                reward=None,
                discount=None,
                observation=obs)

            timesteps.append(ts)
            print("Frame data: ", count)

        print("len(timesteps): ", len(timesteps))
        print("len(actions)  : ", len(actions))
        return timesteps, actions


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset_dir.',
                        default="./datasets", required=False)
    parser.add_argument('--task_name', action='store', type=str, help='Task name.',
                        default="project_test", required=False)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.',
                        default=0, required=False)
    parser.add_argument('--max_timesteps', action='store', type=int, help='Max_timesteps.',
                        default=500, required=False)
    parser.add_argument('--camera_names', action='store', type=str, help='camera_names',
                        default=['cam_top', 'cam_hand'], required=False)
    parser.add_argument('--use_robot_base', action='store', type=bool, help='use_robot_base',
                        default=False, required=False)
    # collect depth image
    parser.add_argument('--use_depth_image', action='store', type=bool, help='use_depth_image',
                        default=False, required=False)
    parser.add_argument('--control', action='store', type=str, help='control with "keyboard", "joycon" or "dualshock"', 
                        default='keyboard', required=False)
    
    args = parser.parse_args()
    return args


def main():
    env = KukaIlEnv(renders=True, isDiscrete=True, numObjects=1)
    env.reset()
    args = get_arguments()
    kuka_operator = KukaOperator(env, args)
    
    p.setRealTimeSimulation(1)
    episode_idx = args.episode_idx
    while True: 
        env.reset()
        timesteps, actions = kuka_operator.process()
        dataset_dir = os.path.join(args.dataset_dir, args.task_name)
        
        if(len(actions) < args.max_timesteps):
            print("\033[31m\nSave failure, please record %s timesteps of data.\033[0m\n" %args.max_timesteps)
            exit(-1)

        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        dataset_path = os.path.join(dataset_dir, "episode_" + str(episode_idx))
        save_data(args, timesteps, actions, dataset_path)
        episode_idx += 1
        time.sleep(5)


if __name__ == '__main__':
    main()

# python collect_data/collect_data.py --dataset_dir ./datasets --max_timesteps 500 --task_name "project_test" --episode_idx 0
