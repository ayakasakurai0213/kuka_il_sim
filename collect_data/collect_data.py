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
from env.kuka_ir_env import KukaIrEnv
from keybord_control import Keyboard


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

        _ = obs.create_dataset('qpos', (data_size, 12))
        _ = obs.create_dataset('qvel', (data_size, 12))
        _ = obs.create_dataset('effort', (data_size, 12))
        _ = root.create_dataset('action', (data_size, 12))
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
        self.init()
        

    def init(self):
        self.joint_ids = []
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
    

    def get_joint(self):
        joint_pos = {"qpos": [], "qvel": [], "torque": [], "effort": [], "base_action": None}
        keys = [i for i in joint_pos.keys()]
        for i in range(len(self.joint_ids)):
            joint_state = p.getJointState(self.kuka_id, self.joint_ids[i])
            for j in range(len(joint_state)):
                joint_pos[keys[j]].append(joint_state[j])
        return joint_pos
    
    
    # def control_pos(self):
    #     for i in range(len(self.param_ids)):
    #         target_joint = p.readUserDebugParameter(self.param_ids[i])
    #         p.setJointMotorControl2(self.kuka_id, self.joint_ids[i], p.POSITION_CONTROL, target_joint, force=5 * 240.)
    #     time.sleep(0.01)
    
    
    def get_top_img(self):
        rgb_img, depth_img = self.env._get_observation()
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        return rgb_img, depth_img
    
    def get_hand_img(self):
        rgb_img, depth_img = self.env._get_hand_cam()
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        return rgb_img, depth_img


    def get_frame(self):
        # self.control_pos()
        img_top, img_top_depth = self.get_top_img()
        # print("img_top:", img_top.shape)
        img_hand, img_hand_depth = self.get_hand_img()
        arm_joint = self.get_joint()

        if self.args.use_depth_image == False:
            img_top_depth = None
            img_hand_depth = None

        robot_base = None
        if self.args.use_robot_base:
            robot_base = arm_joint["base_action"]
        
        return (img_top, img_hand, img_top_depth, img_hand_depth, arm_joint, robot_base)


    def process(self):
        keyboard = Keyboard()
        thread = threading.Thread(target=keyboard.control_key, name="keyboard_thread", daemon=True)
        thread.start()
        timesteps = []
        actions = []
        # image data
        image = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
        image_dict = dict()
        for cam_name in self.args.camera_names:
            image_dict[cam_name] = image
        count = 0

        print_flag = True

        while count < self.args.max_timesteps + 1:
            self.env.arm_control(keyboard.action)
            # collecting image and joint data
            result = self.get_frame()
            if not result:
                if print_flag:
                    print("syn fail")
                    print_flag = False
                continue
            print_flag = True
            count += 1
            (img_top, img_hand, img_top_depth, img_hand_depth, arm_joint, robot_base) = result
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
    
    args = parser.parse_args()
    return args


def main():
    env = KukaIrEnv(renders=True, isDiscrete=True)
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
