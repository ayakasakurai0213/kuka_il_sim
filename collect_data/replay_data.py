import os
import sys
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
import time
import numpy as np
import cv2
import h5py
import argparse
import pybullet as p
from env.kuka_ir_env import KukaIrEnv


def load_hdf5(dataset_dir, dataset_name):
    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        is_sim = root.attrs['sim']
        compressed = root.attrs.get('compress', False)
        qpos = root['/observations/qpos'][()]
        qvel = root['/observations/qvel'][()]
        if 'effort' in root.keys():
            effort = root['/observations/effort'][()]
        else:
            effort = None
        action = root['/action'][()]
        base_action = root['/base_action'][()]
        
        image_dict = dict()
        for cam_name in root[f'/observations/images/'].keys():
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]
        
        if compressed:
            compress_len = root['/compress_len'][()]

    if compressed:
        for cam_id, cam_name in enumerate(image_dict.keys()):
            # un-pad and uncompress
            padded_compressed_image_list = image_dict[cam_name]
            image_list = []
            for frame_id, padded_compressed_image in enumerate(padded_compressed_image_list): # [:1000] to save memory
                image_len = int(compress_len[cam_id, frame_id])
                
                compressed_image = padded_compressed_image
                image = cv2.imdecode(compressed_image, 1)
                image_list.append(image)
            image_dict[cam_name] = image_list

    return qpos, qvel, effort, action, base_action, image_dict
    

def main(args):
    dataset_dir = args.dataset_dir
    episode_idx = args.episode_idx
    task_name   = args.task_name
    dataset_name = f'episode_{episode_idx}'
    
    env = KukaIrEnv(renders=True, isDiscrete=True)
    env.reset()
    kuka_id = env._kuka.kukaUid
    joint_ids = env._kuka.kukaGetJointIndex
    param_ids = []
    joint_name_lst = []
    origin_joint = [
        0.006411874501842649, 0.41318442787173143, -0.01140244401433773, 
        -1.5893163205429706, 0.005379, 1.1376840457008266, -0.006534958891813817, 
        5.800820781903633e-05, -env.finger_angle, env.finger_angle
        ]
    # set joints
    for i in range(p.getNumJoints(kuka_id)):
        info = p.getJointInfo(kuka_id, i)
        # print(info)
        joint_name = info[1]
        if i in joint_ids:
            joint_name_lst.append(joint_name.decode("utf-8"))        
    for i in range(len(joint_ids)):
        param_ids.append(p.addUserDebugParameter(joint_name_lst[i], -4, 4, origin_joint[i]))
    
    # get joint state
    joint_state = []
    for i in range(len(joint_ids)):
        joint_pos = p.getJointState(kuka_id, joint_ids[i])
        joint_state.append(joint_pos[0])
    
    data = load_hdf5(os.path.join(dataset_dir, task_name), dataset_name) 
    actions = data[3]
    last_action = origin_joint
    time_steps = 0
    p.setRealTimeSimulation(1)
    print("=== start simulation ===")
    for action in actions:
        print(f"Time Steps: {time_steps}")
        new_actions = np.linspace(last_action, action, 20)
        last_action = action
        for new_act in new_actions:
            for i in range(len(new_act)):
                p.setJointMotorControl2(kuka_id, joint_ids[i], p.POSITION_CONTROL, new_act[i], force=200)
            last_action = new_act
        time_steps += 1
        time.sleep(0.05)
    print("Success!")
    time.sleep(5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset dir.', default="./datasets", required=False)
    parser.add_argument('--task_name', action='store', type=str, help='Task name.',
                        default="project_test", required=False)

    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.',default=0, required=False)
    
    parser.add_argument('--camera_names', action='store', type=str, help='camera_names',
                        default=['cam_top', 'cam_hand'], required=False)

    parser.add_argument('--use_robot_base', action='store', type=bool, help='use_robot_base',
                        default=False, required=False)
    
    args = parser.parse_args()
    main(args)
    
# python collect_data/replay_data.py --task_name "project_test" --episode_idx 0 