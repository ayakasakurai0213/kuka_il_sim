import os
import sys
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(base_dir)
sys.path.append(base_dir)
import pickle
import argparse
from einops import rearrange

from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy
import collections
# from collections import deque

import torch
import numpy as np
import cv2
# from cv_bridge import CvBridge
import time
import threading
import math
import pybullet as p
from env.kuka_ir_env import KukaIrEnv

import sys
sys.path.append("./")

task_config = {'camera_names': ['cam_top', 'cam_hand']}

inference_thread = None
inference_lock = threading.Lock()
inference_actions = None
inference_timestep = None


def actions_interpolation(args, pre_action, actions, stats):
    steps = np.concatenate((np.array(args.arm_steps_length), np.array(args.arm_steps_length)), axis=0)
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['qpos_std'] + stats['qpos_mean']
    result = [pre_action]
    post_action = post_process(actions[0])
    # print("pre_action:", pre_action[7:])
    # print("actions_interpolation1:", post_action[:, 7:])
    max_diff_index = 0
    max_diff = -1
    for i in range(post_action.shape[0]):
        diff = 0
        for j in range(pre_action.shape[0]):
            if j == 6 or j == 13:
                continue
            diff += math.fabs(pre_action[j] - post_action[i][j])
        if diff > max_diff:
            max_diff = diff
            max_diff_index = i

    for i in range(max_diff_index, post_action.shape[0]):
        step = max([math.floor(math.fabs(result[-1][j] - post_action[i][j])/steps[j]) for j in range(pre_action.shape[0])])
        inter = np.linspace(result[-1], post_action[i], step+2)
        result.extend(inter[1:])
    while len(result) < args.chunk_size+1:
        result.append(result[-1])
    result = np.array(result)[1:args.chunk_size+1]
    # print("actions_interpolation2:", result.shape, result[:, 7:])
    result = pre_process(result)
    result = result[np.newaxis, :]
    return result


def get_model_config(args):
    # 设置随机种子，你可以确保在相同的初始条件下，每次运行代码时生成的随机数序列是相同的。
    set_seed(1)
   
    # 如果是ACT策略
    # fixed parameters
    if args.policy_class == 'ACT':
        policy_config = {'lr': args.lr,
                         'lr_backbone': args.lr_backbone,
                         'backbone': args.backbone,
                         'masks': args.masks,
                         'weight_decay': args.weight_decay,
                         'dilation': args.dilation,
                         'position_embedding': args.position_embedding,
                         'loss_function': args.loss_function,
                         'chunk_size': args.chunk_size,     # 查询
                         'camera_names': task_config['camera_names'],
                         'use_depth_image': args.use_depth_image,
                         'use_robot_base': args.use_robot_base,
                         'kl_weight': args.kl_weight,        # kl散度权重
                         'hidden_dim': args.hidden_dim,      # 隐藏层维度
                         'dim_feedforward': args.dim_feedforward,
                         'enc_layers': args.enc_layers,
                         'dec_layers': args.dec_layers,
                         'nheads': args.nheads,
                         'dropout': args.dropout,
                         'pre_norm': args.pre_norm
                         }
    elif args.policy_class == 'CNNMLP':
        policy_config = {'lr': args.lr,
                         'lr_backbone': args.lr_backbone,
                         'backbone': args.backbone,
                         'masks': args.masks,
                         'weight_decay': args.weight_decay,
                         'dilation': args.dilation,
                         'position_embedding': args.position_embedding,
                         'loss_function': args.loss_function,
                         'chunk_size': 1,     # 查询
                         'camera_names': task_config['camera_names'],
                         'use_depth_image': args.use_depth_image,
                         'use_robot_base': args.use_robot_base
                         }
    elif args.policy_class == 'Diffusion':
        policy_config = {'lr': args.lr,
                         'lr_backbone': args.lr_backbone,
                         'backbone': args.backbone,
                         'masks': args.masks,
                         'weight_decay': args.weight_decay,
                         'dilation': args.dilation,
                         'position_embedding': args.position_embedding,
                         'loss_function': args.loss_function,
                         'chunk_size': args.chunk_size,     # 查询
                         'camera_names': task_config['camera_names'],
                         'use_depth_image': args.use_depth_image,
                         'use_robot_base': args.use_robot_base,
                         'observation_horizon': args.observation_horizon,
                         'action_horizon': args.action_horizon,
                         'num_inference_timesteps': args.num_inference_timesteps,
                         'ema_power': args.ema_power
                         }
    else:
        raise NotImplementedError

    config = {
        'ckpt_dir': args.ckpt_dir,
        'ckpt_name': args.ckpt_name,
        'ckpt_stats_name': args.ckpt_stats_name,
        'episode_len': args.max_publish_step,
        'state_dim': args.state_dim,
        'policy_class': args.policy_class,
        'policy_config': policy_config,
        'temporal_agg': args.temporal_agg,
        'camera_names': task_config['camera_names'],
    }
    return config


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    elif policy_class == 'Diffusion':
        policy = DiffusionPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def get_image(observation, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(observation['images'][cam_name], 'h w c -> c h w')
    
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def get_depth_image(observation, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_images.append(observation['images_depth'][cam_name])
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def inference_process(args, config, kuka_operator, policy, stats, t, pre_action):
    global inference_lock, inference_actions, inference_timestep
    print_flag = True
    pre_pos_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    pre_action_process = lambda next_action: (next_action - stats["action_mean"]) / stats["action_std"]
    while True:
        result = kuka_operator.get_frame()
        if not result:
            if print_flag:
                print("syn fail")
                print_flag = False
            continue
        print_flag = True
        (img_top, img_hand, img_top_depth, img_hand_depth, arm_joint, robot_base) = result
        obs = collections.OrderedDict()
        image_dict = dict()
        image_dict[config['camera_names'][0]] = img_top
        image_dict[config['camera_names'][1]] = img_hand
        obs['images'] = image_dict
        
        if args.use_depth_image:
            image_depth_dict = dict()
            image_depth_dict[config['camera_names'][0]] = img_top_depth
            image_depth_dict[config['camera_names'][1]] = img_hand_depth
            obs['images_depth'] = image_depth_dict

        obs['qpos'] = arm_joint['qpos']
        obs['qvel'] = arm_joint['qvel']
        obs['effort'] = arm_joint['effort']
        if args.use_robot_base:
            obs['base_vel'] = robot_base
            obs['qpos'] = np.concatenate((obs['qpos'], obs['base_vel']), axis=0)
        else:
            obs['base_vel'] = [0.0, 0.0]

        # 归一化处理qpos 并转到cuda
        qpos = pre_pos_process(obs['qpos'])
        qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
        # 当前图像curr_image获取图像
        curr_image = get_image(obs, config['camera_names'])
        curr_depth_image = None
        if args.use_depth_image:
            curr_depth_image = get_depth_image(obs, config['camera_names'])
        start_time = time.time()
        all_actions = policy(curr_image, curr_depth_image, qpos)
        end_time = time.time()
        print("model cost time: ", end_time -start_time)
        inference_lock.acquire()
        inference_actions = all_actions.cpu().detach().numpy()
        if pre_action is None:
            pre_action = obs['qpos']
        if args.use_actions_interpolation:
            inference_actions = actions_interpolation(args, pre_action, inference_actions, stats)
        inference_timestep = t
        inference_lock.release()
        break


def model_inference(args, config, kuka_operator, save_episode=True):
    global inference_lock, inference_actions, inference_timestep, inference_thread
    set_seed(1000)

    policy = make_policy(config['policy_class'], config['policy_config'])
    # print("model structure\n", policy.model)
    
    ckpt_path = os.path.join(config['ckpt_dir'], config['ckpt_name'])
    state_dict = torch.load(ckpt_path)
    new_state_dict = {}
    for key, value in state_dict.items():
        if key in ["model.is_pad_head.weight", "model.is_pad_head.bias"]:
            continue
        if key in ["model.input_proj_next_action.weight", "model.input_proj_next_action.bias"]:
            continue
        new_state_dict[key] = value
    loading_status = policy.deserialize(new_state_dict)
    if not loading_status:
        print("ckpt path not exist")
        return False

    policy.cuda()
    policy.eval()

    stats_path = os.path.join(config['ckpt_dir'], config['ckpt_stats_name'])
    # 统计的数据  # 加载action_mean, action_std, qpos_mean, qpos_std 14维
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    # 数据预处理和后处理函数定义
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['qpos_std'] + stats['qpos_mean']

    max_publish_step = config['episode_len']
    chunk_size = config['policy_config']['chunk_size']

    action = None
    # 推理
    with torch.inference_mode():
        p.setRealTimeSimulation(1)
        while True:
            # 每个回合的步数
            t = 0
            max_t = 0
            if config['temporal_agg']:
                all_time_actions = np.zeros([max_publish_step, max_publish_step + chunk_size, config['state_dim']])
            while t < max_publish_step:
                # start_time = time.time()
                # query policy
                if config['policy_class'] == "ACT":
                    if t >= max_t:
                        pre_action = action
                        inference_thread = threading.Thread(target=inference_process,
                                                            args=(args, config, kuka_operator,
                                                                  policy, stats, t, pre_action))
                        inference_thread.start()
                        inference_thread.join()
                        inference_lock.acquire()
                        if inference_actions is not None:
                            inference_thread = None
                            all_actions = inference_actions
                            inference_actions = None
                            max_t = t + args.pos_lookahead_step
                            if config['temporal_agg']:
                                all_time_actions[[t], t:t + chunk_size] = all_actions
                        inference_lock.release()
                    if config['temporal_agg']:
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = np.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = exp_weights[:, np.newaxis]
                        raw_action = (actions_for_curr_step * exp_weights).sum(axis=0, keepdims=True)
                    else:
                        if args.pos_lookahead_step != 0:
                            raw_action = all_actions[:, t % args.pos_lookahead_step]
                        else:
                            raw_action = all_actions[:, t % chunk_size]
                else:
                    raise NotImplementedError
                action = post_process(raw_action[0])
                kuka_operator.control_pos(action)
                if args.use_robot_base:
                    vel_action = action[10:12]
                    kuka_operator.control_pos(vel_action)
                t += 1
                # end_time = time.time()
                # print("publish: ", t)
                # print("time:", end_time - start_time)
                # print("action:", action)


class KukaOperator:
    def __init__(self, env, args):
        self.env = env
        self.kuka_id = self.env._kuka.kukaUid
        self.args = args
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
    
    def control_pos(self, action):
        for i in range(len(self.param_ids)):
            p.setJointMotorControl2(self.kuka_id, self.joint_ids[i], p.POSITION_CONTROL, action[i], force=5 * 240.)
        time.sleep(0.01)
        
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


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', default='./ckpt', required=False)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', default='project_test', required=False)
    parser.add_argument('--max_publish_step', action='store', type=int, help='max_publish_step', default=10000, required=False)
    parser.add_argument('--ckpt_name', action='store', type=str, help='ckpt_name', default='policy_best.ckpt', required=False)
    parser.add_argument('--ckpt_stats_name', action='store', type=str, help='ckpt_stats_name', default='dataset_stats.pkl', required=False)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', default='ACT', required=False)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', default=8, required=False)
    parser.add_argument('--seed', action='store', type=int, help='seed', default=0, required=False)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', default=2000, required=False)
    parser.add_argument('--lr', action='store', type=float, help='lr', default=1e-5, required=False)
    parser.add_argument('--weight_decay', type=float, help='weight_decay', default=1e-4, required=False)
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)", required=False)
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features", required=False)
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', default=10, required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', default=512, required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', default=3200, required=False)
    parser.add_argument('--temporal_agg', action='store', type=bool, help='temporal_agg', default=True, required=False)

    parser.add_argument('--state_dim', action='store', type=int, help='state_dim', default=10, required=False)
    parser.add_argument('--lr_backbone', action='store', type=float, help='lr_backbone', default=1e-5, required=False)
    parser.add_argument('--backbone', action='store', type=str, help='backbone', default='resnet18', required=False)
    parser.add_argument('--loss_function', action='store', type=str, help='loss_function l1 l2 l1+l2', default='l1', required=False)
    parser.add_argument('--enc_layers', action='store', type=int, help='enc_layers', default=4, required=False)
    parser.add_argument('--dec_layers', action='store', type=int, help='dec_layers', default=7, required=False)
    parser.add_argument('--nheads', action='store', type=int, help='nheads', default=8, required=False)
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer", required=False)
    parser.add_argument('--pre_norm', action='store_true', required=False)

    parser.add_argument('--use_robot_base', action='store', type=bool, help='use_robot_base',
                        default=False, required=False)
    parser.add_argument('--pos_lookahead_step', action='store', type=int, help='pos_lookahead_step',
                        default=0, required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size',
                        default=32, required=False)
    parser.add_argument('--arm_steps_length', action='store', type=float, help='arm_steps_length',
                        default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2], required=False)

    parser.add_argument('--use_actions_interpolation', action='store', type=bool, help='use_actions_interpolation',
                        default=False, required=False)
    parser.add_argument('--use_depth_image', action='store', type=bool, help='use_depth_image',
                        default=False, required=False)

    # for Diffusion
    parser.add_argument('--observation_horizon', action='store', type=int, help='observation_horizon', default=1, required=False)
    parser.add_argument('--action_horizon', action='store', type=int, help='action_horizon', default=8, required=False)
    parser.add_argument('--num_inference_timesteps', action='store', type=int, help='num_inference_timesteps', default=10, required=False)
    parser.add_argument('--ema_power', action='store', type=int, help='ema_power', default=0.75, required=False)
    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    env = KukaIrEnv(renders=True, isDiscrete=True)
    env.reset()
    kuka_operator = KukaOperator(env, args)
    config = get_model_config(args)
    model_inference(args, config, kuka_operator, save_episode=True)


if __name__ == '__main__':
    main()
# python model_train/act/inference.py --ckpt_dir ./ckpt/project_test --task_name "project_test"
