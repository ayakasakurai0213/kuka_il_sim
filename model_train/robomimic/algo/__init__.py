from model_train.robomimic.algo.algo import register_algo_factory_func, algo_name_to_factory_func, algo_factory, Algo, PolicyAlgo, ValueAlgo, PlannerAlgo, HierarchicalAlgo, RolloutPolicy

# note: these imports are needed to register these classes in the global algo registry
from model_train.robomimic.algo.bc import BC, BC_Gaussian, BC_GMM, BC_VAE, BC_RNN, BC_RNN_GMM
from model_train.robomimic.algo.bcq import BCQ, BCQ_GMM, BCQ_Distributional
from model_train.robomimic.algo.cql import CQL
from model_train.robomimic.algo.iql import IQL
from model_train.robomimic.algo.gl import GL, GL_VAE, ValuePlanner
from model_train.robomimic.algo.hbc import HBC
from model_train.robomimic.algo.iris import IRIS
from model_train.robomimic.algo.td3_bc import TD3_BC
from model_train.robomimic.algo.diffusion_policy import DiffusionPolicyUNet
from model_train.robomimic.algo.act import ACT
