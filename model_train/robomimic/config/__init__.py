from model_train.robomimic.config.config import Config
from model_train.robomimic.config.base_config import config_factory, get_all_registered_configs

# note: these imports are needed to register these classes in the global config registry
from model_train.robomimic.config.bc_config import BCConfig
from model_train.robomimic.config.bcq_config import BCQConfig
from model_train.robomimic.config.cql_config import CQLConfig
from model_train.robomimic.config.iql_config import IQLConfig
from model_train.robomimic.config.gl_config import GLConfig
from model_train.robomimic.config.hbc_config import HBCConfig
from model_train.robomimic.config.iris_config import IRISConfig
from model_train.robomimic.config.td3_bc_config import TD3_BCConfig
from model_train.robomimic.config.diffusion_policy_config import DiffusionPolicyConfig
from model_train.robomimic.config.act_config import ACTConfig
