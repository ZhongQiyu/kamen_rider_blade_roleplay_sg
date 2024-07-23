import logging
import os

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 环境配置
os.environ['MODEL_CONFIG'] = 'path/to/config'

# 导入模块
from .model_a import ModelA
from .model_b import ModelB

# 初始化代码
logger.info("Models package has been initialized with configuration from {}".format(os.environ['MODEL_CONFIG']))

# 可以选择性地暴露部分内容
__all__ = ['ModelA', 'ModelB']
