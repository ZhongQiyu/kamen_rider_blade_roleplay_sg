import logging
import os

class LoggerManager:
    def __init__(self, log_directory='logs'):
        # 确保日志目录存在
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)

        # 创建logger
        self.logger = logging.getLogger('kamen_rider_blade')
        self.logger.setLevel(logging.DEBUG)

        # 创建文件处理器
        app_handler = logging.FileHandler(os.path.join(log_directory, 'app.log'))
        error_handler = logging.FileHandler(os.path.join(log_directory, 'error.log'))
        debug_handler = logging.FileHandler(os.path.join(log_directory, 'debug.log'))

        # 设置日志级别
        app_handler.setLevel(logging.INFO)
        error_handler.setLevel(logging.ERROR)
        debug_handler.setLevel(logging.DEBUG)

        # 创建日志格式器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # 将格式器添加到处理器
        app_handler.setFormatter(formatter)
        error_handler.setFormatter(formatter)
        debug_handler.setFormatter(formatter)

        # 将处理器添加到logger
        self.logger.addHandler(app_handler)
        self.logger.addHandler(error_handler)
        self.logger.addHandler(debug_handler)

        # 配置基础日志记录
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[
                                logging.FileHandler(os.path.join(log_directory, 'app.log')),
                                logging.FileHandler(os.path.join(log_directory, 'error.log')),
                                logging.StreamHandler()
                            ])

        # 创建一个单独的错误日志记录器
        self.error_logger = logging.getLogger('error_logger')
        error_handler = logging.FileHandler(os.path.join(log_directory, 'error.log'))
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.error_logger.addHandler(error_handler)

    def get_logger(self):
        return self.logger

    def get_error_logger(self):
        return self.error_logger
