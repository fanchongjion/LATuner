import logging
import os
import threading

LOG_LEVEL_DICT = {
    "logging.debug": logging.DEBUG,
    "logging.info": logging.INFO,
    "logging.warning": logging.WARNING,
    "logging.error": logging.ERROR,
    "logging.critical": logging.CRITICAL
}

DEFAULT_LOG_FORMAT = os.environ.get('DEFAULT_LOG_FORMAT',
                                    '[%(asctime)s:%(filename)s#L%(lineno)d:%(levelname)s]: %(message)s')
DEFAULT_LOG_LEVEL = LOG_LEVEL_DICT[os.environ.get('DEFAULT_LOG_LEVEL', 'logging.INFO').lower()]

class SingletonLogger:
    _instance_lock = threading.Lock()
    _logger_instance = None

    def __new__(cls, log_file_path=None):
        if not cls._logger_instance:
            with cls._instance_lock:
                if not cls._logger_instance:
                    cls._logger_instance = super().__new__(cls)
                    cls._logger_instance.setup_logging("dbenv_log", log_file_path)
        return cls._logger_instance


    def setup_logging(self, logger_name, log_file_path, log_level=DEFAULT_LOG_LEVEL, log_format=DEFAULT_LOG_FORMAT):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(log_level)

        handler = logging.FileHandler(log_file_path)
        handler.setLevel(log_level)
        handler.setFormatter(logging.Formatter(log_format))

        self.logger.addHandler(handler)
