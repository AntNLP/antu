import logging


def dual_channel_logger(
    name: str,
    level: int=logging.DEBUG,
    file_path: str=None,
    file_model: str='w+',
    file_level: int=logging.DEBUG,
    console_level: int=logging.DEBUG,
    formatter: str='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    time_formatter: str='%y-%m-%d %H:%M:%S',) -> logging.Logger:

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(formatter, time_formatter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if file_path:
        file_handler = logging.FileHandler(file_path, file_model)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


