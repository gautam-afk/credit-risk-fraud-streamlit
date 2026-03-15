import logging


def get_logger(name):
    logger = logging.getLogger(name)
    if not any(isinstance(handler, logging.NullHandler) for handler in logger.handlers):
        logger.addHandler(logging.NullHandler())
    return logger
