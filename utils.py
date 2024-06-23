import logging
import sys

class WarningFilter(logging.Filter):
    def filter(self, record):
        if record.levelno == logging.WARNING:
            record.msg = f"Warn!: {record.msg}"
        return True

def create_handler(stream, levels, formatter):
    handler = logging.StreamHandler(stream)
    handler.setLevel(min(levels))
    handler.addFilter(lambda record: record.levelno in levels)
    handler.addFilter(WarningFilter())  # Apply the custom filter
    handler.setFormatter(formatter)
    
    return handler
def setup_logger(logger_name, level, stdout_levels, stderr_levels, formatter):
    logger = logging.getLogger(logger_name)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logger.setLevel(level)
    stdout_handler = create_handler(sys.stdout, stdout_levels, formatter)
    stderr_handler = create_handler(sys.stderr, stderr_levels, formatter)
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)