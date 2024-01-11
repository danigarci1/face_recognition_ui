import logging

logger_names = ["general","tracking","recognition"]
loggers = dict.fromkeys(logger_names)

# Generic logger
# Create a formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Create a console handler and set the level to INFO
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)


# Create a file handler and set the level to DEBUG
file_handler = logging.FileHandler('tmp/tracking_system.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

for name in logger_names:
    tmp_logger = logging.getLogger(name)
    tmp_logger.setLevel(logging.DEBUG)
    tmp_logger.addHandler(console_handler)
    tmp_logger.addHandler(file_handler)
    loggers[name] = tmp_logger
