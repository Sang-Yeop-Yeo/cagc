import torch
import numpy as np
from args_helper import parser_args
import time
from utils.utils import set_seed
import logging


def print_time():
    print("\n\n--------------------------------------")
    print("TIME: The current time is: {}".format(time.ctime()))
    print("TIME: The current time in seconds is: {}".format(time.time()))
    print("--------------------------------------\n\n")


def set_logger(args):
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s(%(name)s): %(message)s'
    )
    consH = logging.StreamHandler()
    consH.setFormatter(formatter)
    consH.setLevel(logging.DEBUG)
    logger.addHandler(consH)
    filehandler = logging.FileHandler(f'{args.save_ckpt}_logfile.log')
    logger.addHandler(filehandler)
    log = logger
    return log