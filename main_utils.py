import torch
import numpy as np
from args_helper import parser_args
import time
from utils.utils import set_seed


def print_time():
    print("\n\n--------------------------------------")
    print("TIME: The current time is: {}".format(time.ctime()))
    print("TIME: The current time in seconds is: {}".format(time.time()))
    print("--------------------------------------\n\n")