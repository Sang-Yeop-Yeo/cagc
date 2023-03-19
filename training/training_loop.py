import torch
import time
import dnnlib
from utils.utils import set_seed
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix

def training_loop(args):
    start_time = time.time()
    device = torch.device("cuda", args.rank)
    set_seed(args.seed * args.num_gpus + args.rank)
    # torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    # torch.backends.cudnn.allow_tf32 = allow_tf32
    conv2d_gradfix.enabled = True
    grid_sample_gradfix.enabled = True

    training_set = dnnlib.util.construct_class_by_name(**args.training_set_kwargs)
    #training_set_sampler = 