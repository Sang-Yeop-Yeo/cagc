import torch
import time
import dnnlib
from utils.utils import set_seed
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix
from torch_utils import misc
import copy

def training_loop(args):
    start_time = time.time()
    device = torch.device("cuda", args.rank)
    set_seed(args.seed * args.num_gpus + args.rank)
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    conv2d_gradfix.enabled = True
    grid_sample_gradfix.enabled = True

    training_set = dnnlib.util.construct_class_by_name(**args.training_set_kwargs)
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=args.rank, num_replicas=args.num_gpus, seed=args.random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=args.batch_size//args.num_gpus, **args.data_loader_kwargs))

    common_kwargs = dict(c_dim=training_set.label_dim, img_resolution=training_set.resolution, img_channels=training_set.num_channels)
    G = dnnlib.util.construct_class_by_name(**args.G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    D = dnnlib.util.construct_class_by_name(**args.D_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()