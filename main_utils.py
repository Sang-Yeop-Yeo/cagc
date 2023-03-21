import torch
import numpy as np
from args_helper import parser_args
import time
from utils.utils import set_random_seed
import logging
import dnnlib
from metrics import metric_main


class UserError(Exception):
    pass


def print_time():
    print("\n\n--------------------------------------")
    print("TIME: The current time is: {}".format(time.ctime()))
    print("TIME: The current time in seconds is: {}".format(time.time()))
    print("--------------------------------------\n\n")


def setup_training_loop_kwargs(args):
    kwargs = ["num_gpus", "snap", "metrics" # general options (not included in desc)
              "data", "cond", "subset", "mirror", # dataset
              "cfg", "gamma", "kimg", "batch", # base config
              "aug", "p", "target", "augpipe", # discriminator augmentation
              "resume", "freezed", # transfer learning
              "fp32", "nhwc", "allow_tf32", "nobench", "workers",  # performance options
              "G_reg_interval", "D_reg_interval", # Regularization
              "ada_interval", "ada_kimg",
              "kimg_per_tick", "resume_pkl", "abort_fn", "progress_fn", "augment_kwargs", "pruning_ratio", "cudnn_benchmark",
              "kd_method", "kd_l1_lambda", "kd_lpips_lambda", "kd_mode"] # compression
    
    for kwarg in kwargs:
        if kwarg not in args:
            setattr(args, kwarg, None)

   
    if args.G_reg_interval is None:
        args.G_reg_interval = 4
    if args.D_reg_interval is None:
        args.D_reg_interval = 16
    if args.ada_interval is None:
        args.ada_interval = 4
    if args.ada_kimg is None:
        args.ada_kimg = 500
    if args.kimg_per_tick is None:
        args.kimg_per_tick = 4
    if args.pruning_ratio is None:
        args.pruning_ratio = 0.0
    if args.cudnn_benchmark is None:
        args.cudnn_benchmark = True
    
    #### ca ########## 
    if args.kd_l1_lambda is None:
        args.kd_l1_lambda = 3
    if args.kd_lpips_lambda is None:
        args.kd_lpips_lambda = 3
    if args.kd_mode is None:
        args.kd_mode = "output_only"
    ############

    if args.num_gpus is None:
        args.num_gpus = 1
    assert isinstance(args.num_gpus, int)
    if not (args.num_gpus >=1 and args.num_gpus & (args.num_gpus - 1) == 0):
        raise UserError("--gpus must be a power of two")
    
    if args.snap is None:
        args.snap = 50
    assert isinstance(args.snap, int)
    if args.snap < 1:
        raise UserError("--snap must be at least 1")
    args.image_snapshot_ticks = args.snap
    args.network_snapshot_ticks = args.snap

    if args.metrics is None:
        args.metrics = ["fid50k_full"]
    assert isinstance(args.metrics, list)
    if not all(metric_main.is_valid_metric(metric) for metric in args.metrics):
        raise UserError('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))
    
    assert isinstance(args.random_seed, int)
    assert ("data" in args) and (args.data is not None)
    assert isinstance(args.data, str)
    args.training_set_kwargs = dnnlib.EasyDict(class_name = "training.dataset.ImageFolderDataset",
                                               path = args.data,
                                               use_labels = True,
                                               max_size = None,
                                               xflip = False)
    args.data_loader_kwargs = dnnlib.EasyDict(pin_memory = True,
                                              num_workers = 3,
                                              prefetch_factor = 2)
    try:
        training_set = dnnlib.util.construct_class_by_name(**args.training_set_kwargs)
        args.training_set_kwargs.resolution = training_set.resolution
        args.training_set_kwargs.use_labels = training_set.has_labels
        args.training_set_kwargs.max_size = len(training_set)
        desc = training_set.name
        del training_set
    except IOError as err:
        raise UserError(f"--data: {err}")
    
    if args.cond is None:
        args.cond = False
    assert isinstance(args.cond, bool)
    if args.cond:
        if not args.training_set_kwargs.use_labels:
            raise UserError('--cond=True requires labels specified in dataset.json')
        desc += "-cond"
    else:
        args.training_set_kwargs.use_labels = False

    if args.subset is not None:
        assert isinstance(args.subset, int)
        if not 1 <= args.subset <= args.training_set_kwargs.max_size:
            raise UserError(f'--subset must be between 1 and {args.training_set_kwargs.max_size}')
        desc += f"-subset{args.subset}"
        if args.subset < args.training_set_kwargs.max_size:
            args.training_set_kwargs.max_size = args.subset
            args.training_set_kwargs.random_seed = args.random_seed

    if args.mirror is None:
        args.mirror = False
    assert isinstance(args.mirror, bool)
    if args.mirror:
        desc += "-mirror"
        args.training_set_kwargs.xflip = True

    if args.cfg is None:
        args.cfg = "auto"
    assert isinstance(args.cfg, str)
    desc += f"-{args.cfg}"

    cfg_specs = {
        'auto':      dict(ref_gpus=-1, kimg=25000,  mb=-1, mbstd=-1, fmaps=-1,  lrate=-1,     gamma=-1,   ema=-1,  ramp=0.05, map=2), # Populated dynamically based on resolution and GPU count.
        'stylegan2': dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=10,   ema=10,  ramp=None, map=8), # Uses mixed-precision, unlike the original StyleGAN2.
        'paper256':  dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=0.5, lrate=0.0025, gamma=1,    ema=20,  ramp=None, map=8),
        'paper512':  dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=1,   lrate=0.0025, gamma=0.5,  ema=20,  ramp=None, map=8),
        'paper1024': dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=2,    ema=10,  ramp=None, map=8),
        'cifar':     dict(ref_gpus=2,  kimg=100000, mb=64, mbstd=32, fmaps=1,   lrate=0.0025, gamma=0.01, ema=500, ramp=0.05, map=2),
    }

    assert args.cfg in cfg_specs
    spec = dnnlib.EasyDict(cfg_specs[args.cfg])
    if args.cfg == "auto":
        desc += f"{args.num_gpus:d}"
        spec.ref_gpus = args.num_gpus
        res = args.training_set_kwargs.resolution
        spec.mb = max(min(args.num_gpus * min(4096 // res, 32), 64), args.num_gpus)
        spec.mbstd = min(spec.mb // args.num_gpus, 4)
        spec.fmaps = 1 if res >= 512 else 0.5
        spec.lrate = 0.002 if res >= 1024 else 0.0025
        spec.gamma = 0.0002 * (res ** 2) / spec.mb
        spec.ema = spec.mb * 10 / 32

    args.G_kwargs = dnnlib.EasyDict(class_name='training.networks.Generator', z_dim=args.z_dim, w_dim=args.w_dim, mapping_kwargs=dnnlib.EasyDict(), synthesis_kwargs=dnnlib.EasyDict())
    args.D_kwargs = dnnlib.EasyDict(class_name='training.networks.Discriminator', block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    args.G_kwargs.synthesis_kwargs.channel_base = args.D_kwargs.channel_base = int(spec.fmaps * 32768)
    args.G_kwargs.synthesis_kwargs.channel_max = args.D_kwargs.channel_max = 512
    args.G_kwargs.mapping_kwargs.num_layers = spec.map
    args.G_kwargs.synthesis_kwargs.num_fp16_res = args.D_kwargs.num_fp16_res = 4
    args.G_kwargs.synthesis_kwargs.conv_clamp = args.D_kwargs.conv_clamp = 256
    args.D_kwargs.epilogue_kwargs.mbstd_group_size = spec.mbstd
    
    args.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0,0.99], eps=1e-8) # SWAD 할 수 있으면 좋겠음
    args.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0,0.99], eps=1e-8) # SWAD 할 수 있으면 좋겠음
    args.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.StyleGAN2Loss', kd_method = args.kd_method, r1_gamma=spec.gamma) # KD 삽입하기
    
    ######################
    args.loss_kwargs.kd_l1_lambda = 3
    args.loss_kwargs.kd_lpips_lambda = 3
    args.loss_kwargs.kd_mode = "output_only"
    ######################

    args.total_kimg = spec.kimg
    args.batch_size = spec.mb
    args.batch_gpu = spec.mb // spec.ref_gpus
    args.ema_kimg = spec.ema
    args.ema_rampup = spec.ramp

    if args.cfg == "cifar":
        args.loss_kwargs.pl_weight = 0
        args.loss_kwargs.style_mixing_prob = 0
        args.D_kwargs.architecture = 'orig'

    if args.gamma is not None:
        assert isinstance(args.gamma, float)
        if not args.gamma >= 0:
            raise UserError('--gamma must be non-negative')
        desc += f'-gamma{args.gamma:g}'
        args.loss_kwargs.r1_gamma = args.gamma

    if args.kimg is not None:
        assert isinstance(args.kimg, int)
        if not args.kimg >= 1:
            raise UserError('--kimg must be at least 1')
        desc += f'-kimg{args.kimg:d}'
        args.total_kimg = args.kimg

    if args.batch is not None:
        assert isinstance(args.batch, int)
        if not (args.batch >= 1 and args.batch % args.num_gpus == 0):
            raise UserError('--batch must be at least 1 and divisible by --gpus')
        desc += f'-batch{args.batch}'
        args.batch_size = args.batch
        args.batch_gpu = args.batch // args.num_gpus

    # ---------------------------------------------------
    # Discriminator augmentation: aug, p, target, augpipe
    # ---------------------------------------------------    

    if args.aug is None:
        args.aug = "ada"
    else:
        assert isinstance(args.aug, str)
        desc += f"-{args.aug}"

    if args.aug == "ada":
        args.ada_target = 0.6

    elif args.aug == "noaug":
        pass
    
    elif args.aug == "fixed":
        if args.p is None:
            raise UserError(f"--aug={args.aug} requires specifying --p")
        
    else:
        raise UserError(f"--aug={args.aug} not supported")
    
    if args.p is not None:
        assert isinstance(args.p, float)
        if args.aug != "fixed":
            raise UserError("--p can only be specified with --aug=fixed")
        if not 0 <= args.p <= 1:
            raise UserError("--p must be between 0 and 1")
        desc += f"-p{args.p:g}"
        args.augment_p = args.p

    if args.target is not None:
        assert isinstance(args.target, float)
        if args.aug != "ada":
            raise UserError('--target can only be specified with --aug=ada')
        if not 0 <= args.target <= 1:
            raise UserError('--target must be between 0 and 1')
        desc += f'-target{args.target:g}'
        args.ada_target = args.target
    
    assert args.augpipe is None or isinstance(args.augpipe, str)
    if args.augpipe is None:
        args.augpipe = 'bgc'
    else:
        if args.aug == 'noaug':
            raise UserError('--augpipe cannot be specified with --aug=noaug')
        desc += f'-{args.augpipe}'

    augpipe_specs = {
        'blit':   dict(xflip=1, rotate90=1, xint=1),
        'geom':   dict(scale=1, rotate=1, aniso=1, xfrac=1),
        'color':  dict(brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
        'filter': dict(imgfilter=1),
        'noise':  dict(noise=1),
        'cutout': dict(cutout=1),
        'bg':     dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1),
        'bgc':    dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
        'bgcf':   dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1),
        'bgcfn':  dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1),
        'bgcfnc': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1, cutout=1),
    }

    assert args.augpipe in augpipe_specs
    if args.aug != 'noaug':
        args.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', **augpipe_specs[args.augpipe])
        
    resume_specs = {
        'ffhq256':     'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res256-mirror-paper256-noaug.pkl',
        'ffhq512':     'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res512-mirror-stylegan2-noaug.pkl',
        'ffhq1024':    'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res1024-mirror-stylegan2-noaug.pkl',
        'celebahq256': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/celebahq-res256-mirror-paper256-kimg100000-ada-target0.5.pkl',
        'lsundog256':  'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/lsundog-res256-paper256-kimg100000-noaug.pkl',
    }    
    
    assert args.resume is None or isinstance(args.resume, str)
    if args.resume is None:
        args.resume = 'noresume'
    elif args.resume == 'noresume':
        desc += '-noresume'
    elif args.resume in resume_specs:
        desc += f'-resume{args.resume}'
        args.resume_pkl = resume_specs[args.resume] # predefined url
    else:
        desc += '-resumecustom'
        args.resume_pkl = args.resume # custom path or url

    if args.resume != 'noresume':
        args.ada_kimg = 100 # make ADA react faster at the beginning
        args.ema_rampup = None # disable EMA rampup

    if args.freezed is not None:
        assert isinstance(args.freezed, int)
        if not args.freezed >= 0:
            raise UserError('--freezed must be non-negative')
        desc += f'-freezed{args.freezed:d}'
        args.D_kwargs.block_kwargs.freeze_layers = args.freezed

    # -------------------------------------------------
    # Performance options: fp32, nhwc, nobench, workers
    # -------------------------------------------------

    if args.fp32 is None:
        args.fp32 = False
    assert isinstance(args.fp32, bool)
    if args.fp32:
        args.G_kwargs.synthesis_kwargs.num_fp16_res = args.D_kwargs.num_fp16_res = 0
        args.G_kwargs.synthesis_kwargs.conv_clamp = args.D_kwargs.conv_clamp = None

    if args.nhwc is None:
        args.nhwc = False
    assert isinstance(args.nhwc, bool)
    if args.nhwc:
        args.G_kwargs.synthesis_kwargs.fp16_channels_last = args.D_kwargs.block_kwargs.fp16_channels_last = True

    if args.nobench is None:
        args.nobench = False
    assert isinstance(args.nobench, bool)
    if args.nobench:
        args.cudnn_benchmark = False

    if args.allow_tf32 is None:
        args.allow_tf32 = False
    assert isinstance(args.allow_tf32, bool)

    if args.workers is not None:
        assert isinstance(args.workers, int)
        if not args.workers >= 1:
            raise UserError('--workers must be at least 1')
        args.data_loader_kwargs.num_workers = args.workers

    return desc