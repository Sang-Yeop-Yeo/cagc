from main_utils import *
import tempfile
import dnnlib
import os
from torch_utils import training_stats
from torch_utils import custom_ops
from training import training_loop
from metrics import metric_main

#num_gpus
class UserError(Exception):
    pass

def setup_training_loop_kwargs(args):
    if ("num_gpus" not in args) or (args.num_gpus is None):
        args.num_gpus = 1
    assert isinstance(args.num_gpus, int)
    if not (args.num_gpus >=1 and args.num_gpus & (args.num_gpus - 1) == 0):
        raise UserError("--gpus must be a power of two")
    
    if ("snap" not in args) or (args.snap is None):
        args.snap = 50
    assert isinstance(args.snap, int)
    if args.snap < 1:
        raise UserError("--snap must be at least 1")
    args.image_snapshot_ticks = args.snap
    args.network_snapshot_ticks = args.snap

    if ("metrics" not in args) or (args.metrics is None):
        args.metrics = ["fid50k_full"]
    assert isinstance(args.metrics, list)
    if not all(metric_main.is_valid_metric(metric) for metric in args.metrics):
        raise UserError('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))
    
    assert isinstance(args.seed, int)
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
    
    if ("cond" not in args) or (args.cond is None):
        args.cond = False
    assert isinstance(args.cond, bool)
    if args.cond:
        if not args.training_set_kwargs.use_labels:
            raise UserError('--cond=True requires labels specified in dataset.json')
        desc += "-cond"
    else:
        args.training_set_kwargs.use_labels = False

    if "subset" not in args:
        args.subset = None
    if args.subset is not None:
        assert isinstance(args.subset, int)
        if not 1 <= args.subset <= args.training_set_kwargs.max_size:
            raise UserError(f'--subset must be between 1 and {args.training_set_kwargs.max_size}')
        desc += f"-subset{args.subset}"
        if args.subset < args.training_set_kwargs.max_size:
            args.training_set_kwargs.max_size = args.subset
            args.training_set_kwargs.random_seed = args.random_seed

    if "mirror" not in args:
        args.mirror = False
    assert isinstance(args.mirror, bool)
    if args.mirror:
        desc += "-mirror"
        args.training_set_kwargs.xflip = True

    if "cfg" not in args:
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
    args.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.StyleGAN2Loss', r1_gamma=spec.gamma) # KD 삽입하기

    args.total_kimg = spec.kimg
    args.batch_size = spec.mb
    args.batch_gpu = spec.mb // spec.ref_gpus
    args.ema_kimg = spec.ema
    args.ema_rampup = spec.ramp





def subprocess_fn(args, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(args.save_ckpt, "log.txt"), file_mode = "a", should_flush = True)

    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, ".torch_distributed_init"))
        if os.name == "nt":
            init_method = "file:///" + init_file.replace("\\", "/")
            torch.distributed.init_process_group(backend = "gloo", init_method = init_method, rank = args.rank, world_size = args.num_gpus)
        else:
            init_method = f"file://{init_file}"
            torch.distributed.init_process_group(backend = "nccl", init_method = init_method, rank = args.rank, world_size = args.num_gpus)

    sync_device = torch.device("cuda", args.rank) if args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank = args.rank, sync_device = sync_device)
    if args.rank != 0:
        custom_ops.verbosity = "none"

    training_loop.training_loop(args)




def main():
    #log = set_logger(parser_args) #logger 저장위치 바꾸기, 시간 반영하기
    #log.info("parser_args: {}".format(parser_args))
    dnnlib.util.Logger(should_flush=True) # setting 어떻게?


    run_desc = setup_training_loop_kwargs(parser_args)



    print("\n\nBeginning of process.")
    print_time()
    set_seed(parser_args.seed)

    torch.multiprocessing.set_start_method("spawn")
    with tempfile.TemporaryDirectory() as temp_dir:
        if parser_args.num_gpus == 1:
            subprocess_fn(args = parser_args,
                          temp_dir = temp_dir)
        else:
            torch.multiprocessing.spawn(fn = subprocess_fn,
                                        args = (parser_args, temp_dir),
                                        nprocs = parser_args.num_gpus)





if __name__ == "__main__":
    main()