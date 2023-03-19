from main_utils import *
import tempfile
import dnnlib
import os
from torch_utils import training_stats
from torch_utils import custom_ops
from training import training_loop
import json


#num_gpus
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


def main(ctx, outdir, dry_run):
    #log = set_logger(parser_args) #logger 저장위치 바꾸기, 시간 반영하기
    #log.info("parser_args: {}".format(parser_args))
    dnnlib.util.Logger(should_flush=True) # setting 어떻게?
    print("\n\nBeginning of process.")
    print_time()


    run_desc = setup_training_loop_kwargs(parser_args)

    parser_args.run_dir = parser_args.ckpt + "_" + time.strftime("%m/%d_%H:%M:%S", time.localtime())

    print()
    print('Training options:')
    print(json.dumps(parser_args, indent=2))
    print()
    print(f'Output directory:   {parser_args.run_dir}')
    print(f'Training data:      {parser_args.training_set_kwargs.path}')
    print(f'Training duration:  {parser_args.total_kimg} kimg')
    print(f'Number of GPUs:     {parser_args.num_gpus}')
    print(f'Number of images:   {parser_args.training_set_kwargs.max_size}')
    print(f'Image resolution:   {parser_args.training_set_kwargs.resolution}')
    print(f'Conditional model:  {parser_args.training_set_kwargs.use_labels}')
    print(f'Dataset x-flips:    {parser_args.training_set_kwargs.xflip}')
    print()


    print('Creating output directory...')
    os.makedirs(parser_args.run_dir)
    with open(os.path.join(parser_args.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(parser_args, f, indent=2)


    print('Launching processes...')
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