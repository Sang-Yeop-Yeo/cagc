from main_utils import *
from training.networks import Generator, Discriminator
from utils.pruning_util import get_pruning_scores
from utils.mask_util import mask_the_generator

import pickle
import dnnlib
import copy
import legacy
import os
from torch_utils import misc



def main():
    print(parser_args)
    print("\n\nBeginning of process.")
    print_time()
    set_random_seed(parser_args.random_seed)
    
    device = torch.device('cuda')
    ####
    parser_args.G_kwargs = dnnlib.EasyDict(class_name='training.networks.Generator', z_dim=parser_args.z_dim, w_dim=parser_args.w_dim, mapping_kwargs=dnnlib.EasyDict(), synthesis_kwargs=dnnlib.EasyDict())
    parser_args.D_kwargs = dnnlib.EasyDict(class_name='training.networks.Discriminator', block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    parser_args.G_kwargs.mapping_kwargs.num_layers = 8
    parser_args.G_kwargs.synthesis_kwargs.channel_base = parser_args.D_kwargs.channel_base = 16384
    parser_args.G_kwargs.synthesis_kwargs.channel_max = parser_args.D_kwargs.channel_max = 512
    parser_args.G_kwargs.synthesis_kwargs.num_fp16_res = parser_args.D_kwargs.num_fp16_res = 4
    parser_args.G_kwargs.synthesis_kwargs.conv_clamp = parser_args.D_kwargs.conv_clamp = 256
    parser_args.D_kwargs.epilogue_kwargs.mbstd_group_size = 8

    common_kwargs = dict(c_dim=parser_args.c_dim, img_resolution=256, img_channels=3)

    G = dnnlib.util.construct_class_by_name(**parser_args.G_kwargs, **common_kwargs).eval().requires_grad_(False).to(device) # subclass of torch.nn.Module
    D = dnnlib.util.construct_class_by_name(**parser_args.D_kwargs, **common_kwargs).eval().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()
    G_pruned = dnnlib.util.construct_class_by_name(pruning_ratio = parser_args.pruning_ratio, **parser_args.G_kwargs, **common_kwargs).eval().requires_grad_(False).to(device) # subclass of torch.nn.Module
    print(f'full model loading from "{parser_args.load_ckpt}"')
    with dnnlib.util.open_url(parser_args.load_ckpt) as f:
        load_data = legacy.load_network_pkl(f)
    for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
        if module is not None:
            misc.copy_params_and_buffers(load_data[name], module, require_all=False) # load하는 부분 체크
    G_ema.requires_grad_(True)
    ####

    start_time = time.time()
    score_list = get_pruning_scores(model = G_ema, 
                                    args = parser_args,
                                    device = device)

    score_array = np.array([np.array(score) for score in score_list])
    pruning_score = np.sum(score_array, axis=0)
    end_time = time.time()

    print("The %s criterion scoring takes: " %parser_args.pruning_criterion, str(round(end_time - start_time, 4)) + ' seconds')

    pruned_generator_dict = mask_the_generator(G_ema.state_dict(), pruning_score, parser_args)
    G_pruned.load_state_dict(pruned_generator_dict)

    for name, module in [('G', G_pruned), ('D', D), ('G_ema', G_pruned)]:
        if module is not None:
            module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
        load_data[name] = module
        del module # conserve memory
    snapshot_pkl = os.path.join(parser_args.save_ckpt, f'pruned_network-{parser_args.pruning_ratio}.pkl')
    
    with open(snapshot_pkl, 'wb') as f:
        pickle.dump(load_data, f)

    print()
    print('Exiting...')

if __name__== "__main__":
    main()