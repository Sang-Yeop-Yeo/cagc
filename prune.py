from main_utils import *
from training.networks import Generator, Discriminator
from utils.pruning_util import get_pruning_scores
from utils.mask_util import mask_the_generator

import pickle



def main():
    print(parser_args)
    print("\n\nBeginning of process.")
    print_time()
    set_random_seed(parser_args.random_seed)
    
    device = torch.device('cuda', parser_args.rank)
    #### model_dict_load
    #dense_model_dict = torch.load(parser_args.load_ckpt, map_location=device)
    #print(dense_model_dict["g_ema"].keys())
    g_ema = Generator(512, 0, 512, 256, 3).to(device)
    dis = Discriminator(0, 256, 3).to(device)
    #with open(parser_args.load_ckpt, "rb") as f:
    #    dense_model_dict = pickle.load(f)
    #### model_dict_load

    start_time = time.time()
    score_list = get_pruning_scores(model = g_ema, 
                                    args = parser_args,
                                    device = device)

    score_array = np.array([np.array(score) for score in score_list])
    pruning_score = np.sum(score_array, axis=0)
    end_time = time.time()

    print("The %s criterion scoring takes: " %parser_args.pruning_criterion, str(round(end_time - start_time, 4)) + ' seconds')

    pruned_generator_dict = mask_the_generator(g_ema.state_dict(), pruning_score, parser_args)
    #pruned_ckpt = {'g': pruned_generator_dict, 'd': model_dict['d'], }

if __name__== "__main__":
    main()