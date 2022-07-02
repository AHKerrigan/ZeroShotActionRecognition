import argparse
import multiprocessing
#import argparge
import torch

def getopt():
    parser = argparse.ArgumentParser()

    opt = parser.parse_args()
    opt.kernels = multiprocessing.cpu_count()
    opt.gpus = 1

    opt.resources = "/home/alec/Documents/BigDatasets/resources/"

    opt.prompt = "a video of "
    opt.text_embed = 'w2vavg'
    opt.train_samples = -1
    opt.network = 'multiplication'

    opt.size = 112
    opt.clip_len = 16
    opt.n_clips = 1

    opt.is_validation = False

    opt.class_overlap = 0.05
    opt.class_total = 350

    opt.n_epochs = 20

    opt.eval_per_epoch = 2
    opt.loss_per_epoch = 40

    #opt.description = 'GeoGuess4-4.2M-Im2GPS3k-F*'
    opt.description = 'Testing'

    opt.archname = 'MixFormer'
    opt.evaluate = False
    opt.cluster = False

    opt.lr = 1e-3
    opt.step_size = 3
    opt.hier_eval = True
    opt.scene = False

    opt.batch_size = 32
    opt.dataset = 'kinetics'

    opt.valset1 = 'ucf101'
    opt.valset2 = 'hmdb51'
    opt.traintype = 'ce'

    opt.device = torch.device('cuda')



    return opt