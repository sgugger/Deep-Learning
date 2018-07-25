from fastai.conv_learner import *
from fastai.fp16 import *
from fastai.models.cifar10.wideresnet import wrn_22
from utils import get_opt_fn, get_phases, log_msg
from callbacks import *
import argparse
from databunch import DataBunch, CustomTfm
import torch.multiprocessing as mp

def main_train(lr, bs, cuda_id, not_distrib, fp16, loss_scale):
    """
    Trains a Language Model

    lr (float): maximum learning rate
    moms (float/tuple): value of the momentum/beta1. If tuple, cyclical momentums will be used
    wd (float): weight decay to be used
    wd_loss (bool): weight decay computed inside the loss if True (l2 reg) else outside (true wd)
    opt_fn (optimizer): name of the optim function to use (should be SGD, RMSProp or Adam)
    bs (int): batch size
    cyc_len (int): length of the cycle
    beta2 (float): beta2 parameter of Adam or alpha parameter of RMSProp
    amsgrad (bool): for Adam, sues amsgrad or not
    div (float): value to divide the maximum learning rate by
    pct (float): percentage to leave for the annealing at the end
    lin_end (bool): if True, the annealing phase goes from the minimum lr to 1/100th of it linearly
                    if False, uses a cosine annealing to 0
    tta (bool): if True, uses Test Time Augmentation to evaluate the model
    """
    torch.backends.cudnn.benchmark = True
    if fp16: assert torch.backends.cudnn.enabled, "missing cudnn"
    stats = (np.array([ 0.4914 ,  0.48216,  0.44653]), np.array([ 0.24703,  0.24349,  0.26159]))
    sz=32
    PATH = Path("../../data/cifar10/")
    tfms = tfms_from_stats(stats, sz, aug_tfms=[RandomCrop(sz), RandomFlip()], pad=sz//8)
    data1 = ImageClassifierData.from_paths(PATH, val_name='test', tfms=tfms, bs=bs)
    m = wrn_22().cuda()
    if not not_distrib: m = nn.parallel.DistributedDataParallel(m, device_ids=[cuda_id], output_device=cuda_id)
    learn = ConvLearner.from_model_data(m, data1)
    learn.crit = nn.CrossEntropyLoss()
    learn.metrics = [accuracy]
    trn_tfms = CustomTfm(0.5, 4, 32, 1)
    val_tfms = None
    data = DataBunch.from_files(PATH, trn_tfms, val_tfms, stats, torch.device('cuda', cuda_id), distrib=not not_distrib, val_name='test', bs=bs)
    learn.data.trn_dl, learn.data.val_dl = data.trn_dl, data.val_dl
    if fp16: learn.half()
    x,y = next(iter(data.trn_dl))
    opt_fn = get_opt_fn('Adam', 0.95, 0.99, False)
    learn.opt_fn = opt_fn
    cyc_len, pct = 30, 0.075
    nbs = [cyc_len * (1-pct) / 2, cyc_len * (1-pct) / 2, cyc_len * pct]
    phases = get_phases(lr, (0.95,0.85), opt_fn, 10, nbs, 0.1, True, False)
    #print_lr = PrintLR(learn)
    learn.fit_opt_sched(phases, loss_scale=loss_scale)
   
class PrintLR(Callback):

    def __init__(self, learner):
        self.learner = learner

    def on_train_begin(self):
        self.n = 0
    
    def on_batch_begin(self):
        if self.n ==0:
            print(self.learner.sched.layer_opt.opt.param_groups[0]['lr'])
        self.n += 1
    
    def on_epoch_end(self, metrics):
        self.n = 0

def main():
    """
    Launches the trainings.

    See main_train for the description of all the arguments.
    name (string): name to be added to the log file
    cuda_id (int): index of the GPU to use
    nb_exp (int): number of experiments to run in a row
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=12e-3)
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument('--fp16', action='store_true', help='Run model fp16 mode.')
    parser.add_argument('--not_distrib', action='store_true', help='Run model fp16 mode.')
    parser.add_argument('--loss_scale', type=float, default=1)
    parser.add_argument("--local_rank", type=int)
    arg = parser.parse_args()
    torch.cuda.set_device(arg.local_rank)
    if not arg.not_distrib: torch.distributed.init_process_group('nccl', init_method='env://')
    main_train(arg.lr, arg.bs, arg.local_rank, arg.not_distrib, arg.fp16, arg.loss_scale)

if __name__ == '__main__': 
    #mp.set_start_method('spawn')
    main()