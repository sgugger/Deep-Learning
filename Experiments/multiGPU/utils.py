from fastai.conv_learner import *

def log_msg(file, msg):
    print(msg)
    file.write('\n' + msg)

def get_opt_fn(opt_fn, mom, beta, amsgrad):
    """
    Helper function to return a proper optim function from its name

    opt_fn (string): name of the optim function (should be SGD, RMSProp or Adam)
    mom (float): momentum to use (or beta1 in the case of Adam)
    beta (float): alpha parameter in RMSProp and beta2 in Adam
    amsgrad (bool): for Adam only, uses amsgrad or not
    """
    assert opt_fn in {'SGD', 'RMSProp', 'Adam'}, 'optim should be SGD, RMSProp or Adam'
    if opt_fn=='SGD': res = optim.SGD
    elif opt_fn=='RMSProp': res = optim.RMSprop if beta is None else partial(optim.RMSProp, alpha=beta)
    else: res = partial(optim.Adam, amsgrad=amsgrad) if beta is None else partial(optim.Adam, betas=(mom,beta), amsgrad=amsgrad)
    return res

def get_one_phase(nb, opt_fn, lr, lr_decay, moms, wd, wd_loss):
    """
    Helper function to create one training phase.

    nb (int): number of epochs
    opt_fn (optimizer): the optim function to use
    lr (float/tuple): the learning rate(s) to use. If tuple, going from the first to the second
    lr_decay (DecayType): the decay type to go from lr1 to lr2
    moms (float/tuple): the momentum(s) to use. If tuple, going from the first to the second linearly
    wd (float): weight decay
    wd_loss (bool): weight decay computed inside the loss if True (l2 reg) else outside (true wd)
    """
    if isinstance(moms, Iterable):
        return TrainingPhase(nb, opt_fn, lr=lr, lr_decay=lr_decay, momentum=moms, 
                             momentum_decay=DecayType.LINEAR, wds=wd, wd_loss=wd_loss)
    else:
        return TrainingPhase(nb, opt_fn, lr=lr, lr_decay=lr_decay, momentum=moms, 
                             wds=wd, wd_loss=wd_loss)

def get_phases(lr, moms, opt_fn, div, nbs, wd, lin_end=False, wd_loss=True):
    """
    Creates the phases for a 1cycle policy (or a variant)

    lr (float): maximum learning rate
    moms (float/tuple): value of the momentum/beta1. If tuple, cyclical momentums will be used
    opt_fn (optimizer): the optim function to use
    div (float): value to divide the maximum learning rate by
    nbs (list): number of epochs for each phase (ascending, constant if len==4, descending, annealing)
    wd (float): weight decay
    lin_end (bool): if True, the annealing phase goes from the minimum lr to 1/100th of it linearly
                    if False, uses a cosine annealing to 0
    wd_loss (bool): weight decay computed inside the loss if True (l2 reg) else outside (true wd)
    """
    max_mom = moms[0] if isinstance(moms, Iterable) else moms
    min_mom = moms[1] if isinstance(moms, Iterable) else moms
    moms_r = (moms[1],moms[0]) if isinstance(moms, Iterable) else moms
    phases = [get_one_phase(nbs[0], opt_fn, (lr/div,lr), DecayType.LINEAR, moms, wd, wd_loss)]
    if len(nbs)==4:
        phases.append(get_one_phase(nbs[1], opt_fn, lr, DecayType.NO, min_mom, wd, wd_loss))
        nbs = [nbs[0]] + nbs[2:]
    phases.append(get_one_phase(nbs[1], opt_fn, (lr,lr/div), DecayType.LINEAR, moms_r, wd, wd_loss))
    if lin_end:
        phases.append(get_one_phase(nbs[2], opt_fn, (lr/div,lr/(100*div)), DecayType.LINEAR, max_mom, wd, wd_loss))
    else:
        phases.append(get_one_phase(nbs[2], opt_fn, lr/div, DecayType.COSINE, max_mom, wd, wd_loss))
    return phases