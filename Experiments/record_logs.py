from fastai.conv_learner import *

class LogResults(Callback):
    """
    Callback to log all the results of the training:
    - at the end of each epoch: training loss, validation loss and metrics
    - at the end of the first batches then every epoch: deciles of the params and their gradients
    """
    
    def __init__(self, learn, fname, init_text=''):
        super().__init__()
        self.learn, self.fname, self.init_text = learn, fname, init_text
        self.pcts = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999]
        self.pnames = {p:n for n,p in learn.model.named_parameters()}
        self.module_names = get_module_names(learn.model)
        
    def on_train_begin(self):
        self.logs, self.epoch, self.n = self.init_text + "\n", 0, 0
        self.deciles = {}
        for name in self.pnames.values(): 
            self.deciles[name] = collections.defaultdict(list)
            self.deciles[name + '.grad'] = collections.defaultdict(list)
        for name in self.module_names.values(): self.deciles[name] = collections.defaultdict(list)
        names = ["epoch", "trn_loss", "val_loss", "metric"]
        layout = "{!s:10} " * len(names)
        self.logs += layout.format(*names) + "\n"
    
    def on_batch_begin(self):
        if self.n == 0 or (self.epoch == 0 and is_power_of_two(self.n+1)):
            self.hooks = []
            self.learn.model.apply(self.register_hook)
    
    def on_batch_end(self, metrics):
        self.loss = metrics
        if self.n == 0 or (self.epoch == 0 and is_power_of_two(self.n+1)):
            self.save_deciles()
        if len(self.hooks) != 0:
            for h in self.hooks: h.remove()
            self.hooks=[]
        self.n += 1
    
    def on_epoch_end(self, metrics):
        self.save_stats(self.epoch, [self.loss] + metrics)
        self.epoch += 1
        self.n=0
        
    def save_stats(self, epoch, values, decimals=6):
        layout = "{!s:^10}" + " {!s:10}" * len(values)
        values = [epoch] + list(np.round(values, decimals))
        self.logs += layout.format(*values) + "\n"
    
    def save_deciles(self):
        for group_param in self.learn.sched.layer_opt.opt_params():
            for param in group_param['params']:
                self.add_deciles(self.pnames[param], to_np(param))
                self.add_deciles(self.pnames[param] + '.grad', to_np(param.grad))
    
    def separate_pcts(self,arr):
        n = len(arr.reshape(-1))
        pos, neg = arr[arr > 0], arr[arr < 0]
        pos_pcts = np.percentile(pos, self.pcts) if len(pos) > 0 else np.array([])
        neg_pcts = np.percentile(neg, self.pcts) if len(neg) > 0 else np.array([])
        return len(pos)/n, len(neg)/n, pos_pcts, neg_pcts
    
    def add_deciles(self, name, arr):
        pos, neg, pct_pos, pct_neg = self.separate_pcts(arr)
        self.deciles[name]['sgn'].append([pos, neg])
        self.deciles[name]['pos'].append(pct_pos)
        self.deciles[name]['neg'].append(pct_neg)
                                                        
    def on_train_end(self):
        with open(self.fname + '.txt', 'a') as f: f.write(self.logs)
        pickle.dump(self.deciles, open(self.fname + '.pkl', 'wb'))
        
    def register_hook(self, module):
        def hook_save_act(module, input, output):
            pos, neg, pct_pos, pct_neg = self.separate_pcts(to_np(output))
            m_name = self.module_names[module]
            self.deciles[m_name]['sgn'].append([pos, neg])
            self.deciles[m_name]['pos'].append(pct_pos)
            self.deciles[m_name]['neg'].append(pct_neg)
        if (not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and not (module == self.learn.model)):
            self.hooks.append(module.register_forward_hook(hook_save_act))

def get_module_names(model):
    def register_names(module):
        if (not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and not (module == model)):
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            m_name = f'{class_name}-{len(names)+1}'
            names[module] = m_name
    names = {}
    model.apply(register_names)
    return names

def is_power_of_two(n):
    while n>1:
        if n%2 != 0: return False
        n = n//2
    return True