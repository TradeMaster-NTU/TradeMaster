from .builder import OPTIMIZERS
from torch import optim
from torch.optim import Adam, AdamW, Adagrad, Adadelta, SGD

@OPTIMIZERS.register_module()
class Optimizer(optim.Optimizer):
    def __init__(self, params, defaults):
        self.params = params
        self.defaults = defaults
        super(Optimizer, self).__init__(params, defaults)

OPTIMIZERS.register_module(name="Adam",force=False, module=Adam)
OPTIMIZERS.register_module(name = "AdamW", force=False, module=AdamW)
OPTIMIZERS.register_module(name = "Adagrad", force=False, module=Adagrad)
OPTIMIZERS.register_module(name = "Adadelta", force=False, module=Adadelta)
OPTIMIZERS.register_module(name = "SGD",force=False, module=SGD)