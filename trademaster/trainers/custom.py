from .builder import TRAINERS

@TRAINERS.register_module()
class Trainer():
    def __init__(self, **kwargs):
        super(Trainer, self).__init__()