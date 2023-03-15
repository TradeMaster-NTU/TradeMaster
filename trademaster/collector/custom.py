from .builder import COLLECTORS

@COLLECTORS.register_module()
class CollectorBase:
    def __init__(self, **kwargs):
        pass