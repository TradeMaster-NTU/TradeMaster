from .builder import AGENTS

@AGENTS.register_module()
class AgentBase:
    def __init__(self, **kwargs):
        pass
