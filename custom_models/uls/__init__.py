from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_uls import ULSConfig
from .modeling_uls import ULSForCausalLM, ULSModel

__all__ = ['ULSConfig', 'ULSForCausalLM', 'ULSModel']

AutoConfig.register('uls', ULSConfig)
AutoModel.register(ULSConfig, ULSModel)
AutoModelForCausalLM.register(ULSConfig, ULSForCausalLM)