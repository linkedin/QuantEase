# Disclaimer: The following code adheres to the AutoGPTQ style.
# Please refer to the implementation details at:
# https://github.com/PanQiWei/AutoGPTQ/blob/main/auto_gptq/modeling

from torch import device
from packaging.version import parse as parse_version

from .falcon import FalconQuantizerForCausalLM
from .llama import LlamaQuantizerForCausalLM
from .mistral import MistralQuantizerForCausalLM
from .bloom import BloomQuantizerForCausalLM
from .opt import OptQuantizerForCausalLM


def compare_transformers_version(
        version: str = "v4.28.0",
        op: str = "eq"
):
    assert op in ["eq", "lt", "le", "gt", "ge"]

    from transformers import __version__

    return getattr(parse_version(__version__), f"__{op}__")(parse_version(version))


SUPPORTED_MODELS = ["llama", "bloom", "opt"]
if compare_transformers_version("v4.33.0", op="ge"):
    SUPPORTED_MODELS.append("falcon")
if compare_transformers_version("v4.34.0", op="ge"):
    SUPPORTED_MODELS.append("mistral")

EXLLAMA_DEFAULT_MAX_INPUT_LENGTH = 2048

CAUSAL_LM_QUANTIZER_MAP = {
    "falcon": FalconQuantizerForCausalLM,
    "mistral": MistralQuantizerForCausalLM,
    "llama": LlamaQuantizerForCausalLM,
    "bloom": BloomQuantizerForCausalLM,
    "opt": OptQuantizerForCausalLM
}

__all__ = ["SUPPORTED_MODELS", "EXLLAMA_DEFAULT_MAX_INPUT_LENGTH", "CAUSAL_LM_QUANTIZER_MAP"]
