from .model_quantizer_base import ModelQuantizerBase


class BloomQuantizerForCausalLM(ModelQuantizerBase):
    has_alibi: bool = True
    inside_layer_modules = [
        ["self_attention.query_key_value"],
        ["self_attention.dense"],
        ["mlp.dense_h_to_4h"],
        ["mlp.dense_4h_to_h"]
    ]

    quantizer_key_prefix = "transformer.h"

    def get_word_embedding_layer(self):
        return self.model.transformer.word_embeddings

    def get_all_layers(self):
        return self.model.transformer.h

    def get_transformer_output(self):
        return self.model.transformer.ln_f

    def get_lm_head(self):
        return self.model.lm_head

    def get_word_embedding_layer_norm(self):
        return self.model.transformer.word_embeddings_layernorm
