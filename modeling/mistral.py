from .model_quantizer_base import ModelQuantizerBase


class MistralQuantizerForCausalLM(ModelQuantizerBase):
    inside_layer_modules = [
        ["self_attn.q_proj"],
        ["self_attn.k_proj"],
        ["self_attn.v_proj"],
        ["self_attn.o_proj"],
        ["mlp.gate_proj"],
        ["mlp.up_proj"],
        ["mlp.down_proj"]
    ]
    quantizer_key_prefix = "model.layers"

    def get_word_embedding_layer(self):
        return self.model.model.embed_tokens

    def get_all_layers(self):
        return self.model.model.layers

    def get_transformer_output(self):
        return self.model.model.norm

    def get_lm_head(self):
        return self.model.lm_head
