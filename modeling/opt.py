from .model_quantizer_base import ModelQuantizerBase


class OptQuantizerForCausalLM(ModelQuantizerBase):
    inside_layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.out_proj"],
        ["fc1"],
        ["fc2"]
    ]
    quantizer_key_prefix = "model.decoder.layers"

    def get_word_embedding_layer(self):
        return self.model.model.decoder.embed_tokens

    def get_embed_positions(self):
        return self.model.model.decoder.embed_positions

    def get_project_out(self):
        return self.model.model.decoder.project_out

    def get_project_in(self):
        return self.model.model.decoder.project_in

    def get_all_layers(self):
        return self.model.model.decoder.layers

    def get_transformer_output(self):
        return self.model.model.decoder.final_layer_norm

    def get_lm_head(self):
        return self.model.lm_head
