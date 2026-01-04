import numpy as np
from dataclasses import dataclass


@dataclass
class AttentionParams:
    """ Parameters for the attention mechanism. """

    query_matrix: np.ndarray
    key_matrix: np.ndarray
    value_matrix: np.ndarray

    def state_dict(self) -> dict:
        """ Convert the attention parameters to a state dictionary. """
        return {
            "query_matrix": self.query_matrix,
            "key_matrix": self.key_matrix,
            "value_matrix": self.value_matrix,
        }
    
    def load_state_dict(self, state_dict: dict):
        """ Load parameters from a state dictionary into the attention mechanism. """
        self.query_matrix = state_dict["query_matrix"]
        self.key_matrix = state_dict["key_matrix"]
        self.value_matrix = state_dict["value_matrix"]


@dataclass
class MultiHeadAttentionParams:
    """ Parameters for teh multi head attention mechanism """

    w_out : np.ndarray
    b_out : np.ndarray
    attention_heads : list[AttentionParams]

    def state_dict(self) -> dict:
        return {
            "w_out": self.w_out.copy(),
            "b_out": self.b_out.copy(),
            "attention_heads": [head.state_dict() for head in self.attention_heads],
        }

    def load_state_dict(self, state_dict: dict):
        self.w_out = state_dict["w_out"]
        self.b_out = state_dict["b_out"]

        for head, head_state in zip(self.attention_heads, state_dict["attention_heads"]):
            head.load_state_dict(head_state)


@dataclass
class MLPParams:
    """ Parameters for the Multi-Layer Perceptron (MLP). """

    w_up: np.ndarray
    b_up: np.ndarray
    w_down: np.ndarray
    b_down: np.ndarray

    def state_dict(self) -> dict:
        """ Convert the MLP parameters to a state dictionary. """
        return {
            "w_up": self.w_up,
            "b_up": self.b_up,
            "w_down": self.w_down,
            "b_down": self.b_down,
        }
    
    def load_state_dict(self, state_dict: dict):
        """ Load parameters from a state dictionary into the MLP. """
        self.w_up = state_dict["w_up"]
        self.b_up = state_dict["b_up"]
        self.w_down = state_dict["w_down"]
        self.b_down = state_dict["b_down"]


@dataclass
class LayerNormParams:
    """ Parameters for Layer Normalization. """

    gamma: np.ndarray
    beta: np.ndarray
    eps: float

    def state_dict(self) -> dict:
        """ Convert the layer normalization parameters to a state dictionary. """
        return {
            "gamma": self.gamma,
            "beta": self.beta,
            "eps": self.eps,
        }
    
    def load_state_dict(self, state_dict: dict):
        """ Load parameters from a state dictionary into the layer normalization. """
        self.gamma = state_dict["gamma"]
        self.beta = state_dict["beta"]
        self.eps = state_dict["eps"]


@dataclass
class TransformerBlockParams:
    """ Parameters for a Transformer block. """

    ln1: LayerNormParams
    self_attention: AttentionParams
    ln2: LayerNormParams
    mlp: MLPParams

    def state_dict(self) -> dict:
        return {
            "ln1": self.ln1.state_dict(),
            "self_attention": self.self_attention.state_dict(),
            "ln2": self.ln2.state_dict(),
            "mlp": self.mlp.state_dict(),
        }

    def load_state_dict(self, state_dict: dict):
        self.ln1.load_state_dict(state_dict["ln1"])
        self.self_attention.load_state_dict(state_dict["self_attention"])
        self.ln2.load_state_dict(state_dict["ln2"])
        self.mlp.load_state_dict(state_dict["mlp"])


@dataclass
class ModelParams:
    """ Parameters for the entire model. """

    embedding_matrix: np.ndarray
    position_matrix: np.ndarray
    transformer_block_params: list[TransformerBlockParams]
    w_out: np.ndarray
    b_out: np.ndarray

    def state_dict(self) -> dict:
        return {
            "embedding_matrix": self.embedding_matrix.copy(),
            "position_matrix": self.position_matrix.copy(),
            "transformer_block_params": [
                block.state_dict() for block in self.transformer_block_params
            ],
            "w_out": self.w_out.copy(),
            "b_out": self.b_out.copy(),
        }
    
    def load_state_dict(self, state_dict: dict):
        """ Load parameters from a state dictionary into the model. """
        self.embedding_matrix = state_dict["embedding_matrix"]
        self.position_matrix = state_dict["position_matrix"]
        for block, block_state in zip(
            self.transformer_block_params,
            state_dict["transformer_block_params"]
        ):
            block.load_state_dict(block_state)
        self.w_out = state_dict["w_out"]
        self.b_out = state_dict["b_out"]

    @classmethod
    def from_state_dict(cls, state_dict: dict) -> "ModelParams":
        transformer_blocks = []

        for block_state in state_dict["transformer_block_params"]:

            attention_heads = [
                AttentionParams(
                    query_matrix=head_state["query_matrix"],
                    key_matrix=head_state["key_matrix"],
                    value_matrix=head_state["value_matrix"],
                )
                for head_state in block_state["self_attention"]["attention_heads"]
            ]

            self_attention = MultiHeadAttentionParams(
                w_out=block_state["self_attention"]["w_out"],
                b_out=block_state["self_attention"]["b_out"],
                attention_heads=attention_heads,
            )

            block = TransformerBlockParams(
                ln1=LayerNormParams(
                    gamma=block_state["ln1"]["gamma"],
                    beta=block_state["ln1"]["beta"],
                    eps=block_state["ln1"]["eps"],
                ),
                self_attention=self_attention,
                ln2=LayerNormParams(
                    gamma=block_state["ln2"]["gamma"],
                    beta=block_state["ln2"]["beta"],
                    eps=block_state["ln2"]["eps"],
                ),
                mlp=MLPParams(
                    w_up=block_state["mlp"]["w_up"],
                    b_up=block_state["mlp"]["b_up"],
                    w_down=block_state["mlp"]["w_down"],
                    b_down=block_state["mlp"]["b_down"],
                ),
            )

            transformer_blocks.append(block)

        return cls(
            embedding_matrix=state_dict["embedding_matrix"],
            position_matrix=state_dict["position_matrix"],
            transformer_block_params=transformer_blocks,
            w_out=state_dict["w_out"],
            b_out=state_dict["b_out"],
        )
