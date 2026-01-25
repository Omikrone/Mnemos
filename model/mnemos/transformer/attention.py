from mnemos import xp
from mnemos.transformer.gradient import Param
from mnemos.transformer.save_model import AttentionParams, MultiHeadAttentionParams
from mnemos.config.params import EMBEDDING_DIM, NB_ATTENTION_HEADS, EPS


class SelfAttention:
    """ SelfAttention class for implementing the attention mechanism in the Transformer model. """

    input_embeddings : xp.ndarray
    query_matrix : Param
    key_matrix : Param
    value_matrix : Param
    output : xp.ndarray
    attention_weights : xp.ndarray
    value : xp.ndarray
    key : xp.ndarray
    query : xp.ndarray


    def __init__(self):
        """ Initialize the attention matrices with random values. """

        # Random initialization of attention matrices
        std = xp.sqrt(2.0 / (EMBEDDING_DIM + EMBEDDING_DIM))
        self.query_matrix = Param(xp.random.randn(EMBEDDING_DIM, EMBEDDING_DIM) * std)
        self.key_matrix = Param(xp.random.randn(EMBEDDING_DIM, EMBEDDING_DIM) * std)
        self.value_matrix = Param(xp.random.randn(EMBEDDING_DIM, EMBEDDING_DIM) * std)


    @classmethod
    def from_params(cls, params: AttentionParams):
        """ Create an Attention instance from saved parameters. """

        instance = cls()
        instance.query_matrix = Param(params.query_matrix)
        instance.key_matrix = Param(params.key_matrix)
        instance.value_matrix = Param(params.value_matrix)
        return instance


    def add_attention(self, input_embeddings: xp.ndarray) -> xp.ndarray:
        self.input_embeddings = input_embeddings
        
        # Projections
        self.query = input_embeddings @ self.query_matrix.value
        self.key   = input_embeddings @ self.key_matrix.value
        self.value = input_embeddings @ self.value_matrix.value

        # Attention scores
        d_k = input_embeddings.shape[-1]
        scale = xp.sqrt(d_k)  # Retirer EPS
        scores = self.query @ self.key.transpose(0, 2, 1) / scale

        # Apply softmax to the scores
        seq_len = input_embeddings.shape[1]
        mask = xp.tril(xp.ones((seq_len, seq_len), dtype=xp.bool_))[None, :, :]
        scores = xp.where(mask, scores, -xp.inf)

        # Softmax with numerical stability
        max_scores = xp.max(scores, axis=-1, keepdims=True)
        scores = scores - max_scores
        exp_scores = xp.exp(scores)
        sum_exp = xp.sum(exp_scores, axis=-1, keepdims=True) + 1e-9
        self.attention_weights = exp_scores / sum_exp

        self.output = self.attention_weights @ self.value
        return self.output
    

    def backward(self, d_output) -> xp.ndarray:
        B, T, D = d_output.shape
        sqrt_d = xp.sqrt(D)

        # Gradient with respect to value
        d_value = self.attention_weights.transpose(0, 2, 1) @ d_output  # (B, T, D)

        # Gradient with respect to attention_weights
        d_weights = d_output @ self.value.transpose(0, 2, 1)  # (B, T, T)

        # Backpropagation through softmax (stabilized)
        softmax = self.attention_weights
        d_softmax = d_weights  # (B, T, T)
        temp = (d_softmax * softmax).sum(axis=-1, keepdims=True)  # (B, T, 1)
        d_scores = softmax * (d_softmax - temp)  # (B, T, T)

        # Gradient with respect to query and key
        d_query = d_scores @ self.key / sqrt_d
        d_key = d_scores.transpose(0, 2, 1) @ self.query / sqrt_d

        # Gradient with respect to weights
        self.query_matrix.gradient += self.input_embeddings.reshape(B*T, D).T @ d_query.reshape(B*T, D)
        self.key_matrix.gradient += self.input_embeddings.reshape(B*T, D).T @ d_key.reshape(B*T, D)
        self.value_matrix.gradient += self.input_embeddings.reshape(B*T, D).T @ d_value.reshape(B*T, D)

        # Gradient with respect to input embeddings
        d_input_q = d_query @ self.query_matrix.value.T
        d_input_k = d_key @ self.key_matrix.value.T
        d_input_v = d_value @ self.value_matrix.value.T

        return d_input_q + d_input_k + d_input_v
    

    def step(self, lr : float):
        """ Update the parameters of the attention mechanism with gradient descent. """

        self.query_matrix.step(lr)
        self.value_matrix.step(lr)
        self.key_matrix.step(lr)


    def zero_grad(self):
        """ Reset the gradients of the attention parameters. """

        self.query_matrix.zero_grad()
        self.value_matrix.zero_grad()
        self.key_matrix.zero_grad()
        self.attention_weights = None
        self.output = None
        self.input_embeddings = None
        self.query = None
        self.key = None
        self.value = None


    def get_parameters(self):
        """ Return the parameters of the attention mechanism for saving. """

        return AttentionParams(
            query_matrix=self.query_matrix.value,
            key_matrix=self.key_matrix.value,
            value_matrix=self.value_matrix.value
        )
    

class MultiHeadAttention:

    attention_heads : list[SelfAttention]
    concat : xp.ndarray
    w_out : Param
    b_out : Param

    def __init__(self):
        
        self.attention_heads = [SelfAttention() for _ in range(NB_ATTENTION_HEADS)]

        self.w_out = Param(xp.random.randn(NB_ATTENTION_HEADS * EMBEDDING_DIM, EMBEDDING_DIM) * 0.01)
        self.b_out = Param(xp.zeros(EMBEDDING_DIM))

    
    @classmethod
    def from_params(cls, params: MultiHeadAttentionParams) -> 'MultiHeadAttention':
        """ Create an Attention instance from saved parameters. """

        instance = cls()
        instance.attention_heads = [SelfAttention.from_params(param) for param in params.attention_heads]
        instance.b_out.value = params.b_out
        instance.w_out.value = params.w_out
        return instance
    

    def forward(self, inputs):
        heads_out = [head.add_attention(inputs) for head in self.attention_heads]
        self.concat = xp.concatenate(heads_out, axis=-1)
        output = self.concat @ self.w_out.value + self.b_out.value
        return output
    

    def backward(self, grad_out : xp.ndarray):

        B, T, D_concat = self.concat.shape
        _, _, D_out = grad_out.shape
        self.w_out.gradient += self.concat.reshape(B*T, D_concat).T @ grad_out.reshape(B*T, D_out)
        self.b_out.gradient += grad_out.sum(axis=(0,1))

        grad_concat = grad_out @ self.w_out.value.T
        split_grads = xp.split(grad_concat, NB_ATTENTION_HEADS, axis=-1)
        grad_x_heads = [h.backward(g) for h, g in zip(self.attention_heads, split_grads)]
        grad_x = sum(grad_x_heads)
        return grad_x
    

    def step(self, lr : int):
        for head in self.attention_heads:
            head.step(lr)
        self.w_out.step(lr)
        self.b_out.step(lr)


    def zero_grad(self):
        for head in self.attention_heads:
            head.zero_grad()
        self.w_out.zero_grad()
        self.b_out.zero_grad()

    
    def get_params(self) -> MultiHeadAttentionParams:
        return MultiHeadAttentionParams(
            w_out=self.w_out.value,
            b_out=self.b_out.value,
            attention_heads=[head.get_parameters() for head in self.attention_heads]
        )