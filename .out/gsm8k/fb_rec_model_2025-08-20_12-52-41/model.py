import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    def __init__(self, ndim, weight=False, bias=False):
        super().__init__()
        self.ndim = ndim
        self.weight = nn.Parameter(torch.ones(ndim)) if weight else None
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, [self.ndim], self.weight, self.bias, 1e-5)

class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embeddings (RoPE) module.
    Applies rotational transformations to query and key vectors.
    """
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        # Using register_buffer for parameters that should be part of the model's state but not trained.
        # persistent=False means they won't be saved in the state_dict, as they are re-computed.
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)

    def _build_cache(self, seq_len, device, dtype):
        # Check if cache is already built and sufficient
        if self.cos_cached is not None and seq_len <= self.cos_cached.shape[0] and self.cos_cached.device == device and self.cos_cached.dtype == dtype:
            return
        
        self.max_seq_len = max(self.max_seq_len, seq_len)
        # Inverse frequency term for sinusoidal embeddings
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim))
        # Time steps
        t = torch.arange(self.max_seq_len, device=device, dtype=torch.float32)
        # Frequencies for each position
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        # Concatenate for both halves of the embedding dimension
        emb = torch.cat((freqs, freqs), dim=-1)
        # Cache cosine and sine values
        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)
        
    # Function to apply rotation to the last dimension
    def rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
        
    def rotate_qk(self, q, k, T):
        cos = self.cos_cached[:T]
        sin = self.sin_cached[:T]
        q = (q * cos) + (self.rotate_half(q) * sin)
        k = (k * cos) + (self.rotate_half(k) * sin)
        return q, k

class CausalSelfAttention(nn.Module):
    def __init__(self, model_config, rotary_emb=None):
        super().__init__()
        self.model_config = model_config
        assert model_config.n_embd % model_config.n_head == 0
        #self.c_attn = nn.Linear(model_config.n_embd, 3 * model_config.n_embd, bias=False)
        self.q_attn = nn.Linear(model_config.n_embd, model_config.n_embd, bias=False)
        self.k_attn = nn.Linear(model_config.n_embd, model_config.n_embd, bias=False)
        self.v_attn = nn.Linear(model_config.n_embd, model_config.n_embd, bias=False)
        self.c_proj = nn.Linear(model_config.n_embd, model_config.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(model_config.dropout)
        self.resid_dropout = nn.Dropout(model_config.dropout)
        self.n_head = model_config.n_head
        self.n_embd = model_config.n_embd
        self.dropout = model_config.dropout
        self.rotary_emb = rotary_emb

    def forward(self, x, qkvo_delta_weights=None):
        B, T, C = x.size()
        if qkvo_delta_weights is None:
            #q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
            q, k, v = self.q_attn(x), self.k_attn(x), self.v_attn(x)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            
            if self.rotary_emb is not None:
                q, k = self.rotary_emb.rotate_qk(q, k, T)

            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=False)
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            y = self.resid_dropout(self.c_proj(y))
            return y
        else:
            #qkvo_delta_weights = qkvo_delta_weights
            delta_q_A, delta_q_B, delta_k_A, delta_k_B, delta_v_A, delta_v_B, delta_o_A, delta_o_B = qkvo_delta_weights.split(self.model_config.lora_rank, dim=1)
            assert delta_q_A.shape == delta_o_B.shape, f"All tensors should have the same shape"
            
            # manually performing linear map so we can add the delta weights
            # Performs (A:(rank, emb), B:(rank, emb)) -> B.T @ A:(emb, rank) @ (rank, emb) = (emb, emb)
            #q = F.linear(x, self.q_attn.weight + delta_q_B.T @ delta_q_A)
            #k = F.linear(x, self.k_attn.weight + delta_k_B.mTT @ delta_k_A)
            #v = F.linear(x, self.v_attn.weight + delta_v_B.T @ delta_v_A)
            q = torch.bmm(x, (self.q_attn.weight + delta_q_B.mT @ delta_q_A).mT)
            k = torch.bmm(x, (self.k_attn.weight + delta_k_B.mT @ delta_k_A).mT)
            v = torch.bmm(x, (self.v_attn.weight + delta_v_B.mT @ delta_v_A).mT)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            
            if self.rotary_emb is not None:
                q, k = self.rotary_emb.rotate_qk(q, k, T)

            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=False)
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            # manually performing linear map so we can add the delta weights
            #o = F.linear(y, self.c_proj.weight + delta_o_B.T @ delta_o_A)
            o = torch.bmm(y, (self.c_proj.weight + delta_o_B.mT @ delta_o_A).mT)
            y = self.resid_dropout(o)
            return y
            
class MLP(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.c_fc = nn.Linear(model_config.n_embd, 4 * model_config.n_embd, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * model_config.n_embd, model_config.n_embd, bias=False)
        self.dropout = nn.Dropout(model_config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
class   FeedbackwardBlock(nn.Module):
    def __init__(self, model_config, rotary_emb=None):
        super().__init__()
        self.model_config = model_config
        self.ln_1 = LayerNorm(model_config.n_embd)
        self.attn_1 = CausalSelfAttention(model_config, rotary_emb=rotary_emb)
        self.ln_2 = LayerNorm(model_config.n_embd)
        self.mlp_1 = MLP(model_config)
        
        self.h_qkvo = nn.Parameter(torch.normal(mean=0.0, std=0.02, size=(1, 2 * 4 * model_config.lora_rank, model_config.n_embd)))
        self.ln_3 = LayerNorm(model_config.n_embd)
        self.attn_2 = CausalSelfAttention(model_config, rotary_emb=rotary_emb)
        self.ln_4 = LayerNorm(model_config.n_embd)
        self.mlp_2 = MLP(model_config)
        
        
    def forward(self, x, qkvo_delta_weights):
        B, T, E = x.size()
        
        x = x + self.attn_1(self.ln_1(x))
        x = x + self.mlp_1(self.ln_2(x))
        
        # concat along sequence length. Assumes the same batch size and emb dim
        qkvo_context = torch.cat((x, self.h_qkvo.repeat(B, 1, 1)), dim=1)
        qkvo_context = qkvo_context + self.attn_2(self.ln_3(qkvo_context))
        qkvo_context = qkvo_context + self.mlp_2(self.ln_4(qkvo_context))
        #qkvo_delta_weights = torch.split(qkvo_context[:, -2 * 4 * self.model_config.lora_rank:, :], 2 * 4, dim=1)
        
        qkvo_delta_weights.copy_(qkvo_context[:, -2 * 4 * self.model_config.lora_rank:, :])
        
        return x
        

class FeedforwardBlock(nn.Module):
    def __init__(self, model_config, rotary_emb=None):
        super().__init__()
        self.ln_1 = LayerNorm(model_config.n_embd)
        self.attn = CausalSelfAttention(model_config, rotary_emb=rotary_emb)
        self.ln_2 = LayerNorm(model_config.n_embd)
        self.mlp = MLP(model_config)

    def forward(self, x, qkvo_delta_weights=None):
        x = x + self.attn(self.ln_1(x), qkvo_delta_weights)
        x = x + self.mlp(self.ln_2(x))
        return x
    
class FeedbackwardRecurrentModel(nn.Module):
    """
    A transformer based model to generate weights for a feedforward base model that is recurrent in the block.
    """
    def __init__(self, model_config, dataset_config):
        super().__init__()
        self.model_config = model_config
        self.dataset_config = dataset_config
        
        head_dim = model_config.n_embd // model_config.n_head
        rotary_emb = RotaryEmbedding(dim=head_dim, max_seq_len=dataset_config.max_seq_length)
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(dataset_config.vocab_size, model_config.n_embd),
            drop = nn.Dropout(model_config.dropout),
            h = FeedbackwardBlock(model_config, rotary_emb=rotary_emb),
            le = nn.Embedding(model_config.n_layer, model_config.n_embd)
        ))
        
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.model_config.n_layer))
                
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, input_ids):
        """
        Forward pass for the masked language model.
        Args:
            input_ids (torch.Tensor): Token indices of shape (B, T).
        Returns:
            qkvo_blocks_delta_weights (List[Tuple[2 * 4 * torch.Tensor]]): List of q(A, B), k(A, B), v(A, B), o(A, B) weights for each block.
                Has length 8 even though q, k, v, and o are 4 elements is due to being passed the lora pairs per weight.
                Must be ordered so that the last element in the list is for this first block
        """
        assert input_ids.dtype == torch.long, "Input must be token indices"
        tok_emb = self.transformer.wte(input_ids)
        
        B, T, C = tok_emb.size()
        assert T <= self.dataset_config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.dataset_config.block_size}"
        
        # The rotary_emb module is shared across all blocks. We can pre-build the cache once.
        # The first block's attention layer holds the rotary_emb instance.
        self.transformer.h.attn_1.rotary_emb._build_cache(T, tok_emb.device, tok_emb.dtype)

        x = self.transformer.drop(tok_emb)
        
        feedback_reverse = self.model_config.feedback_reverse
        assert isinstance(feedback_reverse, bool), f"feedback_reverse must be a boolean, got {feedback_reverse} instead."
        
        lora_weight_tokens_length = 2 * 4 * self.model_config.lora_rank
        qkvo_blocks_delta_weights = torch.zeros((B, lora_weight_tokens_length * self.model_config.n_layer, C), device=x.device, dtype=x.dtype)
        for index in range(self.model_config.n_layer):
            if feedback_reverse:
                # populate the blocks from last to first corresponding to feedback starting from the last going back to the first
                reverse_index = self.model_config.n_layer - index - 1
                qkvo_delta_weights = qkvo_blocks_delta_weights[
                    :,
                    lora_weight_tokens_length * reverse_index : lora_weight_tokens_length * (reverse_index + 1),
                    :
                ]
                layer_emb = self.transformer.le(torch.tensor([reverse_index], device=x.device, dtype=torch.long))
            else:
                qkvo_delta_weights = qkvo_blocks_delta_weights[
                    :,
                    lora_weight_tokens_length * index : lora_weight_tokens_length * (index + 1),
                    :
                ]
                layer_emb = self.transformer.le(torch.tensor([index], device=x.device, dtype=torch.long))
                
            block = self.transformer.h
            layer_embedding_expanded = layer_emb.expand(B, T, -1)
            x = block(x + layer_embedding_expanded, qkvo_delta_weights)
        return qkvo_blocks_delta_weights
    
class FeedbackwardModel(nn.Module):
    """
    A transformer based model to generate weights for a feedforward base model that is not recurrent and each block is distinct.
    """
    def __init__(self, model_config, dataset_config):
        super().__init__()
        self.model_config = model_config
        self.dataset_config = dataset_config
        
        head_dim = model_config.n_embd // model_config.n_head
        rotary_emb = RotaryEmbedding(dim=head_dim, max_seq_len=dataset_config.max_seq_length)
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(dataset_config.vocab_size, model_config.n_embd),
            drop = nn.Dropout(model_config.dropout),
            h = nn.ModuleList([FeedbackwardBlock(model_config, rotary_emb=rotary_emb) for _ in range(model_config.n_layer)])
        ))
        
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.model_config.n_layer))
                
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, input_ids):
        """
        Forward pass for the masked language model.
        Args:
            input_ids (torch.Tensor): Token indices of shape (B, T).
        Returns:
            qkvo_blocks_delta_weights (List[Tuple[2 * 4 * torch.Tensor]]): List of q(A, B), k(A, B), v(A, B), o(A, B) weights for each block.
                Has length 8 even though q, k, v, and o are 4 elements is due to being passed the lora pairs per weight.
                Must be ordered so that the last element in the list is for this first block
        """
        assert input_ids.dtype == torch.long, "Input must be token indices"
        tok_emb = self.transformer.wte(input_ids)
        
        B, T, C = tok_emb.size()
        assert T <= self.dataset_config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.dataset_config.block_size}"
        
        # The rotary_emb module is shared across all blocks. We can pre-build the cache once.
        # The first block's attention layer holds the rotary_emb instance.
        self.transformer.h[0].attn_1.rotary_emb._build_cache(T, tok_emb.device, tok_emb.dtype)

        x = self.transformer.drop(tok_emb)
        
        feedback_reverse = self.model_config.feedback_reverse
        assert isinstance(feedback_reverse, bool), f"feedback_reverse must be a boolean, got {feedback_reverse} instead."
        
        lora_weight_tokens_length = 2 * 4 * self.model_config.lora_rank
        qkvo_blocks_delta_weights = torch.zeros((B, lora_weight_tokens_length * self.model_config.n_layer, C), device=x.device, dtype=x.dtype)
        for index, block in enumerate(self.transformer.h):
            if feedback_reverse:
                # populate the blocks from last to first corresponding to feedback starting from the last going back to the first
                reverse_index = self.model_config.n_layer - index - 1
                qkvo_delta_weights = qkvo_blocks_delta_weights[
                    :,
                    lora_weight_tokens_length * reverse_index : lora_weight_tokens_length * (reverse_index + 1),
                    :
                ]
                
            else:
                qkvo_delta_weights = qkvo_blocks_delta_weights[
                    :,
                    lora_weight_tokens_length * index : lora_weight_tokens_length * (index + 1),
                    :
                ]
            x = block(x, qkvo_delta_weights)
        return qkvo_blocks_delta_weights
        

class FeedforwardModel(nn.Module):
    """
    A Transformer-based model for LLaDA style Diffusion.
    It predicts the clean data x_0 from a noisy input x_t.
    """
    def __init__(self, model_config, dataset_config):
        super().__init__()
        self.model_config = model_config
        self.dataset_config = dataset_config

        head_dim = model_config.n_embd // model_config.n_head
        rotary_emb = RotaryEmbedding(dim=head_dim, max_seq_len=dataset_config.max_seq_length)

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(dataset_config.vocab_size, model_config.n_embd),
            drop = nn.Dropout(model_config.dropout),
            h = nn.ModuleList([FeedforwardBlock(model_config, rotary_emb=rotary_emb) for _ in range(model_config.n_layer)]),
            ln_f = LayerNorm(model_config.n_embd),
        ))
        self.lm_head = nn.Linear(model_config.n_embd, dataset_config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # Weight tying

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.model_config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def qkvo_blocks_grads(self):
        """
        Get all q, k, v, o gradients from each block as a single tensor that is detached.
        Returns:
            list (List[Tuple[4 * torch.Tensor]]): List of gradients for each block's q, k, v, o weights.
        """
        first_q_block = self.transformer.h[0].attn.q_attn.weight.grad
        T, C = first_q_block.size()
        ret = torch.zeros(
            (1, T * 4 * len(self.transformer.h), C), 
            device=first_q_block.device, dtype=first_q_block.dtype, requires_grad=False
        )
        for index, block in enumerate(self.transformer.h):
            ret[0, (0 + 4 * index) * T : (1 + 4 * index) * T, :] = block.attn.q_attn.weight.grad
            ret[0, (1 + 4 * index) * T : (2 + 4 * index) * T, :] = block.attn.k_attn.weight.grad
            ret[0, (2 + 4 * index) * T : (3 + 4 * index) * T, :] = block.attn.v_attn.weight.grad
            ret[0, (3 + 4 * index) * T : (4 + 4 * index) * T, :] = block.attn.c_proj.weight.grad # c_proj corresponds to o (out projection) in attention
        return ret


    def forward(self, input_ids, qkvo_blocks_delta_weights=None):
        """
        Forward pass for the masked language model.
        Args:
            input_ids (torch.Tensor): Token indices of shape (B, T).
            qkvo_blocks_delta_weights (List[Tuple[2 * 4 * torch.Tensor]]): List of q(A, B), k(A, B), v(A, B), o(A, B) weights for each block.
                Has length 8 even though q, k, v, and o are 4 elements is due to being passed the lora pairs per weight.
                Must be ordered so that the last element in the list is for this first block
        Returns:
            logits (torch.Tensor): Predicted logits for the original tokens. Shape (B, T, vocab_size).
        """
        assert input_ids.dtype == torch.long, "Input must be token indices"
        tok_emb = self.transformer.wte(input_ids)
        
        B, T, C = tok_emb.size()
        assert T <= self.dataset_config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.dataset_config.block_size}"
        
        # The rotary_emb module is shared across all blocks. We can pre-build the cache once.
        # The first block's attention layer holds the rotary_emb instance.
        self.transformer.h[0].attn.rotary_emb._build_cache(T, tok_emb.device, tok_emb.dtype)

        x = self.transformer.drop(tok_emb)
        
        if not qkvo_blocks_delta_weights is None:
            lora_weight_tokens_length = 2 * 4 * self.model_config.lora_rank
            
        for index, block in enumerate(self.transformer.h):
            if not qkvo_blocks_delta_weights is None:
                qkvo_delta_weights = qkvo_blocks_delta_weights[
                    :,
                    lora_weight_tokens_length * index : lora_weight_tokens_length * (index + 1),
                    :
                ]
                x = block(x, qkvo_delta_weights)
            else:
                x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits
