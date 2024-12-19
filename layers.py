from functools import wraps
import torch
from torch import nn, einsum
from torch import Tensor
import torch.nn.functional as F

from typing import Optional, Tuple
from einops import rearrange, repeat
from functools import partial

from timm.models.layers import DropPath, trunc_normal_
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn

# Helpers
def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cache_fn(f):
    cache = dict()

    @wraps(f)
    def cached_fn(*args, _cache=True, key=None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if key in cache:
            return cache[key]
        result = f(*args, **kwargs)
        cache[key] = result
        return result

    return cached_fn


# Helper classes
class PreNorm(nn.Module):
    def __init__(self, channel, fn, context_channel=None):
        super(PreNorm, self).__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(channel)
        self.norm_context = nn.LayerNorm(context_channel) if exists(context_channel) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, channel, mult=4, dropout=0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(channel, channel * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(channel * mult, channel)
        )

    def forward(self, x):
        return self.net(x)
#######################################Cross Attention ###################################################
class Cross_Mamba_Attention(nn.Module):
    def __init__(
            self, query_channel, context_channel=None, output_channel=None,
            heads_num=8, heads_channel=64, dropout=0.,
    mamba_or_transformer='mamba', input_or_output='input', 
    args=None):
        super(Cross_Mamba_Attention, self).__init__()
        inner_channel = heads_channel * heads_num
        context_dim = default(context_channel, query_channel)
        output_dim = default(output_channel, query_channel)

        # print("query_channel :" , query_channel)
        # print("context_dim :" , context_dim)
        # print("output_channel : ", output_channel)
        # assert False
        self.input_or_output = input_or_output
        self.scale = heads_channel ** -0.5
        self.heads = heads_num
        self.args = args

        self.mamba_or_transformer = mamba_or_transformer
        # print("we are using:  ",self.mamba_or_transformer)
        if mamba_or_transformer == 'transformer' or input_or_output=='output':
            self.to_q = nn.Linear(query_channel, query_channel, bias=False)
            self.to_kv = nn.Linear(context_dim, query_channel * 2, bias=False)
            self.dropout = nn.Dropout(dropout)
            self.to_out = nn.Linear(query_channel, output_dim)

        elif mamba_or_transformer == 'mamba' and input_or_output == 'input':
            self.dropout = nn.Dropout(dropout)
            self.to_q = nn.Linear(query_channel, query_channel, bias=False)
            self.to_kv = nn.Linear(context_dim, query_channel*2, bias=False)
            self.mamba_block = create_block(query_channel)
            self.mamba_block_addition = [create_block(query_channel).cuda(),create_block(query_channel).cuda()]
            self.to_out = nn.Linear(query_channel, output_dim)
            
    def forward(self, x, context=None, mask=None):
        if self.mamba_or_transformer == 'mamba' and self.input_or_output=='input':
            # print("x shape: ", x.shape)
            # print("context shape: ", context.shape)
            h = self.heads
            q = self.to_q(x)
            context = default(context, x)
            # print("context.shape : " ,context.shape) 
            k, v = self.to_kv(context).chunk(2, dim=-1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
            
            if q.shape[0]!=k.shape[0]:
                q = q.reshape(k.shape[0], -1, q.shape[-1])

            sim = einsum('b d i, b d j -> b i j', k, v) * self.scale
            
            attn = sim.softmax(dim=-1)
            attn = self.dropout(attn)
            # out_before = q@attn/(q.shape[0])
            out_before = q@attn

            if self.args.data_name == 'elasticity':
                out = self.mamba_block(out_before, out_before)[0]
                # for i in range(2):
                #     out= self.mamba_block(out, out_before)[0] + out_before
                    
            elif self.args.data_name == 'airfoil':
                out = self.mamba_block(out_before, out_before)[0] 
                # for i in range(2):
                #     out= self.mamba_block(out, out_before)[0] + out_before
            elif self.args.data_name == 'plasticity':
                out = self.mamba_block(out_before)[0] + out_before
                # for i in range(2):
                #     out= self.mamba_block(out, out_before)[0] + out_before
            elif self.args.data_name == 'darcyflow':
                out = self.mamba_block(out_before, out_before)[0] 
            elif self.args.data_name == 'darcyflow':
                out = self.mamba_block(out_before, out_before)[0] 
                # for i in range(2):
                #     out= self.mamba_block(out, out_before)[0] + out_before
                

            out = self.to_out(out)
            
            return out
            
        
        elif self.mamba_or_transformer == 'transformer' or self.input_or_output=='output':
            # print("x shape: ", x.shape)
            # print("context shape: ", context.shape)
            h = self.heads
            q = self.to_q(x)
            # print("x : ", x.shape)
            # print("after to_q : q.shape : " , q.shape )

            context = default(context, x)
            # print("after defaullt context shape: ", context.shape)
            
            k, v = self.to_kv(context).chunk(2, dim=-1)
            # print("after to_kv : k.shape : " , k.shape , "v.shape: ", v.shape)
            # assert False
            # print("context : " , context.shape)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
            # print("q.shape : ", q.shape)
            # print("k.shape : ", k.shape)
            # print("v.shape : ", v.shape)
            if q.shape[0]!=k.shape[0]:
                q = q.reshape(k.shape[0], -1, q.shape[-1])
            # print("q.shape : ", q.shape)
            # print("k.shape : ", k.shape)
            # print("v.shape : ", v.shape)
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
    
            if exists(mask):
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                sim.masked_fill_(~mask, max_neg_value)
            
            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            attn = self.dropout(attn)
            # print("attn : ", attn.shape)
            # print("v    : ", v.shape)
            out = einsum('b i j, b j d -> b i d', attn, v)

            # out = einsum('b i j, b j d -> b i d', attn, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
            # print(" out : " , out.shape)
            return self.to_out(out)
            
#######################################################################################################

class Mamba_Attention(nn.Module):
    def __init__(
            self, query_channel, context_channel=None, output_channel=None,
            heads_num=8, heads_channel=64, dropout=0.,
            mamba_or_transformer='transformer'):
        super(Mamba_Attention, self).__init__()
        inner_channel = heads_channel * heads_num
        context_dim = default(context_channel, query_channel)
        output_dim = default(output_channel, query_channel)

        print("query_channel :" , query_channel)
        print("context_dim :" , context_dim)
        self.scale = heads_channel ** -0.5
        self.heads = heads_num

        self.mamba_or_transformer = mamba_or_transformer
        print("we are using:  ",self.mamba_or_transformer)
        if mamba_or_transformer == 'transformer':
            self.to_q = nn.Linear(query_channel, inner_channel, bias=False)
            self.to_kv = nn.Linear(context_dim, inner_channel * 2, bias=False)
            self.dropout = nn.Dropout(dropout)
            self.to_out = nn.Linear(inner_channel, output_dim)
        elif mamba_or_transformer == 'mamba':
            self.mamba_block = create_block(output_dim)
            self.output_dim = output_dim
    
    def forward(self, x, context=None, mask=None):
        if self.mamba_or_transformer == 'mamba':
            # print("x.shape                       : ",x.shape, self.output_dim) 
            # print("context.shape                 : ", context.shape)
            # print("self.mamba_block (x)[0].shape : ", self.mamba_block (x)[0].shape)
            # print("self.mamba_block (x)[1].shape : ", self.mamba_block (x)[1].shape)
            return self.mamba_block(x, x)[0] +x
    
        elif self.mamba_or_transformer == 'transformer':
            h = self.heads
            q = self.to_q(x)
            context = default(context, x)
            k, v = self.to_kv(context).chunk(2, dim=-1)
    
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
    
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
    
            if exists(mask):
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                sim.masked_fill_(~mask, max_neg_value)
    
            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            attn = self.dropout(attn)
    
            out = einsum('b i j, b j d -> b i d', attn, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
            return self.to_out(out)
        
    



class Block(nn.Module):
    def __init__(
            self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Mamba block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)

        # drop path 
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (self.drop_path(hidden_states) + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        drop_path=0.,
        device=None,
        dtype=None, ):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        drop_path=drop_path,
    )
    block.layer_idx = layer_idx
    return block

