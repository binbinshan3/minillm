import math
from typing import Optional, Tuple, List, Union

import torch
from torch import nn
from torch.xpu import device
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast


class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
            self,
            dropout: float = 0.0,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            hidden_act: str = 'silu',
            hidden_size: int = 512,
            intermediate_size: int = None,
            max_position_embeddings: int = 32768,
            num_attention_heads: int = 8,
            num_hidden_layers: int = 8,
            num_key_value_heads: int = 2,
            vocab_size: int = 6400,
            rms_norm_eps: float = 1e-05,
            rope_theta: int = 1000000.0,
            inference_rope_scaling: bool = False,
            flash_attn: bool = True,
            ####################################################
            # Here are the specific configurations of MOE
            # When use_moe is false, the following is invalid
            ####################################################
            use_moe: bool = False,
            num_experts_per_tok: int = 2,
            n_routed_experts: int = 4,
            n_shared_experts: int = 1,
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.1,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        # 外推长度 = factor * original_max_position_embeddings = 32768
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        self.flash_attn = flash_attn
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率

class RMSNorm(nn.Module):
    def __init__(self,dim:int,eps:float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.Tensor(dim))
    def _norm(self,x):
        return x*torch.rsqt(x.pow(2).mean(-1,keepdim=True)+self.eps)
    def forward(self,x):
        # 归一化先转精度再转回来
        return self.weight*self._norm(x.float()).type_as(x)


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6,
                         rope_scaling: Optional[dict] = None):
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


def repeat_kv(x:torch.Tensor,n_rep:int):
    bsz,seqlen,num_kvheads,dim=x.shape
    if n_rep==1:
        return x
    else:
        return x.unsqueeze(3).expand(bsz,seqlen,num_kvheads,n_rep,dim).view(bsz,seqlen,-1,dim)


class Attention(nn.Module):
    def __init__(self, args:MiniMindConfig):
        super().__init__()
        self.args=args
        self.num_key_value_heads=args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        assert args.num_attention_heads % self.num_key_value_heads == 0
        self.n_local_heads=args.num_attention_heads
        self.n_local_kv_heads=args.num_key_value_heads
        self.n_rep=self.n_local_heads//self.n_local_kv_heads
        self.head_dim=args.hidden_size//self.num_attention_heads
        self.qw=nn.Linear(args.hidden_size,self.n_local_heads*self.head_dim,bias=False)
        self.kw=nn.Linear(args.hidden_size,self.n_local_kv_heads*self.head_dim,bias=False)
        self.vw=nn.Linear(args.hidden_size,self.n_local_kv_heads*self.head_dim,bias=False)
        self.ow=nn.Linear(self.n_local_heads*self.head_dim,args.hidden_size,bias=False)
        self.att_dropout=nn.Dropout(args.dropout)
        self.res_dropout=nn.Dropout(args.dropout)
        self.dropout=args.dropout
        # args.flash是bool值
        self.flash=hasattr(torch.nn.functional,'scaled_dot_product_attention') and args.flash
    def forward(self,
                x:torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],
                # 这个地方是一个加速的kv缓存机制
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: Optional[bool] = False,
                attn_mask: Optional[torch.Tensor] = None
                ):
        bsz,seqlen,_=x.shape
        xq=self.qw(x)
        xk=self.kw(x)
        xv=self.vw(x)
        xq=xq.view(bsz,seqlen,self.n_local_heads,self.head_dim)
        xk=xk.view(bsz,seqlen,self.n_local_kv_heads,self.head_dim)
        xv=xv.view(bsz,seqlen,self.n_local_kv_heads,self.head_dim)
        cos,sin=position_embeddings
        xq,xk=apply_rotary_pos_emb(xq,xk,cos[:seqlen],sin[:seqlen])
        if past_key_value is not None:
            xk=torch.cat([past_key_value[0],xk],dim=1)
            xv=torch.cat([past_key_value[1],xv],dim=1)
        past_kv=(xk,xv) if use_cache else None
        xq=xq.transpose(1,2)
        xk=repeat_kv(xk,self.n_rep).transpose(1,2)
        xv=repeat_kv(xv,self.n_rep).transpose(1,2)
        if self.flash and seqlen>1 and(attn_mask is None or torch.all(attn_mask==1)):
            output= F.scaled_dot_product_attention(xq,xk,xv,attn_mask=attn_mask,
                                                  dropout_p=self.dropout if self.training else None,
                                                  is_causal=True)
        else:
            atten_score=torch.matmul(xq,xk.transpose(-2,-1))/math.sqrt(self.head_dim)
            atten_score=atten_score+torch.triu(
                torch.full((seqlen,seqlen),float('-inf'),device=atten_score.device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)
            # 这个地方是对如果有添加额外的掩码，例如pad等情况补充的掩码机制
            if attn_mask is not None:
                extended_attn_mask=attn_mask.unsqueeze(1).unsqueeze(2)
                extended_attn_mask=(1-extended_attn_mask)*-1e9
                atten_score=atten_score+extended_attn_mask
            scores = F.softmax(atten_score,dim=-1)
            scores=self.att_dropout(scores)
            output=torch.matmul(scores,xv)
        output=output.transpose(1,2).contiguous().view(bsz,seqlen,-1)
        output=self.ow(output)
        output=self.res_dropout(output)
        return output,past_kv

class FFN(nn.Module):
    def __init__(self,config:MiniMindConfig):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size=int(config.hidden_size*8/3)
            config.intermediate_size=64*((intermediate_size+63)//64)
        self.fc1=nn.Linear(config.hidden_size,config.intermediate_size,bias=False)
        self.fc2=nn.Linear(config.intermediate_size,config.hidden_size,bias=False)
        self.fc3=nn.Linear(config.hidden_size,config.intermediate_size,bias=False)
        self.dropout=nn.Dropout(config.dropout)
    def forward(self,x):
        return self.dropout(self.fc2(F.silu(self.fc1(x))*self.fc3(x)))

class MinimindBlock(nn.Module):
    def __init__(self,layer_id:int,config:MiniMindConfig):
        super().__init__()
        self.layer_id=layer_id
        self.atten_norm=RMSNorm(config.hidden_size,config.rms_norm_eps)
        self.attention=Attention(config)
        self.FFN_norm=RMSNorm(config.hidden_size,config.rms_norm_eps)
        self.FFN=FFN(config)
    def forward(self,hidden_state,position_embeddings,past_key_value=None,use_cache=False,attn_mask=None):
        residual=hidden_state
        hidden_state,past_key_value=self.attention.forward(self.atten_norm(hidden_state),
                                                           position_embeddings,
                                                           past_key_value,
                                                           use_cache,attn_mask)
        hidden_state=hidden_state+residual
        hidden_state=self.FFN(self.FFN_norm(hidden_state))+hidden_state
        return hidden_state,past_key_value

class MinimindModel(nn.Module):
    def __init__(self,config:MiniMindConfig):
        super().__init__()
        self.config=config
        self.embed=nn.Embedding(config.vocab_size,config.hidden_size)
        self.dropout=nn.Dropout(config.dropout)
        self.layers=nn.ModuleList([MinimindBlock(layer,config) for layer in range(config.layers)])
        self.norm=RMSNorm(config.hidden_size,config.rms_norm_eps)
        freqs_cos,freqs_sin=precompute_freqs_cis(config.hidden_size//config.num_attention_heads,
                                                 end=config.max_position_embeddings,
                                                 rope_base=config.rope_base,
                                                 rope_scaling=config.rope_scaling)
        self.register_buffer('freqs_cos',freqs_cos)
        self.register_buffer('freqs_sin',freqs_sin)

    def forward(self,
                input_ids:Optional[torch.Tensor]=None,
                attention_mask:Optional[torch.Tensor]=None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]=None,
                use_cache=False,
                **kwargs) :
        bsz, seqlen=input_ids.shape
        past_key_values=past_key_values or [None]*len(self.layers)
        stat_pos=past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        x=self.embed(input_ids)
        hidden_state=self.dropout(x)
        position_embeddings=(self.freqs_cos[stat_pos:stat_pos+seqlen],self.freqs_sin[stat_pos:stat_pos+seqlen])
        presents=[]
        for layer_idx,(layer,past_key_value) in enumerate(zip(self.layers,past_key_values)):
            hidden_state,past_key_value=layer(hidden_state,position_embeddings,past_key_value,use_cache,attention_mask)
            presents.append(past_key_value)
        hidden_state=self.norm(hidden_state)
        return hidden_state,presents

class MinimindForCausalLM(PreTrainedModel,GenerationMixin):
    config_class=MiniMindConfig
    def __init__(self,config:MiniMindConfig):
        self.config=config or MiniMindConfig()
        super().__init__(self.config)
        self.model=MinimindModel(config)
        self.lm_head=nn.Linear(config.hidden_size,config.vocab_size,bias=False)
        self.model.embed.weight=self.lm_head.weight

    def forward(self,input_ids:Optional[torch.Tensor]=None,
            attention_mask:Optional[torch.Tensor]=None,
            past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]=None,
            use_cache=False,
            logits_to_keep:Union[int,torch.Tensor]=0,
            **args):
        hidden_state,presents=self.model(input_ids,attention_mask,past_key_values,use_cache,**args)
        logit_slice=slice(-logits_to_keep,None) if isinstance(logits_to_keep,int) else logits_to_keep
        logits=self.lm_head(hidden_state[:,logit_slice,:])
        output=CausalLMOutputWithPast(logits=logits,past_key_values=presents,hidden_states=hidden_state)
        return output







