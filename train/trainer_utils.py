import gc
import math
import random
from logging import Logger

import numpy as np
import torch
import torch.distributed as dist
import os

from transformers import AutoTokenizer

from model import MinimindForCausalLM

def Logger(content):
    if is_main_process():
        print(content)


def get_lr(current_step,total_steps,lr):
    return (lr/10)+0.5*lr*(1+math.cos(math.pi*(current_step/total_steps)))

def init_distributed_mode():
    if int(os.environ.get('RANK',-1))==-1:
        return 0
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def setup_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def lm_checkpoint(lm_config,weight='full_sft',model=None,optimizer=None,epoch=0,
                  step=0,wandb=None,save_dir='../checkpoints',**kwargs):
    os.makedirs(save_dir,exist_ok=True)
    moe_path='_moe' if lm_config.use_moe else ''
    ckpt_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    resume_path=f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'
    if model is not None:
        from torch.nn.parallel import DistributedDataParallel
        state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
        state_dict={k:v.half().cpu() for k,v in state_dict.items()}
        ckp_tmp=ckpt_path+'.tmp'
        torch.save(state_dict,ckp_tmp)
        os.replace(ckp_tmp,ckpt_path)
        wandb_id=None
        if wandb:
            # 下面这个地方没有逻辑，完全是有些wandb的版本有 getrun这个属性有些没有
            if hasattr(wandb,'get_run'):
                run=wandb.get_run()
                wandb_id=getattr(run,'id',None) if run else None
            else:
                wandb_id=getattr(wandb,'id',None)
        resume_data={
            'model':state_dict,
            'optimizer':optimizer.state_dict(),
            'epoch':epoch,
            'step':step,
            'world_size':dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id':wandb_id
        }
        # 下面这个是还原数据
        for key,value in kwargs.items():
            if value is not None:
                if hasattr(value,'state_dict'):
                    if isinstance(value,DistributedDataParallel):
                        resume_data[key]=value.state_dict()
                    else:
                        resume_data[key]=value.state_dict()
                else:
                    resume_data[key]=value
        resume_tmp=resume_path+'.tmp'
        torch.save(resume_data,resume_tmp)
        os.replace(resume_tmp,resume_path)
        del state_dict,resume_data
        gc.collect()
        torch.cuda.empty_cache()
    # 加载模式
    else:
        if os.path.exists(resume_path):
            ckp_data=torch.load(resume_path,map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def init_model(lm_config, from_weight='pretrain', tokenizer_path='/content/drive/MyDrive/minimind', save_dir='../out', device='cuda'):
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_path)
    model=MinimindForCausalLM(lm_config)

    if from_weight !='none':
        moe_suffix='_moe' if lm_config.use_moe else ''
        weight_path=f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights=torch.load(weight_path,map_location=device)
        model.load_state_dict(weights,strict=False)
    Logger(f'所加载Model可训练参数：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model.to(device), tokenizer

class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

