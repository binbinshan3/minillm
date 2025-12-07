import json

import torch
from torch.utils.data import Dataset, DataLoader


class PretrainDataset(Dataset):
    def __init__(self,data_path,tokenizer,max_length=512):
        super().__init__()
        self.data_path=data_path
        self.tokenizer=tokenizer
        self.max_length=max_length
        self.samples=self.loaddata(self.data_path)
    def loaddata(self,path):
        samples=[]
        with open(path,encoding='utf-8') as f:
            for line_num,line in enumerate(f):
                data=json.loads(line)
                samples.append(data)
        return samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        sample=self.samples[idx]
        encoding=self.tokenizer(str(sample['text']),
                                max_length=self.max_length,
                                padding='max_length',
                                truncation=True,
                                return_tensors='pt'
                                )
        input_ids=encoding.input_ids.squeeze()
        loss_mask=(input_ids !=self.tokenizer.pad_token_id)
        X = input_ids[:-1].clone().long()  # 若input_ids已是long，可省略.long()
        Y = input_ids[1:].clone().long()
        loss_mask = loss_mask[1:].long()
        return X,Y,loss_mask
#
#
#
# import mmap
#
#
# class PretrainDataset(Dataset):
#     def __init__(self, data_path, tokenizer, max_length=512):
#         super().__init__()
#         self.data_path = data_path
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         print("正在构建文件索引...")
#         self.offsets = [0]
#         with open(data_path, "r", encoding="utf-8") as f:
#             while f.readline():
#                 self.offsets.append(f.tell())
#         self.offsets.pop()
#         print(f"索引完成，共 {len(self.offsets)} 条。")
#
#     def __len__(self):
#         return len(self.offsets)
#
#     def __getitem__(self, idx):
#         offset = self.offsets[idx]
#         with open(self.data_path, "r", encoding="utf-8") as f:
#             f.seek(offset)
#             line = f.readline()
#
#         try:
#             data = json.loads(line)
#             text = data['text']
#         except Exception:
#             return self.__getitem__(idx + 1 if idx + 1 < len(self) else 0)
#
#         encoding = self.tokenizer(
#             str(text),
#             max_length=self.max_length,
#             padding='max_length',
#             truncation=True,
#             return_tensors='pt'
#         )
#         input_ids = encoding.input_ids.squeeze()
#
#         # === 核心逻辑修改 ===
#         pad_id = self.tokenizer.pad_token_id
#
#         # 1. 切分 X 和 Y
#         X = input_ids[:-1]
#         Y = input_ids[1:].clone()  # 必须 clone，防止后续修改 Y 影响 X
#
#         # 2. 生成真实的 loss_mask (维度与 Y 一致)
#         # 如果 Y 不等于 pad_id，则 mask 为 1，否则为 0
#         loss_mask = (Y != pad_id).long()
#
#         # 3. 将 Y 中的 Pad 设置为 -100 (Loss函数需要)
#         Y[Y == pad_id] = -100
#
#         # 4. 返回完整的三样东西，保持 API 不变
#         return X, Y, loss_mask
