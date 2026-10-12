import torch
from torch.utils.data import Dataset

from utils.vocab import tokenize, BOS_IDX, EOS_IDX


class Multi30kDataset(Dataset):
    """Multi30k 平行语料：字符串世界的文件 → id 世界的张量对。"""

    def __init__(self, src_path, trg_path, src_vocab, trg_vocab):
        # TODO 1：读两个文件 → 两组"分好词的句子列表"
        with open(src_path, encoding="utf-8") as f:
            src_sentences = [tokenize(line.strip()) for line in f]
        with open(trg_path, encoding="utf-8") as f:
            trg_sentences = [tokenize(line.strip()) for line in f]


        # TODO 2：把两列句子配成一对一对
        # TODO 3：逐对 encode —— src 直接 encode；trg 首尾加 BOS/EOS 再 encode
        # TODO 4：把每一对 (src_ids, trg_ids) 存进 self.data
        self.data = [] #data变量存储的是src和trg的id对
        for src_tokens, trg_tokens in zip(src_sentences, trg_sentences):
            src_ids = src_vocab.encode(src_tokens)
            trg_ids = [BOS_IDX] + trg_vocab.encode(trg_tokens) + [EOS_IDX]
            self.data.append((src_ids, trg_ids))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 解包data
        src_ids, trg_ids = self.data[idx]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(trg_ids, dtype=torch.long)
        
        
