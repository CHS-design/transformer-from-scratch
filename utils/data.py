import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from utils.vocab import tokenize, BOS_IDX, EOS_IDX, PAD_IDX


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

def collate_fn(batch):
    """把一批长短不齐的 (src, trg) 对，补 pad 后堆成两个矩形张量。"""

    # ① 拆两摞：batch 的每个元素是元组 (src, trg)，pair[0] 是 src，pair[1] 是 trg
    srcs, trgs = zip(*batch)
    # ② 堆成矩形pad_sequence 专为长短不齐设计
    src_batch = pad_sequence(srcs, batch_first=True, padding_value=PAD_IDX)
    trg_batch = pad_sequence(trgs, batch_first=True, padding_value=PAD_IDX)

    #第二种方法
    # # ② 找到最长的 src 和最长的 trg
    # max_len_src = max(len(src) for src in srcs)
    # max_len_trg = max(len(trg) for trg in trgs)

    # # ③ 逐条补 pad：F.pad(张量, (前面补几个, 后面补几个), value=填充值)
    # #    "后面补几个" = 目标长度 - 当前长度，动态计算
    # padded_srcs = [F.pad(t, (0, max_len_src - t.size(0)), value=PAD_IDX) for t in srcs]
    # padded_trgs = [F.pad(t, (0, max_len_trg - t.size(0)), value=PAD_IDX) for t in trgs]

    # # ④ 堆成矩形：torch.stack 要求列表里所有张量形状相同
    # src_batch = torch.stack(padded_srcs)               # 参照行
    # trg_batch = torch.stack(padded_trgs)               # 参照列

    return src_batch, trg_batch






        
        
