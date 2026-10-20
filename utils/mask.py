import torch
from utils.vocab import PAD_IDX

# 两个函数：src_mask 和 trg_mask，分别表示：源句子的遮罩、目标句子的遮罩
def make_src_mask(src):
    """src: (B, S) id 张量 → (B, 1, 1, S) bool，True=允许关注"""
    mask = src != PAD_IDX                      # 参照行：真词判定
    return mask.unsqueeze(1).unsqueeze(2)      # 参照行：垫成 4 维


def make_trg_mask(trg):
    """trg: (B, S) id 张量 → (B, 1, S, S) bool：pad 与未来都要挡"""
    # 空 1：pad 挡板——仿照 make_src_mask 的两行做法，但先不 unsqueeze，
    #        停留在 (B, S)，起个名字 trg_pad_mask
    trg_pad_mask = trg != PAD_IDX
    # 空 2：未来挡板——用 torch.ones(S, S, dtype=torch.bool) 造全 True 方阵，
    #        再 torch.tril 取下三角，S = trg.shape[1]
    S = trg.shape[1]
    mask_2 = torch.tril(torch.ones(S, S, dtype=torch.bool, device=trg.device))
    # 空 3：合并——trg_pad_mask 先 unsqueeze(1).unsqueeze(2)，再 & 未来挡板
    #        （想想谁广播谁：4 维 & 2 维，结果什么形状？）
    trg_mask = trg_pad_mask.unsqueeze(1).unsqueeze(2) & mask_2
    return trg_mask
