"""分词与词表：字符串世界和整数世界的双向翻译官。

模型的数学世界只认整数 id；人类的数据世界是字符串。
本模块负责两个世界之间的全部转换，模型代码永远不直接碰字符串。

组成：
- tokenize(text)  : 一句话 -> token 列表（小写化 + 标点独立成 token）
- Vocab 类        : token <-> id 双向映射，支持构建 / 编码 / 解码 / 存取
"""

import json
import re
from collections import Counter

# 四个特殊 token 的 id 是全局约定（固定 0~3），全项目任何地方都不许变。
# pad=0 的约定是为了配合 nn.Embedding 的 padding_idx 参数（阶段 3 见）。
PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]

# 分词正则：两种东西都抓——
#   \w+      连续的文字块（字母/数字/下划线，Python3 默认 Unicode，ü ö ß ä 都算）
#   [^\w\s]  单个"既非文字又非空白"的字符，即标点，独立成 token
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")


def tokenize(text: str) -> list:
    """一句话 -> token 列表。词级分词的全部实现就这一行正则。"""
    return TOKEN_PATTERN.findall(text.lower())


class Vocab:
    """词表：token <-> id 的双向字典。

    id 分配规则：0~3 永远是特殊 token；之后按语料词频降序编号。
    """

    def __init__(self):
        # 正向表：查词得 id（编码用）；反向表：查 id 得词（解码用）
        self.token2id = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
        self.id2token = {idx: tok for tok, idx in self.token2id.items()}

    def build(self, sentences, min_freq: int = 2):
        """从分好词的语料（句子的列表）构建词表。

        min_freq=2：只出现 1 次的词不收入词表（推理时它们会变成 <unk>）。
        理由：29000 句的小语料里，singleton 几乎都是噪声或专名，
        为它们各开一个词表席位、各养一份 512 维向量，不值。

        排序规则：频次降序；同频按字母序——固定规则保证每次构建结果可复现。
        """
        counter = Counter(tok for sent in sentences for tok in sent)
        words = sorted(
            (w for w, c in counter.items() if c >= min_freq),
            key=lambda w: (-counter[w], w),
        )
        for word in words:
            self.token2id[word] = len(self.token2id)
            self.id2token[len(self.id2token)] = word
        return self

    def encode(self, tokens) -> list:
        """token 列表 -> id 列表。查不到的一律 <unk>（模型承认不认识，但不崩溃）。"""
        unk = self.token2id["<unk>"]
        return [self.token2id.get(tok, unk) for tok in tokens]

    def decode(self, ids) -> list:
        """id 列表 -> token 列表。阶段 5 推理时把模型输出翻回人类语言。"""
        return [self.id2token.get(i, "<unk>") for i in ids]

    def __len__(self) -> int:
        return len(self.token2id)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token2id, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str):
        vocab = cls()
        with open(path, encoding="utf-8") as f:
            vocab.token2id = json.load(f)
        vocab.id2token = {idx: tok for tok, idx in vocab.token2id.items()}
        return vocab
