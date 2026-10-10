"""构建英/德两张词表，存入 vocab/ 文件夹。

跑一次即可：python build_vocab.py
之后训练、推理一律从 vocab/ 加载，保证各环节用的是同一张表。
"""

from utils.vocab import Vocab, tokenize

EN_PATH = "data/train.en"
DE_PATH = "data/train.de"
MIN_FREQ = 2


def read_sentences(path):
    """读语料文件：每行一句 -> 分好词的句子列表。"""
    with open(path, encoding="utf-8") as f:
        return [tokenize(line.strip()) for line in f]


def report(name, sentences, vocab):
    total = sum(len(s) for s in sentences)
    print(f"[{name}] 句子数: {len(sentences)} | 总 token 数: {total} "
          f"| 词表大小: {len(vocab)} (含 4 个特殊 token)")


def main():
    en_sents = read_sentences(EN_PATH)
    de_sents = read_sentences(DE_PATH)

    en_vocab = Vocab().build(en_sents, min_freq=MIN_FREQ)
    de_vocab = Vocab().build(de_sents, min_freq=MIN_FREQ)

    en_vocab.save("vocab/vocab.en.json")
    de_vocab.save("vocab/vocab.de.json")

    report("英语", en_sents, en_vocab)
    report("德语", de_sents, de_vocab)

    demo = en_sents[0]
    print(f"\n演示 encode/decode：")
    print(f"  原句 tokens : {demo}")
    print(f"  encode 成 id: {en_vocab.encode(demo)}")
    print(f"  decode 回来 : {en_vocab.decode(en_vocab.encode(demo))}")


if __name__ == "__main__":
    main()
