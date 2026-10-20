"""loss 曲线可视化：读取 checkpoints/loss_history.json，画出 train/val 双曲线。

用法：python plot_loss.py
前提：train.py 至少完整跑过一次（loss_history.json 存在）
"""

import json

import matplotlib.pyplot as plt

with open("checkpoints/loss_history.json", encoding="utf-8") as f:
    hist = json.load(f)

train, val = hist["train"], hist["val"]
epochs = range(1, len(train) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, train, marker="o", markersize=3, label="train loss")
plt.plot(epochs, val, marker="s", markersize=3, label="val loss")

plt.axhline(y=8.97, color="gray", linestyle="--", linewidth=1, label="random guess (ln 7882)")
plt.xlabel("Epoch")
plt.ylabel("CrossEntropy Loss")
plt.title("Transformer Training Curve (Multi30k en->de)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("checkpoints/loss_curve.png", dpi=150)
plt.show()
print("已保存: checkpoints/loss_curve.png")
