import json

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.vocab import Vocab, PAD_IDX
from utils.data import Multi30kDataset, collate_fn
from utils.mask import make_src_mask, make_trg_mask
from models.transformer import Transformer
from configs.config import *
import os
#可视化进度依赖
import time
from tqdm import tqdm

def plot_loss_curve(train_losses, val_losses):
    """训练结束自动画出 train/val 双曲线并保存。"""
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker="o", markersize=3, label="train loss")
    plt.plot(epochs, val_losses, marker="s", markersize=3, label="val loss")
    plt.axhline(y=8.97, color="gray", linestyle="--", linewidth=1, label="random guess (ln 7882)")
    plt.xlabel("Epoch")
    plt.ylabel("CrossEntropy Loss")
    plt.title("Transformer Training Curve (Multi30k en->de)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"checkpoints/{RUN_NAME}/loss_curve.png", dpi=150)
    plt.close()
    print("loss 曲线已保存: checkpoints/loss_curve.png")


def train():
    RUN_DIR = f"checkpoints/{RUN_NAME}"
    os.makedirs(RUN_DIR, exist_ok=True)    # 目录不存在就建；已存在则静默通过
    # TODO 1：读词表
    src_vocab = Vocab.load("vocab/vocab.en.json")
    trg_vocab = Vocab.load("vocab/vocab.de.json")

    # TODO 2：加载训练数据
    train_data = Multi30kDataset("data/train.en", "data/train.de", src_vocab, trg_vocab)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_data = Multi30kDataset("data/valid.en", "data/valid.de", src_vocab, trg_vocab)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)


    best_loss = float("inf")
    train_losses, val_losses = [], []      # 每个 epoch 的 loss 历史（供可视化）
    early_stop_count = 0
    # TODO 3：加载模型
    model = Transformer(src_vocab_size=len(src_vocab),
                        trg_vocab_size=len(trg_vocab),
                        d_model=D_MODEL,
                        n_heads=N_HEADS,
                        n_layers=N_LAYERS,
                        d_ff=D_FF,
                        dropout=DROPOUT,
                        max_len=MAX_LEN)

    # TODO 4：加载优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # TODO 5：训练模型
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    train_start =time.time()#记录训练开始时间

    for epoch in range(EPOCHS):
        epoch_start = time.time()#记录每个epoch开始时间
        print(f"\n[{time.strftime('%H:%M:%S')}] ── Epoch {epoch+1}/{EPOCHS} 开始 ──")#打印当前时间
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")   # 空⑤：包进度条

        total_loss = 0
        for src_batch, trg_batch in pbar:
            src_batch, trg_batch = src_batch.to(device), trg_batch.to(device)

            optimizer.zero_grad()  #标准5步：1.清零梯度 2.前向传播 3.计算损失 4.反向传播 5.优化器更新
            # TODO 5.5 ：计算mask
            src_mask = make_src_mask(src_batch)
            trg_input = trg_batch[:, :-1] #这是为了去掉最后一个单词，因为最后一个单词是标签，不能作为输入
            trg_mask = make_trg_mask(trg_input)
            trg_target = trg_batch[:, 1:] #这是为了去掉第一个单词，因为第一个单词是起始符，不能作为标签
            # TODO 6：前向传播
            logits = model(src_batch, trg_input, src_mask, trg_mask)
            # TODO 7：计算损失 
            loss = criterion(
                logits.reshape(-1, logits.shape[-1]),
                trg_target.reshape(-1)
            )
            # TODO 8：反向传播
            loss.backward()
            # TODO 9：优化器更新
            optimizer.step()

            pbar.set_postfix(loss=f"{loss.item():.3f}") #空⑥：进度条显示 loss
            # TODO 10：记录损失
            total_loss += loss.item()
        # TODO 11：打印损失
        epoch_time = time.time() - epoch_start
        train_losses.append(total_loss / len(train_loader))
        print(f"[{time.strftime('%H:%M:%S')}] Epoch {epoch+1}/{EPOCHS} 结束 | "
              f"平均train loss: {train_losses[-1]:.4f} | 耗时 {epoch_time:.1f}s")

        #验证

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for src_batch, trg_batch in val_loader:
                src_batch, trg_batch = src_batch.to(device), trg_batch.to(device)
                src_mask = make_src_mask(src_batch)
                trg_input = trg_batch[:, :-1]
                trg_mask = make_trg_mask(trg_input)
                trg_target = trg_batch[:, 1:]
                logits = model(src_batch, trg_input, src_mask, trg_mask)
                loss = criterion(
                    logits.reshape(-1, logits.shape[-1]),
                    trg_target.reshape(-1)
                )
                val_loss += loss.item()
        print(f"验证集 loss: {val_loss/len(val_loader):.4f}")
        val_losses.append(val_loss / len(val_loader))
        model.train()
        # TODO 12：保存模型
        if val_loss<best_loss:
            best_loss = val_loss
            early_stop_count = 0
            torch.save(model.state_dict(), f"{RUN_DIR}/best.pt")
        else:
            early_stop_count += 1

        if early_stop_count >= PATIENCE:
            print(f"早停触发：验证 loss 连续 {PATIENCE} 轮无改善，停止训练")
            break

    # 训练结束：保存 loss 历史供可视化
    with open(f"{RUN_DIR}/loss_history.json", "w", encoding="utf-8") as f:
        json.dump({"train": train_losses, "val": val_losses}, f)
    print("loss 历史已保存: checkpoints/loss_history.json")
    plot_loss_curve(train_losses, val_losses)
    print(f"\n总训练耗时: {(time.time() - train_start) / 60:.1f} 分钟")

if __name__ == "__main__":
    train()

        


