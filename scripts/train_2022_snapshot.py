import os
from model import CascadeSRModel
from utils import *
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import time

torch.cuda.empty_cache()

# 显式指定设备
device = torch.device("cuda:0")

# 模型与损失函数
model = CascadeSRModel().to(device)
criterion = nn.L1Loss().to(device)

# 优化器与学习率
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=10, verbose=True,
    threshold=0.0001, threshold_mode='rel', cooldown=10, min_lr=1e-8, eps=1e-8)

os.makedirs('model', exist_ok=True)
writer = SummaryWriter("logs_train")

epoch = 100
start_time = time.time()

for i in range(epoch):
    model.train()
    total_train_loss = 0
    print(f"------第 {i+1} 轮训练开始------ ")

    # 训练
    for low_da, low2, high_da, _ in train_load:
        low_da, low2, high_da = low_da.to(device), low2.to(device), high_da.to(device)

        optimizer.zero_grad()

        x1, x2, outputs, mid = model(low_da)
        
        loss_a = criterion(outputs, high_da)
        loss_b = 0.0
            # Accumulate loss for each element in the mid list with respect to low2
        for m in mid:
            loss_b += criterion(m, low2)
#         loss2 = criterion(o1, low2)
#         loss3 = criterion(o2, low2)
#         loss4 = criterion(o3, low2)
#         loss = loss1 + loss2 + loss3 + loss4
        # 新增 loss
# 假设 x1 的 shape 是 [batch, 36]，low_da 的 shape 是 [batch, 16]

        # 与 low_da[:, 0:4] 比较，权重从 9/9 到 1/9（降序）
        loss1  = criterion(x1[:,  0:  4], low_da[:,  0:  4]) * (11/11)
        loss2  = criterion(x1[:,  4:  8], low_da[:,  0:  4]) * (10/11)
        loss3  = criterion(x1[:,  8: 12], low_da[:,  0:  4]) * (9/11)
        loss4  = criterion(x1[:, 12: 16], low_da[:,  0:  4]) * (8/11)
        loss5  = criterion(x1[:, 16: 20], low_da[:,  0:  4]) * (7/11)
        loss6  = criterion(x1[:, 20: 24], low_da[:,  0:  4]) * (6/11)
        loss7  = criterion(x1[:, 24: 28], low_da[:,  0:  4]) * (5/11)
        loss8  = criterion(x1[:, 28: 32], low_da[:,  0:  4]) * (4/11)
        loss9  = criterion(x1[:, 32: 36], low_da[:,  0:  4]) * (3/11)
        loss10 = criterion(x1[:, 36: 40], low_da[:,  0:  4]) * (2/11)
        loss11 = criterion(x1[:, 40: 44], low_da[:,  0:  4]) * (1/11)
        # 与 low_da[:, 4:8] 比较，权重从 1/9 到 9/9（升序）
        loss12 = criterion(x1[:,  0: 4], low_da[:,  4:  8]) * (1/11)
        loss13 = criterion(x1[:, 4: 8], low_da[:,  4:  8]) * (2/11)
        loss14 = criterion(x1[:, 8: 12], low_da[:,  4:  8]) * (3/11)
        loss15 = criterion(x1[:, 12: 16], low_da[:,  4:  8]) * (4/11)
        loss16 = criterion(x1[:, 16: 20], low_da[:,  4:  8]) * (5/11)
        loss17 = criterion(x1[:, 20: 24], low_da[:,  4:  8]) * (6/11)
        loss18 = criterion(x1[:, 24: 28], low_da[:,  4:  8]) * (7/11)
        loss19 = criterion(x1[:,  28:  32], low_da[:,  4:  8]) * (8/11)
        loss20 = criterion(x1[:,  32:  36], low_da[:,  4:  8]) * (9/11)
        loss21 = criterion(x1[:,  36: 40], low_da[:,  4:  8]) * (10/11)
        loss22 = criterion(x1[:, 40: 44], low_da[:,  4:  8]) * (11/11)
        # 与 low_da[:, 8:12] 比较，权重从 9/9 到 1/9（降序）

        loss23 = criterion(x1[:,  0:  4], low_da[:,  8:  ]) * (1/11)
        loss24 = criterion(x1[:, 4: 8], low_da[:,  8: ]) * (2/11)
        loss25 = criterion(x1[:, 8: 12], low_da[:,  8: ]) * (3/11)
        loss26 = criterion(x1[:, 12: 16], low_da[:,  8: ]) * (4/11)
        loss27 = criterion(x1[:, 16: 20], low_da[:,  8: ]) * (5/11)
        loss28 = criterion(x1[:,  20:  24], low_da[:, 8: ]) * (6/11)
        loss29 = criterion(x1[:,  24:  28], low_da[:, 8: ]) * (7/11)
        loss30 = criterion(x1[:,  28: 32], low_da[:, 8: ]) * (8/11)
        loss31 = criterion(x1[:, 32: 36], low_da[:, 8: ]) * (9/11)
        loss32 = criterion(x1[:, 36: 40], low_da[:, 8: ]) * (10/11)
        loss33 = criterion(x1[:, 40: 44], low_da[:, 8: ]) * (11/11)
        # 与 low_da[:, 12:16] 比较，权重从 1/9 到 9/9（升序）


        loss34 = criterion(x1[:,  0:  4], low_da[:, 8: ]) * (11/11)
        loss35 = criterion(x1[:, 4: 8], low_da[:, 8: ]) * (10/11)
        loss36 = criterion(x1[:, 8: 12], low_da[:, 8: ]) * (9/11)
        loss37 = criterion(x1[:, 12: 16], low_da[:, 8: ]) * (8/11)
        loss38 = criterion(x1[:, 16: 20], low_da[:, 8: ]) * (7/11)
        loss39 = criterion(x1[:, 20: 24], low_da[:, 8: ]) * (6/11)
        loss40 = criterion(x1[:, 24: 28], low_da[:, 8: ]) * (5/11)
        loss41 = criterion(x1[:, 28: 32], low_da[:, 8: ]) * (4/11)
        loss42 = criterion(x1[:, 32: 36], low_da[:, 8: ]) * (3/11)
        loss43 = criterion(x1[:, 36: 40], low_da[:, 8: ]) * (2/11)
        loss44 = criterion(x1[:, 40: 44], low_da[:, 8: ]) * (1/11)

        # 最终累加
        extra_loss = (
            loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8 + loss9 +
            loss10 + loss11 + loss12 + loss13 + loss14 + loss15 + loss16 + loss17 + loss18 +
            loss19 + loss20 + loss21 + loss22 + loss23 + loss24 + loss25 + loss26 + loss27 +
            loss28 + loss29 + loss30 + loss31 + loss32 + loss33 + loss34 + loss35 + loss36 +
            loss37 + loss38 + loss39 + loss40 + loss41 + loss42 + loss43 + loss44
        )

#         extra_loss = (loss5 + loss8 + loss11 + loss14)
        loss = loss_a + loss_b + extra_loss

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        del low_da, low2, high_da, mid, outputs, loss
        torch.cuda.empty_cache()

    avg_train_loss = total_train_loss / (len(train_dataset) / batch_size)
    print(f"整体训练集上的loss: {avg_train_loss:.6f}")
    writer.add_scalar("train_loss", avg_train_loss, i + 1)

    # 验证
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for low_da, low2, high_da, _ in test_load:
            low_da, low2, high_da = low_da.to(device), low2.to(device), high_da.to(device)

            x1, x2, outputs, mid = model(low_da)
        
            loss1 = criterion(outputs, high_da)
            loss2 = 0.0
                # Accumulate loss for each element in the mid list with respect to low2
            for m in mid:
                loss2 += criterion(m, low2)
    #         loss2 = criterion(o1, low2)
    #         loss3 = criterion(o2, low2)
    #         loss4 = criterion(o3, low2)
    #         loss = loss1 + loss2 + loss3 + loss4
            # 新增 loss
#             loss5 = criterion(x1[:, :4], low_da[:, :4])*3/3
    #         loss6 = criterion(x1[:, 4:8], low_da[:, :4])*2/3
    #         loss7 = criterion(x1[:, 8:12], low_da[:, :4])*1/3

#             loss8  = criterion(x1[:, :4], low_da[:, 4:8])*1/3
    #         loss9  = criterion(x1[:, 4:8], low_da[:, 4:8])*2/3
    #         loss10 = criterion(x1[:, 8:12], low_da[:, 4:8])*3/3

#             loss11 = criterion(x2[:, :4], low_da[:, 4:8])*3/3
    #         loss12 = criterion(x2[:, 4:8], low_da[:, 4:8])*2/3
    #         loss13 = criterion(x2[:, 8:12], low_da[:, 4:8])*1/3

#             loss14 = criterion(x2[:, :4], low_da[:, 8:])*1/3
    #         loss15 = criterion(x2[:, 4:8], low_da[:, 8:])*2/3
    #         loss16 = criterion(x2[:, 8:12], low_da[:, 8:])*3/3

    #         extra_loss = (loss5 + loss6 + loss7 + loss8 + loss9 + loss10 + loss11 + loss12 + loss13 + loss14 + loss15 + loss16)
#             extra_loss = (loss5 + loss8 + loss11 + loss14)
            loss = loss1 + loss2
#     + extra_loss

            total_test_loss += loss.item()

    avg_test_loss = total_test_loss / (len(test_dataset) / batch_size)
    print(f"整体验证集上的loss: {avg_test_loss:.6f}")
    scheduler.step(avg_test_loss)
    writer.add_scalar("test_loss", avg_test_loss, i + 1)

    # 保存模型权重
    torch.save(model.base_model.state_dict(), f'./model/base_model_{i+1}.pth')
    torch.save(model.base_model2.state_dict(), f'./model/base_model2_{i+1}.pth')
    print("模型已保存")

    # 测试集额外评估
    total_t_loss = 0
    with torch.no_grad():
        for low_da, low2, high_da, _ in new_load:
            low_da, high_da = low_da.to(device), high_da.to(device)

            _, _, outputs, _ = model(low_da)
            loss1 = criterion(outputs, high_da)
            total_t_loss += loss1.item()

    avg_t_loss = total_t_loss / (len(new_dataset) / batch_size)
    print(f"整体测试集上的loss: {avg_t_loss:.6f}")
    writer.add_scalar("t_loss", avg_t_loss, i + 1)

writer.close()
