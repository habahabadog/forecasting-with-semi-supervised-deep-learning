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

        x1, x2, o1, o2, o3, outputs = model(low_da)
        
        loss1 = criterion(outputs, high_da)
        loss2 = criterion(o1, low2)
        loss3 = criterion(o2, low2)
        loss4 = criterion(o3, low2)
#         loss = loss1 + loss2 + loss3 + loss4
        # 新增 loss
        loss5 = criterion(x1[:, :4], low_da[:, :4])*3/3
        loss6 = criterion(x1[:, 4:8], low_da[:, :4])*2/3
        loss7 = criterion(x1[:, 8:12], low_da[:, :4])*1/3

        loss8  = criterion(x1[:, :4], low_da[:, 4:8])*1/3
        loss9  = criterion(x1[:, 4:8], low_da[:, 4:8])*2/3
        loss10 = criterion(x1[:, 8:12], low_da[:, 4:8])*3/3

        loss11 = criterion(x2[:, :4], low_da[:, 4:8])*3/3
        loss12 = criterion(x2[:, 4:8], low_da[:, 4:8])*2/3
        loss13 = criterion(x2[:, 8:12], low_da[:, 4:8])*1/3

        loss14 = criterion(x2[:, :4], low_da[:, 8:])*1/3
        loss15 = criterion(x2[:, 4:8], low_da[:, 8:])*2/3
        loss16 = criterion(x2[:, 8:12], low_da[:, 8:])*3/3
        
        extra_loss = (loss5 + loss6 + loss7 + loss8 + loss9 + loss10 + loss11 + loss12 + loss13 + loss14 + loss15 + loss16)
        loss = loss1 + loss2 + loss3 + loss4 + extra_loss

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        del low_da, low2, high_da, o1, o2, o3, outputs, loss
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

            x1, x2, o1, o2, o3, outputs = model(low_da)

            loss1 = criterion(outputs, high_da)
            loss2 = criterion(o1, low2)
            loss3 = criterion(o2, low2)
            loss4 = criterion(o3, low2)
    #         loss = loss1 + loss2 + loss3 + loss4
            # 新增 loss
            loss5 = criterion(x1[:, :4], low_da[:, :4])*3/3
            loss6 = criterion(x1[:, 4:8], low_da[:, :4])*2/3
            loss7 = criterion(x1[:, 8:12], low_da[:, :4])*1/3

            loss8  = criterion(x1[:, :4], low_da[:, 4:8])*1/3
            loss9  = criterion(x1[:, 4:8], low_da[:, 4:8])*2/3
            loss10 = criterion(x1[:, 8:12], low_da[:, 4:8])*3/3

            loss11 = criterion(x2[:, :4], low_da[:, 4:8])*3/3
            loss12 = criterion(x2[:, 4:8], low_da[:, 4:8])*2/3
            loss13 = criterion(x2[:, 8:12], low_da[:, 4:8])*1/3

            loss14 = criterion(x2[:, :4], low_da[:, 8:])*1/3
            loss15 = criterion(x2[:, 4:8], low_da[:, 8:])*2/3
            loss16 = criterion(x2[:, 8:12], low_da[:, 8:])*3/3

            extra_loss = (loss5 + loss6 + loss7 + loss8 + loss9 + loss10 + loss11 + loss12 + loss13 + loss14 + loss15 + loss16)
            loss = loss1 + loss2 + loss3 + loss4 + extra_loss

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

            _, _, _, _, _, outputs = model(low_da)
            loss1 = criterion(outputs, high_da)
            total_t_loss += loss1.item()

    avg_t_loss = total_t_loss / (len(new_dataset) / batch_size)
    print(f"整体测试集上的loss: {avg_t_loss:.6f}")
    writer.add_scalar("t_loss", avg_t_loss, i + 1)

writer.close()
