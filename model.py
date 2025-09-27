import common

import torch.nn as nn
import torch
def make_model(args, parent=False):
    return RCAN(args)

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
    
class AverageChannelAttention(nn.Module):
    def __init__(self, channels):
        super(AverageChannelAttention, self).__init__()
        self.channels = channels
        # 使用1x1卷积进行通道重塑
        self.channel_weights = nn.Conv2d(1, channels, kernel_size=1)

    def forward(self, x):
        # 计算在所有通道上的平均值
        # 这里不使用全局平均池化，而是计算所有通道的平均值
        avg_values = x.mean(dim=1, keepdim=True)
        
        # 通道重塑，每个通道都会被赋予一个权重
        channel_weights = self.channel_weights(avg_values)
        
        # 使用sigmoid函数来获取权重因子，范围为(0, 1)
        scale = torch.sigmoid(channel_weights)
        
        # 将计算得到的权重应用于输入的每个通道
        return x * scale

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat))
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(AverageChannelAttention(n_feat))
#         modules_body.append(AverageChannelAttention(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

## Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    def __init__(self, in_chan=6, out_chan=1, conv=common.default_conv, n_resgroups=4, n_resblocks=4, n_feats=108, reduction=6,scale=10):
        super(RCAN, self).__init__()
        
        # n_resgroups = args.n_resgroups
        # n_resblocks = args.n_resblocks
        # n_feats = args.n_feats
        kernel_size = 3
        # reduction = args.reduction
        # scale = args.scale[0]
        act = nn.ReLU(True)
        
        # RGB mean for DIV2K
        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # rgb_std = (1.0, 1.0, 1.0)
        # self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        
        # define head module
        modules_head = [conv(in_chan, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False)]
            # conv(n_feats, args.n_colors, kernel_size)]

        # self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        # x = self.add_mean(x)

        return x 

    
class RCANtime(nn.Module):
    def __init__(self, in_chan=24, out_chan=36, conv=common.default_conv, n_resgroups=4, n_resblocks=4, n_feats=108, reduction=6,scale=1):
        super(RCANtime, self).__init__()
        
        # n_resgroups = args.n_resgroups
        # n_resblocks = args.n_resblocks
        # n_feats = args.n_feats
        kernel_size = 3
        # reduction = args.reduction
        # scale = args.scale[0]
        act = nn.ReLU(True)
        
        # RGB mean for DIV2K
        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # rgb_std = (1.0, 1.0, 1.0)
        # self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        
        # define head module
        modules_head = [conv(in_chan, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False)]
            # conv(n_feats, args.n_colors, kernel_size)]

        # self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        # x = self.add_mean(x)

        return x 

    
    
class CascadeSRModel(nn.Module):
    def __init__(self, base_model=RCANtime(in_chan=8, out_chan=12, n_resgroups=2, n_resblocks=2, n_feats=96, reduction=16, scale=1), base_model2=RCAN(in_chan=4, out_chan=4, n_resgroups=2, n_resblocks=2, n_feats=96, reduction=16, scale=5)):
        super(CascadeSRModel, self).__init__()
        self.base_model = base_model  # 假设base_model是已经定义的2倍SR模型
        self.base_model2 = base_model2
    def forward(self, x):
        # 第一次放大
        x1 = self.base_model(x[:, :8, :, :])
        # 第二次放大
        x2 = self.base_model(x[:, 4:, :, :])

        x_concat1 = torch.cat([x1[:, :4, :, :], x2[:, :4, :, :]], dim=1)
        x_concat2 = torch.cat([x1[:, 4:8, :, :], x2[:, 4:8, :, :]], dim=1)
        x_concat3 = torch.cat([x1[:, 8:, :, :], x2[:, 8:, :, :]], dim=1)

        x3 = self.base_model(x_concat1)[:, 8:, :, :]
        x4 = self.base_model(x_concat2)[:, 4:8, :, :]
        x5 = self.base_model(x_concat3)[:, :4, :, :]

        x6 = self.base_model2((x3 + x4 + x5) / 3)

        return x1, x2, x3, x4, x5, x6
