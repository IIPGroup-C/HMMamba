Mimport torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from mamba_ssm import Mamba  # 假设使用Mamba实现

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#普通卷积
# class LocalFeatureExtractor(nn.Module):
#     def __init__(self, channels, kernel_size=3, dilation=1):
#         super().__init__()
#         padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
#         self.conv = nn.Sequential(
#             # 普通卷积提取空间特征
#             nn.Conv2d(channels, channels, kernel_size=kernel_size,
#                       padding=padding, dilation=dilation, bias=False),
#             nn.BatchNorm2d(channels),
#             nn.SiLU(),
#             # 点卷积混合通道信息（保留）
#             nn.Conv2d(channels, channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(channels)
#         )
#
#     def forward(self, x):
#         return self.conv(x) + x  # 残差连接

# 深度可分离
class LocalFeatureExtractor(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Sequential(
            # 深度可分离卷积提取空间特征
            nn.Conv2d(channels, channels, kernel_size=kernel_size,
                      padding=padding, dilation=dilation, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            # 点卷积混合通道信息
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return self.conv(x) + x  # 残差连接


class SpectralCompressionMamba(nn.Module):
    def __init__(self):
        super().__init__()
        # 空间压缩后的序列处理
        self.mamba = Mamba(
            d_model=1,  # 输入维度
            d_state=16,  # SSM状态维度
            d_conv=4,  # 卷积核大小
            expand=2  # 扩展因子
        )

        # 共享参数的Mamba结构
        self.shared_mamba = self.mamba  # 共享参数

        self.weights = nn.Parameter(torch.ones(2) / 2)
        self.softmax = nn.Softmax(dim=0)

        # 投影层 - 恢复维度并应用非线性变换

        self.proj = nn.Sequential(
            nn.GroupNorm(1, 1),  # 对序列维度应用GroupNorm
            nn.SiLU()  # 应用非线性激活
        )

    def forward(self, x):
        B, C, H, W = x.shape
        identity = x
        x_permuted = x.permute(0, 2, 3, 1)  # B H W C
        # 1. 沿波段方向压缩 (W,C) -> (W,1)
        x_pool_avg = F.adaptive_avg_pool2d(x_permuted, (W, 1))
        x_pool_max = F.adaptive_max_pool2d(x_permuted, (W, 1))

        # 2. 处理压缩后的序列 (B,H,W,1) -> (B,H,W)
        x_avg_re1 = rearrange(x_pool_avg, 'b h w 1 -> (b h) w 1')
        x_max_re1 = rearrange(x_pool_max, 'b h w 1 -> (b h) w 1')
        x_avg_re2 = rearrange(x_pool_avg, 'b h w 1 -> (b w) h 1')
        x_max_re2 = rearrange(x_pool_max, 'b h w 1 -> (b w) h 1')

        # 3. 共享Mamba处理
        x_avg_for1 = self.shared_mamba(x_avg_re1)
        x_max_for1 = self.shared_mamba(x_max_re1)
        x_avg_for2 = self.shared_mamba(x_avg_re2)
        x_max_for2 = self.shared_mamba(x_max_re2)




        x_avg_reverse1 = torch.flip(x_avg_for1, dims=[1])
        x_max_reverse1 = torch.flip(x_max_for1, dims=[1])
        x_avg_reverse2 = torch.flip(x_avg_for2, dims=[1])
        x_max_reverse2 = torch.flip(x_max_for2, dims=[1])
        x_avg_backward1 = self.shared_mamba(x_avg_reverse1)
        x_max_backward1 = self.shared_mamba(x_max_reverse1)
        x_avg_backward2 = self.shared_mamba(x_avg_reverse2)
        x_max_backward2 = self.shared_mamba(x_max_reverse2)
        x_avg_back1 = torch.flip(x_avg_backward1, dims=[1])
        x_max_back1 = torch.flip(x_max_backward1, dims=[1])
        x_avg_back2 = torch.flip(x_avg_backward2, dims=[1])
        x_max_back2 = torch.flip(x_max_backward2, dims=[1])
        x_max1 = (x_max_for1 + x_max_back1) * 0.5
        x_avg1 = (x_avg_for1 + x_avg_back1) * 0.5
        x_max2 = (x_max_for2 + x_max_back2) * 0.5
        x_avg2 = (x_avg_for2 + x_avg_back2) * 0.5
        # x_max1 = x_max_for1
        # x_avg1 = x_avg_for1
        # x_max2 = x_max_for2
        # x_avg2 = x_avg_for2
        x_max_rea1 = rearrange(x_max1, '(b h) w 1 -> b h w 1', b=B, h=H)
        x_avg_rea1 = rearrange(x_avg1, '(b h) w 1 -> b h w 1', b=B, h=H)
        x_max_rea2 = rearrange(x_max2, '(b w) h 1 -> b h w 1', b=B, w=W)
        x_avg_rea2 = rearrange(x_avg2, '(b w) h 1 -> b h w 1', b=B, w=W)

        # x_avg_rea1 = rearrange(y_avg1, '(b h) w 1 -> b h w 1', b=B, h=H)
        # x_max_rea1 = rearrange(y_max1, '(b h) w 1 -> b h w 1', b=B, h=H)
        #
        # x_avg_rea2 = rearrange(y_avg2, '(b w) h 1 -> b h w 1', b=B, w=W)
        # x_max_rea2 = rearrange(y_max2, '(b w) h 1 -> b h w 1', b=B, w=W)
        x_avg_add = (x_avg_rea1 + x_avg_rea2)*0.5
        x_max_add = (x_max_rea1 + x_max_rea2)*0.5


        y_avg_re = x_avg_add.permute(0, 3, 1, 2)  # B H W 1 -> B 1 H W
        y_max_re = x_max_add.permute(0, 3, 1, 2)


        # 4. 恢复维度并投影处理
        y_avg_proj = self.proj(y_avg_re)
        y_max_proj = self.proj(y_max_re)


        # 7. 扩展到原始空间尺寸
        # y_out = y_add.expand(-1, C, -1, -1)
        y_max_out = y_max_proj.expand(-1, C, -1, -1)
        y_avg_out = y_avg_proj.expand(-1, C, -1, -1)



        # 5. 融合并恢复形状
        # y_add = y_proj1 + y_proj2
        weights = self.softmax(self.weights)
        fusion = y_max_out * weights[0] + y_avg_out * weights[1]

        # 8. 残差连接
        return fusion + x


class WidthCompressionMamba(nn.Module):
    def __init__(self, shared_mamba, share_proj):
        super().__init__()

        # 共享参数的Mamba结构 # 共享参数
        self.shared_mamba = shared_mamba
        # 共享投影层 - 恢复维度并应用非线性变换
        self.share_proj = share_proj

        self.weights = nn.Parameter(torch.ones(2) / 2)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        B, C, H, W = x.shape
        identity = x

        # 1. 沿行方向压缩 (H,W) -> (H,1)
        x_pool_avg = F.adaptive_avg_pool2d(x, (H, 1))
        x_pool_max = F.adaptive_max_pool2d(x, (H, 1))

        # 2. 处理压缩后的序列 (B,C,H,1) -> (B,H,C)
        x_avg = rearrange(x_pool_avg, 'b c h 1 -> b h c')
        x_max = rearrange(x_pool_max, 'b c h 1 -> b h c')

        # 3. 共享Mamba处理
        # 正
        x_avg_for = self.shared_mamba(x_avg)
        x_max_for = self.shared_mamba(x_max)

        # 反
        x_avg_reverse = torch.flip(x_avg, dims=[1])
        x_max_reverse = torch.flip(x_max, dims=[1])
        x_avg_backward = self.shared_mamba(x_avg_reverse)
        x_max_backward = self.shared_mamba(x_max_reverse)
        x_avg_back = torch.flip(x_avg_backward, dims=[1])
        x_max_back = torch.flip(x_max_backward, dims=[1])

        # x_avg_rea = rearrange(x_avg, 'b h c -> b c h 1')
        # x_max_rea = rearrange(x_max, 'b h c -> b c h 1')

        # 4. 融合
        x_max = (x_max_for + x_max_back) * 0.5
        x_avg = (x_avg_for + x_avg_back) * 0.5
        # x_max = x_max_for
        # x_avg = x_avg_for
        x_max_re = rearrange(x_max, 'b h c -> b c h 1')
        x_avg_re = rearrange(x_avg, 'b h c -> b c h 1')



        # 5. 恢复维度并投影处理
        x_max_proj = self.share_proj(x_max_re)
        x_avg_proj = self.share_proj(x_avg_re)
        # x_proj = self.share_proj(fu)  # B C H 1

        # 6. 扩展到原始空间尺寸
        x_max_out = x_max_proj.expand(-1, -1, -1, W)
        x_avg_out = x_avg_proj.expand(-1, -1, -1, W)

        weights = self.softmax(self.weights)
        # fusion_spa = x_max * weights[0] + weights[1] * x_avg + x_avg_back * weights[2] + x_max_back * weights[3]
        fusion = x_max_out * weights[0] + x_avg_out * weights[1]

        return fusion + x


class HeightCompressionMamba(nn.Module):
    def __init__(self, shared_mamba, share_proj):
        super().__init__()

        # 共享参数的Mamba结构
        self.shared_mamba = shared_mamba
        # 共享投影层 - 恢复维度并应用非线性变换
        self.share_proj = share_proj

        self.weights = nn.Parameter(torch.ones(2) / 2)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        B, C, H, W = x.shape
        identity = x

        # 1. 沿列方向压缩 (H,W) -> (1,W)
        x_pool_avg = F.adaptive_avg_pool2d(x, (1, W))
        x_pool_max = F.adaptive_max_pool2d(x, (1, W))

        # 2. 处理压缩后的序列 (B,C,1,W) -> (B,W,C)
        x_avg = rearrange(x_pool_avg, 'b c 1 w -> b w c')
        x_max = rearrange(x_pool_max, 'b c 1 w -> b w c')

        # 3. 共享Mamba处理
        # 正
        x_avg_for = self.shared_mamba(x_avg)
        x_max_for = self.shared_mamba(x_max)
        # 反向
        x_avg_reverse = torch.flip(x_avg, dims=[1])
        x_max_reverse = torch.flip(x_max, dims=[1])
        x_avg_backward = self.shared_mamba(x_avg_reverse)
        x_max_backward = self.shared_mamba(x_max_reverse)
        x_avg_back = torch.flip(x_avg_backward, dims=[1])
        x_max_back = torch.flip(x_max_backward, dims=[1])

        # 4. 融合
        # x_add = x_avg_rea + x_max_rea

        x_max = (x_max_for + x_max_back)*0.5
        x_avg = (x_avg_for + x_avg_back)*0.5
        # x_max = x_max_for
        # x_avg = x_avg_for
        x_max_re = rearrange(x_max, 'b w c -> b c 1 w')
        x_avg_re = rearrange(x_avg, 'b w c -> b c 1 w')

        # 5. 恢复维度并投影处理
        x_max_proj = self.share_proj(x_max_re)
        x_avg_proj = self.share_proj(x_avg_re)

        # 6. 扩展到原始空间尺寸
        x_max_out = x_max_proj.expand(-1, -1, H, -1)
        x_avg_out = x_avg_proj.expand(-1, -1, H, -1)

        weights = self.softmax(self.weights)
        fusion = x_max_out * weights[0] + x_avg_out * weights[1]

        return fusion +x


class ACmambaBlock(nn.Module):
    def __init__(self, channels, use_residual):
        super(ACmambaBlock, self).__init__()
        self.use_residual = use_residual

        self.shared_mamba = Mamba(
            d_model=channels,  # 输入维度
            d_state=16,  # SSM状态维度
            d_conv=4,  # 卷积核大小
            expand=2  # 扩展因子
        )
        self.share_proj = nn.Sequential(
            nn.GroupNorm(1, channels),  # 对序列维度应用
            nn.SiLU()  # 应用非线性激活
        )
        #####不共享
        # self.width_mamba = Mamba(
        #     d_model=channels,
        #     d_state=16,
        #     d_conv=4,
        #     expand=2
        # )
        # self.width_proj = nn.Sequential(
        #     nn.GroupNorm(1, channels),
        #     nn.SiLU()
        # )
        #
        # # 为Height单独创建
        # self.height_mamba = Mamba(
        #     d_model=channels,
        #     d_state=16,
        #     d_conv=4,
        #     expand=2
        # )
        # self.height_proj = nn.Sequential(
        #     nn.GroupNorm(1, channels),
        #     nn.SiLU()
        # )
        # ####结束

        self.weights = nn.Parameter(torch.ones(3) / 3)
        self.softmax = nn.Softmax(dim=0)

        self.x_w = WidthCompressionMamba(self.shared_mamba, self.share_proj)
        self.x_h = HeightCompressionMamba(self.shared_mamba, self.share_proj)
        # self.x_w = WidthCompressionMamba(self.width_mamba, self.width_proj)
        # self.x_h = HeightCompressionMamba(self.height_mamba, self.height_proj)
        self.x_c = SpectralCompressionMamba()

    def forward(self, x):
        # # 1. 并行处理

        x_w = self.x_w(x)
        x_h = self.x_h(x)
        x_c = self.x_c(x)

        # 2. 融合策略
        weights = self.softmax(self.weights)
        fusion_spa = x_w * weights[0] + weights[1] * x_h + x_c * weights[2]
        # fusion_spa = x_h * weights[0] + weights[1] * x_c #去W压缩


        # fusion_spa = x_w + x_h +x_c
        # 3. 残差连接
        # if self.use_residual:
        #     return fusion_spa + x
        # else:
        return fusion_spa + x


class MidRangeConnector(nn.Module):
    def __init__(self, channels):
        super(MidRangeConnector, self).__init__()
        self.channels = channels
        dim = self.channels  # 特征维度与通道数相同

        # # 水平方向的卷积操作 使用3*7长卷积核保持空间尺寸不变
        # #groups=dim 表示使用深度可分离卷积，每个通道单独处理
        # self.fc_h = nn.Conv2d(dim, dim, kernel_size=(3, 7), stride=1, padding=(1, 7 // 2), groups=dim, bias=False)
        # # 垂直方向的卷积操作 使用7*3长卷积核保持空间尺寸不变
        # self.fc_w = nn.Conv2d(dim, dim, kernel_size=(7, 3), stride=1, padding=(7 // 2, 1), groups=dim, bias=False)
        # self.fc_h = nn.Conv2d(dim, dim, kernel_size=(3, 11), stride=1, padding=(1, 5), groups=dim, bias=False)
        # self.fc_w = nn.Conv2d(dim, dim, kernel_size=(11, 3), stride=1, padding=(5, 1), groups=dim, bias=False)
        self.fc_h = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(3, 1), padding=(1, 0), groups=dim, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=(1, 11), padding=(0, 5), groups=dim, bias=False)
        )

        self.fc_w = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(1, 3), padding=(0, 1), groups=dim, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=(11, 1), padding=(5, 0), groups=dim, bias=False)

        )

        # self.fc_h = nn.Conv2d(dim, dim, kernel_size=(3, 11), stride=1, padding=(1, 5), groups=dim, bias=False)
        # self.fc_w = nn.Conv2d(dim, dim, kernel_size=(11, 3), stride=1, padding=(5, 1), groups=dim, bias=False)
        # 普通卷积
        # self.fc_h = nn.Conv2d(dim, dim, kernel_size=(3, 7), stride=1, padding=(1, 7 // 2), bias=False)
        # self.fc_w = nn.Conv2d(dim, dim, kernel_size=(7, 3), stride=1, padding=(7 // 2, 1), bias=False)
        # self.fc_h = nn.Sequential(
        #     nn.Conv2d(dim, dim, kernel_size=(3, 1), padding=(1, 0), bias=False),
        #     nn.GELU(),
        #     nn.Conv2d(dim, dim, kernel_size=(1, 11), padding=(0, 5), bias=False)
        # )
        #
        # self.fc_w = nn.Sequential(
        #     nn.Conv2d(dim, dim, kernel_size=(1, 3), padding=(0, 1), bias=False),
        #     nn.GELU(),
        #     nn.Conv2d(dim, dim, kernel_size=(11, 1), padding=(5, 0), bias=False)
        #
        # )

        
        self.weights = nn.Parameter(torch.ones(2) / 2)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        B, C, H, W = x.shape
        # 水平方向特征提取
        x_w = self.fc_w(x)
        # 垂直方向特征提取
        x_h = self.fc_h(x)

        weights = self.softmax(self.weights)
        # x_att = x_h * weights[0] + x_w * weights[1]
        x_att = x_w
        return x_att + x




class MBSBlock(nn.Module):
    def __init__(self, channels, use_residual, use_att=True):
        super(MBSBlock, self).__init__()
        self.use_att = use_att
        self.use_residual = use_residual
        if self.use_att:
            self.weights = nn.Parameter(torch.ones(3) / 3)
            self.softmax = nn.Softmax(dim=0)

        self.LFE = LocalFeatureExtractor(channels)
        self.MRC = MidRangeConnector(channels)
        self.TAFmamba = ACmambaBlock(channels, use_residual=use_residual)

    def forward(self, x):
        # # 1. 并行处理
        x_tafmamba = self.TAFmamba(x)
        x_lfe = self.LFE(x)
        x_mrc = self.MRC(x)

        # 2. 融合策略
        if self.use_att:
            weights = self.softmax(self.weights)  # 计算注意力权重
            # fusion_x = x_tafmamba * weights[0] + x_lfe * weights[1]
            fusion_x = x_tafmamba * weights[0] + x_lfe * weights[1] + x_mrc * weights[2]
            # fusion_x = x_lfe
            # fusion_x = x_tafmamba + x_lfe + x_mrc
        else:
            fusion_x = x_tafmamba + x_lfe + x_mrc  # 简单相加
            # fusion_x = x_tafmamba + x_lfe   # 简单相加
            # fusion_x = x_lfe    # 简单相加

        # 3. 残差连接
        if self.use_residual:
            return fusion_x + x
        else:
            return fusion_x


class ACmamba(nn.Module):
    def __init__(self, in_channels=128, hidden_dim=64, num_classes=10, use_residual=True, mamba_type='both',
                 token_num=4, group_num=4, use_att=True):
        super(ACmamba, self).__init__()
        self.mamba_type = mamba_type

        # 初始投影层（1x1卷积）
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU())

        self.mamba = nn.Sequential(MBSBlock(channels=hidden_dim, use_residual=use_residual),
                                   nn.AvgPool2d(kernel_size=3, stride=2, padding=1),  # 空间下采样
                                   MBSBlock(channels=hidden_dim, use_residual=use_residual),
                                   # nn.Dropout2d(0.03),
                                   nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                                   MBSBlock(channels=hidden_dim, use_residual=use_residual)
                                   )

        # 分类头
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, 128),
            nn.SiLU(),
            # nn.Dropout(0.3),
            nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        # 1. 初始投影
        x = self.patch_embedding(x)  # [B,C,H,W] → [B,hidden_dim,H,W]
        # 2. Mamba处理
        x = self.mamba(x)
        # 3. 分类预测
        logits = self.cls_head(x)  # [B,num_classes,H,W]
        return logits





# if __name__ == '__main__':
#     input = torch.randn(1, 32, 224, 224).to(device)
#     model = LMM(channels=32).to(device)
#     output = model(input)
#     print(input.shape)
#     print(output.shape)
if __name__ == "__main__":
    # 创建测试输入
    x = torch.randn(2, 128, 32, 32).to(device)

    # 初始化模型
    model = ACmamba(
        in_channels=128,
        hidden_dim=64,
        num_classes=10,
        use_att=True
    ).to(device)

    # 测试前向传播
    out = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")  # 应该输出 [2, 10, 8, 8]