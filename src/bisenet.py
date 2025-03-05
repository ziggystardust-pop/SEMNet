import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary


class StemBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=16):
        super(StemBlock, self).__init__()

        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv_branch = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 2, 1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(),
            nn.Conv2d(out_channels // 2, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)

        self.fusion = nn.Sequential(
            nn.Conv2d(2 * out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv_in(x)

        x_branch = self.conv_branch(x)
        x_downsample = self.pool(x)
        out = torch.cat([x_branch, x_downsample], dim=1)
        out = self.fusion(out)

        return out


class depthwise_separable_conv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1,
                                   groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class GELayer(nn.Module):
    def __init__(self, in_channels, out_channels, exp_ratio=6, stride=1):
        super(GELayer, self).__init__()
        mid_channel = in_channels * exp_ratio
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

        if stride == 1:
            self.dwconv = nn.Sequential(
                # ReLU in ConvModule not shown in paper
                nn.Conv2d(in_channels, mid_channel, 3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(mid_channel),
                nn.ReLU(),

                depthwise_separable_conv(mid_channel, mid_channel, stride=1),
                nn.BatchNorm2d(mid_channel),
            )
            self.shortcut = None
        else:
            self.dwconv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channel, 3, stride=1, padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(mid_channel),
                nn.ReLU(),

                # ReLU in ConvModule not shown in paper
                depthwise_separable_conv(mid_channel, mid_channel, stride=stride),
                nn.BatchNorm2d(mid_channel),

                depthwise_separable_conv(mid_channel, mid_channel, stride=1),
                nn.BatchNorm2d(mid_channel),
            )

            self.shortcut = nn.Sequential(
                depthwise_separable_conv(in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels),

                nn.Conv2d(out_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )

        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.act = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.conv2(x)

        if self.shortcut is not None:
            shortcut = self.shortcut(identity)
            x = x + shortcut
        else:
            x = x + identity
        x = self.act(x)
        return x


class CEBlock(nn.Module):
    def __init__(self, in_channels=16, out_channels=16):
        super(CEBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            # AdaptiveAvgPool2d 把形状变为(Batch size, N, 1, 1)后，batch size=1不能正常通过BatchNorm2d， 但是batch size>1是可以正常通过的。如果想开启BatchNorm，训练时batch size>1即可，测试时使用model.eval()即不会报错。
            # nn.BatchNorm2d(self.in_channels)
        )

        self.conv_gap = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 1, stride=1, padding=0),
            # nn.BatchNorm2d(self.out_channels), 同上
            nn.ReLU()
        )

        # Note: in paper here is naive conv2d, no bn-relu
        self.conv_last = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1)

    def forward(self, x):
        identity = x
        x = self.gap(x)
        x = self.conv_gap(x)
        x = identity + x
        x = self.conv_last(x)
        return x


class DetailBranch(nn.Module):
    def __init__(self, detail_channels=(64, 64, 128), in_channels=3):
        super(DetailBranch, self).__init__()
        self.detail_branch = nn.ModuleList()

        for i in range(len(detail_channels)):
            if i == 0:
                self.detail_branch.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, detail_channels[i], 3, stride=2, padding=1),
                        nn.BatchNorm2d(detail_channels[i]),
                        nn.ReLU(),

                        nn.Conv2d(detail_channels[i], detail_channels[i], 3, stride=1, padding=1),
                        nn.BatchNorm2d(detail_channels[i]),
                        nn.ReLU(),
                    )
                )
            else:
                self.detail_branch.append(
                    nn.Sequential(
                        nn.Conv2d(detail_channels[i - 1], detail_channels[i], 3, stride=2, padding=1),
                        nn.BatchNorm2d(detail_channels[i]),
                        nn.ReLU(),

                        nn.Conv2d(detail_channels[i], detail_channels[i], 3, stride=1, padding=1),
                        nn.BatchNorm2d(detail_channels[i]),
                        nn.ReLU(),

                        nn.Conv2d(detail_channels[i], detail_channels[i], 3, stride=1, padding=1),
                        nn.BatchNorm2d(detail_channels[i]),
                        nn.ReLU()
                    )
                )

    def forward(self, x):
        for stage in self.detail_branch:
            x = stage(x)
        return x


class SemanticBranch(nn.Module):
    def __init__(self, semantic_channels=(16, 32, 64, 128), in_channels=3, exp_ratio=6):
        super(SemanticBranch, self).__init__()
        self.in_channels = in_channels
        self.semantic_channels = semantic_channels
        self.semantic_stages = nn.ModuleList()

        for i in range(len(semantic_channels)):
            if i == 0:
                self.semantic_stages.append(StemBlock(self.in_channels, semantic_channels[i]))

            elif i == (len(semantic_channels) - 1):
                self.semantic_stages.append(
                    nn.Sequential(
                        GELayer(semantic_channels[i - 1], semantic_channels[i], exp_ratio, 2),
                        GELayer(semantic_channels[i], semantic_channels[i], exp_ratio, 1),

                        GELayer(semantic_channels[i], semantic_channels[i], exp_ratio, 1),
                        GELayer(semantic_channels[i], semantic_channels[i], exp_ratio, 1)
                    )
                )

            else:
                self.semantic_stages.append(
                    nn.Sequential(
                        GELayer(semantic_channels[i - 1], semantic_channels[i],
                                exp_ratio, 2),
                        GELayer(semantic_channels[i], semantic_channels[i],
                                exp_ratio, 1)
                    )
                )

        self.semantic_stages.append(CEBlock(semantic_channels[-1], semantic_channels[-1]))

    def forward(self, x):
        semantic_outs = []
        for semantic_stage in self.semantic_stages:
            x = semantic_stage(x)
            semantic_outs.append(x)
        return semantic_outs


class AggregationLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AggregationLayer, self).__init__()
        self.Conv_DetailBranch_1 = nn.Sequential(
            depthwise_separable_conv(in_channels, out_channels, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 1)
        )

        self.Conv_DetailBranch_2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.Conv_SemanticBranch_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            nn.Sigmoid()
        )

        self.Conv_SemanticBranch_2 = nn.Sequential(
            depthwise_separable_conv(in_channels, out_channels, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, Detail_x, Semantic_x):
        DetailBranch_1 = self.Conv_DetailBranch_1(Detail_x)
        DetailBranch_2 = self.Conv_DetailBranch_2(Detail_x)

        SemanticBranch_1 = self.Conv_SemanticBranch_1(Semantic_x)
        SemanticBranch_2 = self.Conv_SemanticBranch_2(Semantic_x)

        out_1 = torch.matmul(DetailBranch_1, SemanticBranch_1)
        out_2 = torch.matmul(DetailBranch_2, SemanticBranch_2)
        out_2 = F.interpolate(out_2, scale_factor=4, mode="bilinear", align_corners=True)

        out = torch.matmul(out_1, out_2)
        out = self.conv_out(out)
        return out


class SegHead(nn.Module):
    def __init__(self, channels, num_classes):
        super().__init__()
        self.cls_seg = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, num_classes, 1),
        )

    def forward(self, x):
        return self.cls_seg(x)


class BiSeNetV2(nn.Module):
    def __init__(self, in_channels=3,
                 detail_channels=(64, 64, 128),
                 semantic_channels=(16, 32, 64, 128),
                 semantic_expansion_ratio=6,
                 aggregation_channels=128,
                 out_indices=(0, 1, 2, 3, 4),
                 num_classes=3):
        super(BiSeNetV2, self).__init__()

        self.in_channels = in_channels
        self.detail_channels = detail_channels
        self.semantic_expansion_ratio = semantic_expansion_ratio
        self.semantic_channels = semantic_channels
        self.aggregation_channels = aggregation_channels
        self.out_indices = out_indices
        self.num_classes = num_classes

        self.detail = DetailBranch(detail_channels=self.detail_channels, in_channels=self.in_channels)
        self.semantic = SemanticBranch(semantic_channels=self.semantic_channels, in_channels=self.in_channels,
                                       exp_ratio=self.semantic_expansion_ratio)
        self.AggregationLayer = AggregationLayer(in_channels=self.aggregation_channels,
                                                 out_channels=self.aggregation_channels)

        self.seg_head_aggre = SegHead(semantic_channels[-1], self.num_classes)
        self.seg_heads = nn.ModuleList()
        self.seg_heads.append(self.seg_head_aggre)
        for channel in semantic_channels:
            self.seg_heads.append(SegHead(channel, self.num_classes))

    def forward(self, x):
        _, _, h, w = x.size()
        x_detail = self.detail(x)
        x_semantic_lst = self.semantic(x)
        x_head = self.AggregationLayer(x_detail, x_semantic_lst[-1])
        outs = [x_head] + x_semantic_lst[:-1]
        outs = [outs[i] for i in self.out_indices]

        out = tuple(outs)

        seg_out = []
        for index, stage in enumerate(self.seg_heads):
            seg_out.append(F.interpolate(stage(out[index]), size=(h, w), mode="bilinear", align_corners=True))

        return seg_out

if __name__ == '__main__':

    model = BiSeNetV2(in_channels=3,num_classes=3)
    input_tensor = torch.rand(1, 3, 256, 256)  # 生成一个随机输入
    output = model(input_tensor)  # 将输入通过UNet处理
    print(type(output))
    print(output[0])
    print(output[0].size())
    # print(output["out"].size())  # 打印输出尺寸
    # summary(model, (3, 256, 256))