import math
import torch
import torch.nn as nn
import torch
import torch._utils
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


BN_MOMENTUM = 0.1


class TransEncoder(nn.Module):
    def __init__(self, channel, num_head, num_layer, num_patches, use_pos_embed):
        super(TransEncoder, self).__init__()
        self.channel = channel
        self.use_pos_embed = use_pos_embed
        translayer = nn.TransformerEncoderLayer(d_model=channel, nhead=num_head)
        self.trans = nn.TransformerEncoder(translayer, num_layers=num_layer)
        if self.use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, channel))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        if self.use_pos_embed:
            x = x.flatten(2).transpose(1, 2) + self.pos_embed
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.trans(x)
        x = x.transpose(1, 2).view(-1, self.channel, int(h), int(w))
        return x


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class CBAM(nn.Module):
    def __init__(self, in_channel):
        super(CBAM, self).__init__()
        self.Cam = ChannelAttentionModul(in_channel=in_channel)
        self.Sam = SpatialAttentionModul(in_channel=in_channel)

    def forward(self, x):
        out = self.Cam(x)
        out = self.Sam(out)
        x = torch.cat([out, x], 1)
        return x


class ChannelAttentionModul(nn.Module):
    def __init__(self, in_channel, r=0.5):
        super(ChannelAttentionModul, self).__init__()
        self.MaxPool = nn.AdaptiveMaxPool2d(1)

        self.fc_MaxPool = nn.Sequential(
            nn.Linear(in_channel, int(in_channel * r)),
            nn.ReLU(),
            nn.Linear(int(in_channel * r), in_channel),
            nn.Sigmoid(),
        )

        self.AvgPool = nn.AdaptiveAvgPool2d(1)

        self.fc_AvgPool = nn.Sequential(
            nn.Linear(in_channel, int(in_channel * r)),
            nn.ReLU(),
            nn.Linear(int(in_channel * r), in_channel),
            nn.Sigmoid(),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_branch = self.MaxPool(x)
        max_in = max_branch.view(max_branch.size(0), -1)
        max_weight = self.fc_MaxPool(max_in)

        avg_branch = self.AvgPool(x)
        avg_in = avg_branch.view(avg_branch.size(0), -1)
        avg_weight = self.fc_AvgPool(avg_in)

        weight = max_weight + avg_weight
        weight = self.sigmoid(weight)

        h, w = weight.shape
        Mc = torch.reshape(weight, (h, w, 1, 1))

        x = Mc * x

        return x


class SpatialAttentionModul(nn.Module):
    def __init__(self, in_channel):
        super(SpatialAttentionModul, self).__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        MaxPool = torch.max(x, dim=1).values
        AvgPool = torch.mean(x, dim=1)

        MaxPool = torch.unsqueeze(MaxPool, dim=1)
        AvgPool = torch.unsqueeze(AvgPool, dim=1)

        x_cat = torch.cat((MaxPool, AvgPool), dim=1)

        x_out = self.conv(x_cat)
        Ms = self.sigmoid(x_out)

        x = Ms * x

        return x


class ChannelAttention(nn.Module):
    def __init__(self):
        super(ChannelAttention, self).__init__()
        self.beta = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.beta.data.fill_(0.0)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        b, c, h, w = x.size()
        X = self.softmax(
            torch.matmul(x.view(b, c, h * w), x.view(b, c, h * w).transpose(1, 2))
        )
        X = torch.matmul(X.transpose(1, 2), x.view(b, c, h * w)).view(b, c, h, w)
        X = self.beta * X + x

        return X


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

        self.attention = CoordAtt(planes, planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.attention(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, padding=2, dilation=2, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, padding=2, dilation=2, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv4 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn4 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.attention = CoordAtt(planes * self.expansion, planes * self.expansion)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.conv4(out)
        out = self.bn4(out)

        out = self.attention(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(
        self,
        num_branches,
        blocks,
        num_blocks,
        num_inchannels,
        num_channels,
        multi_scale_output=True,
    ):
        super(HighResolutionModule, self).__init__()
        self.num_inchannels = num_inchannels
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels
        )
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):

        downsample = None
        if (
            stride != 1
            or self.num_inchannels[branch_index]
            != num_channels[branch_index] * block.expansion
        ):
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion, momentum=BN_MOMENTUM
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample,
            )
        )
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(self.num_inchannels[branch_index], num_channels[branch_index])
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for branch_index in range(num_branches):
            branches.append(
                self._make_one_branch(branch_index, block, num_blocks, num_channels)
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        fuse_layers = []
        for i in range(self.num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(self.num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                self.num_inchannels[j],
                                self.num_inchannels[i],
                                1,
                                1,
                                0,
                                bias=False,
                            ),
                            nn.BatchNorm2d(
                                self.num_inchannels[i], momentum=BN_MOMENTUM
                            ),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        self.num_inchannels[j],
                                        self.num_inchannels[i],
                                        3,
                                        2,
                                        1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(
                                        self.num_inchannels[i], momentum=BN_MOMENTUM
                                    ),
                                )
                            )
                        else:
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        self.num_inchannels[j],
                                        self.num_inchannels[j],
                                        3,
                                        2,
                                        1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(
                                        self.num_inchannels[j], momentum=BN_MOMENTUM
                                    ),
                                    nn.ReLU(inplace=True),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = 0
            # y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(0, self.num_branches):
                if j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode="bilinear",
                        align_corners=True,
                    )
                elif i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


class HighResolutionNet_Classification(nn.Module):
    def __init__(self, num_classes, backbone):
        super(HighResolutionNet_Classification, self).__init__()
        num_filters = {
            "hrnetv2_w18": [18, 36, 72, 144],
            "hrnetv2_w32": [32, 64, 128, 256],
            "hrnetv2_w48": [48, 96, 192, 384],
        }[backbone]
        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)

        pre_stage_channels = [Bottleneck.expansion * 64]
        num_channels = [num_filters[0], num_filters[1]]
        self.transition1 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            1, 2, BasicBlock, [4, 4], num_channels, num_channels
        )

        num_channels = [num_filters[0], num_filters[1], num_filters[2]]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            4, 3, BasicBlock, [4, 4, 4], num_channels, num_channels
        )

        num_channels = [num_filters[0], num_filters[1], num_filters[2], num_filters[3]]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            3, 4, BasicBlock, [4, 4, 4, 4], num_channels, num_channels
        )

        self.pre_stage_channels = pre_stage_channels

        self.incre_modules, self.downsamp_modules, self.final_layer = self._make_head(
            Bottleneck, pre_stage_channels
        )

        self.classifier = nn.Linear(2048, num_classes)

    def _make_layer(self, block, inplanes, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_transition_layer(self, num_inchannels, num_channels):

        num_branches_pre = len(num_inchannels)
        num_branches_cur = len(num_channels)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels[i] != num_inchannels[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[i], num_channels[i], 3, 1, 1, bias=False
                            ),
                            nn.BatchNorm2d(num_channels[i], momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = [
                    nn.Sequential(
                        nn.Conv2d(
                            num_inchannels[-1], num_channels[i], 3, 2, 1, bias=False
                        ),
                        nn.BatchNorm2d(num_channels[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True),
                    )
                ]
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_stage(
        self,
        num_modules,
        num_branches,
        block,
        num_blocks,
        num_inchannels,
        num_channels,
        multi_scale_output=True,
    ):
        modules = []
        for i in range(num_modules):
            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    multi_scale_output,
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def _make_head(self, block, pre_stage_channels):
        head_channels = [32, 64, 128, 256]

        incre_modules = []
        for i, channels in enumerate(pre_stage_channels):
            incre_module = self._make_layer(
                block, channels, head_channels[i], 1, stride=1
            )
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)

        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = head_channels[i] * block.expansion
            out_channels = head_channels[i + 1] * block.expansion

            downsamp_module = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
            )

            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=head_channels[3] * block.expansion,
                out_channels=2048,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(2048, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

        return incre_modules, downsamp_modules, final_layer

    def forward(self, x):
        # The stem part
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        # The stage 1 to 2
        x_list = []
        for i in range(2):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        # The stage 2 to 3
        x_list = []
        for i in range(3):
            if self.transition2[i] is not None:
                if i < 2:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        # The stage 3 to 4
        x_list = []
        for i in range(4):
            if self.transition3[i] is not None:
                if i < 3:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        y = self.incre_modules[0](y_list[0])
        for i in range(len(self.downsamp_modules)):
            y = self.incre_modules[i + 1](y_list[i + 1]) + self.downsamp_modules[i](y)

        y = self.final_layer(y)

        if torch._C._get_tracing_state():
            y = y.flatten(start_dim=2).mean(dim=2)
        else:
            y = F.avg_pool2d(y, kernel_size=y.size()[2:]).view(y.size(0), -1)

        y = self.classifier(y)

        return y


def hrnet_classification(pretrained=False, backbone="hrnetv2_w18"):
    model = HighResolutionNet_Classification(num_classes=1000, backbone=backbone)
    if pretrained:
        model_urls = {
            "hrnetv2_w18": "https://github.com/bubbliiiing/hrnet-pytorch/releases/download/v1.0/hrnetv2_w18_imagenet_pretrained.pth",
            "hrnetv2_w32": "https://github.com/bubbliiiing/hrnet-pytorch/releases/download/v1.0/hrnetv2_w32_imagenet_pretrained.pth",
            "hrnetv2_w48": "https://github.com/bubbliiiing/hrnet-pytorch/releases/download/v1.0/hrnetv2_w48_imagenet_pretrained.pth",
        }
        state_dict = load_state_dict_from_url(
            model_urls[backbone], model_dir="./model_data"
        )
        model.load_state_dict(state_dict)

    return model


class Down_Sampling(nn.Module):
    def __init__(self, in_channels, out_channles):
        super(Down_Sampling, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channles, kernel_size=3, stride=2, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channles)
        self.conv2 = nn.Conv2d(out_channles, out_channles, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channles)
        self.relu = nn.ReLU()

        self.CA = ChannelAttention()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = self.CA(out)
        out = self.relu(out)

        return out


class Down_T_Sampling(nn.Module):
    def __init__(self, in_channels, out_channles):
        super(Down_T_Sampling, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channles, kernel_size=3, stride=2, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channles)
        self.conv2 = nn.Conv2d(out_channles, out_channles, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channles)
        self.relu = nn.ReLU()

        self.TA = TransEncoder(
            out_channles, num_head=4, num_layer=6, num_patches=16, use_pos_embed=False
        )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = self.TA(out)
        out = self.relu(out)

        return out


class HRnet_Backbone(nn.Module):
    def __init__(self, backbone="hrnetv2_w18", pretrained=False):
        super(HRnet_Backbone, self).__init__()
        self.model = hrnet_classification(backbone=backbone, pretrained=pretrained)
        del self.model.incre_modules
        del self.model.downsamp_modules
        del self.model.final_layer
        del self.model.classifier

        pre_channel1 = np.int_(self.model.pre_stage_channels[0])
        self.DSP1 = Down_Sampling(in_channels=64 * 4, out_channles=pre_channel1 * 2)
        pre_channel2 = np.int_(self.model.pre_stage_channels[1])
        self.DSP2 = Down_Sampling(
            in_channels=pre_channel2, out_channles=pre_channel2 * 2
        )
        pre_channel3 = np.int_(self.model.pre_stage_channels[2])
        self.DSP3 = Down_T_Sampling(
            in_channels=pre_channel3, out_channles=pre_channel3 * 2
        )

    def forward(self, x):
        # The stem part
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.conv2(x)
        x = self.model.bn2(x)
        x = self.model.relu(x)
        x = self.model.layer1(x)

        # The stage 1 to 2
        x_list = []
        for i in range(2):
            if self.model.transition1[i] is not None:
                x_list.append(self.model.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.model.stage2(x_list)

        _, c1, h1, w1 = x.shape
        _, c2, h2, w2 = y_list[-1].shape
        # down_sample1 = Down_Sampling(c1, c2//2)(x)
        down_sample1 = self.DSP1(x)
        y_list[-1] = torch.add(down_sample1, y_list[-1])

        # The stage 2 to 3
        x_list = []
        for i in range(3):
            if self.model.transition2[i] is not None:
                if i < 2:
                    x_list.append(self.model.transition2[i](y_list[i]))
                else:
                    x_list.append(self.model.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.model.stage3(x_list)

        _, c3, h3, w3 = down_sample1.shape
        # down_sample2 = Down_Sampling(c3, c3)(down_sample1)
        down_sample2 = self.DSP2(down_sample1)
        y_list[-1] = torch.add(down_sample2, y_list[-1])

        # The stage 3 to 4
        x_list = []
        for i in range(4):
            if self.model.transition3[i] is not None:
                if i < 3:
                    x_list.append(self.model.transition3[i](y_list[i]))
                else:
                    x_list.append(self.model.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.model.stage4(x_list)

        _, c4, h4, w4 = down_sample2.shape
        # down_sample3 = Down_Sampling(c4, c4)(down_sample2)
        down_sample3 = self.DSP3(down_sample2)
        y_list[-1] = torch.add(down_sample3, y_list[-1])

        return y_list


class HRnet(nn.Module):
    def __init__(self, num_classes=2, backbone="hrnetv2_w48", pretrained=False):
        super(HRnet, self).__init__()
        self.backbone = HRnet_Backbone(backbone=backbone, pretrained=pretrained)

        last_inp_channels = np.int_(np.sum(self.backbone.model.pre_stage_channels))

        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        inp_channel1 = np.int_(self.backbone.model.pre_stage_channels[-1])
        self.CA1 = CoordAtt(inp=inp_channel1, oup=inp_channel1)
        inp_channel2 = np.int_(np.sum(self.backbone.model.pre_stage_channels[:-3:-1]))
        self.CA2 = CoordAtt(inp=inp_channel2, oup=inp_channel2)
        inp_channel3 = np.int_(np.sum(self.backbone.model.pre_stage_channels[:-4:-1]))
        self.CA3 = CoordAtt(inp=inp_channel3, oup=inp_channel3)

        self.Tconv3 = nn.ConvTranspose2d(
            inp_channel1, inp_channel1, kernel_size=4, stride=2, padding=1
        )
        self.Tconv2 = nn.ConvTranspose2d(
            inp_channel2, inp_channel2, kernel_size=4, stride=2, padding=1
        )
        self.Tconv1 = nn.ConvTranspose2d(
            inp_channel3, inp_channel3, kernel_size=4, stride=2, padding=1
        )

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        x = self.backbone(inputs)

        # Upsampling + attention
        # x3_c, x3_h, x3_w = x[3].size(1), x[3].size(2), x[3].size(3)
        # x3 = CoordAtt(x3_c, x3_c)(x[3])
        x3 = self.CA1(x[3])

        x2_c, x2_h, x2_w = x[2].size(1), x[2].size(2), x[2].size(3)
        # x3 = F.interpolate(x3, size=(x2_h, x2_w), mode='bilinear', align_corners=True)
        x3 = self.Tconv3(x3)
        out3 = torch.cat([x[2], x3], 1)

        x1_c, x1_h, x1_w = x[1].size(1), x[1].size(2), x[1].size(3)
        # x2 = CoordAtt(out3.size(1), out3.size(1))(out3)
        x2 = self.CA2(out3)
        # x2 = F.interpolate(x2, size=(x1_h, x1_w), mode='bilinear', align_corners=True)
        x2 = self.Tconv2(x2)
        out2 = torch.cat([x[1], x2], 1)

        x0_c, x0_h, x0_w = x[0].size(1), x[0].size(2), x[0].size(3)
        # x1 = CoordAtt(out2.size(1), out2.size(1))(out2)
        x1 = self.CA3(out2)
        # x1 = F.interpolate(x1, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x1 = self.Tconv1(x1)
        out = torch.cat([x[0], x1], 1)

        x = self.last_layer(out)
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=True)
        return x
