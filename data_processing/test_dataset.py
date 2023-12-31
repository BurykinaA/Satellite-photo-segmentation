import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


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


if __name__ == "__main__":
    net = HRnet()
    x = torch.Tensor(1, 3, 512, 512)
    out = net(x)
    print(out.shape)


class TestDataset(Dataset):
    def __init__(self, root_img, patch_size=(512, 512)):
        self.image_name = root_img
        self.patch_size = patch_size
        self.image = Image.open(self.image_name).convert("RGB")
        self.post_transform = torchvision.transforms.ToTensor()

        self.image_patches = self._split_image_into_patches(self.image, self.patch_size)

        a = (self.image.size[0] + self.patch_size[0] - 1) // self.patch_size[0]
        b = (self.image.size[1] + self.patch_size[1] - 1) // self.patch_size[1]
        self.patches_count = a * b

    def print_mask(self, item):
        mask = Image.open(os.path.join(self.mask_dir, self.mask_names[item]))
        mask.show()

    def print_img(self, item):
        image = Image.open(os.path.join(self.image_dir, self.image_names[item]))
        image.show()

    def print_img_mask(self, item):
        image = Image.open(os.path.join(self.image_dir, self.image_names[item]))
        mask = Image.open(os.path.join(self.mask_dir, self.mask_names[item]))

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(image)
        axes[0].set_title("Image")
        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("Mask")
        plt.show()

    def print_mask_on_img(self, item):
        image_path = os.path.join(self.image_dir, self.image_names[item])
        mask_path = os.path.join(self.mask_dir, self.mask_names[item])
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize(image.size, Image.NEAREST)
        mask_np = np.array(mask)
        cmap = plt.cm.get_cmap("viridis", 2)

        plt.figure(figsize=(10, 5))
        plt.imshow(image)
        plt.imshow(mask_np, cmap=cmap, alpha=0.5, vmin=0, vmax=1)
        plt.title("Image with Overlayed Mask")
        plt.axis("off")
        plt.show()

    def _split_image_into_patches(self, image, patch_size):

        image_patches = []

        for i in range(0, image.width, patch_size[0]):
            for j in range(0, image.height, patch_size[1]):
                patch_image = image.crop((i, j, i + patch_size[0], j + patch_size[1]))

                image_patches.append(patch_image)

        # возвращает списки патчей в формате Image
        return image_patches

    def __getitem__(self, item):
        image_patch = np.array(self.image_patches[item])
        image_patch = self.post_transform(image_patch)

        return image_patch

    def __len__(self):
        # return len(self.image_names)
        return self.patches_count
