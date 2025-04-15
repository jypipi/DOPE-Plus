"""
NVIDIA from jtremblay@gmail.com
"""

# Networks
import torch
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.models as models
import timm
import torch.nn.functional as F

import torchvision.utils as vutils

class DopeNetwork(nn.Module):
    def __init__(
        self,
        pretrained=False,
        numBeliefMap=9,
        numAffinity=16,
        stop_at_stage=6,  # number of stages to process (if less than total number of stages)
    ):
        super(DopeNetwork, self).__init__()

        self.stop_at_stage = stop_at_stage

        """
        vgg_full = models.vgg19(pretrained=False).features
        self.vgg = nn.Sequential()
        for i_layer in range(24):
            self.vgg.add_module(str(i_layer), vgg_full[i_layer])

        # Add some layers
        i_layer = 23
        self.vgg.add_module(
            str(i_layer), nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        )
        self.vgg.add_module(str(i_layer + 1), nn.ReLU(inplace=True))
        self.vgg.add_module(
            str(i_layer + 2), nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        )
        self.vgg.add_module(str(i_layer + 3), nn.ReLU(inplace=True))
        """

        # === 1. ViT backbone using timm ===
        self.vit = timm.create_model(
            'vit_base_patch16_224', pretrained=True, features_only=True
        )
        #self.vit = timm.create_model(
        #    'swin_base_patch4_window7_224',
        #    pretrained=True,
        #    features_only=True
        #)

        # Vit输出的最后一层特征大小（默认是 [B, 768, 14, 14]）
        vit_out_channels = self.vit.feature_info[-1]['num_chs']  # usually 768

        # === 2. 将Vit输出映射到128通道 ===
        self.feature_proj = nn.Sequential(
            nn.Conv2d(vit_out_channels, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        #---above is transformer

        # print('---Belief------------------------------------------------')
        # _2 are the belief map stages
        self.m1_2 = DopeNetwork.create_stage(128, numBeliefMap, True)
        self.m2_2 = DopeNetwork.create_stage(
            128 + numBeliefMap + numAffinity, numBeliefMap, False
        )
        self.m3_2 = DopeNetwork.create_stage(
            128 + numBeliefMap + numAffinity, numBeliefMap, False
        )
        self.m4_2 = DopeNetwork.create_stage(
            128 + numBeliefMap + numAffinity, numBeliefMap, False
        )
        self.m5_2 = DopeNetwork.create_stage(
            128 + numBeliefMap + numAffinity, numBeliefMap, False
        )
        self.m6_2 = DopeNetwork.create_stage(
            128 + numBeliefMap + numAffinity, numBeliefMap, False
        )

        # print('---Affinity----------------------------------------------')
        # _1 are the affinity map stages
        self.m1_1 = DopeNetwork.create_stage(128, numAffinity, True)
        self.m2_1 = DopeNetwork.create_stage(
            128 + numBeliefMap + numAffinity, numAffinity, False
        )
        self.m3_1 = DopeNetwork.create_stage(
            128 + numBeliefMap + numAffinity, numAffinity, False
        )
        self.m4_1 = DopeNetwork.create_stage(
            128 + numBeliefMap + numAffinity, numAffinity, False
        )
        self.m5_1 = DopeNetwork.create_stage(
            128 + numBeliefMap + numAffinity, numAffinity, False
        )
        self.m6_1 = DopeNetwork.create_stage(
            128 + numBeliefMap + numAffinity, numAffinity, False
        )

    def forward(self, x):
        """Runs inference on the neural network"""

        #out1 = self.vgg(x)
        #--------------------
        #vutils.save_image(x[0], 'input_image.png')
        x = F.interpolate(x, size=(224, 224), mode='bilinear')
        #rint("x shape: ",x.shape)
        #vutils.save_image(x[0], 'output_image.png')
        feats = self.vit(x)
        out1 = self.feature_proj(feats[-1])  # 使用最后一层特征
        #out1 = self.feature_proj(out1)  # Conv 映射到 128 channels
        out1 = F.interpolate(out1, size=(50, 50), mode='bilinear', align_corners=False)
        #----------------

        out1_2 = self.m1_2(out1)
        out1_1 = self.m1_1(out1)

        if self.stop_at_stage == 1:
            return [out1_2], [out1_1]

        out2 = torch.cat([out1_2, out1_1, out1], 1)
        out2_2 = self.m2_2(out2)
        out2_1 = self.m2_1(out2)

        if self.stop_at_stage == 2:
            return [out1_2, out2_2], [out1_1, out2_1]

        out3 = torch.cat([out2_2, out2_1, out1], 1)
        out3_2 = self.m3_2(out3)
        out3_1 = self.m3_1(out3)

        if self.stop_at_stage == 3:
            return [out1_2, out2_2, out3_2], [out1_1, out2_1, out3_1]

        out4 = torch.cat([out3_2, out3_1, out1], 1)
        out4_2 = self.m4_2(out4)
        out4_1 = self.m4_1(out4)

        if self.stop_at_stage == 4:
            return [out1_2, out2_2, out3_2, out4_2], [out1_1, out2_1, out3_1, out4_1]

        out5 = torch.cat([out4_2, out4_1, out1], 1)
        out5_2 = self.m5_2(out5)
        out5_1 = self.m5_1(out5)

        if self.stop_at_stage == 5:
            return [out1_2, out2_2, out3_2, out4_2, out5_2], [
                out1_1,
                out2_1,
                out3_1,
                out4_1,
                out5_1,
            ]

        out6 = torch.cat([out5_2, out5_1, out1], 1)
        out6_2 = self.m6_2(out6)
        out6_1 = self.m6_1(out6)

        return [out1_2, out2_2, out3_2, out4_2, out5_2, out6_2], [
            out1_1,
            out2_1,
            out3_1,
            out4_1,
            out5_1,
            out6_1,
        ]

    @staticmethod
    def create_stage(in_channels, out_channels, first=False):
        """Create the neural network layers for a single stage."""

        model = nn.Sequential()
        mid_channels = 128
        if first:
            padding = 1
            kernel = 3
            count = 6
            final_channels = 512
        else:
            padding = 3
            kernel = 7
            count = 10
            final_channels = mid_channels

        # First convolution
        model.add_module(
            "0",
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=kernel, stride=1, padding=padding
            ),
        )

        # Middle convolutions
        i = 1
        while i < count - 1:
            model.add_module(str(i), nn.ReLU(inplace=True))
            i += 1
            model.add_module(
                str(i),
                nn.Conv2d(
                    mid_channels,
                    mid_channels,
                    kernel_size=kernel,
                    stride=1,
                    padding=padding,
                ),
            )
            i += 1

        # Penultimate convolution
        model.add_module(str(i), nn.ReLU(inplace=True))
        i += 1
        model.add_module(
            str(i), nn.Conv2d(mid_channels, final_channels, kernel_size=1, stride=1)
        )
        i += 1

        # Last convolution
        model.add_module(str(i), nn.ReLU(inplace=True))
        i += 1
        model.add_module(
            str(i), nn.Conv2d(final_channels, out_channels, kernel_size=1, stride=1)
        )
        i += 1

        return model
