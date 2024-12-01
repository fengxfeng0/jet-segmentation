import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import layers

class MobileHDC(nn.Module):
    def __init__(self,cls_num=2):
        super(MobileHDC, self).__init__()

        self.numclass = cls_num

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(),
        )

        self.layer2 = nn.Sequential(*layers.get_inverted_residual_block_arr(32, 16, 1, 1, 1))

        self.layer3 = nn.Sequential(*layers.get_inverted_residual_block_arr(16, 24, 6, 2, 2))

        self.layer4 = nn.Sequential(*layers.get_inverted_residual_block_arr(24, 32, 6, 2, 3))

        self.layer5 = nn.Sequential(*layers.get_inverted_residual_block_arr(32, 64, 6, 2, 4))

        self.layer6 = nn.Sequential(*layers.get_inverted_residual_block_arr(64, 96, 6, 1, 3))

        self.layer7 = nn.Sequential(*layers.get_inverted_residual_block_arr(96, 160, 6, 1, 3))

        self.dilated_layer1 = layers.InvertedResidual(160, 256, t=6, s=1, dilation=2)

        self.dilated_layer2 = layers.InvertedResidual(256, 256, t=6, s=1, dilation=4)

        self.dilated_layer3 = layers.InvertedResidual(256, 256, t=6, s=1, dilation=4)

        self.dilated_layer4 = layers.InvertedResidual(256, 256, t=6, s=1, dilation=7)

        # ===================decoder===========================
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(160, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.cat_conv = nn.Sequential(
            nn.Conv2d(256 + 256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )

        self.final_layer = nn.Conv2d(512,  self.numclass, 1)



    def forward(self, x):

        # h = x.size()[2]
        # w = x.size()[3]

        h = 512
        w = 512

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        low_level_features = x

        x = self.dilated_layer1(x)
        x = self.dilated_layer2(x)
        x = self.dilated_layer3(x)
        x = self.dilated_layer4(x)


        low_level_features = self.shortcut_conv(low_level_features)
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear',
                          align_corners=False)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        x = self.final_layer(x)

        output = F.upsample(x, size=(h, w), mode='bilinear', align_corners=False)

        return output
