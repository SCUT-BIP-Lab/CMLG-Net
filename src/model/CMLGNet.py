# Code for Paper:
# [Title]  - "Robust and Accurate Hand Gesture Authentication With Cross-Modality Local-Global Behavior Analysis"
# [Author] - Yufeng Zhang, Wenwei Song, Wenxiong Kang
# [Github] - https://github.com/SCUT-BIP-Lab/CMLG-Net

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from src.loss.loss import AMSoftmax


class TSP(nn.Module):
    def __init__(self, in_channel, frames_len):
        super().__init__()
        self.frames_len = frames_len
        self.conv1 = nn.Conv3d(in_channel, in_channel, kernel_size=(2, 1, 1), stride=1, padding=(0, 0, 0), bias=True, groups=in_channel)
        self.conv2 = nn.Conv3d(in_channel, in_channel, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0), bias=True, groups=in_channel)
        self.conv3 = nn.Conv3d(in_channel, in_channel, kernel_size=(4, 1, 1), stride=1, padding=(0, 0, 0), bias=True, groups=in_channel)
        nn.init.constant_(self.conv1.weight, 0)
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.constant_(self.conv2.weight, 0)
        nn.init.constant_(self.conv2.bias, 0)
        nn.init.constant_(self.conv3.weight, 0)
        nn.init.constant_(self.conv3.bias, 0)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn1 = nn.BatchNorm3d(in_channel)
        self.bn2 = nn.BatchNorm3d(in_channel)
        self.bn3 = nn.BatchNorm3d(in_channel)
        self.bn = nn.BatchNorm2d(in_channel)
        self.down_conv = nn.Sequential(nn.Conv3d(in_channels=3 * in_channel, out_channels=in_channel, kernel_size=1, groups=in_channel),
                                       nn.BatchNorm3d(num_features=in_channel),
                                       nn.ReLU(inplace=True),
                                       )

    def channel_interlace(self, x1, x2, x3):
        x = torch.stack((x1, x2, x3), dim=2)
        batch, channel, stack, frame, h, w = x.shape[:]
        x = x.reshape(batch, -1, frame, h, w)
        x = self.down_conv(x)
        return x

    def forward(self, x):
        x = x.reshape(-1, self.frames_len, *x.shape[-3:])
        x = x.permute(0, 2, 1, 3, 4)
        x1 = self.bn1(self.conv1(F.pad(x, (0, 0, 0, 0) + (0, 1), mode='replicate')))
        x2 = self.bn2(self.conv2(F.pad(x, (0, 0, 0, 0) + (1, 1), mode='replicate')))
        x3 = self.bn3(self.conv3(F.pad(x, (0, 0, 0, 0) + (2, 1), mode='replicate')))
        x_t = self.channel_interlace(x1, x2, x3)
        x = x + x_t
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(-1, *x.shape[-3:])
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=20):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class SA(nn.Module):
    def __init__(self, dim_model, nhead=8, dim_ff=64, dropout=0.1):
        super(SA, self).__init__()
        self.self_attention = nn.MultiheadAttention(dim_model, nhead, dropout=dropout, batch_first=True)
        self.ff = FFN(dim_model, dim_ff, dropout)
        self.norm1 = nn.LayerNorm(dim_model)

    def forward(self, src):
        src_sa = self.self_attention(src, src, src)[0]
        src = self.norm1(src + src_sa)
        src_ff = self.ff(src)  # FFN

        return src_ff


class CA(nn.Module):
    def __init__(self, dim_model, nhead=8, dim_ff=64, dropout=0.1):
        super(CA, self).__init__()
        self.crs_attention1 = nn.MultiheadAttention(dim_model, nhead, dropout=dropout, batch_first=True)
        self.ff1 = FFN(dim_model, dim_ff, dropout)
        self.norm11 = nn.LayerNorm(dim_model)
        self.norm12 = nn.LayerNorm(dim_model)

        self.crs_attention2 = nn.MultiheadAttention(dim_model, nhead, dropout=dropout, batch_first=True)
        self.ff2 = FFN(dim_model, dim_ff, dropout)
        self.norm21 = nn.LayerNorm(dim_model)
        self.norm22 = nn.LayerNorm(dim_model)

    def forward(self, src1, src2):
        src1_cross = self.crs_attention1(query=src1, key=src2, value=src2)[0]
        src2_cross = self.crs_attention2(query=src2, key=src1, value=src1)[0]
        src1 = self.norm11(src1 + src1_cross)
        src2 = self.norm21(src2 + src2_cross)
        src1_ff = self.ff1(src1)  # FFN
        src2_ff = self.ff2(src2)  # FFN
        return src1_ff, src2_ff


class FFN(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)

        return x


class CMTNL(torch.nn.Module):
    def __init__(self, dim, reduce_factor=2, nheads=8, dropout=0.1, layers=1, n_pos=200):
        super().__init__()
        dim_ff = dim // reduce_factor
        self.pos_1 = PositionalEncoding(dim, n_position=n_pos)
        self.pos_2 = PositionalEncoding(dim, n_position=n_pos)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.sa1 = SA(dim, nheads, dim_ff, dropout)
        self.sa2 = SA(dim, nheads, dim_ff, dropout)
        self.ca_list = nn.ModuleList([CA(dim, nheads, dim_ff, dropout) for _ in range(layers)])
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, feat1, feat2):
        # (N, L, E),where L is the target sequence length, N is the batch size, E is the embedding dimension.
        feat1 = self.layer_norm(self.dropout(self.pos_1(feat1)))
        feat2 = self.layer_norm(self.dropout(self.pos_2(feat2)))
        feat1_sa = self.sa1(feat1)
        feat2_sa = self.sa2(feat2)
        for ca in self.ca_list:
            feat1, feat2 = ca(feat1_sa, feat2_sa)

        return feat1, feat2, feat1_sa, feat2_sa


class Model_CMLGNet(torch.nn.Module):
    def __init__(self, conf):
        super(Model_CMLGNet, self).__init__()

        self.frame_length = 64  # there are 64 frames in each dynamic hand gesture video
        self.frame_size = 224
        self.feature_dim = 512
        self.out_dim = 512  # the feature dim of the two branches

        self.R_Branch = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        self.D_Branch = torchvision.models.resnet18(weights='IMAGENET1K_V1')

        self.R_Branch.fc = nn.Linear(in_features=self.feature_dim, out_features=self.out_dim)
        self.D_Branch.fc = nn.Linear(in_features=self.feature_dim, out_features=self.out_dim)

        # temp pyramid
        self.tsp_r = TSP(64, self.frame_length)
        self.tsp_d = TSP(64, self.frame_length)


        self.fc = nn.Linear(1024, 1024)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.cmtnl = CMTNL(dim=512, reduce_factor=2, nheads=8, dropout=0.1, layers=1, n_pos=512)

        # loss function
        if conf["mode"] == "train":
            self.criterian_meg = AMSoftmax(in_feats=self.feature_dim * 2, n_classes=int(conf["classes_num"]))
            self.criterian_rgb = AMSoftmax(in_feats=self.feature_dim, n_classes=int(conf["classes_num"]))
            self.criterian_dep = AMSoftmax(in_feats=self.feature_dim, n_classes=int(conf["classes_num"]))

    def forward(self, data, label=None):
        fis = {}
        data_rgb = data['RGB']
        data_dep = data['Depth']
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            data_rgb = data_rgb.cuda()
            data_dep = data_dep.cuda()
            if label is not None:
                label = label.cuda()


        data_rgb = data_rgb.view(-1, 3, self.frame_size, self.frame_size).contiguous()
        data_dep = data_dep.view(-1, 3, self.frame_size, self.frame_size).contiguous()

        # R-Branch
        x_r = self.R_Branch.conv1(data_rgb)
        x_r = self.tsp_r1(x_r)
        x_r = self.R_Branch.bn1(x_r)
        x_r = self.R_Branch.relu(x_r)
        x_r = self.R_Branch.maxpool(x_r)
        for i in range(0, 4):
            layer_name = "layer" + str(i + 1)
            layer = getattr(self.R_Branch, layer_name)
            x_r = layer(x_r)
        x_r = self.avgpool(x_r)
        x_r = torch.flatten(x_r, 1)
        x_r = self.R_Branch.fc(x_r)
        x_r = x_r.view(-1, self.frame_length, self.out_dim)
        x_r_d = x_r.detach()
        x_r = torch.mean(x_r, dim=1, keepdim=False)
        x_r_norm = torch.div(x_r, torch.norm(x_r, p=2, dim=1, keepdim=True).clamp(min=1e-12))

        # D-Branch
        x_d = self.D_Branch.conv1(data_dep)
        x_d = self.tsp_d(x_d)
        x_d = self.D_Branch.bn1(x_d)
        x_d = self.D_Branch.relu(x_d)
        x_d = self.D_Branch.maxpool(x_d)
        for i in range(0, 4):
            layer_name = "layer" + str(i + 1)
            layer = getattr(self.D_Branch, layer_name)
            x_d = layer(x_d)
        x_d = self.avgpool(x_d)
        x_d = torch.flatten(x_d, 1)
        x_d = self.D_Branch.fc(x_d)
        x_d = x_d.view(-1, self.frame_length, self.out_dim)
        x_d_d = x_d.detach()
        x_d = torch.mean(x_d, dim=1, keepdim=False)
        x_d_norm = torch.div(x_d, torch.norm(x_d, p=2, dim=1, keepdim=True).clamp(min=1e-12))

        # CMTNL
        x_r_cat, x_d_cat, x_r_sa, x_d_sa = self.cmtnl(x_r_d, x_d_d)
        x_r_cat = torch.mean(x_r_cat, dim=1, keepdim=False)  # batch, 512
        x_d_cat = torch.mean(x_d_cat, dim=1, keepdim=False)  # batch, 512
        x_r_norm_cat = torch.div(x_r_cat, torch.norm(x_r_cat, p=2, dim=1, keepdim=True).clamp(min=1e-12))  # normalization
        x_d_norm_cat = torch.div(x_d_cat, torch.norm(x_d_cat, p=2, dim=1, keepdim=True).clamp(min=1e-12))  # normalization
        id_feature = torch.cat((x_r_norm_cat, x_d_norm_cat), dim=1)

        if label is None:
            fis["id_feature"] = id_feature
            return fis
        else:
            loss_meg, _ = self.criterian_meg(id_feature, label)
            loss_rgb, _ = self.criterian_rgb(x_r_norm, label)
            loss_dep, _ = self.criterian_dep(x_d_norm, label)
            fis["loss"] = loss_meg + loss_rgb + loss_dep

            return fis
