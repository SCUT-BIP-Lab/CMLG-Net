# Demo Code for Paper:
# [Title]  - "Robust and Accurate Hand Gesture Authentication with Cross-Modality Local-Global Behavior Analysis"
# [Author] -Yufeng Zhang, Wenxiong Kang, Wenwei Song
# [Github] - https://github.com/SCUT-BIP-Lab/CMLG-Net.git

import torch
import torch.nn.functional as F
from model.PBNet import Model_PBNet
from loss.loss import AMSoftmax

def feedforward_demo(model, out_dim, is_train=False):

    if is_train:
        # AMSoftmax loss function
        # there are 143 identities in the training set
        criterian_r = AMSoftmax(in_feats=out_dim, n_classes=143)
        criterian_d = AMSoftmax(in_feats=out_dim, n_classes=143)
        criterian_f = AMSoftmax(in_feats=out_dim * 2, n_classes=143)

    data_rgb = torch.randn(2, 64, 3, 224, 224)  #batch, frame, channel, h, w
    data_dep = torch.randn(2, 64, 3, 224, 224)  # batch, frame, channel, h, w
    data_rgb = data_rgb.view(-1, 3, 224, 224)  #regard the frame as batch (TSN paradigm)
    data_dep = data_dep.view(-1, 3, 224, 224)  # regard the frame as batch (TSN paradigm)
    id_feature, x_r_norm, x_d_norm = model(data_rgb, data_dep) # feedforward

    if is_train is False:
        # Use the id_feature to calculate the EER when testing
        return id_feature
    else:
        # Use the id_feature, x_r_norm, and x_d_norm to calculate loss when training
        label = torch.randint(0, 143, size=(2,))
        loss_r, _ = criterian_r(x_r_norm, label)
        loss_d, _ = criterian_d(x_d_norm, label)
        loss_f, _ = criterian_f(id_feature, label)
        return loss_f + 0.7 * loss_r + 0.3 * loss_d

if __name__ == '__main__':
    # there are 64 frames in each dynamic hand gesture video
    frame_length = 64
    # the feature dim of last feature map (layer4) from ResNet18 is 512
    feature_dim = 512
    # the identity feature dim
    out_dim = 512

    model = Model_PBNet(frame_length=frame_length, feature_dim=feature_dim, out_dim=out_dim, sample_rate=sample_rate, clip_size=clip_size)
    # feedforward_test
    id_feature = feedforward_demo(model, out_dim, is_train=False)
    # feedforward_train
    loss = feedforward_demo(model, out_dim, is_train=True)
    print("Demo is finished!")

