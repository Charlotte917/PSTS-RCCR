from .VNet import VNet, MCNet3d_v1, MCNet3d_v2, Mine3d_v1, Mine3d_v2, VNet_v1
from .unet2d import UNet2D, Mine2dUNetV1


def net_factory(net_type="unet", in_chns=1, class_num=4, mode="train"):
    if net_type == "vnet" and mode == "train":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "mcnet3d_v1" and mode == "train":
        net = MCNet3d_v1(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "mcnet3d_v2" and mode == "train":
        net = MCNet3d_v2(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "mine3d_v1" and mode == "train":
        net = Mine3d_v1(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "mine3d_v2" and mode == "train":
        net = Mine3d_v2(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "unet2d" and mode == "train":
        net = UNet2D(n_channels=in_chns, n_classes=class_num, has_dropout=True).cuda()
    elif net_type == "mine2d_unet" and mode == "train":
        net = Mine2dUNetV1(n_channels=in_chns, n_classes=class_num, has_dropout=True).cuda()
    elif net_type == "vnet" and mode == "test":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "mcnet3d_v1" and mode == "test":
        net = MCNet3d_v1(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "mcnet3d_v2" and mode == "test":
        net = MCNet3d_v2(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "mine3d_v1" and mode == "test":
        net = Mine3d_v1(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "mine3d_v2" and mode == "test":
        net = Mine3d_v2(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "unet2d" and mode == "test":
        net = UNet2D(n_channels=in_chns, n_classes=class_num, has_dropout=False).cuda()
    elif net_type == "mine2d_unet" and mode == "test":
        net = Mine2dUNetV1(n_channels=in_chns, n_classes=class_num, has_dropout=False).cuda()
    return net
