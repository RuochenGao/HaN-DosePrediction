# FROM: https://github.com/GhTara/Dose_Prediction
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Sequence, Union, Tuple
import numpy as np
from monai.networks.nets.vit import ViT
from monai.networks.blocks.unetr_block import (
    UnetrBasicBlock,
    UnetrPrUpBlock,
    UnetrUpBlock,
)
from monai.utils import ensure_tuple_rep

from modeling.base_block import ModifiedUnetrUpBlock
import torch.nn.functional as F
import torch.nn as nn
import torch


class SingleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding):
        super(SingleConv, self).__init__()

        self.single_conv = nn.Sequential(
            nn.Conv3d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=True,
            ),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.single_conv(x)


class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, stride=1, bias=True),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=True)
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_ch, list_ch):
        super(Encoder, self).__init__()
        self.encoder_1 = nn.Sequential(
            SingleConv(in_ch, list_ch[1], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[1], list_ch[1], kernel_size=3, stride=1, padding=1),
        )
        self.encoder_2 = nn.Sequential(
            SingleConv(list_ch[1], list_ch[2], kernel_size=3, stride=2, padding=1),
            SingleConv(list_ch[2], list_ch[2], kernel_size=3, stride=1, padding=1),
        )
        self.encoder_3 = nn.Sequential(
            SingleConv(list_ch[2], list_ch[3], kernel_size=3, stride=2, padding=1),
            SingleConv(list_ch[3], list_ch[3], kernel_size=3, stride=1, padding=1),
        )
        self.encoder_4 = nn.Sequential(
            SingleConv(list_ch[3], list_ch[4], kernel_size=3, stride=2, padding=1),
            SingleConv(list_ch[4], list_ch[4], kernel_size=3, stride=1, padding=1),
        )
        self.encoder_5 = nn.Sequential(
            SingleConv(list_ch[4], list_ch[5], kernel_size=3, stride=2, padding=1),
            SingleConv(list_ch[5], list_ch[5], kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        out_encoder_1 = self.encoder_1(x)
        out_encoder_2 = self.encoder_2(out_encoder_1)
        out_encoder_3 = self.encoder_3(out_encoder_2)
        out_encoder_4 = self.encoder_4(out_encoder_3)
        out_encoder_5 = self.encoder_5(out_encoder_4)

        return [
            out_encoder_1,
            out_encoder_2,
            out_encoder_3,
            out_encoder_4,
            out_encoder_5,
        ]


class Decoder(nn.Module):
    def __init__(self, list_ch):
        super(Decoder, self).__init__()

        self.upconv_4 = UpConv(list_ch[5], list_ch[4])
        self.decoder_conv_4 = nn.Sequential(
            SingleConv(2 * list_ch[4], list_ch[4], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[4], list_ch[4], kernel_size=3, stride=1, padding=1),
        )
        self.upconv_3 = UpConv(list_ch[4], list_ch[3])
        self.decoder_conv_3 = nn.Sequential(
            SingleConv(2 * list_ch[3], list_ch[3], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[3], list_ch[3], kernel_size=3, stride=1, padding=1),
        )
        self.upconv_2 = UpConv(list_ch[3], list_ch[2])
        self.decoder_conv_2 = nn.Sequential(
            SingleConv(2 * list_ch[2], list_ch[2], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[2], list_ch[2], kernel_size=3, stride=1, padding=1),
        )
        self.upconv_1 = UpConv(list_ch[2], list_ch[1])
        self.decoder_conv_1 = nn.Sequential(
            SingleConv(2 * list_ch[1], list_ch[1], kernel_size=3, stride=1, padding=1)
        )

    def forward(self, out_encoder):
        out_encoder_1, out_encoder_2, out_encoder_3, out_encoder_4, out_encoder_5 = (
            out_encoder
        )

        out_decoder_4 = self.decoder_conv_4(
            torch.cat((self.upconv_4(out_encoder_5), out_encoder_4), dim=1)
        )
        out_decoder_3 = self.decoder_conv_3(
            torch.cat((self.upconv_3(out_decoder_4), out_encoder_3), dim=1)
        )
        out_decoder_2 = self.decoder_conv_2(
            torch.cat((self.upconv_2(out_decoder_3), out_encoder_2), dim=1)
        )
        out_decoder_1 = self.decoder_conv_1(
            torch.cat((self.upconv_1(out_decoder_2), out_encoder_1), dim=1)
        )

        return out_decoder_1


class BaseUNet(nn.Module):
    def __init__(self, in_ch, list_ch):
        super(BaseUNet, self).__init__()
        self.encoder = Encoder(in_ch, list_ch)
        self.decoder = Decoder(list_ch)

        # init
        self.initialize()

    @staticmethod
    def init_conv_IN(modules):
        for m in modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.InstanceNorm3d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def initialize(self):
        print("# random init encoder weight using nn.init.kaiming_uniform !")
        self.init_conv_IN(self.decoder.modules)
        print("# random init decoder weight using nn.init.kaiming_uniform !")
        self.init_conv_IN(self.encoder.modules)

    def forward(self, x):
        out_encoder = self.encoder(x)
        out_decoder = self.decoder(out_encoder)

        # Output is a list: [Output]
        return out_decoder


##############################
#        Encoder
##############################


class ViTEncoder(nn.Module):

    def __init__(
        self,
        in_channels: int,
        img_size: Union[Sequence[int], int],
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 12,
        proj_type: str = "conv",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
    ) -> None:

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.num_layers = num_layers
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(16, spatial_dims)
        self.feat_size = tuple(
            img_d // p_d for img_d, p_d in zip(img_size, self.patch_size)
        )
        self.hidden_size = hidden_size
        self.classification = False
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            proj_type=proj_type,
            classification=self.classification,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )

        self.skip1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.skip2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.skip3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.skip4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )

        self.proj_axes = (0, spatial_dims + 1) + tuple(
            d + 1 for d in range(spatial_dims)
        )
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x_in):
        ##############################
        # model a
        ##############################
        i = self.num_layers // 4
        z12, hidden_states_out = self.vit(x_in)
        # 16 x 128 x 128 x 128
        out_encoder_1 = self.skip1(x_in)
        z3 = hidden_states_out[i]
        # 32 x 64 x 64 x 64
        out_encoder_2 = self.skip2(self.proj_feat(z3))
        z6 = hidden_states_out[i * 2]
        # 64 x 32 x 32 x 32
        out_encoder_3 = self.skip3(self.proj_feat(z6))
        z9 = hidden_states_out[i * 3]
        # 128 x 16 x 16 x 16
        out_encoder_4 = self.skip4(self.proj_feat(z9))
        # 786 x 8 x 8 x 8
        out_encoder_5 = self.proj_feat(z12)

        return [
            out_encoder_1,
            out_encoder_2,
            out_encoder_3,
            out_encoder_4,
            out_encoder_5,
        ]


##############################
#        Decoder
##############################
class PyMSCDecoder(nn.Module):

    def __init__(
        self,
        feature_size: int = 16,
        hidden_size: int = 768,
        norm_name: Union[Tuple, str] = "instance",
        spatial_dims: int = 3,
        mode_multi: bool = False,
        act="relu",
        multiS_conv=True,
    ) -> None:
        super().__init__()

        self.decoder4 = (
            UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=hidden_size,
                out_channels=feature_size * 8,
                upsample_kernel_size=2,
                kernel_size=3,
                norm_name=norm_name,
            )
            if not mode_multi
            else ModifiedUnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=hidden_size,
                out_channels=feature_size * 8,
                upsample_kernel_size=2,
                act=act,
                multiS_conv=multiS_conv,
            )
        )

        self.decoder3 = (
            UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size * 8,
                out_channels=feature_size * 4,
                upsample_kernel_size=2,
                kernel_size=3,
                norm_name=norm_name,
            )
            if not mode_multi
            else ModifiedUnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size * 8,
                out_channels=feature_size * 4,
                upsample_kernel_size=2,
                act=act,
                multiS_conv=multiS_conv,
            )
        )

        self.decoder2 = (
            UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size * 4,
                out_channels=feature_size * 2,
                upsample_kernel_size=2,
                kernel_size=3,
                norm_name=norm_name,
            )
            if not mode_multi
            else ModifiedUnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size * 4,
                out_channels=feature_size * 2,
                upsample_kernel_size=2,
                act=act,
                multiS_conv=multiS_conv,
            )
        )

        self.decoder1 = (
            UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size * 2,
                out_channels=feature_size,
                upsample_kernel_size=2,
                kernel_size=3,
                norm_name=norm_name,
            )
            if not mode_multi
            else ModifiedUnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size * 2,
                out_channels=feature_size,
                upsample_kernel_size=2,
                act=act,
                multiS_conv=multiS_conv,
            )
        )

    def forward(self, out_encoder):
        out_encoder_1, out_encoder_2, out_encoder_3, out_encoder_4, out_encoder_5 = (
            out_encoder
        )
        dec4 = self.decoder4(out_encoder_5, out_encoder_4)
        dec3 = self.decoder3(dec4, out_encoder_3)
        dec2 = self.decoder2(dec3, out_encoder_2)
        dec1 = self.decoder1(dec2, out_encoder_1)

        return [dec1, dec2, dec3, dec4]


##############################
#        Generator
##############################
class MainSubsetModel(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        img_size,
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 12,
        conv_block: bool = True,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        mode_multi_dec=False,
        act="relu",
        multiS_conv=True,
    ):
        super().__init__()

        # ----- Encoder part ----- #
        self.encoder = ViTEncoder(
            in_channels=in_ch,
            img_size=img_size,
            # 16 => 4
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            proj_type="perceptron",
            norm_name="instance",
            res_block=res_block,
            conv_block=conv_block,
            dropout_rate=dropout_rate,
        )

        # ----- Decoder part ----- #
        if True:
            self.decoder = PyMSCDecoder(
                feature_size=feature_size,
                hidden_size=hidden_size,
                mode_multi=mode_multi_dec,
                act=act,
                multiS_conv=multiS_conv,
            )

        def to_out(in_feature):
            return nn.Sequential(
                nn.Conv3d(in_feature, out_ch, kernel_size=1, padding=0, bias=True),
                # nn.Tanh(),
                # nn.Sigmoid()
            )

        self.dose_convertors = nn.ModuleList([to_out(feature_size)])
        # depth: 4
        for i in range(1, 4):
            self.dose_convertors.append(to_out(feature_size * np.power(2, i)))
        self.out = nn.Sequential(
            nn.Conv3d(feature_size, out_ch, kernel_size=1, padding=0, bias=True),
            # nn.Tanh(),
            # nn.Sigmoid()
        )

    def update_config(self, config_hparam):
        self.encoder.hidden_size = config_hparam["hidden_size"]
        self.encoder.num_layers = config_hparam["hidden_size"]

    def forward(self, x):
        out_encoder = self.encoder(x)

        out_decoders = self.decoder(out_encoder)
        outputs = []
        for out_dec, convertor in zip(out_decoders, self.dose_convertors):
            outputs.append(convertor(out_dec))

        return outputs


##############################
#        Model
##############################
class Pyfer(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        list_ch_A,
        feature_size=16,
        img_size=(128, 128, 128),
        num_layers=8,  # 4, 8, 12
        num_heads=6,  # 3, 6, 12
        act="mish",
        mode_multi_dec=True,
        multiS_conv=True,
    ):
        super(Pyfer, self).__init__()

        # list_ch records the number of channels in each stage, eg. [-1, 32, 64, 128, 256, 512]
        self.net_A = BaseUNet(in_ch, list_ch_A)
        self.net_B = MainSubsetModel(
            in_ch=in_ch + list_ch_A[1],
            out_ch=out_ch,
            feature_size=feature_size,
            img_size=img_size,
            num_layers=num_layers,  # 4, 8, 12
            num_heads=num_heads,  # 3, 6, 12
            act=act,
            mode_multi_dec=mode_multi_dec,
            multiS_conv=multiS_conv,
        )

        self.conv_out_A = nn.Conv3d(
            list_ch_A[1], out_ch, kernel_size=1, padding=0, bias=True
        )

    def forward(self, x):
        out_net_A = self.net_A(x)
        out_net_B = self.net_B(torch.cat((out_net_A, x), dim=1))

        output_A = self.conv_out_A(out_net_A)
        return [output_A, out_net_B[0]]


if __name__ == "__main__":

    model = Pyfer(
        in_ch=9,
        out_ch=1,
        list_ch_A=[-1, 16, 32, 64, 128, 256],
        feature_size=16,
        img_size=(128, 128, 128),
    )
    x = torch.rand(1, 9, 128, 128, 128)
    y1, y2 = model(x)
    print(y1.shape, y2.shape)
