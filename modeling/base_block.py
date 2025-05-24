import torch
import torch.nn as nn
from typing import Optional, Sequence, Tuple, Union

from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock
from monai.networks.blocks.dynunet_block import get_conv_layer
from monai.networks.blocks.dynunet_block import get_padding, get_output_padding
from monai.networks.blocks import ADN
from monai.networks.layers.convutils import same_padding
from monai.networks.layers.factories import Act, Norm


import torch
import torch.nn as nn


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class conv_block_1(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class conv_block_2(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=2, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=2, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=2, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=2, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class conv_block_3(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super(conv_block_3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True) if act == 'relu' else nn.Mish(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True) if act == 'relu' else nn.Mish(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class conv_block_5(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_5, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class conv_block_7(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_7, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class conv_block_9(nn.Module):
    def __init_(self, ch_in, ch_out):
        super(conv_block_9, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=9, stride=1, padding=4, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=9, stride=1, padding=4, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class conv_3_1(nn.Module):
    def __init__(self, ch_in, ch_out, act):
        super(conv_3_1, self).__init__()

        self.conv_3 = nn.Sequential(
            conv_block_3(ch_in, ch_out),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True) if act == 'relu' else nn.Mish(inplace=True))
        self.conv_7 = nn.Sequential(
            conv_block_7(ch_in, ch_out),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True) if act == 'relu' else nn.Mish(inplace=True))

        self.conv = nn.Sequential(
            nn.Conv3d(ch_out * 2, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True) if act == 'relu' else nn.Mish(inplace=True))

    def forward(self, x):
        x3 = self.conv_3(x)
        x7 = self.conv_7(x)

        x = torch.cat((x3, x7), dim=1)
        x = self.conv(x)

        return x


class dilated_conv_block_5(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super(dilated_conv_block_5, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True) if act == 'relu' else nn.Mish(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True) if act == 'relu' else nn.Mish(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class dilated_conv_block_7(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super(dilated_conv_block_7, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=3, dilation=3, bias=True),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True) if act == 'relu' else nn.Mish(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=3, dilation=3, bias=True),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True) if act == 'relu' else nn.Mish(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class DualDilatedBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super(DualDilatedBlock, self).__init__()

        self.conv_3 = conv_block_3(ch_in, ch_out, act)
        self.conv_5 = dilated_conv_block_5(ch_in, ch_out, act)
        self.conv_7 = dilated_conv_block_7(ch_in, ch_out, act)

        self.conv = nn.Sequential(
            nn.Conv3d(ch_out * 3, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True) if act == 'relu' else nn.Mish(inplace=True))

    def forward(self, x):
        x3 = self.conv_3(x)
        x5 = self.conv_5(x)
        x7 = self.conv_7(x)

        x = torch.cat((x3, x5, x7), dim=1)
        x = self.conv(x)

        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out, bilinear=False):
        super(up_conv, self).__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv3d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm3d(ch_in),
                nn.ReLU(inplace=True)
            )
        else:
            self.up = nn.ConvTranspose3d(ch_in, ch_out, kernel_size=2, stride=2)

    def forward(self, x):

        x = self.up(x)

        return x
    
class conv_block(nn.Module):
    """
    Convolutional block with one convolutional layer
    and ReLU activation function.
    """

    def __init__(self, ch_in, ch_out, kernel_size, padding=1, bias=False):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=kernel_size, stride=1, padding=padding, bias=bias),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class conv_trans_block(nn.Module):
    """
    Convolutional block with one convolutional layer
    and ReLU activation function.
    """

    def __init__(self, ch_in, ch_out, kernel_size, padding=1, bias=False):
        super(conv_trans_block, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose3d(ch_in, ch_out, kernel_size=kernel_size, stride=1, padding=padding, bias=bias),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class MultiScaleConvolution(nn.Sequential):
    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            strides: Union[Sequence[int], int] = 1,
            kernel_size: Union[Sequence[int], int] = 3,
            adn_ordering: str = "NDA",
            act: Optional[Union[Tuple, str]] = "PRELU",
            norm: Optional[Union[Tuple, str]] = "INSTANCE",
            dropout: Optional[Union[Tuple, str, float]] = None,
            dropout_dim: Optional[int] = 1,
            dilation: Union[Sequence[int], int] = 1,
            groups: int = 1,
            bias: bool = True,
            conv_only: bool = False,
            is_transposed: bool = False,
            padding: Optional[Union[Sequence[int], int]] = None,
            output_padding: Optional[Union[Sequence[int], int]] = None,
            dimensions: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dimensions = spatial_dims if dimensions is None else dimensions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_transposed = is_transposed

        if padding is None:
            padding = same_padding(kernel_size, dilation)

        conv: nn.Module
        conv = MultiScaleConv(
            ch_in=in_channels,
            ch_out=out_channels,
        )
        self.add_module("conv", conv)

        if conv_only:
            return
        if act is None and norm is None and dropout is None:
            return
        self.add_module(
            "adn",
            ADN(
                ordering=adn_ordering,
                in_channels=out_channels,
                act=act,
                norm=norm,
                norm_dim=self.dimensions,
                dropout=dropout,
                dropout_dim=dropout_dim,
            ),
        )


def get_multi_conv_layer(
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int] = 3,
        stride: Union[Sequence[int], int] = 1,
        act: Optional[Union[Tuple, str]] = Act.PRELU,
        norm: Optional[Union[Tuple, str]] = Norm.INSTANCE,
        dropout: Optional[Union[Tuple, str, float]] = None,
        bias: bool = False,
        conv_only: bool = True,
        is_transposed: bool = False,
):
    padding = get_padding(kernel_size, stride)
    output_padding = None
    if is_transposed:
        output_padding = get_output_padding(kernel_size, stride, padding)
    return MultiScaleConvolution(
        spatial_dims,
        in_channels,
        out_channels,
        strides=stride,
        kernel_size=kernel_size,
        act=act,
        norm=norm,
        dropout=dropout,
        bias=bias,
        conv_only=conv_only,
        is_transposed=is_transposed,
        padding=padding,
        output_padding=output_padding,
    )


class MultiScaleConv(nn.Module):
    """
      Multiscale convolutional block with 3 convolutional blocks
      with kernel size of 3x3, 5x5 and 7x7. Which is then concatenated
      and fed into a 1x1 convolutional block.
      """

    def __init__(self, ch_in, ch_out):
        super(MultiScaleConv, self).__init__()
        self.conv3x3x3 = conv_block(ch_in=ch_in, ch_out=ch_out, kernel_size=3, padding=1)
        self.conv5x5x5 = conv_block(ch_in=ch_in, ch_out=ch_out, kernel_size=5, padding=2)
        self.conv7x7x7 = conv_block(ch_in=ch_in, ch_out=ch_out, kernel_size=7, padding=3)
        self.conv1x1x1 = conv_block(ch_in=ch_out * 3, ch_out=ch_out, kernel_size=1, padding=0)

    def forward(self, x):
        x1 = self.conv3x3x3(x)
        x2 = self.conv5x5x5(x)
        x3 = self.conv7x7x7(x)
        comb = torch.cat((x1, x2, x3), 1)
        out = self.conv1x1x1(comb)
        return out

class MultiUnetBasicBlock(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            multiS_conv=True,
            act='relu',
    ):
        super().__init__()

        self.cov_ = conv_3_1(ch_in=in_channels, ch_out=out_channels, act=act) if multiS_conv else DualDilatedBlock(
            ch_in=in_channels, ch_out=out_channels, act=act)

    def forward(self, inp):
        out = self.cov_(inp)
        return out


class MultiUnetResBlock(UnetResBlock):
    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
            stride: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            norm_name=norm_name,
        )

        self.conv1 = get_multi_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            act=act_name,
            norm=norm_name,
            conv_only=False,
        )
        self.conv2 = get_multi_conv_layer(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            act=None,
            norm=None,
            conv_only=False,
        )

    def forward(self, inp):
        residual = inp
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if hasattr(self, "conv3"):
            residual = self.conv3(residual)
        if hasattr(self, "norm3"):
            residual = self.norm3(residual)
        out += residual
        out = self.lrelu(out)
        return out


class ModifiedUnetrUpBlock(nn.Module):
    """
    An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            upsample_kernel_size: Union[Sequence[int], int],
            act='relu',
            norm='instance',
            multiS_conv=True
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.act = act
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
            norm=norm
        )

        self.conv_block = MultiUnetBasicBlock(  # type: ignore
            out_channels + out_channels,
            out_channels,
            act=act,
            multiS_conv=multiS_conv,
        )

    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out


class ModifiedUnetOutBlock(nn.Module):

    def __init__(
            self, spatial_dims: int, in_channels: int, out_channels: int,
            dropout: Optional[Union[Tuple, str, float]] = None
    ):
        super().__init__()
        self.conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            dropout=dropout,
            bias=True,
            act=None,
            norm=None,
            conv_only=False,
        )

    def forward(self, inp):
        return self.conv(inp)


class ModifiedUnetrPrUpBlock(nn.Module):
    """
    A projection upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            num_layer: int,
            kernel_size: Union[Sequence[int], int],
            stride: Union[Sequence[int], int],
            upsample_kernel_size: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            conv_block: bool = False,
            res_block: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            num_layer: number of upsampling blocks.
            kernel_size: convolution kernel size.
            stride: convolution stride.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
        """

        super().__init__()

        upsample_stride = upsample_kernel_size
        self.transp_conv_init = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )
        if conv_block:
            if res_block:
                self.blocks = nn.ModuleList(
                    [
                        nn.Sequential(
                            get_conv_layer(
                                spatial_dims,
                                out_channels,
                                out_channels,
                                kernel_size=upsample_kernel_size,
                                stride=upsample_stride,
                                conv_only=True,
                                is_transposed=True,
                            ),
                            UnetResBlock(
                                spatial_dims=spatial_dims,
                                in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                norm_name=norm_name,
                            ),
                        )
                        for _ in range(num_layer)
                    ]
                )
            else:
                self.blocks = nn.ModuleList(
                    [
                        nn.Sequential(
                            get_conv_layer(
                                spatial_dims,
                                out_channels,
                                out_channels,
                                kernel_size=upsample_kernel_size,
                                stride=upsample_stride,
                                conv_only=True,
                                is_transposed=True,
                            ),
                            UnetBasicBlock(
                                spatial_dims=spatial_dims,
                                in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                norm_name=norm_name,
                            ),
                        )
                        for _ in range(num_layer)
                    ]
                )
        else:
            self.blocks = nn.ModuleList(
                [
                    get_conv_layer(
                        spatial_dims,
                        out_channels,
                        out_channels,
                        kernel_size=upsample_kernel_size,
                        stride=upsample_stride,
                        conv_only=True,
                        is_transposed=True,
                    )
                    for _ in range(num_layer)
                ]
            )

    def forward(self, x):
        x = self.transp_conv_init(x)
        return x



class MultiUnetBasicBlock(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            multiS_conv=True,
            act='relu',
    ):
        super().__init__()

        self.cov_ = conv_3_1(ch_in=in_channels, ch_out=out_channels, act=act) if multiS_conv else DualDilatedBlock(
            ch_in=in_channels, ch_out=out_channels, act=act)

    def forward(self, inp):
        out = self.cov_(inp)
        return out


class MultiUnetResBlock(UnetResBlock):
    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
            stride: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            norm_name=norm_name,
        )

        self.conv1 = get_multi_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            act=act_name,
            norm=norm_name,
            conv_only=False,
        )
        self.conv2 = get_multi_conv_layer(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            act=None,
            norm=None,
            conv_only=False,
        )

    def forward(self, inp):
        residual = inp
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if hasattr(self, "conv3"):
            residual = self.conv3(residual)
        if hasattr(self, "norm3"):
            residual = self.norm3(residual)
        out += residual
        out = self.lrelu(out)
        return out


