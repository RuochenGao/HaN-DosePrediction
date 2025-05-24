# FROM: MONAI MedNeXt implementation
import torch
import torch.nn as nn
from collections.abc import Sequence


def get_conv_layer(spatial_dim: int = 3, transpose: bool = False):
    if spatial_dim == 2:
        return nn.ConvTranspose2d if transpose else nn.Conv2d
    else:  # spatial_dim == 3
        return nn.ConvTranspose3d if transpose else nn.Conv3d


class MedNeXtBlock(nn.Module):
    """
    MedNeXtBlock class for the MedNeXt model.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        expansion_ratio (int): Expansion ratio for the block. Defaults to 4.
        kernel_size (int): Kernel size for convolutions. Defaults to 7.
        use_residual_connection (int): Whether to use residual connection. Defaults to True.
        norm_type (str): Type of normalization to use. Defaults to "group".
        dim (str): Dimension of the input. Can be "2d" or "3d". Defaults to "3d".
        global_resp_norm (bool): Whether to use global response normalization. Defaults to False.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_ratio: int = 4,
        kernel_size: int = 7,
        use_residual_connection: int = True,
        norm_type: str = "group",
        dim="3d",
        global_resp_norm=False,
    ):

        super().__init__()

        self.do_res = use_residual_connection

        self.dim = dim
        conv = get_conv_layer(spatial_dim=2 if dim == "2d" else 3)
        global_resp_norm_param_shape = (1,) * (2 if dim == "2d" else 3)
        # First convolution layer with DepthWise Convolutions
        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=in_channels,
        )

        # Normalization Layer. GroupNorm is used by default.
        if norm_type == "group":
            self.norm = nn.GroupNorm(num_groups=in_channels, num_channels=in_channels)  # type: ignore
        elif norm_type == "layer":
            self.norm = nn.LayerNorm(
                normalized_shape=[in_channels] + [kernel_size] * (2 if dim == "2d" else 3)  # type: ignore
            )
        # Second convolution (Expansion) layer with Conv3D 1x1x1
        self.conv2 = conv(
            in_channels=in_channels, out_channels=expansion_ratio * in_channels, kernel_size=1, stride=1, padding=0
        )

        # GeLU activations
        self.act = nn.GELU()

        # Third convolution (Compression) layer with Conv3D 1x1x1
        self.conv3 = conv(
            in_channels=expansion_ratio * in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0
        )

        self.global_resp_norm = global_resp_norm
        if self.global_resp_norm:
            global_resp_norm_param_shape = (1, expansion_ratio * in_channels) + global_resp_norm_param_shape
            self.global_resp_beta = nn.Parameter(torch.zeros(global_resp_norm_param_shape), requires_grad=True)
            self.global_resp_gamma = nn.Parameter(torch.zeros(global_resp_norm_param_shape), requires_grad=True)

    def forward(self, x):
        """
        Forward pass of the MedNeXtBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x1 = x
        x1 = self.conv1(x1)
        x1 = self.act(self.conv2(self.norm(x1)))

        if self.global_resp_norm:
            # gamma, beta: learnable affine transform parameters
            # X: input of shape (N,C,H,W,D)
            if self.dim == "2d":
                gx = torch.norm(x1, p=2, dim=(-2, -1), keepdim=True)
            else:
                gx = torch.norm(x1, p=2, dim=(-3, -2, -1), keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
            x1 = self.global_resp_gamma * (x1 * nx) + self.global_resp_beta + x1
        x1 = self.conv3(x1)
        if self.do_res:
            x1 = x + x1
        return x1


class MedNeXtDownBlock(MedNeXtBlock):
    """
    MedNeXtDownBlock class for downsampling in the MedNeXt model.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        expansion_ratio (int): Expansion ratio for the block. Defaults to 4.
        kernel_size (int): Kernel size for convolutions. Defaults to 7.
        use_residual_connection (bool): Whether to use residual connection. Defaults to False.
        norm_type (str): Type of normalization to use. Defaults to "group".
        dim (str): Dimension of the input. Can be "2d" or "3d". Defaults to "3d".
        global_resp_norm (bool): Whether to use global response normalization. Defaults to False.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_ratio: int = 4,
        kernel_size: int = 7,
        use_residual_connection: bool = False,
        norm_type: str = "group",
        dim: str = "3d",
        global_resp_norm: bool = False,
    ):

        super().__init__(
            in_channels,
            out_channels,
            expansion_ratio,
            kernel_size,
            use_residual_connection=False,
            norm_type=norm_type,
            dim=dim,
            global_resp_norm=global_resp_norm,
        )

        conv = get_conv_layer(spatial_dim=2 if dim == "2d" else 3)
        self.resample_do_res = use_residual_connection
        if use_residual_connection:
            self.res_conv = conv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2)

        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels,
        )

    def forward(self, x):
        """
        Forward pass of the MedNeXtDownBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x1 = super().forward(x)

        if self.resample_do_res:
            res = self.res_conv(x)
            x1 = x1 + res

        return x1


class MedNeXtUpBlock(MedNeXtBlock):
    """
    MedNeXtUpBlock class for upsampling in the MedNeXt model.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        expansion_ratio (int): Expansion ratio for the block. Defaults to 4.
        kernel_size (int): Kernel size for convolutions. Defaults to 7.
        use_residual_connection (bool): Whether to use residual connection. Defaults to False.
        norm_type (str): Type of normalization to use. Defaults to "group".
        dim (str): Dimension of the input. Can be "2d" or "3d". Defaults to "3d".
        global_resp_norm (bool): Whether to use global response normalization. Defaults to False.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_ratio: int = 4,
        kernel_size: int = 7,
        use_residual_connection: bool = False,
        norm_type: str = "group",
        dim: str = "3d",
        global_resp_norm: bool = False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            expansion_ratio,
            kernel_size,
            use_residual_connection=False,
            norm_type=norm_type,
            dim=dim,
            global_resp_norm=global_resp_norm,
        )

        self.resample_do_res = use_residual_connection

        self.dim = dim
        conv = get_conv_layer(spatial_dim=2 if dim == "2d" else 3, transpose=True)
        if use_residual_connection:
            self.res_conv = conv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2)

        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels,
        )

    def forward(self, x):
        """
        Forward pass of the MedNeXtUpBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x1 = super().forward(x)
        # Asymmetry but necessary to match shape

        if self.dim == "2d":
            x1 = torch.nn.functional.pad(x1, (1, 0, 1, 0))
        else:
            x1 = torch.nn.functional.pad(x1, (1, 0, 1, 0, 1, 0))

        if self.resample_do_res:
            res = self.res_conv(x)
            if self.dim == "2d":
                res = torch.nn.functional.pad(res, (1, 0, 1, 0))
            else:
                res = torch.nn.functional.pad(res, (1, 0, 1, 0, 1, 0))
            x1 = x1 + res

        return x1


class MedNeXtOutBlock(nn.Module):
    """
    MedNeXtOutBlock class for the output block in the MedNeXt model.

    Args:
        in_channels (int): Number of input channels.
        n_classes (int): Number of output classes.
        dim (str): Dimension of the input. Can be "2d" or "3d".
    """

    def __init__(self, in_channels, n_classes, dim):
        super().__init__()

        conv = get_conv_layer(spatial_dim=2 if dim == "2d" else 3, transpose=True)
        self.conv_out = conv(in_channels, n_classes, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the MedNeXtOutBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.conv_out(x)
    



class MedNeXt(nn.Module):
    """
    MedNeXt model class from paper: https://arxiv.org/pdf/2303.09975

    Args:
        spatial_dims: spatial dimension of the input data. Defaults to 3.
        init_filters: number of output channels for initial convolution layer. Defaults to 32.
        in_channels: number of input channels for the network. Defaults to 1.
        out_channels: number of output channels for the network. Defaults to 2.
        encoder_expansion_ratio: expansion ratio for encoder blocks. Defaults to 2.
        decoder_expansion_ratio: expansion ratio for decoder blocks. Defaults to 2.
        bottleneck_expansion_ratio: expansion ratio for bottleneck blocks. Defaults to 2.
        kernel_size: kernel size for convolutions. Defaults to 7.
        deep_supervision: whether to use deep supervision. Defaults to False.
        use_residual_connection: whether to use residual connections in standard, down and up blocks. Defaults to False.
        blocks_down: number of blocks in each encoder stage. Defaults to [2, 2, 2, 2].
        blocks_bottleneck: number of blocks in bottleneck stage. Defaults to 2.
        blocks_up: number of blocks in each decoder stage. Defaults to [2, 2, 2, 2].
        norm_type: type of normalization layer. Defaults to 'group'.
        global_resp_norm: whether to use Global Response Normalization. Defaults to False. Refer: https://arxiv.org/abs/2301.00808
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        init_filters: int = 32,
        in_channels: int = 1,
        out_channels: int = 2,
        encoder_expansion_ratio: Sequence[int] | int = 2,
        decoder_expansion_ratio: Sequence[int] | int = 2,
        bottleneck_expansion_ratio: int = 2,
        kernel_size: int = 7,
        deep_supervision: bool = False,
        use_residual_connection: bool = False,
        blocks_down: Sequence[int] = (2, 2, 2, 2),
        blocks_bottleneck: int = 2,
        blocks_up: Sequence[int] = (2, 2, 2, 2),
        norm_type: str = "group",
        global_resp_norm: bool = False,
    ):
        """
        Initialize the MedNeXt model.

        This method sets up the architecture of the model, including:
        - Stem convolution
        - Encoder stages and downsampling blocks
        - Bottleneck blocks
        - Decoder stages and upsampling blocks
        - Output blocks for deep supervision (if enabled)
        """
        super().__init__()

        self.do_ds = deep_supervision
        assert spatial_dims in [2, 3], "`spatial_dims` can only be 2 or 3."
        spatial_dims_str = f"{spatial_dims}d"
        enc_kernel_size = dec_kernel_size = kernel_size

        if isinstance(encoder_expansion_ratio, int):
            encoder_expansion_ratio = [encoder_expansion_ratio] * len(blocks_down)

        if isinstance(decoder_expansion_ratio, int):
            decoder_expansion_ratio = [decoder_expansion_ratio] * len(blocks_up)

        conv = nn.Conv2d if spatial_dims_str == "2d" else nn.Conv3d

        self.stem = conv(in_channels, init_filters, kernel_size=1)

        enc_stages = []
        down_blocks = []

        for i, num_blocks in enumerate(blocks_down):
            enc_stages.append(
                nn.Sequential(
                    *[
                        MedNeXtBlock(
                            in_channels=init_filters * (2**i),
                            out_channels=init_filters * (2**i),
                            expansion_ratio=encoder_expansion_ratio[i],
                            kernel_size=enc_kernel_size,
                            use_residual_connection=use_residual_connection,
                            norm_type=norm_type,
                            dim=spatial_dims_str,
                            global_resp_norm=global_resp_norm,
                        )
                        for _ in range(num_blocks)
                    ]
                )
            )

            down_blocks.append(
                MedNeXtDownBlock(
                    in_channels=init_filters * (2**i),
                    out_channels=init_filters * (2 ** (i + 1)),
                    expansion_ratio=encoder_expansion_ratio[i],
                    kernel_size=enc_kernel_size,
                    use_residual_connection=use_residual_connection,
                    norm_type=norm_type,
                    dim=spatial_dims_str,
                )
            )

        self.enc_stages = nn.ModuleList(enc_stages)
        self.down_blocks = nn.ModuleList(down_blocks)

        self.bottleneck = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=init_filters * (2 ** len(blocks_down)),
                    out_channels=init_filters * (2 ** len(blocks_down)),
                    expansion_ratio=bottleneck_expansion_ratio,
                    kernel_size=dec_kernel_size,
                    use_residual_connection=use_residual_connection,
                    norm_type=norm_type,
                    dim=spatial_dims_str,
                    global_resp_norm=global_resp_norm,
                )
                for _ in range(blocks_bottleneck)
            ]
        )

        up_blocks = []
        dec_stages = []
        for i, num_blocks in enumerate(blocks_up):
            up_blocks.append(
                MedNeXtUpBlock(
                    in_channels=init_filters * (2 ** (len(blocks_up) - i)),
                    out_channels=init_filters * (2 ** (len(blocks_up) - i - 1)),
                    expansion_ratio=decoder_expansion_ratio[i],
                    kernel_size=dec_kernel_size,
                    use_residual_connection=use_residual_connection,
                    norm_type=norm_type,
                    dim=spatial_dims_str,
                    global_resp_norm=global_resp_norm,
                )
            )

            dec_stages.append(
                nn.Sequential(
                    *[
                        MedNeXtBlock(
                            in_channels=init_filters * (2 ** (len(blocks_up) - i - 1)),
                            out_channels=init_filters * (2 ** (len(blocks_up) - i - 1)),
                            expansion_ratio=decoder_expansion_ratio[i],
                            kernel_size=dec_kernel_size,
                            use_residual_connection=use_residual_connection,
                            norm_type=norm_type,
                            dim=spatial_dims_str,
                            global_resp_norm=global_resp_norm,
                        )
                        for _ in range(num_blocks)
                    ]
                )
            )

        self.up_blocks = nn.ModuleList(up_blocks)
        self.dec_stages = nn.ModuleList(dec_stages)

        self.out_0 = MedNeXtOutBlock(in_channels=init_filters, n_classes=out_channels, dim=spatial_dims_str)

        if deep_supervision:
            out_blocks = [
                MedNeXtOutBlock(in_channels=init_filters * (2**i), n_classes=out_channels, dim=spatial_dims_str)
                for i in range(1, len(blocks_up) + 1)
            ]

            out_blocks.reverse()
            self.out_blocks = nn.ModuleList(out_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor | Sequence[torch.Tensor]:
        """
        Forward pass of the MedNeXt model.

        This method performs the forward pass through the model, including:
        - Stem convolution
        - Encoder stages and downsampling
        - Bottleneck blocks
        - Decoder stages and upsampling with skip connections
        - Output blocks for deep supervision (if enabled)

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor or Sequence[torch.Tensor]: Output tensor(s).
        """
        # Apply stem convolution
        x = self.stem(x)

        # Encoder forward pass
        enc_outputs = []
        for enc_stage, down_block in zip(self.enc_stages, self.down_blocks):
            x = enc_stage(x)
            enc_outputs.append(x)
            x = down_block(x)

        # Bottleneck forward pass
        x = self.bottleneck(x)

        # Initialize deep supervision outputs if enabled
        if self.do_ds:
            ds_outputs = []

        # Decoder forward pass with skip connections
        for i, (up_block, dec_stage) in enumerate(zip(self.up_blocks, self.dec_stages)):
            if self.do_ds and i < len(self.out_blocks):
                ds_outputs.append(self.out_blocks[i](x))

            x = up_block(x)
            x = x + enc_outputs[-(i + 1)]
            x = dec_stage(x)

        # Final output block
        x = self.out_0(x)

        # Return output(s)
        if self.do_ds and self.training:
            return (x, *ds_outputs[::-1])
        else:
            return x


# Define the MedNeXt variants as reported in 10.48550/arXiv.2303.09975
def create_mednext(
    variant: str,
    spatial_dims: int = 3,
    in_channels: int = 1,
    out_channels: int = 2,
    kernel_size: int = 3,
    deep_supervision: bool = False,
    init_filters: int = 32,
) -> MedNeXt:
    """
    Factory method to create MedNeXt variants.

    Args:
        variant (str): The MedNeXt variant to create ('S', 'B', 'M', or 'L').
        spatial_dims (int): Number of spatial dimensions. Defaults to 3.
        in_channels (int): Number of input channels. Defaults to 1.
        out_channels (int): Number of output channels. Defaults to 2.
        kernel_size (int): Kernel size for convolutions. Defaults to 3.
        deep_supervision (bool): Whether to use deep supervision. Defaults to False.

    Returns:
        MedNeXt: The specified MedNeXt variant.

    Raises:
        ValueError: If an invalid variant is specified.
    """
    common_args = {
        "spatial_dims": spatial_dims,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "deep_supervision": deep_supervision,
        "use_residual_connection": True,
        "norm_type": "group",
        "global_resp_norm": False,
    }


    if variant.upper() == "B":
        return MedNeXt(
            encoder_expansion_ratio=(2, 3, 4, 4),
            decoder_expansion_ratio=(4, 4, 3, 2),
            bottleneck_expansion_ratio=4,
            blocks_down=(2, 2, 2, 2),
            blocks_bottleneck=2,
            blocks_up=(2, 2, 2, 2),
            init_filters=init_filters,
            **common_args,  # type: ignore
        )
  

if __name__ == "__main__":
    model = create_mednext("B", 
                           spatial_dims=3, 
                           in_channels=9, out_channels=1, 
                           kernel_size=3, deep_supervision=False)
    
    x = torch.randn(1, 9, 128, 128, 128)  
    output = model(x)
    print(output.shape)     