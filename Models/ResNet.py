from typing import List, Optional
from torch import nn
import torch


class ResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: Optional[int] = 1,
        identity_downscale: nn.Sequential = None,
    ) -> None:
        """
        Create Residual block in ResNet

        Args:
            in_channels (int): In channels size
            out_channels (int): Out channels size
            stride (Optional[int], optional):
                Stride for image size cut. Defaults to 1.
            identity_downscale (nn.Sequential, optional):
                Way of scaling identity. Defaults to None.
        """
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                stride=stride,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU()
        self.identity_downscale = identity_downscale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pytorch forward function

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: block output
        """
        identity = x.clone()
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)

        if self.identity_downscale:
            identity = self.identity_downscale(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        layers: Optional[List] = [2, 2, 2, 2],
        block: ResNetBlock = ResNetBlock,
    ) -> None:
        """
        Create ResNet architecture model

        Args:
            in_channels (int): initial image number of channels
            num_classes (int): number of classes for classification
            layers (Optional[List], optional):
                List with numbers of layers for each residual block.
                Defaults to [2, 2, 2, 2].
            block (ResNetBlock, optional):
                Residual custom block. Defaults to ResNetBlock.
        """
        super().__init__()
        self.initial_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.layer_64 = self.create_residual_layer(
            block=block,
            out_channels=64,
            num_residual_blocks=layers[0],
        )
        self.layer_128 = self.create_residual_layer(
            block=block,
            out_channels=128,
            num_residual_blocks=layers[1],
            stride=2,
        )
        self.layer_256 = self.create_residual_layer(
            block=block,
            out_channels=256,
            num_residual_blocks=layers[2],
            stride=2,
        )
        self.layer_512 = self.create_residual_layer(
            block=block,
            out_channels=512,
            num_residual_blocks=layers[3],
            stride=2,
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # adapt if model structure changes
        self.fc = nn.Linear(512, num_classes)

    def create_residual_layer(
        self,
        block: ResNetBlock,
        out_channels: int,
        num_residual_blocks: int,
        stride=1,
    ) -> nn.Sequential:
        """
        Build residual parts of model

        Args:
            block (ResNetBlock): residual block
            out_channels (int): output feature maps after residual
            num_residual_blocks (int): number of that blocks
            stride (int, optional): stride for image cut. Defaults to 1.

        Returns:
            (nn.Sequential): built residual block
        """
        identity_downscale = None
        in_channels = out_channels // 2 if out_channels != 64 else out_channels
        if stride != 1:
            identity_downscale = nn.Sequential(
                *[
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                ]
            )
        layers = []
        layers.append(
            block(
                in_channels,
                out_channels,
                stride=stride,
                identity_downscale=identity_downscale,
            )
        )
        for _ in range(num_residual_blocks - 1):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pytorch forward function

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: model output
        """
        x = self.initial_layers(x)
        x = self.layer_64(x)
        x = self.layer_128(x)
        x = self.layer_256(x)
        x = self.layer_512(x)
        x = self.global_avg_pool(x)
        x = x.squeeze()
        x = self.fc(x)
        return x


if __name__ == "__main__":
    device = torch.device("mps")
    model = ResNet(3, 36, [2, 2, 2, 2], ResNetBlock).to(device)
    test_tensor = torch.rand(size=(16, 3, 224, 224), device=device)
    out = model(test_tensor)
    print(out.shape)
