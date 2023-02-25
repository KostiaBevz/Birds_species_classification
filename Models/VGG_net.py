from typing import List, Optional, Tuple

import torch
from torch import nn

# TODO: add weight initialization

VGG = [
    64,
    64,
    "M",
    128,
    128,
    "M",
    256,
    256,
    256,
    "M",
    512,
    512,
    512,
    "M",
    512,
    512,
    512,
    "M",
]
VGG_head = [4096, 4096]


class VGG_Net(nn.Module):
    """
    Custom model implementing VGG net paper from scratch with some modern
    modifications like global pooling layers insted of FC layers
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        model_structure: List[int or str],
        image_size: int or Tuple,
        head: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.backbone = []
        number_max_pool_layers = len(
            [i for i in model_structure if type(i) == str]
        )
        if isinstance(image_size, tuple):
            image_size = image_size[
                0
            ]  # TODO: think about the case when I have not square image
        flatten_size = int(image_size / (2**number_max_pool_layers))
        for filter in model_structure:
            if type(filter) == str:
                self.backbone.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                out_channels = filter
                self.backbone.extend(
                    [
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(out_channels),  # Optional
                        nn.ReLU(),
                    ]
                )
                in_channels = filter
        """
        Here we have 2 ways to configure head with usage of FC layers
        or proceding with Global Max Pooling layer
        learn more:
        https://blog.paperspace.com/global-pooling-in-convolutional-neural-networks/
        Better to use global layers
        """
        # 1
        if head:
            self.backbone.append(nn.Flatten())
            in_features = out_channels * (flatten_size**2)
            for size in head:
                out_channels = size
                self.backbone.extend(
                    [
                        nn.Linear(
                            in_features=in_features,
                            out_features=out_channels,
                        ),
                        nn.ReLU(),
                        nn.Dropout(0.4),
                    ]
                )
                in_features = size
            else:
                self.backbone.append(nn.Linear(in_features, num_classes))
        # 2
        else:
            self.backbone.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=num_classes,
                    kernel_size=1,
                )
            )
            self.backbone.append(nn.MaxPool2d(kernel_size=flatten_size))

        self.backbone = nn.Sequential(*self.backbone)

    def forward(self, x: torch.Tensor):
        x = torch.squeeze(self.backbone(x))
        return x


if __name__ == "__main__":
    device = torch.device("mps")
    net = VGG_Net(
        in_channels=3,
        num_classes=36,
        model_structure=VGG,
        image_size=512,
        # head=VGG_head,
    ).to(device)

    print(net)

    x = torch.randn(64, 3, 224, 224, device=device)
    out = net(x)
    print(out.shape)
