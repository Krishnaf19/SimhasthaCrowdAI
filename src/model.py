import torch
import torch.nn as nn
from torchvision import models


class SELayer(nn.Module):
    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def clear_device_cache(device: torch.device) -> None:
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()


class CSRNet(nn.Module):
    def __init__(
        self,
        load_weights: bool = True,
        freeze_frontend: bool = True,
        use_se: bool = True,
    ):
        super().__init__()
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]

        self.frontend = self._make_layers(self.frontend_feat, in_channels=3, dilation=False)
        self.backend = self._make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.se = SELayer(channel=64) if use_se else nn.Identity()
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        if load_weights:
            self._load_vgg16_weights()
        else:
            self._init_backend_weights()

        if freeze_frontend:
            self._freeze_frontend()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.frontend(x)
        x = self.backend(x)
        x = self.se(x)
        x = self.output_layer(x)
        return torch.relu(x)

    def _make_layers(self, cfg, in_channels: int = 3, dilation: bool = False) -> nn.Sequential:
        d_rate = 2 if dilation else 1
        layers = []
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate))
                layers.append(nn.ReLU(inplace=True))
                in_channels = v
        return nn.Sequential(*layers)

    def _load_vgg16_weights(self) -> None:
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        vgg_state = vgg16.features.state_dict()
        front_state = self.frontend.state_dict()
        new_state = {}
        transferred = 0
        for fk, vk in zip(front_state.keys(), vgg_state.keys()):
            if front_state[fk].shape == vgg_state[vk].shape:
                new_state[fk] = vgg_state[vk]
                transferred += 1
            else:
                new_state[fk] = front_state[fk]
        self.frontend.load_state_dict(new_state)
        self._init_backend_weights()
        print(f"CSRNet: transferred {transferred}/{len(front_state)} frontend weights from VGG16.")

    def _init_backend_weights(self) -> None:
        for m in self.backend.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.output_layer.weight, std=0.01)
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, 0)

    def _freeze_frontend(self) -> None:
        for param in self.frontend.parameters():
            param.requires_grad = False

    def unfreeze_frontend(self) -> None:
        for param in self.frontend.parameters():
            param.requires_grad = True

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


if __name__ == '__main__':
    device = get_device()
    print(f"Using device: {device}")
    model = CSRNet(load_weights=True, freeze_frontend=True, use_se=True).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}, trainable: {trainable:,}")
    test_input = torch.randn(1, 3, 512, 512).to(device)
    with torch.no_grad():
        output = model(test_input)
    print(f"Output shape: {output.shape}")
