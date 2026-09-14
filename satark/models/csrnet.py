import torch
import torch.nn as nn
from torchvision import models

# Default 4 classes: head, turban, veil, cap
DEFAULT_OUTPUT_CHANNELS = 4


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        return x * self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def clear_device_cache(device):
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()


class CSRNet(nn.Module):
    FRONTEND_CFG = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
    BACKEND_CFG = [512, 512, 512, 256, 128, 64]

    def __init__(self, load_weights=True, freeze_frontend=True, use_se=True,
                 output_channels=None):
        super().__init__()
        self.frontend = self._make_layers(self.FRONTEND_CFG, in_channels=3, dilation=False)
        self.backend = self._make_layers(self.BACKEND_CFG, in_channels=512, dilation=True)
        self.se = SELayer(64) if use_se else nn.Identity()
        self.output_layer = nn.Conv2d(
            64, output_channels if output_channels is not None else DEFAULT_OUTPUT_CHANNELS, kernel_size=1
        )
        if load_weights:
            self._load_vgg16_weights()
        else:
            self._init_backend_weights()
        if freeze_frontend:
            self._freeze_frontend()

    def forward(self, x):
        return torch.relu(self.output_layer(self.se(self.backend(self.frontend(x)))))

    def _make_layers(self, cfg, in_channels=3, dilation=False):
        d, layers = 2 if dilation else 1, []
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers += [nn.Conv2d(in_channels, v, 3, padding=d, dilation=d), nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def _load_vgg16_weights(self):
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        vs, fs = vgg.features.state_dict(), self.frontend.state_dict()
        self.frontend.load_state_dict({fk: (vs[vk] if fs[fk].shape == vs[vk].shape else fs[fk])
                                       for fk, vk in zip(fs, vs)})
        self._init_backend_weights()

    def _init_backend_weights(self):
        for m in list(self.backend.modules()) + [self.output_layer]:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _freeze_frontend(self):
        for p in self.frontend.parameters():
            p.requires_grad = False

    def unfreeze_frontend(self):
        for p in self.frontend.parameters():
            p.requires_grad = True

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
