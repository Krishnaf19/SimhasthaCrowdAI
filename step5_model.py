# import torch.nn as nn
# import torch
# from torchvision import models

# class CSRNet(nn.Module):
#     def __init__(self, load_weights=True):
#         super(CSRNet, self).__init__()
#         # VGG-16 Front-end (First 10 layers are standard)
#         self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
#         # Dilated Back-end (Dilated convs increase receptive field without losing resolution)
#         self.backend_feat  = [512, 512, 512, 256, 128, 64]
        
#         self.frontend = self._make_layers(self.frontend_feat)
#         self.backend = self._make_layers(self.backend_feat, in_channels=512, dilation=True)
#         self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

#         if load_weights:
#             # 1. Load Pre-trained VGG16 weights for the frontend
#             vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
#             self._initialize_weights(vgg16)
#             print("CSRNet: Frontend initialized with pre-trained VGG16 weights.")
#         else:
#             self._initialize_weights()

#     def forward(self, x):
#         x = self.frontend(x)
#         x = self.backend(x)
#         x = self.output_layer(x)
#         return x

#     def _make_layers(self, cfg, in_channels=3, batch_norm=False, dilation=False):
#         d_rate = 2 if dilation else 1
#         layers = []
#         for v in cfg:
#             if v == 'M':
#                 layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#             else:
#                 conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
#                 layers += [conv2d, nn.ReLU(inplace=True)]
#                 in_channels = v
#         return nn.Sequential(*layers)

#     def _initialize_weights(self, vgg16_model=None):
#         if vgg16_model:
#             # Transfer weights from VGG16 to our frontend
#             vgg_items = list(vgg16_model.features.state_dict().items())
#             frontend_items = list(self.frontend.state_dict().items())
            
#             new_state_dict = {}
#             for i in range(len(frontend_items)):
#                 new_state_dict[frontend_items[i][0]] = vgg_items[i][1]
#             self.frontend.load_state_dict(new_state_dict)
        
#         # Initialize Backend and Output layer with Gaussian distribution
#         for m in self.backend.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.normal_(m.weight, std=0.01)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#         nn.init.normal_(self.output_layer.weight, std=0.01)
#         if self.output_layer.bias is not None:
#             nn.init.constant_(self.output_layer.bias, 0)

# if __name__ == "__main__":
#     # Quick Test
#     model = CSRNet()
#     test_input = torch.randn(1, 3, 512, 512)
#     output = model(test_input)
#     print(f"Input Shape: {test_input.shape}")
#     print(f"Output Shape: {output.shape}") # Should be [1, 1, 64, 64]

import torch
import torch.nn as nn
from torchvision import models


def get_device() -> torch.device:
    """
    Returns the best available device: MPS (Apple Silicon) → CUDA → CPU.
    Use this in your training script instead of hardcoding a device string.
    """
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def clear_device_cache(device: torch.device) -> None:
    """
    FIX 2: Cache-clearing utility to call at the end of every training epoch.
    Handles MPS, CUDA, and CPU (no-op) transparently.
    """
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()
    # CPU: nothing to clear


class CSRNet(nn.Module):
    def __init__(self, load_weights: bool = True, freeze_frontend: bool = True):
        super(CSRNet, self).__init__()

        # VGG-16 frontend feature config (matches first 13 conv layers of VGG16)
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        # Dilated backend
        self.backend_feat  = [512, 512, 512, 256, 128, 64]

        self.frontend     = self._make_layers(self.frontend_feat, in_channels=3, dilation=False)
        self.backend      = self._make_layers(self.backend_feat,  in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        if load_weights:
            self._load_vgg16_weights()
        else:
            self._init_backend_weights()

        # FIX 1: Freeze frontend AFTER weights are loaded
        if freeze_frontend:
            self._freeze_frontend()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        # FIX 5: Clamp output to non-negative — density maps cannot be negative
        x = torch.relu(x)
        return x

    # ── Layer construction ────────────────────────────────────────────────────

    def _make_layers(
        self,
        cfg: list,
        in_channels: int = 3,
        dilation: bool = False
    ) -> nn.Sequential:
        d_rate = 2 if dilation else 1
        layers = []
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(
                    nn.Conv2d(in_channels, v, kernel_size=3,
                              padding=d_rate, dilation=d_rate)
                )
                layers.append(nn.ReLU(inplace=True))
                in_channels = v
        return nn.Sequential(*layers)

    # ── Weight initialisation ─────────────────────────────────────────────────

    def _load_vgg16_weights(self) -> None:
        """
        FIX 3: Transfer VGG16 weights by matching layer names, not positional
        index — safe against any structural mismatch.
        """
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        vgg_state   = vgg16.features.state_dict()
        front_state = self.frontend.state_dict()

        # Only copy keys that exist in both and have matching shapes
        transferred = 0
        new_state   = {}
        vgg_keys    = list(vgg_state.keys())
        front_keys  = list(front_state.keys())

        for fk, vk in zip(front_keys, vgg_keys):
            if front_state[fk].shape == vgg_state[vk].shape:
                new_state[fk] = vgg_state[vk]
                transferred  += 1
            else:
                print(f"  Shape mismatch — skipping '{fk}': "
                      f"{front_state[fk].shape} vs {vgg_state[vk].shape}")
                new_state[fk] = front_state[fk]   # keep random init

        self.frontend.load_state_dict(new_state)
        print(f"CSRNet: {transferred}/{len(front_keys)} frontend layers "
              f"initialized from VGG16 ImageNet weights.")

        self._init_backend_weights()

    def _init_backend_weights(self) -> None:
        """Gaussian init for backend and output layer (standard CSRNet practice)."""
        for m in self.backend.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        nn.init.normal_(self.output_layer.weight, std=0.01)
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, 0)

    # ── FIX 1: Frontend freezing ──────────────────────────────────────────────

    def _freeze_frontend(self) -> None:
        """
        Freeze all VGG16 frontend parameters so they are not updated during
        training. Critical for small datasets (19 images) to prevent overfitting.
        """
        frozen = 0
        for param in self.frontend.parameters():
            param.requires_grad = False
            frozen += 1
        print(f"CSRNet: Frontend frozen ({frozen} parameter tensors, "
              f"requires_grad=False).")

    def unfreeze_frontend(self) -> None:
        """Optional: call this for fine-tuning after initial training converges."""
        for param in self.frontend.parameters():
            param.requires_grad = True
        print("CSRNet: Frontend unfrozen — all parameters now trainable.")

    def trainable_parameters(self) -> list:
        """
        Returns only parameters where requires_grad=True.
        Pass this to your optimizer so frozen frontend params are excluded.
        Usage: optimizer = Adam(model.trainable_parameters(), lr=1e-4)
        """
        return [p for p in self.parameters() if p.requires_grad]


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")

    model = CSRNet(load_weights=True, freeze_frontend=True).to(device)

    # Verify parameter split
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable
    print(f"\nParameter summary:")
    print(f"  Total      : {total:,}")
    print(f"  Trainable  : {trainable:,}  (backend + output layer)")
    print(f"  Frozen     : {frozen:,}  (VGG16 frontend)")

    # Forward pass test
    test_input = torch.randn(1, 3, 512, 512).to(device)
    with torch.no_grad():
        output = model(test_input)

    print(f"\nForward pass:")
    print(f"  Input  shape : {test_input.shape}")
    print(f"  Output shape : {output.shape}")    # expect [1, 1, 64, 64]
    print(f"  Output min   : {output.min():.4f}")  # should be >= 0.0 after relu
    print(f"  Output max   : {output.max():.4f}")

    # Epoch-end cache clear test
    clear_device_cache(device)
    print(f"\nDevice cache cleared for '{device.type}'.")