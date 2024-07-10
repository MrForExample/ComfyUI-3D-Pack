import os
import torch
import torch.nn as nn
import torch.nn.init as init
from einops import rearrange

class PoseGuider(nn.Module):
    def __init__(self, noise_latent_channels=4):
        super(PoseGuider, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Final projection layer
        self.final_proj = nn.Conv2d(in_channels=128, out_channels=noise_latent_channels, kernel_size=1)

        # Initialize layers
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize weights with Gaussian distribution and zero out the final layer
        for m in self.conv_layers:
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    init.zeros_(m.bias)

        init.zeros_(self.final_proj.weight)
        if self.final_proj.bias is not None:
            init.zeros_(self.final_proj.bias)

    def forward(self, pose_image):
        x = self.conv_layers(pose_image)
        x = self.final_proj(x)

        return x

    @classmethod
    def from_pretrained(pretrained_model_path):
        if not os.path.exists(pretrained_model_path):
            print(f"There is no model file in {pretrained_model_path}")
        print(f"loaded PoseGuider's pretrained weights from {pretrained_model_path} ...")

        state_dict = torch.load(pretrained_model_path, map_location="cpu")
        model = PoseGuider(noise_latent_channels=4)
        m, u = model.load_state_dict(state_dict, strict=False)
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")        
        params = [p.numel() if "temporal" in n else 0 for n, p in model.named_parameters()]
        print(f"### PoseGuider's Parameters: {sum(params) / 1e6} M")
        
        return model
