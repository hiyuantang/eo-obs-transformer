import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from typing import List, Optional, Tuple

class PyTorchResNetBlock(nn.Module):
    IMAGE_SIZE = 224
    DEFAULT_MS_CHANNELS = ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'TEMP1']

    def __init__(self,
                 ms_channels: Optional[List[str]] = None,
                 pretrained_ms: bool = True,
                 pretrained_nl: bool = True,
                 freeze_model: bool = False,
                 batch_norm_final: bool = True,
                 **kwargs):
        super().__init__()

        self.freeze_model = freeze_model
        self.batch_norm_final = batch_norm_final

        self.ms_channels = ms_channels if ms_channels else self.DEFAULT_MS_CHANNELS
        n_ms_bands = len(self.ms_channels)
        n_nl_bands = 1

        # --- Initialize MS Model ---
        self.ms_model = self._get_resnet_feature_extractor(
            num_channels=n_ms_bands,
            keep_rgb=True,
        )

        # --- Initialize NL Model ---
        self.nl_model = self._get_resnet_feature_extractor(
            num_channels=n_nl_bands,
            keep_rgb=False,
        )

        # --- Feature dimension ---
        # ResNet-18's output feature dimension before the final FC layer
        self._feature_dim = 512 # Output channels of the final conv block in ResNet-18

        # Concatenated feature dimension
        self.combined_feature_dim = self._feature_dim * 2

        # Optional final batch normalization
        self.final_bn = nn.BatchNorm1d(self.combined_feature_dim) if batch_norm_final else nn.Identity()

        # Freeze models if requested
        if self.freeze_model:
            for param in self.ms_model.parameters():
                param.requires_grad = False
            for param in self.nl_model.parameters():
                param.requires_grad = False


    def _get_resnet_feature_extractor(self,
                                      num_channels: int,
                                      keep_rgb: bool) -> nn.Module:
        
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        original_conv1 = model.conv1
        new_conv1_weights = self._adapt_first_conv_weights(
            num_channels=num_channels,
            original_weights=original_conv1.weight.data,
            keep_rgb=keep_rgb
        )
        model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.conv1.weight.data = new_conv1_weights
        model.fc = nn.Identity()
        return model

    @staticmethod
    def _adapt_first_conv_weights(num_channels: int,
                                 original_weights: torch.Tensor,
                                 keep_rgb: bool) -> torch.Tensor:
        rgb_mean = original_weights.mean(dim=1, keepdim=True)

        if keep_rgb and num_channels >= 3:
            num_new_channels = num_channels - 3
            mean_weights = rgb_mean.repeat(1, num_new_channels, 1, 1)
            scale_factor = 3.0 / num_channels
            new_weights = torch.cat((original_weights * scale_factor, mean_weights * scale_factor), dim=1)
        else:
            new_weights = rgb_mean.repeat(1, num_channels, 1, 1)
            new_weights /= num_channels

        assert new_weights.shape[1] == num_channels, "Channel dimension mismatch"
        return new_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_ms_bands = self.ms_model.conv1.in_channels
        # Split input tensor into MS and NL parts
        ms_input = x[:, :n_ms_bands, :, :]
        nl_input = x[:, n_ms_bands:, :, :]

        # Process through respective ResNet models
        ms_features = self.ms_model(ms_input) # Shape: (batch_size, feature_dim)
        nl_features = self.nl_model(nl_input) # Shape: (batch_size, feature_dim)
        # Concatenate features from both models
        combined_features = torch.cat((ms_features, nl_features), dim=1) # Shape: (batch_size, feature_dim*2)

        # Apply final batch normalization if enabled
        output = self.final_bn(combined_features)
        return output

