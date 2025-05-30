# stdlib
import os

# third party
import torch

synthcity_device = os.environ.get("SYNTHCITY_DEVICE", "cpu")
DEVICE = torch.device(synthcity_device)
