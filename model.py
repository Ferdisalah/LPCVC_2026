import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

# Constants for the specific task
NUM_CLASSES  = 92
NUM_FAMILIES = 13
# =============================================================================
# LAYER4 SURGERY
# =============================================================================
def modify_layer4_stride(layer4):
    """
    Stride 2→1 in spatial dims → 14×14 output
    
    Block 0:
    - conv1[0][0]: (1,2,2) → (1,1,1)  spatial conv
    - downsample[0]: (2,2,2) → (2,1,1) keep temporal!
    """
    block0 = layer4[0]

    # Fix main path — spatial conv (first Conv3d in Conv2Plus1D)
    spatial_conv = block0.conv1[0][0]
    print(f"  Before: conv1 spatial stride = {spatial_conv.stride}")
    spatial_conv.stride = (1, 1, 1)
    print(f"  After:  conv1 spatial stride = {spatial_conv.stride}")

    # Fix downsample — must match! (keep temporal stride=2)
    if block0.downsample is not None:
        ds_conv = block0.downsample[0]
        print(f"  Before: downsample stride = {ds_conv.stride}")
        ds_conv.stride = (2, 1, 1)  # keep temporal=2, remove spatial!
        print(f"  After:  downsample stride = {ds_conv.stride}")

    # Block 1 already has stride (1,1,1) → no change needed!
    print(f"  Block 1 stride: {layer4[1].conv1[0][0].stride} ✅")

    return layer4
# =============================================================================
# MODEL
# =============================================================================
class SurgicalNovelEdgeAI(nn.Module):
    def __init__(self):
        super().__init__()

        # QuantStubs → ready for w8a8 later!
        self.quant   = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        bb = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
        self.stem   = bb.stem
        self.layer1 = bb.layer1
        self.layer2 = bb.layer2
        self.layer3 = bb.layer3
        self.layer4 = modify_layer4_stride(bb.layer4)
        # Output: (B, 512, 1, 14, 14) ✅

        self.dropout     = nn.Dropout(0.3)
        self.fine_head   = nn.Linear(512, NUM_CLASSES)
        self.coarse_head = nn.Linear(512, NUM_FAMILIES)
        # Add to __init__
# Vertical Branch (tracks Height/Y-axis)
        self.v_strip = nn.Sequential(
    nn.Conv2d(512, 128, kernel_size=(3, 1), padding=(1, 0), bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True),
    nn.Conv2d(128, 1, kernel_size=1, bias=False)
)

# Horizontal Branch (tracks Width/X-axis)
        self.h_strip = nn.Sequential(
    nn.Conv2d(512, 128, kernel_size=(1, 3), padding=(0, 1), bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True),
    nn.Conv2d(128, 1, kernel_size=1, bias=False)
)
        # Temporal attention
        self.temporal_fc = nn.Sequential(
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Linear(128, 1)
)  
        self.motion_alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        x = self.quant(x)

        # Backbone
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
# --- MOTION ENCODING ---
        if x.shape[2] > 1:
          motion = x[:, :, 1:] - x[:, :, :-1]
          motion = torch.cat([motion[:, :, :1], motion], dim=2)
        else:
          motion = torch.zeros_like(x)

        motion = torch.tanh(motion)  # stabilize magnitude
        x = x + self.motion_alpha * motion
        # x: (B, 512, 1, 14, 14)

        # Temporal attention (weighted sum across T)
        # Better temporal attention
        B, C, T, H, W = x.shape

        t = x.mean(dim=(3, 4))          # (B, C, T)
        t = t.permute(0, 2, 1)          # (B, T, C)

        attn_scores = self.temporal_fc(t).squeeze(-1)  # (B, T)
        t_attn = torch.softmax(attn_scores, dim=1)

        x = (x * t_attn[:, None, :, None, None]).sum(dim=2)

        # Spatial attention (weighted sum across H,W)
        # Now meaningful because 14×14 not 7×7!
# x shape: (B, 512, 14, 14) after temporal attention
        B, C, H, W = x.shape
# --- VERTICAL STRIP (Tracks Up/Down) ---
        v_feat = x.mean(dim=3, keepdim=True)       # Pool width: (B, 512, 14, 1)
        v_attn = torch.sigmoid(self.v_strip(v_feat)) # (B, 1, 14, 1)

# --- HORIZONTAL STRIP (Tracks Left/Right) ---
        h_feat = x.mean(dim=2, keepdim=True)       # Pool height: (B, 512, 1, 14)
        h_attn = torch.sigmoid(self.h_strip(h_feat)) # (B, 1, 1, 14)

# --- FUSE ATTENTION ---
# The person is located where vertical and horizontal importance intersect
        x = x * v_attn * h_attn

# --- FINAL GLOBAL COLLAPSE ---
# Now that we've highlighted the person, pool everything to 512 channels
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(B, -1) # (B, 512)

        x = self.dropout(x)

# ... inside the SurgicalNovelEdgeAI class ...
        fine   = self.dequant(self.fine_head(x))
        coarse = self.dequant(self.coarse_head(x))
        return fine, coarse
