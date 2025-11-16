import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# ==================== Phase Locking Value (PLV) Module ====================
class PLVConnectivity(nn.Module):
    """Compute Phase Locking Value for functional connectivity"""
    def __init__(self, n_channels: int, sampling_rate: int = 250):
        super().__init__()
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        
    def compute_plv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute PLV from EEG signals
        Args:
            x: (batch, channels, time)
        Returns:
            plv: (batch, channels, channels) connectivity matrix
        """
        # Apply Hilbert transform to get instantaneous phase
        x_analytic = torch.fft.fft(x, dim=-1)
        x_analytic[..., x.shape[-1]//2:] = 0
        x_analytic = torch.fft.ifft(x_analytic, dim=-1)
        
        # Get phase
        phase = torch.angle(x_analytic)  # (batch, channels, time)
        
        # Compute phase differences for all channel pairs
        phase_i = phase.unsqueeze(2)  # (batch, channels, 1, time)
        phase_j = phase.unsqueeze(1)  # (batch, 1, channels, time)
        phase_diff = phase_i - phase_j  # (batch, channels, channels, time)
        
        # Compute PLV
        plv = torch.abs(torch.mean(torch.exp(1j * phase_diff), dim=-1))
        
        return plv.real
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.compute_plv(x)


# ==================== Vision Transformer Components ====================
class PatchEmbedding(nn.Module):
    """Convert image patches to embeddings"""
    def __init__(self, img_size: int = 64, patch_size: int = 8, in_channels: int = 3, embed_dim: int = 512):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (batch, embed_dim, n_patches_h, n_patches_w)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ==================== EEG Encoder ====================
class EEGEncoder(nn.Module):
    """Encode EEG signals to latent representation"""
    def __init__(self, n_channels: int = 64, time_points: int = 500, 
                 embed_dim: int = 512, use_plv: bool = False):
        super().__init__()
        self.n_channels = n_channels
        self.time_points = time_points
        self.use_plv = use_plv
        
        # Temporal convolution
        self.temp_conv = nn.Sequential(
            nn.Conv1d(n_channels, 128, kernel_size=25, stride=2, padding=12),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Conv1d(256, 512, kernel_size=10, stride=2, padding=4),
            nn.BatchNorm1d(512),
            nn.GELU(),
        )
        
        # PLV connectivity encoder
        if use_plv:
            self.plv_module = PLVConnectivity(n_channels)
            self.plv_encoder = nn.Sequential(
                nn.Linear(n_channels * n_channels, 1024),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(1024, 512),
                nn.GELU(),
            )
        
        # Calculate output size after convolutions
        out_time = time_points // 8  # After 3 stride-2 convolutions
        
        # Projection to embedding dimension
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * out_time, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: EEG signal (batch, channels, time)
        Returns:
            eeg_embed: (batch, embed_dim)
            plv_embed: (batch, 512) if use_plv else None
        """
        # Temporal encoding
        eeg_features = self.temp_conv(x)
        eeg_embed = self.projection(eeg_features)
        
        # PLV connectivity encoding
        plv_embed = None
        if self.use_plv:
            plv_matrix = self.plv_module(x)  # (batch, channels, channels)
            plv_flat = plv_matrix.view(plv_matrix.size(0), -1)
            plv_embed = self.plv_encoder(plv_flat)
        
        return eeg_embed, plv_embed


# ==================== Stage 1: Low Resolution Generator ====================
class LowResGenerator(nn.Module):
    """Generate low resolution images from EEG embeddings"""
    def __init__(self, eeg_embed_dim: int = 512, img_size: int = 64, 
                 n_transformer_blocks: int = 6, use_plv: bool = False):
        super().__init__()
        self.img_size = img_size
        self.use_plv = use_plv
        
        # Combine EEG and PLV embeddings if using PLV
        input_dim = eeg_embed_dim + 512 if use_plv else eeg_embed_dim
        
        # Learnable patch tokens
        self.n_patches = (img_size // 4) ** 2
        self.patch_dim = 512
        
        # Project EEG embedding to patch space
        self.eeg_to_patches = nn.Sequential(
            nn.Linear(input_dim, self.patch_dim * self.n_patches),
            nn.GELU(),
        )
        
        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_patches, self.patch_dim))
        
        # Transformer blocks
        self.transformer = nn.ModuleList([
            TransformerBlock(self.patch_dim, n_heads=8)
            for _ in range(n_transformer_blocks)
        ])
        
        # Decode to image
        self.to_image = nn.Sequential(
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, 4 * 4 * 64),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', 
                     h=img_size//4, w=img_size//4, p1=4, p2=4, c=64),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, eeg_embed: torch.Tensor, plv_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            eeg_embed: (batch, embed_dim)
            plv_embed: (batch, 512) if use_plv
        Returns:
            low_res_img: (batch, 3, img_size, img_size)
        """
        # Combine embeddings if using PLV
        if self.use_plv and plv_embed is not None:
            combined_embed = torch.cat([eeg_embed, plv_embed], dim=-1)
        else:
            combined_embed = eeg_embed
        
        # Generate patch tokens
        x = self.eeg_to_patches(combined_embed)
        x = rearrange(x, 'b (n d) -> b n d', n=self.n_patches, d=self.patch_dim)
        
        # Add positional embedding
        x = x + self.pos_embedding
        
        # Apply transformer blocks
        for block in self.transformer:
            x = block(x)
        
        # Decode to image
        img = self.to_image(x)
        
        return img


# ==================== Stage 2: High Resolution Refiner ====================
class HighResRefiner(nn.Module):
    """Refine low resolution images to high resolution using ViT"""
    def __init__(self, low_res_size: int = 64, high_res_size: int = 256,
                 embed_dim: int = 768, n_transformer_blocks: int = 12,
                 eeg_embed_dim: int = 512, use_plv: bool = False):
        super().__init__()
        self.low_res_size = low_res_size
        self.high_res_size = high_res_size
        self.use_plv = use_plv
        
        # Patch embedding for low-res image
        patch_size = 8
        self.patch_embed = PatchEmbedding(low_res_size, patch_size, 3, embed_dim)
        n_patches = (low_res_size // patch_size) ** 2
        
        # EEG conditioning
        cond_dim = eeg_embed_dim + 512 if use_plv else eeg_embed_dim
        self.eeg_proj = nn.Linear(cond_dim, embed_dim)
        
        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, n_patches + 1, embed_dim))
        self.eeg_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer blocks
        self.transformer = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads=12, dropout=0.1)
            for _ in range(n_transformer_blocks)
        ])
        
        # Upsampling decoder
        self.decoder = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 16 * 16 * 256),
            Rearrange('b n (h w c) -> b c (n h) w', h=16, w=16, c=256),
            
            # Upsample to 128x128
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            # Upsample to 256x256
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            # Final refinement
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, low_res_img: torch.Tensor, eeg_embed: torch.Tensor,
                plv_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            low_res_img: (batch, 3, low_res_size, low_res_size)
            eeg_embed: (batch, eeg_embed_dim)
            plv_embed: (batch, 512) if use_plv
        Returns:
            high_res_img: (batch, 3, high_res_size, high_res_size)
        """
        B = low_res_img.shape[0]
        
        # Combine embeddings if using PLV
        if self.use_plv and plv_embed is not None:
            combined_embed = torch.cat([eeg_embed, plv_embed], dim=-1)
        else:
            combined_embed = eeg_embed
        
        # Patch embedding
        x = self.patch_embed(low_res_img)  # (B, n_patches, embed_dim)
        
        # Add EEG conditioning token
        eeg_cond = self.eeg_proj(combined_embed).unsqueeze(1)  # (B, 1, embed_dim)
        eeg_token = self.eeg_token.expand(B, -1, -1) + eeg_cond
        
        x = torch.cat([eeg_token, x], dim=1)  # (B, n_patches+1, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embedding
        
        # Apply transformer blocks
        for block in self.transformer:
            x = block(x)
        
        # Remove EEG token and decode
        x = x[:, 1:, :]  # Remove first token
        
        # Decode to high-res image
        high_res_img = self.decoder(x)
        
        return high_res_img


# ==================== Complete EEG-to-Image Model ====================
class EEGToImageGenerator(nn.Module):
    """Complete two-stage EEG to image generation model"""
    def __init__(self, 
                 n_channels: int = 64,
                 time_points: int = 500,
                 low_res_size: int = 64,
                 high_res_size: int = 256,
                 use_plv: bool = False):
        super().__init__()
        self.use_plv = use_plv
        
        # EEG Encoder
        self.eeg_encoder = EEGEncoder(
            n_channels=n_channels,
            time_points=time_points,
            embed_dim=512,
            use_plv=use_plv
        )
        
        # Stage 1: Low-res generator
        self.low_res_generator = LowResGenerator(
            eeg_embed_dim=512,
            img_size=low_res_size,
            n_transformer_blocks=6,
            use_plv=use_plv
        )
        
        # Stage 2: High-res refiner
        self.high_res_refiner = HighResRefiner(
            low_res_size=low_res_size,
            high_res_size=high_res_size,
            embed_dim=768,
            n_transformer_blocks=12,
            eeg_embed_dim=512,
            use_plv=use_plv
        )
        
    def forward(self, eeg_signal: torch.Tensor, 
                return_low_res: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            eeg_signal: (batch, n_channels, time_points)
            return_low_res: whether to return low-res image
        Returns:
            high_res_img: (batch, 3, high_res_size, high_res_size)
            low_res_img: (batch, 3, low_res_size, low_res_size) if return_low_res
        """
        # Encode EEG
        eeg_embed, plv_embed = self.eeg_encoder(eeg_signal)
        
        # Stage 1: Generate low-res image
        low_res_img = self.low_res_generator(eeg_embed, plv_embed)
        
        # Stage 2: Refine to high-res
        high_res_img = self.high_res_refiner(low_res_img, eeg_embed, plv_embed)
        
        if return_low_res:
            return high_res_img, low_res_img
        return high_res_img, None


# ==================== Training Example ====================
def train_step_example():
    """Example training step"""
    # Hyperparameters
    batch_size = 8
    n_channels = 64
    time_points = 500
    use_plv = True  # Set to True to use PLV connectivity
    
    # Initialize model
    model = EEGToImageGenerator(
        n_channels=n_channels,
        time_points=time_points,
        low_res_size=64,
        high_res_size=256,
        use_plv=use_plv
    )
    
    # Dummy data
    eeg_data = torch.randn(batch_size, n_channels, time_points)
    target_images = torch.randn(batch_size, 3, 256, 256)
    
    # Forward pass
    generated_img, low_res_img = model(eeg_data, return_low_res=True)
    
    print(f"EEG input shape: {eeg_data.shape}")
    print(f"Low-res output shape: {low_res_img.shape}")
    print(f"High-res output shape: {generated_img.shape}")
    print(f"Using PLV connectivity: {use_plv}")
    
    # Loss computation (example with MSE and perceptual loss)
    mse_loss = F.mse_loss(generated_img, target_images)
    low_res_loss = F.mse_loss(
        F.interpolate(low_res_img, size=256, mode='bilinear'),
        target_images
    )
    
    total_loss = mse_loss + 0.5 * low_res_loss
    
    print(f"\nLoss: {total_loss.item():.4f}")
    
    return model, total_loss


if __name__ == "__main__":
    print("EEG-to-Image Generator with Vision Transformer")
    print("=" * 60)
    train_step_example()
