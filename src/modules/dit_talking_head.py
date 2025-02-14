import torch
import torch.nn as nn
import torch.nn.functional as F
import platform

from .common import PositionalEncoding, enc_dec_mask, pad_audio
from ..config.base_config import make_abs_path
import math
from logger import logger


class DiffusionSchedule(nn.Module):
    def __init__(self, num_steps, mode='linear', beta_1=1e-4, beta_T=0.02, s=0.008):
        super().__init__()

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, num_steps)
        elif mode == 'quadratic':
            betas = torch.linspace(beta_1 ** 0.5, beta_T ** 0.5, num_steps) ** 2
        elif mode == 'sigmoid':
            betas = torch.sigmoid(torch.linspace(-5, 5, num_steps)) * (beta_T - beta_1) + beta_1
        elif mode == 'cosine':
            steps = num_steps + 1
            x = torch.linspace(0, num_steps, steps)
            alpha_bars = torch.cos(((x / num_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alpha_bars = alpha_bars / alpha_bars[0]
            betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
            betas = torch.clip(betas, 0.0001, 0.999)
        else:
            raise ValueError(f'Unknown diffusion schedule {mode}!')
        betas = torch.cat([torch.zeros(1), betas], dim=0)  # Padding beta_0 = 0

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.shape[0]):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.shape[0]):
            sigmas_inflex[i] = ((1 - alpha_bars[i - 1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.num_steps = num_steps
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = torch.randint(1, self.num_steps + 1, (batch_size,))
        return ts.tolist()

    def get_sigmas(self, t, flexibility=0):
        assert 0 <= flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas


class LSHTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.norm = nn.LayerNorm(decoder_layer.linear1.in_features)
        self.num_layers = num_layers
        
        logger.debug(f"Initializing LSH Transformer Decoder with {num_layers} layers")
        logger.debug(f"Layer normalization dimension: {decoder_layer.linear1.in_features}")
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        Args:
            tgt: Target sequence (N, L, d_model)
            memory: Memory from encoder (N, S, d_model)
            tgt_mask: Target sequence mask (optional)
            memory_mask: Memory mask (optional)
        """
        output = tgt
        
        logger.debug(f"\nLSH Transformer Decoder Forward Pass")
        logger.debug(f"Input shapes - tgt: {tgt.shape}, memory: {memory.shape}")
        if tgt_mask is not None:
            logger.debug(f"Target mask shape: {tgt_mask.shape}")
        if memory_mask is not None:
            logger.debug(f"Memory mask shape: {memory_mask.shape}")

        # Debug tensor statistics
        with torch.no_grad():
            logger.debug(f"Input statistics:")
            logger.debug(f"  tgt - mean: {tgt.mean():.4f}, std: {tgt.std():.4f}")
            logger.debug(f"  memory - mean: {memory.mean():.4f}, std: {memory.std():.4f}")
            
        # Process through decoder layers
        for idx, layer in enumerate(self.layers):
            logger.debug(f"\nProcessing decoder layer {idx + 1}/{self.num_layers}")
            
            # Record statistics before layer
            with torch.no_grad():
                pre_mean = output.mean().item()
                pre_std = output.std().item()
            
            # Forward pass through layer
            output = layer(output, memory, tgt_mask, memory_mask)
            
            # Record statistics after layer
            with torch.no_grad():
                post_mean = output.mean().item()
                post_std = output.std().item()
                logger.debug(f"Layer {idx + 1} statistics:")
                logger.debug(f"  Pre  - mean: {pre_mean:.4f}, std: {pre_std:.4f}")
                logger.debug(f"  Post - mean: {post_mean:.4f}, std: {post_std:.4f}")
            
            # Check for NaN/Inf values
            if torch.isnan(output).any():
                logger.error(f"NaN detected in output of layer {idx + 1}")
            if torch.isinf(output).any():
                logger.error(f"Inf detected in output of layer {idx + 1}")

        # Final normalization
        output = self.norm(output)
        
        # Final output statistics
        with torch.no_grad():
            logger.debug(f"\nFinal output statistics:")
            logger.debug(f"  Shape: {output.shape}")
            logger.debug(f"  Mean: {output.mean():.4f}")
            logger.debug(f"  Std: {output.std():.4f}")
            logger.debug(f"  Min: {output.min():.4f}")
            logger.debug(f"  Max: {output.max():.4f}")
            
        return output
        
    def debug_attention_patterns(self, attn_weights):
        """Debug helper to analyze attention patterns"""
        with torch.no_grad():
            # Average attention weights across heads
            avg_attn = attn_weights.mean(dim=1)  # Average across heads
            
            logger.debug(f"\nAttention Pattern Analysis:")
            logger.debug(f"  Shape: {attn_weights.shape}")
            logger.debug(f"  Mean attention weight: {avg_attn.mean():.4f}")
            logger.debug(f"  Max attention weight: {avg_attn.max():.4f}")
            
            # Check for attention concentration
            top_k = 5
            values, _ = torch.topk(avg_attn.flatten(), top_k)
            logger.debug(f"  Top {top_k} attention weights: {values.tolist()}")
            
            # Check for uniform attention
            uniformity = -(avg_attn * torch.log(avg_attn + 1e-9)).sum(dim=-1).mean()
            logger.debug(f"  Attention uniformity (entropy): {uniformity:.4f}")


import math

class LSHAttention(nn.Module):
    def __init__(self, feature_dim, n_heads=8, n_hashes=4, bucket_size=64):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_heads = n_heads
        self.n_hashes = n_hashes
        self.bucket_size = bucket_size
        self.head_dim = feature_dim // n_heads
        
        logger.debug(f"Initializing LSH Attention with dims: feature={feature_dim}, heads={n_heads}, head_dim={self.head_dim}")
        
        self.qk_proj = nn.Linear(feature_dim, feature_dim * 2)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
        # Random rotation matrix for LSH
        self.register_buffer('random_rotations', 
            torch.randn(n_heads, n_hashes, self.head_dim, device='cuda') * 0.02)

    def hash_vectors(self, vectors):
        """Hash vectors to buckets using random rotations"""
        # vectors: [batch, seq_len, n_heads, head_dim]
        logger.debug(f"Hashing vectors shape: {vectors.shape}")
        logger.debug(f"Random rotations shape: {self.random_rotations.shape}")
        
        # Reshape for batch and heads
        B, L, H, D = vectors.shape
        vectors = vectors.permute(0, 2, 1, 3)  # [B, H, L, D]
        
        # Multiply by random rotations
        rotated = torch.einsum('bhld,hrd->bhlr', vectors, self.random_rotations)
        logger.debug(f"Rotated vectors shape: {rotated.shape}")
        
        # Convert to buckets
        buckets = torch.argmax(torch.cat([rotated, -rotated], dim=-1), dim=-1)  # [B, H, L]
        logger.debug(f"Buckets shape: {buckets.shape}")
        
        return buckets

    def sort_by_buckets(self, buckets, *tensors):
        """Sort tensors according to bucket assignments"""
        B, H, L = buckets.shape  # [batch_size, n_heads, seq_len]
        logger.debug(f"Sort by buckets - buckets shape: {buckets.shape}")
        
        for i, tensor in enumerate(tensors):
            logger.debug(f"Sort by buckets - tensor {i} shape: {tensor.shape}")
        
        # Get indices for sorting
        indices = buckets.argsort(dim=-1)  # [B, H, L]
        logger.debug(f"Sorted indices shape: {indices.shape}")
        
        # Sort all tensors according to bucket ordering
        sorted_tensors = []
        for tensor in tensors:
            # tensor shape: [batch, seq_len, n_heads, head_dim]
            # Reshape to [batch, n_heads, seq_len, head_dim]
            tensor = tensor.permute(0, 2, 1, 3)
            
            # Gather elements using sorted indices
            gather_index = indices.unsqueeze(-1).expand(-1, -1, -1, tensor.size(-1))
            sorted_tensor = torch.gather(tensor, dim=2, index=gather_index)
            
            # Reshape back to [batch, seq_len, n_heads, head_dim]
            sorted_tensor = sorted_tensor.permute(0, 2, 1, 3)
            logger.debug(f"Sorted tensor shape: {sorted_tensor.shape}")
            
            sorted_tensors.append(sorted_tensor)
            
        return tuple(sorted_tensors) + (indices,)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        logger.debug(f"\nLSH Attention Forward - Input shape: {x.shape}")
        if mask is not None:
            logger.debug(f"Input mask shape: {mask.shape}")
        
        # Project queries, keys and values
        qk = self.qk_proj(x)
        q, k = qk.chunk(2, dim=-1)
        v = self.v_proj(x)
        
        # Split heads
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        logger.debug(f"After head split - Q shape: {q.shape}")
        
        # Compute LSH buckets
        buckets = self.hash_vectors(q)
        logger.debug(f"Computed buckets shape: {buckets.shape}")
        
        # Sort according to buckets
        sorted_q, sorted_k, sorted_v, indices = self.sort_by_buckets(buckets, q, k, v)
        logger.debug(f"After sorting - Q shape: {sorted_q.shape}")
        
        # Compute attention scores
        sorted_q = sorted_q.permute(0, 2, 1, 3)  # [B, H, L, D]
        sorted_k = sorted_k.permute(0, 2, 1, 3)  # [B, H, L, D]
        sorted_v = sorted_v.permute(0, 2, 1, 3)  # [B, H, L, D]
        
        dots = torch.matmul(sorted_q, sorted_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        logger.debug(f"Attention scores shape: {dots.shape}")
        
        # Handle masking
        if mask is not None:
            # Create proper mask for bucketed attention
            if mask.dim() == 2:
                # Expand mask to match batch size and heads
                mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, L]
                logger.debug(f"Expanded 2D mask shape: {mask.shape}")
            mask = mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)  # [B, H, L, L]
            logger.debug(f"Final mask shape before sorting: {mask.shape}")
            
            # Sort mask according to indices
            indices_expanded_q = indices.unsqueeze(-1).expand(-1, -1, -1, seq_len)
            indices_expanded_k = indices.unsqueeze(-2).expand(-1, -1, seq_len, -1)
            
            # Apply the same sorting to mask rows and columns
            mask_sorted = mask.gather(2, indices_expanded_q)
            mask_sorted = mask_sorted.gather(3, indices_expanded_k)
            logger.debug(f"Sorted mask shape: {mask_sorted.shape}")
            
            dots = dots.masked_fill(~mask_sorted, float('-inf'))
        
        # Apply softmax
        attn = F.softmax(dots, dim=-1)
        logger.debug(f"Attention weights shape: {attn.shape}")
        
        # Apply attention to values
        out = torch.matmul(attn, sorted_v)  # [B, H, L, D]
        logger.debug(f"After attention shape: {out.shape}")
        
        # Restore original ordering
        restore_indices = indices.argsort(dim=-1)
        restore_indices = restore_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        out = torch.gather(out, dim=2, index=restore_indices)
        
        # Reshape to [B, L, H, D]
        out = out.permute(0, 2, 1, 3)
        logger.debug(f"After restoring order shape: {out.shape}")
        
        # Combine heads
        out = out.reshape(batch_size, seq_len, self.feature_dim)
        logger.debug(f"Final output shape: {out.shape}")
        
        return self.out_proj(out)

    
class LSHTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048):
        super().__init__()
        self.self_attn = LSHAttention(d_model, nhead)
        self.cross_attn = LSHAttention(d_model, nhead)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(0.1)
        self.activation = F.gelu

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        logger.debug(f"\nLSH Transformer Decoder Layer - Input shapes: tgt={tgt.shape}, memory={memory.shape}")
        
        # Self attention
        tgt2 = self.self_attn(self.norm1(tgt), mask=tgt_mask)
        tgt = tgt + self.dropout(tgt2)
        logger.debug(f"After self attention shape: {tgt.shape}")
        
        # Cross attention
        tgt2 = self.cross_attn(self.norm2(tgt), mask=memory_mask)
        tgt = tgt + self.dropout(tgt2)
        logger.debug(f"After cross attention shape: {tgt.shape}")
        
        # Feedforward
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm3(tgt)))))
        tgt = tgt + self.dropout(tgt2)
        logger.debug(f"After feedforward shape: {tgt.shape}")
        
        return tgt
      
class DitTalkingHead(nn.Module):
    def __init__(self, device='cuda', target="sample", architecture="decoder",
                 motion_feat_dim=76, fps=25, n_motions=100, n_prev_motions=10, 
                 audio_model="hubert", feature_dim=512, n_diff_steps=500, diff_schedule="cosine", 
                 cfg_mode="incremental", guiding_conditions="audio,"):
        super().__init__()

        # Model parameters
        self.target = target # 预测原始图像还是预测噪声
        self.architecture = architecture
        self.motion_feat_dim = motion_feat_dim # motion 特征维度
        self.fps = fps
        self.n_motions = n_motions # 当前motion100个, window_length, T_w
        self.n_prev_motions = n_prev_motions # 前续motion 10个, T_p
        self.feature_dim = feature_dim

        # Audio encoder
        self.audio_model = audio_model
        if self.audio_model == 'wav2vec2':
            print("using wav2vec2 audio encoder ...")
            from .wav2vec2 import Wav2Vec2Model
            self.audio_encoder = Wav2Vec2Model.from_pretrained(make_abs_path('../../pretrained_weights/wav2vec2-base-960h'))
            self.audio_encoder.feature_extractor._freeze_parameters()
        elif self.audio_model == 'hubert': # 根据经验，hubert特征提取器效果更好
            from .hubert import HubertModel
            self.audio_encoder = HubertModel.from_pretrained(make_abs_path('../../pretrained_weights/hubert-base-ls960'))
            self.audio_encoder.feature_extractor._freeze_parameters()
        elif self.audio_model == 'hubert_zh_ori' or self.audio_model == 'hubert_zh': # 根据经验，hubert特征提取器效果更好
            print("using hubert chinese ori")
            model_path = '../../pretrained_weights/TencentGameMate:chinese-hubert-base'
            if platform.system() == "Windows":
                model_path = '../../pretrained_weights/chinese-hubert-base'
            from .hubert import HubertModel
            self.audio_encoder = HubertModel.from_pretrained(make_abs_path(model_path))
            self.audio_encoder.feature_extractor._freeze_parameters()
        else:
            raise ValueError(f'Unknown audio model {self.audio_model}!')

        if architecture == 'decoder':
            self.audio_feature_map = nn.Linear(768, feature_dim)
            self.start_audio_feat = nn.Parameter(torch.randn(1, self.n_prev_motions, feature_dim))
        else:
            raise ValueError(f'Unknown architecture {architecture}!')

        self.start_motion_feat = nn.Parameter(torch.randn(1, self.n_prev_motions, self.motion_feat_dim)) # 1, 10, 76

        # Diffusion model
        self.denoising_net = DenoisingNetwork(device=device, n_motions=self.n_motions, n_prev_motions=self.n_prev_motions, 
                                              motion_feat_dim=self.motion_feat_dim, feature_dim=feature_dim)
        # diffusion schedule
        self.diffusion_sched = DiffusionSchedule(n_diff_steps, diff_schedule)

        # Classifier-free settings
        self.cfg_mode = cfg_mode
        guiding_conditions = guiding_conditions.split(',') if guiding_conditions else []
        self.guiding_conditions = [cond for cond in guiding_conditions if cond in ['audio']]
        if 'audio' in self.guiding_conditions:
            audio_feat_dim = feature_dim
            self.null_audio_feat = nn.Parameter(torch.randn(1, 1, audio_feat_dim)) # 1, 1, 512

        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, motion_feat, audio_or_feat, prev_motion_feat=None, prev_audio_feat=None, time_step=None, indicator=None):
        """
        Args:
            motion_feat: (N, L, d_coef) motion coefficients or features
            audio_or_feat: (N, L_audio) raw audio or audio feature
            prev_motion_feat: (N, n_prev_motions, d_motion) previous motion coefficients or feature
            prev_audio_feat: (N, n_prev_motions, d_audio) previous audio features
            time_step: (N,)
            indicator: (N, L) 0/1 indicator of real (unpadded) motion coefficients

        Returns:
           motion_feat_noise: (N, L, d_motion)
        """
        batch_size = motion_feat.shape[0]

        # 加载语音特征
        if audio_or_feat.ndim == 2: # 原始语音
            # Extract audio features
            assert audio_or_feat.shape[1] == 16000 * self.n_motions / self.fps, \
                f'Incorrect audio length {audio_or_feat.shape[1]}'
            audio_feat_saved = self.extract_audio_feature(audio_or_feat)  # (N, L, feature_dim)
        elif audio_or_feat.ndim == 3: # 语音特征
            assert audio_or_feat.shape[1] == self.n_motions, f'Incorrect audio feature length {audio_or_feat.shape[1]}'
            audio_feat_saved = audio_or_feat
        else:
            raise ValueError(f'Incorrect audio input shape {audio_or_feat.shape}')
        audio_feat = audio_feat_saved.clone()

        # 前续motion特征
        if prev_motion_feat is None:
            prev_motion_feat = self.start_motion_feat.expand(batch_size, -1, -1)  # (N, n_prev_motions, d_motion)
        
        # 前续语音特征
        if prev_audio_feat is None:
            # (N, n_prev_motions, feature_dim)
            prev_audio_feat = self.start_audio_feat.expand(batch_size, -1, -1)

        # Classifier-free guidance
        if len(self.guiding_conditions) > 0:
            assert len(self.guiding_conditions) <= 2, 'Only support 1 or 2 CFG conditions!'
            if len(self.guiding_conditions) == 1 or self.cfg_mode == 'independent':
                null_cond_prob = 0.5 if len(self.guiding_conditions) >= 2 else 0.1
                if 'audio' in self.guiding_conditions:
                    mask_audio = torch.rand(batch_size, device=self.device) < null_cond_prob
                    audio_feat = torch.where(mask_audio.view(-1, 1, 1),
                                             self.null_audio_feat.expand(batch_size, self.n_motions, -1),
                                             audio_feat)
            else:
                # len(self.guiding_conditions) > 1 and self.cfg_mode == 'incremental'
                # full (0.45), w/o style (0.45), w/o style or audio (0.1)
                mask_flag = torch.rand(batch_size, device=self.device)
                if 'audio' in self.guiding_conditions:
                    mask_audio = mask_flag > 0.9
                    audio_feat = torch.where(mask_audio.view(-1, 1, 1),
                                             self.null_audio_feat.expand(batch_size, self.n_motions, -1),
                                             audio_feat)

        if time_step is None:
            # Sample time step
            time_step = self.diffusion_sched.uniform_sample_t(batch_size)  # (N,)

        # The forward diffusion process
        alpha_bar = self.diffusion_sched.alpha_bars[time_step]  # (N,)
        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)  # (N, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)  # (N, 1, 1)

        eps = torch.randn_like(motion_feat)  # (N, L, d_motion)
        motion_feat_noisy = c0 * motion_feat + c1 * eps

        # The reverse diffusion process
        motion_feat_target = self.denoising_net(motion_feat_noisy, audio_feat, 
                                                prev_motion_feat, prev_audio_feat, time_step, indicator)

        return eps, motion_feat_target, motion_feat.detach(), audio_feat_saved.detach()

    def extract_audio_feature(self, audio, frame_num=None):
        frame_num = frame_num or self.n_motions

        # # Strategy 1: resample during audio feature extraction
        # hidden_states = self.audio_encoder(pad_audio(audio), self.fps, frame_num=frame_num).last_hidden_state  # (N, L, 768)

        # Strategy 2: resample after audio feature extraction (BackResample)
        hidden_states = self.audio_encoder(pad_audio(audio), self.fps,
                                           frame_num=frame_num * 2).last_hidden_state  # (N, 2L, 768)
        hidden_states = hidden_states.transpose(1, 2)  # (N, 768, 2L)
        hidden_states = F.interpolate(hidden_states, size=frame_num, align_corners=False, mode='linear')  # (N, 768, L)
        hidden_states = hidden_states.transpose(1, 2)  # (N, L, 768)

        audio_feat = self.audio_feature_map(hidden_states)  # (N, L, feature_dim)
        return audio_feat

    @torch.no_grad()
    def sample(self, audio_or_feat, prev_motion_feat=None, prev_audio_feat=None,
               motion_at_T=None, indicator=None, cfg_mode=None, cfg_cond=None, cfg_scale=1.15, flexibility=0,
               dynamic_threshold=None, ret_traj=False):
        # Check and convert inputs
        batch_size = audio_or_feat.shape[0]

        # Check CFG conditions
        if cfg_mode is None:  # Use default CFG mode
            cfg_mode = self.cfg_mode
        if cfg_cond is None:  # Use default CFG conditions
            cfg_cond = self.guiding_conditions
        cfg_cond = [c for c in cfg_cond if c in ['audio', ]]

        if not isinstance(cfg_scale, list):
            cfg_scale = [cfg_scale] * len(cfg_cond)

        # sort cfg_cond and cfg_scale
        if len(cfg_cond) > 0:
            cfg_cond, cfg_scale = zip(*sorted(zip(cfg_cond, cfg_scale), key=lambda x: ['audio', ].index(x[0])))
        else:
            cfg_cond, cfg_scale = [], []

        if audio_or_feat.ndim == 2:
            # Extract audio features
            assert audio_or_feat.shape[1] == 16000 * self.n_motions / self.fps, \
                f'Incorrect audio length {audio_or_feat.shape[1]}'
            audio_feat = self.extract_audio_feature(audio_or_feat)  # (N, L, feature_dim)
        elif audio_or_feat.ndim == 3:
            assert audio_or_feat.shape[1] == self.n_motions, f'Incorrect audio feature length {audio_or_feat.shape[1]}'
            audio_feat = audio_or_feat
        else:
            raise ValueError(f'Incorrect audio input shape {audio_or_feat.shape}')

        if prev_motion_feat is None:
            prev_motion_feat = self.start_motion_feat.expand(batch_size, -1, -1)  # (N, n_prev_motions, d_motion)
        if prev_audio_feat is None:
            # (N, n_prev_motions, feature_dim)
            prev_audio_feat = self.start_audio_feat.expand(batch_size, -1, -1)

        if motion_at_T is None:
            motion_at_T = torch.randn((batch_size, self.n_motions, self.motion_feat_dim)).to(self.device)

        # Prepare input for the reverse diffusion process (including optional classifier-free guidance)
        if 'audio' in cfg_cond:
            audio_feat_null = self.null_audio_feat.expand(batch_size, self.n_motions, -1)
        else:
            audio_feat_null = audio_feat

        audio_feat_in = [audio_feat_null]
        for cond in cfg_cond:
            if cond == 'audio':
                audio_feat_in.append(audio_feat)

        n_entries = len(audio_feat_in)
        audio_feat_in = torch.cat(audio_feat_in, dim=0)
        prev_motion_feat_in = torch.cat([prev_motion_feat] * n_entries, dim=0)
        prev_audio_feat_in = torch.cat([prev_audio_feat] * n_entries, dim=0)
        indicator_in = torch.cat([indicator] * n_entries, dim=0) if indicator is not None else None

        traj = {self.diffusion_sched.num_steps: motion_at_T}
        for t in range(self.diffusion_sched.num_steps, 0, -1):
            if t > 1:
                z = torch.randn_like(motion_at_T)
            else:
                z = torch.zeros_like(motion_at_T)

            alpha = self.diffusion_sched.alphas[t]
            alpha_bar = self.diffusion_sched.alpha_bars[t]
            alpha_bar_prev = self.diffusion_sched.alpha_bars[t - 1]
            sigma = self.diffusion_sched.get_sigmas(t, flexibility)

            motion_at_t = traj[t]
            motion_in = torch.cat([motion_at_t] * n_entries, dim=0)
            step_in = torch.tensor([t] * batch_size, device=self.device)
            step_in = torch.cat([step_in] * n_entries, dim=0)

            results = self.denoising_net(motion_in, audio_feat_in, prev_motion_feat_in,
                                         prev_audio_feat_in, step_in, indicator_in)

            # Apply thresholding if specified
            if dynamic_threshold:
                dt_ratio, dt_min, dt_max = dynamic_threshold
                abs_results = results[:, -self.n_motions:].reshape(batch_size * n_entries, -1).abs()
                s = torch.quantile(abs_results, dt_ratio, dim=1)
                s = torch.clamp(s, min=dt_min, max=dt_max)
                s = s[..., None, None]
                results = torch.clamp(results, min=-s, max=s)

            results = results.chunk(n_entries)

            # Unconditional target (CFG) or the conditional target (non-CFG)
            target_theta = results[0][:, -self.n_motions:]
            # Classifier-free Guidance (optional)
            for i in range(0, n_entries - 1):
                if cfg_mode == 'independent':
                    target_theta += cfg_scale[i] * (
                                results[i + 1][:, -self.n_motions:] - results[0][:, -self.n_motions:])
                elif cfg_mode == 'incremental':
                    target_theta += cfg_scale[i] * (
                                results[i + 1][:, -self.n_motions:] - results[i][:, -self.n_motions:])
                else:
                    raise NotImplementedError(f'Unknown cfg_mode {cfg_mode}')

            if self.target == 'noise':
                c0 = 1 / torch.sqrt(alpha)
                c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)
                motion_next = c0 * (motion_at_t - c1 * target_theta) + sigma * z
            elif self.target == 'sample':
                c0 = (1 - alpha_bar_prev) * torch.sqrt(alpha) / (1 - alpha_bar)
                c1 = (1 - alpha) * torch.sqrt(alpha_bar_prev) / (1 - alpha_bar)
                motion_next = c0 * motion_at_t + c1 * target_theta + sigma * z
            else:
                raise ValueError('Unknown target type: {}'.format(self.target))

            traj[t - 1] = motion_next.detach()  # Stop gradient and save trajectory.
            traj[t] = traj[t].cpu()  # Move previous output to CPU memory.
            if not ret_traj:
                del traj[t]

        if ret_traj:
            return traj, motion_at_T, audio_feat
        else:
            return traj[0], motion_at_T, audio_feat


class DenoisingNetwork(nn.Module):
    def __init__(self, device='cuda', motion_feat_dim=76, 
                 use_indicator=None, architecture="decoder", feature_dim=512, n_heads=8, 
                 n_layers=8, mlp_ratio=4, align_mask_width=1, no_use_learnable_pe=True, n_prev_motions=10,
                 n_motions=100, n_diff_steps=500):
        super().__init__()

        # Model parameters
        self.motion_feat_dim = motion_feat_dim 
        self.use_indicator = use_indicator
        self.architecture = architecture
        self.feature_dim = feature_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.mlp_ratio = mlp_ratio
        self.align_mask_width = align_mask_width
        self.use_learnable_pe = not no_use_learnable_pe
        self.n_prev_motions = n_prev_motions
        self.n_motions = n_motions

        # Important: Initialize feature_proj first since it's used in forward
        if self.architecture == 'decoder':
            self.feature_proj = nn.Linear(self.motion_feat_dim + (1 if self.use_indicator else 0),
                                        self.feature_dim)

        # Temporal embedding for diffusion time step
        self.TE = PositionalEncoding(self.feature_dim, max_len=n_diff_steps + 1)
        self.diff_step_map = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

        # Positional encoding
        if self.use_learnable_pe:
            self.PE = nn.Parameter(torch.randn(1, 1 + self.n_prev_motions + self.n_motions, self.feature_dim))
        else:
            self.PE = PositionalEncoding(self.feature_dim)

        # LSH Transformer decoder
        if self.architecture == 'decoder':
            decoder_layer = LSHTransformerDecoderLayer(
                d_model=self.feature_dim,
                nhead=self.n_heads,
                dim_feedforward=self.mlp_ratio * self.feature_dim
            )
            self.transformer = LSHTransformerDecoder(decoder_layer, num_layers=self.n_layers)
            
            if self.align_mask_width > 0:
                motion_len = self.n_prev_motions + self.n_motions
                alignment_mask = enc_dec_mask(motion_len, motion_len, 
                                           frame_width=1, 
                                           expansion=self.align_mask_width - 1)
                self.register_buffer('alignment_mask', alignment_mask)
            else:
                self.alignment_mask = None
        else:
            raise ValueError(f'Unknown architecture: {self.architecture}')

        # Motion decoder
        self.motion_dec = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.GELU(),
            nn.Linear(self.feature_dim // 2, self.motion_feat_dim),
        )

        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, motion_feat, audio_feat, prev_motion_feat, prev_audio_feat, step, indicator=None):
        # Handle indicator if present
        if indicator is not None:
            indicator = torch.cat([
                torch.zeros((indicator.shape[0], self.n_prev_motions), device=indicator.device),
                indicator
            ], dim=1).unsqueeze(-1)

        # Concat features
        feats_in = torch.cat([prev_motion_feat, motion_feat], dim=1)
        if self.use_indicator:
            feats_in = torch.cat([feats_in, indicator], dim=-1)

        # Project features
        feats_in = self.feature_proj(feats_in)

        # Add positional and temporal embeddings
        diff_step_embedding = self.diff_step_map(self.TE.pe[0, step]).unsqueeze(1)
        if self.use_learnable_pe:
            feats_in = feats_in + self.PE + diff_step_embedding
        else:
            feats_in = self.PE(feats_in) + diff_step_embedding

        # Transformer with LSH attention
        audio_feat_in = torch.cat([prev_audio_feat, audio_feat], dim=1)
        feat_out = self.transformer(feats_in, audio_feat_in, memory_mask=self.alignment_mask)

        # Decode motion features
        motion_feat_target = self.motion_dec(feat_out)

        return motion_feat_target
if __name__ == "__main__":
    device = "cuda"
    motion_feat_dim = 76
    n_motions = 100 # L
    n_prev_motions = 10 # L_p

    L_audio = int(16000 * n_motions / 25) # 64000
    d_audio = 768

    N = 5
    feature_dim = 512

    motion_feat = torch.ones((N, n_motions, motion_feat_dim)).to(device)
    prev_motion_feat = torch.ones((N, n_prev_motions, motion_feat_dim)).to(device)

    audio_or_feat = torch.ones((N, L_audio)).to(device)
    prev_audio_feat = torch.ones((N, n_prev_motions, d_audio)).to(device)

    time_step = torch.ones(N, dtype=torch.long).to(device)

    model = DitTalkingHead().to(device)

    z = model(motion_feat, audio_or_feat, prev_motion_feat=None, 
              prev_audio_feat=None, time_step=None, indicator=None)
    traj, motion_at_T, audio_feat = z[0], z[1], z[2]
    print(motion_at_T.shape, audio_feat.shape)