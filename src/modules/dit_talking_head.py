import torch
import torch.nn as nn
import torch.nn.functional as F
import platform

from .common import PositionalEncoding, enc_dec_mask, pad_audio
from ..config.base_config import make_abs_path
import math
from logger import logger
import numpy as np


def format_shape_info(self, x):
    """Helper to safely get shape info for different types."""
    if torch.is_tensor(x):
        return f"{x.shape} (tensor)"
    elif isinstance(x, list):
        return f"list of len {len(x)}"
    elif isinstance(x, np.ndarray):
        return f"{x.shape} (numpy)"
    else:
        return f"type: {type(x)}"
        

        
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


class DeformableExpressionAttention(nn.Module):
    def __init__(self, dim, n_heads=8, n_points=4):
        super().__init__()
        self.n_heads = n_heads
        self.n_points = n_points
        self.dim_per_head = dim // n_heads
        
        # Value projection
        self.value_proj = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)
        
        # Query/Key transformations
        self.qk_proj = nn.Linear(dim, dim * 2)
        
        # Reference point prediction
        self.sampling_offsets = nn.Linear(dim, n_heads * n_points * 2)
        
        # Attention weights
        self.attention_weights = nn.Linear(dim, n_heads * n_points)
        
        self._reset_parameters()

    def log_tensor_stats(self, name, tensor):
        """Helper to log tensor statistics."""
        if torch.is_tensor(tensor):
            logger.debug(f"  {name}:")
            logger.debug(f"    shape: {tensor.shape}")
            logger.debug(f"    mean: {tensor.float().mean():.4f}")
            logger.debug(f"    std: {tensor.float().std():.4f}")
            logger.debug(f"    min: {tensor.float().min():.4f}")
            logger.debug(f"    max: {tensor.float().max():.4f}")
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qk_proj.weight)
        nn.init.zeros_(self.qk_proj.bias)
        
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)
        
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        
        # Initialize sampling offsets
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([torch.cos(thetas), torch.sin(thetas)], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        grid_init = grid_init.view(self.n_heads, 1, 2).repeat(1, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, i, :] *= 0.5 * (i + 1) / self.n_points
        self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        
        # Initialize attention weights
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        
    def forward(self, query, key, value, pos=None, attn_mask=None):
        """
        Args:
            query: (B, L, D)
            key: (B, L, D)
            value: (B, L, D)
            pos: Optional positional encoding (B, L, D)
            attn_mask: Optional attention mask (L, L) or (B, L, L)
        """
        logger.debug("\n=== DeformableExpressionAttention Forward Pass ===")
        
        batch_size, seq_len, _ = value.shape
        logger.debug(f"Batch size: {batch_size}, Sequence length: {seq_len}")
        
        # Add positional encodings if provided
        if pos is not None:
            query = query + pos
            key = key + pos
            
        # Project queries and keys
        logger.debug("\nProjecting queries and keys...")
        qk = self.qk_proj(query).chunk(2, dim=-1)
        q, k = qk[0], qk[1]
        
        # Project values
        logger.debug("\nProjecting values...")
        v = self.value_proj(value)
        
        # Generate reference points (normalized positions)
        ref_y = torch.linspace(0, 1, seq_len, device=query.device)
        ref_x = torch.linspace(0, 1, seq_len, device=query.device)
        ref_y, ref_x = torch.meshgrid(ref_y, ref_x, indexing='ij')
        reference_points = torch.stack([ref_x, ref_y], dim=-1)  # [L, L, 2]
        
        # Expand reference points for batch and heads
        reference_points = reference_points.unsqueeze(0).unsqueeze(2)  # [1, L, 1, L, 2]
        reference_points = reference_points.expand(batch_size, -1, self.n_heads, -1, -1)  # [B, L, H, L, 2]
        
        logger.debug(f"Reference points shape: {reference_points.shape}")
        self.log_tensor_stats("Reference points", reference_points)
        
        # Predict sampling offsets and attention weights
        logger.debug("\nPredicting sampling offsets and attention weights...")
        sampling_offsets = self.sampling_offsets(query).view(
            batch_size, seq_len, self.n_heads, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            batch_size, seq_len, self.n_heads, self.n_points
        )
        
        # Apply attention mask if provided
        if attn_mask is not None:
            logger.debug("\nApplying attention mask...")
            # Convert mask to boolean if it's not already
            if not attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.to(torch.bool)
            
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)  # [1, L, L]
                
            # Log original mask shape
            logger.debug(f"Original mask shape: {attn_mask.shape}")
            
            # Reshape mask to [B, L, 1, 1]
            attn_mask = attn_mask.unsqueeze(2).unsqueeze(3)  # [B, L, 1, 1]
            logger.debug(f"Reshaped mask shape: {attn_mask.shape}")
            
            # Create a broadcasting mask for the attention weights
            # attention_weights shape: [B, L, H, P]
            # Expand mask to [B, L, H, P]
            expanded_mask = attn_mask.expand(batch_size, seq_len, self.n_heads, self.n_points)
            logger.debug(f"Expanded mask shape: {expanded_mask.shape}")
            logger.debug(f"Attention weights shape before masking: {attention_weights.shape}")
            
            # Apply the mask
            mask_value = float('-inf')
            attention_weights = torch.where(expanded_mask, attention_weights, torch.tensor(mask_value, device=attention_weights.device))
        
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        logger.debug(f"Sampling offsets shape: {sampling_offsets.shape}")
        logger.debug(f"Attention weights shape after softmax: {attention_weights.shape}")
        
        # Calculate sampling locations with proper broadcasting
        logger.debug("\nCalculating sampling locations...")
        sampling_locations = (
            reference_points[:, :, :, :1, :] + 
            sampling_offsets
        )
        sampling_locations = sampling_locations.clamp(0, 1)
        
        logger.debug(f"Sampling locations shape: {sampling_locations.shape}")
        self.log_tensor_stats("Sampling locations", sampling_locations)
        
        # Sample features
        logger.debug("\nPreparing for feature sampling...")
        v = v.view(batch_size, seq_len, self.n_heads, self.dim_per_head)
        v = v.permute(0, 2, 3, 1).contiguous()  # [B, H, D/H, L]
        
        # Reshape for grid_sample
        v = v.view(batch_size * self.n_heads, self.dim_per_head, 1, seq_len)
        sampling_locs = sampling_locations.view(
            batch_size * self.n_heads, seq_len, self.n_points, 2
        )
        
        # Sample features
        logger.debug("\nPerforming grid sampling...")
        sampled_feats = F.grid_sample(
            v, sampling_locs,
            mode='bilinear', padding_mode='zeros', align_corners=False
        )
        
        # Reshape sampled features to match attention weights
        sampled_feats = sampled_feats.view(
            batch_size, self.n_heads, self.dim_per_head, seq_len, self.n_points
        )
        
        # Reshape attention weights for broadcasting
        attention_weights = attention_weights.permute(0, 2, 1, 3)  # [B, H, L, P]
        attention_weights = attention_weights.unsqueeze(2)  # [B, H, 1, L, P]
        
        logger.debug(f"Final sampled features shape: {sampled_feats.shape}")
        logger.debug(f"Final attention weights shape: {attention_weights.shape}")
        
        # Apply attention weights
        output = (sampled_feats * attention_weights).sum(-1)  # [B, H, D/H, L]
        
        # Final reshaping and projection
        output = output.permute(0, 3, 1, 2).contiguous()  # [B, L, H, D/H]
        output = output.view(batch_size, seq_len, -1)
        output = self.output_proj(output)
        
        logger.debug(f"Final output shape: {output.shape}")
        self.log_tensor_stats("Final output", output)
        
        return output 
    
class DeformableExpressionTransformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        n_heads=8,
        n_points=4,
        n_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        n_prev_motions=25,
        n_motions=100
    ):
        super().__init__()
        
        self.n_prev_motions = n_prev_motions
        self.n_motions = n_motions
        self.total_length = n_prev_motions + n_motions
        
        # Input projections
        self.exp_input_proj = nn.Linear(d_model, d_model)
        
        # Position encodings
        self.pos_encoding = nn.Parameter(torch.zeros(1, self.total_length, d_model))
        self.memory_pos = nn.Parameter(torch.zeros(1, self.total_length, d_model))
        
        # Initialize position encodings
        self._init_position_encodings()
        
        # Transformer layers
        self.layers = nn.ModuleList([
            DeformableExpressionTransformerLayer(
                d_model=d_model,
                n_heads=n_heads,
                n_points=n_points,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ) for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
        
    def _init_position_encodings(self):
        """Initialize position encodings with sinusoidal values"""
        position = torch.arange(self.total_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.pos_encoding.shape[-1], 2) * (-math.log(10000.0) / self.pos_encoding.shape[-1]))
        
        self.pos_encoding.data[0, :, 0::2] = torch.sin(position * div_term)
        self.pos_encoding.data[0, :, 1::2] = torch.cos(position * div_term)
        self.memory_pos.data = self.pos_encoding.data.clone()
        
    def forward(self, x, memory, memory_mask=None):
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            memory: Memory tensor from encoder (batch_size, seq_len, d_model)
            memory_mask: Optional attention mask
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        if seq_len != self.total_length:
            raise ValueError(f"Expected sequence length {self.total_length} but got {seq_len}")
            
        # Project input
        x = self.exp_input_proj(x)
        
        # Process through transformer layers
        attentions = []
        for layer in self.layers:
            layer_out = layer(
                x, memory,
                pos=self.pos_encoding,
                memory_pos=self.memory_pos,
                mask=memory_mask
            )
            x = layer_out[0] if isinstance(layer_out, tuple) else layer_out
            if isinstance(layer_out, tuple) and len(layer_out) > 1:
                attentions.append(layer_out[1])
        
        # Project to output dimension
        output = self.output_proj(x)
        
        if attentions:
            return output, attentions
        return output
    
class DeformableExpressionTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, n_points, dim_feedforward, dropout):
        super().__init__()
        
        # Self attention
        self.self_attn = DeformableExpressionAttention(
            dim=d_model,
            n_heads=n_heads,
            n_points=n_points
        )
        
        # Cross attention
        self.cross_attn = DeformableExpressionAttention(
            dim=d_model,
            n_heads=n_heads,
            n_points=n_points
        )
        
        # FFN and norms remain the same
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
        
    def forward(self, x, memory, pos=None, memory_pos=None, mask=None):
        """
        Args:
            x: Input tensor 
            memory: Memory tensor for cross attention
            pos: Position embedding for input
            memory_pos: Position embedding for memory
            mask: Attention mask
        """
        # Self attention
        residual = x
        x = self.norm1(x)
        q = k = self.with_pos_embed(x, pos)
        self_attn_out = self.self_attn(q, k, x, pos) 
        x = residual + self.dropout1(self_attn_out)
        
        # Cross attention with memory
        if memory is not None:
            residual = x
            x = self.norm2(x)
            
            # Apply mask if provided
            if mask is not None:
                # Convert boolean mask to float mask
                attn_mask = torch.zeros_like(mask, dtype=torch.float)
                attn_mask.masked_fill_(~mask, float('-inf'))
            else:
                attn_mask = None
                
            cross_attn_out = self.cross_attn(
                self.with_pos_embed(x, pos),
                self.with_pos_embed(memory, memory_pos),
                memory,
                memory_pos,
                attn_mask=attn_mask
            )
            x = residual + self.dropout2(cross_attn_out)
        
        # Feed forward
        residual = x
        x = self.norm3(x)
        x = residual + self.dropout3(self.ffn(x))
        
        return x, self_attn_out
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
            model_path = '../../pretrained_weights/chinese-hubert-base'
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
                 n_motions=100, n_diff_steps=500, ):
        super().__init__()

        # Model parameters
        self.motion_feat_dim = motion_feat_dim 
        self.use_indicator = use_indicator

        # Transformer
        self.architecture = architecture
        self.feature_dim = feature_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.mlp_ratio = mlp_ratio
        self.align_mask_width = align_mask_width
        self.use_learnable_pe = not no_use_learnable_pe

        # sequence length
        self.n_prev_motions = n_prev_motions
        self.n_motions = n_motions

        # Temporal embedding for the diffusion time step
        self.TE = PositionalEncoding(self.feature_dim, max_len=n_diff_steps + 1)
        self.diff_step_map = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

        if self.use_learnable_pe:
            # Learnable positional encoding
            self.PE = nn.Parameter(torch.randn(1, 1 + self.n_prev_motions + self.n_motions, self.feature_dim))
        else:
            self.PE = PositionalEncoding(self.feature_dim)

        # Transformer decoder
        if self.architecture == 'decoder':
            self.feature_proj = nn.Linear(self.motion_feat_dim + (1 if self.use_indicator else 0),
                                          self.feature_dim)
            # decoder_layer = nn.TransformerDecoderLayer(
            #     d_model=self.feature_dim, nhead=self.n_heads, dim_feedforward=self.mlp_ratio * self.feature_dim,
            #     activation='gelu', batch_first=True
            # )
            # self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=self.n_layers)

            self.transformer = DeformableExpressionTransformer(
                d_model=feature_dim,
                n_heads=n_heads,
                n_points=4,
                n_layers=n_layers,
                dim_feedforward=feature_dim * mlp_ratio,
                dropout=0.1,
                n_prev_motions=n_prev_motions,
                n_motions=n_motions
            )   



            if self.align_mask_width > 0:
                motion_len = self.n_prev_motions + self.n_motions
                alignment_mask = enc_dec_mask(motion_len, motion_len, frame_width=1, expansion=self.align_mask_width - 1)
                # print(f"alignment_mask: ", alignment_mask.shape)
                # alignment_mask = F.pad(alignment_mask, (0, 0, 1, 0), value=False)
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
            # nn.Tanh() # 增加了一个tanh
            # nn.Softmax()
        )

        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def format_shape_info(self, x):
        """Helper to safely get shape info for different types."""
        if torch.is_tensor(x):
            return f"{x.shape} (tensor)"
        elif isinstance(x, list):
            return f"list of len {len(x)}"
        elif isinstance(x, np.ndarray):
            return f"{x.shape} (numpy)"
        else:
            return f"type: {type(x)}"


    def display_mask_info(self, mask):
        """Helper to format mask information."""
        info = []
        if torch.is_tensor(mask):
            info.append(f"Shape: {mask.shape}")
            info.append(f"Active positions: {mask.sum().item()}")
            info.append(f"Sparsity: {(mask > 0).float().mean().item()*100:.1f}%")
            first_block = mask[:5, :5].cpu().numpy()
            info.append(f"First 5x5 block:\n{first_block}")
        return "\n".join(info)

    def forward(self, motion_feat, audio_feat, prev_motion_feat, prev_audio_feat, step, indicator=None):
        """
        Forward pass with detailed debug logging.
        """
        logger.debug("\n=== DenoisingNetwork Forward Pass Start ===")
        
        # Log input shapes
        for name, tensor in [
            ('motion_feat', motion_feat),
            ('audio_feat', audio_feat),
            ('prev_motion_feat', prev_motion_feat),
            ('prev_audio_feat', prev_audio_feat),
            ('step', step)
        ]:
            logger.debug(f"  {name}: {self.format_shape_info(tensor)}")

        # Convert step to tensor if needed
        if isinstance(step, list):
            logger.debug("Converting step from list to tensor")
            step = torch.tensor(step, device=self.device)
            logger.debug(f"  step tensor shape: {self.format_shape_info(step)}")

        # Time step embedding
        logger.debug("\n=== Time Step Embedding ===")
        diff_step_embedding = self.diff_step_map(self.TE.pe[0, step]).unsqueeze(1)
        logger.debug(f"Embedding: {self.format_shape_info(diff_step_embedding)}")
        if torch.is_tensor(diff_step_embedding):
            self.log_tensor_stats("Step embedding", diff_step_embedding)

        # Process indicator
        if indicator is not None:
            logger.debug("\n=== Processing Indicator ===")
            logger.debug(f"Original: {self.format_shape_info(indicator)}")
            padding = torch.zeros((indicator.shape[0], self.n_prev_motions), device=indicator.device)
            indicator = torch.cat([padding, indicator], dim=1)
            logger.debug(f"After padding: {self.format_shape_info(indicator)}")
            if torch.is_tensor(indicator):
                logger.debug(f"  Sum per batch: {indicator.sum(dim=1).tolist()}")
            indicator = indicator.unsqueeze(-1)
            logger.debug(f"Final: {self.format_shape_info(indicator)}")

        # Feature processing
        logger.debug("\n=== Feature Processing ===")
        if self.architecture == 'decoder':
            # Concatenate features
            feats_in = torch.cat([prev_motion_feat, motion_feat], dim=1)
            logger.debug(f"Combined features: {self.format_shape_info(feats_in)}")
            
            # Add indicator if needed
            if self.use_indicator:
                feats_in = torch.cat([feats_in, indicator], dim=-1)
                logger.debug(f"With indicator: {self.format_shape_info(feats_in)}")

            # Project features
            feats_in = self.feature_proj(feats_in)
            logger.debug(f"Projected: {self.format_shape_info(feats_in)}")
            if torch.is_tensor(feats_in):
                self.log_tensor_stats("Projected features", feats_in)

            # Position encoding
            if self.use_learnable_pe:
                logger.debug(f"Using learnable PE: {self.format_shape_info(self.PE)}")
                feats_in = feats_in + self.PE + diff_step_embedding
            else:
                feats_in = self.PE(feats_in) + diff_step_embedding
            logger.debug(f"After PE: {self.format_shape_info(feats_in)}")

            # Audio features
            audio_feat_in = torch.cat([prev_audio_feat, audio_feat], dim=1)
            logger.debug(f"Audio features: {self.format_shape_info(audio_feat_in)}")

            # Alignment mask
            if self.alignment_mask is not None:
                logger.debug("\nAlignment Mask Info:")
                logger.debug(self.display_mask_info(self.alignment_mask))

            # Transformer
            logger.debug("\n=== Transformer ===")
            feat_out = self.transformer(feats_in, audio_feat_in, memory_mask=self.alignment_mask)
            logger.debug(f"Output: {self.format_shape_info(feat_out)}")
            if torch.is_tensor(feat_out):
                self.log_tensor_stats("Transformer output", feat_out)

            # Motion decoding
            logger.debug("\n=== Motion Decoder ===")
            motion_feat_target = self.motion_dec(feat_out)
            logger.debug(f"Final output: {self.format_shape_info(motion_feat_target)}")
            if torch.is_tensor(motion_feat_target):
                self.log_tensor_stats("Final output", motion_feat_target)
                logger.debug(f"  Expected length: {self.n_prev_motions + self.n_motions}")
                logger.debug(f"  Actual length: {motion_feat_target.shape[1]}")

        else:
            raise ValueError(f'Unknown architecture: {self.architecture}')

        logger.debug("=== Forward Pass Complete ===\n")
        return motion_feat_target

    def log_tensor_stats(self, name, tensor):
        """Helper to log tensor statistics."""
        logger.debug(f"  {name} stats:")
        logger.debug(f"    mean: {tensor.mean():.4f}")
        logger.debug(f"    std: {tensor.std():.4f}")
        logger.debug(f"    min: {tensor.min():.4f}")
        logger.debug(f"    max: {tensor.max():.4f}")

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