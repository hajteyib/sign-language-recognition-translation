import torch
import torch.nn as nn
import math

class SpatialEmbedding(nn.Module):
    def __init__(self, input_dims, embed_dim):
        """
        Embeds different body parts separately with proportional allocation.
        Hands are most important in sign language, followed by face, then pose.
        
        input_dims: dict with keys 'pose', 'left_hand', 'right_hand', 'face'
        embed_dim: total embedding dimension
        """
        super().__init__()
        
        # Proportional allocation based on importance for sign language
        # Hands: most critical (33% each)
        # Face: very important for expressions (21%)
        # Pose: body orientation/position (13%)
        # Total must sum to embed_dim
        
        pose_dim = embed_dim // 8          # ~13% for d_model=512 -> 64
        hand_dim = embed_dim // 3          # ~33% for d_model=512 -> 170 (each)
        face_dim = embed_dim // 5          # ~20% for d_model=512 -> 102
        
        # Adjust to ensure exact sum
        total_dim = pose_dim + 2 * hand_dim + face_dim
        adjustment = embed_dim - total_dim
        face_dim += adjustment  # Add any remainder to face
        
        self.pose_proj = nn.Linear(input_dims['pose'], pose_dim)
        self.lh_proj = nn.Linear(input_dims['left_hand'], hand_dim)
        self.rh_proj = nn.Linear(input_dims['right_hand'], hand_dim)
        self.face_proj = nn.Linear(input_dims['face'], face_dim)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
        print(f"SpatialEmbedding allocation: pose={pose_dim}, hands={hand_dim}x2, face={face_dim}, total={embed_dim}")

    def forward(self, x):
        # x is a dict of tensors: (Batch, Time, Features)
        
        p = self.relu(self.pose_proj(x['pose']))
        lh = self.relu(self.lh_proj(x['left_hand']))
        rh = self.relu(self.rh_proj(x['right_hand']))
        f = self.relu(self.face_proj(x['face']))
        
        # Concatenate
        combined = torch.cat([p, lh, rh, f], dim=-1)
        return self.dropout(combined)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class SignLanguageTransformer(nn.Module):
    def __init__(self, input_dims, vocab_size, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        self.embedding = SpatialEmbedding(input_dims, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)
        self.generator = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, src_dict, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, tgt_mask=None):
        """
        src_dict: dict of tensors (Batch, Time, Features)
        tgt: (Time, Batch) - target sequence indices
        """
        # Embed source
        src = self.embedding(src_dict) # (Batch, Time, d_model)
        src = src.permute(1, 0, 2) # (Time, Batch, d_model) for Transformer
        
        src = self.pos_encoder(src)
        
        # Embed target
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        
        output = self.transformer(
            src, tgt, 
            tgt_mask=tgt_mask, 
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask, 
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        return self.generator(output)

    def generate_square_subsequent_mask(self, sz):
        return self.transformer.generate_square_subsequent_mask(sz)
    
    def create_src_padding_mask(self, src_dict, src_lengths):
        """
        Create padding mask for source sequences.
        
        Args:
            src_dict: Dict of tensors (Batch, Time, Features)
            src_lengths: Tensor of shape (Batch,) with actual sequence lengths
        
        Returns:
            mask: Boolean tensor (Batch, Time) where True indicates padding
        """
        batch_size = src_dict['pose'].shape[0]
        max_len = src_dict['pose'].shape[1]
        
        # Create mask based on lengths
        # True for positions that should be masked (padding positions)
        mask = torch.arange(max_len, device=src_lengths.device).unsqueeze(0) >= src_lengths.unsqueeze(1)
        
        return mask  # (Batch, Time)
