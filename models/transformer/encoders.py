from torch.nn import functional as F
from models.transformer.utils import PositionWiseFeedForward
import torch
from torch import nn
from models.transformer.attention import MultiHeadAttention, ScaledDotProductAttention


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        att = self.mhatt(queries, keys, values, attention_mask, attention_weights)
        ff = self.pwff(att)
        return ff


class MultiLevelEncoder(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])
        self.padding_idx = padding_idx

    def forward(self, input, attention_weights=None):
        # input (b_s, seq_len, d_in)
        attention_mask = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)
        out = input
        for l in self.layers:
            out = l(out, out, out, attention_mask, attention_weights)
        return out, attention_mask

class MultiLevelEncoderWithMasks(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoderWithMasks, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.visual_layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])

        self.segmask_layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])

        self.visual_segmask_layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])

        self.segmask_visual_layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])

        self.padding_idx = padding_idx

        self.fc_mask = nn.Linear(1, self.d_model)
        self.dropout_mask = nn.Dropout(p=self.dropout)
        self.layer_norm_mask = nn.LayerNorm(self.d_model)

        self.visual_seg_fusion = nn.Linear(self.d_model*2, self.d_model)
        self.visual_seg_fusion_dropout = nn.Dropout(p=self.dropout)
        self.visual_seg_fusion_norm = nn.LayerNorm(self.d_model)

    def forward(self, input, masks, attention_weights=None):
        # input (b_s, seq_len, d_in)
        attention_mask = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)
        segmentation_mask = (masks > 0.5)[:,:,0].unsqueeze(1).unsqueeze(1)

        masks = F.relu(self.fc_mask(masks))
        masks = self.dropout_mask(masks)
        masks = self.layer_norm_mask(masks)

        # Fuse masks and inputs
        visual_out = input
        seg_out = masks

        for idx, (l_visual, l_seg, l_vs, l_sv) in enumerate(zip(self.visual_layers, self.segmask_layers, self.visual_segmask_layers, self.segmask_visual_layers)):
            visual_out = l_visual(visual_out, visual_out, visual_out, attention_mask, attention_weights)
            seg_out = l_seg(seg_out, seg_out, seg_out, segmentation_mask, attention_weights)
            
            t_visual_out, t_seg_out = visual_out, seg_out
            
            visual_out = l_vs(visual_out, t_seg_out, t_seg_out, torch.mul(segmentation_mask, attention_mask), attention_weights)
            seg_out = l_sv(seg_out, t_visual_out, t_visual_out, torch.mul(segmentation_mask, attention_mask), attention_weights)

        out = F.relu(self.visual_seg_fusion(torch.cat([visual_out, seg_out], dim=-1)))
        out = self.visual_seg_fusion_dropout(out)
        out = self.visual_seg_fusion_norm(out)
        
        return out, attention_mask

class MemoryAugmentedEncoder(MultiLevelEncoder):
    def __init__(self, N, padding_idx, d_in=2048, **kwargs):
        super(MemoryAugmentedEncoder, self).__init__(N, padding_idx, **kwargs)
        self.fc = nn.Linear(d_in, self.d_model)
        self.dropout = nn.Dropout(p=self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, input, attention_weights=None):
        out = F.relu(self.fc(input))
        out = self.dropout(out)
        out = self.layer_norm(out)
        return super(MemoryAugmentedEncoder, self).forward(out, attention_weights=attention_weights)

class MemoryAugmentedEncoderWithMasks(MultiLevelEncoderWithMasks):
    def __init__(self, N, padding_idx, d_in=2048, **kwargs):
        super(MemoryAugmentedEncoderWithMasks, self).__init__(N, padding_idx, **kwargs)
        self.fc = nn.Linear(d_in, self.d_model)
        self.dropout = nn.Dropout(p=self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, input, masks, attention_weights=None):
        out = F.relu(self.fc(input))
        out = self.dropout(out)
        out = self.layer_norm(out)
        
        return super(MemoryAugmentedEncoderWithMasks, self).forward(out, masks, attention_weights=attention_weights)