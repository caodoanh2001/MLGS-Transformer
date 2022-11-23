import torch
from torch import nn
import copy
from models.containers import ModuleList
from ..captioning_model import CaptioningModel, CaptioningModelWithMasks
import torch.nn.functional as F
import torchvision.models as models
from .attention import ScaledDotProductAttention
import timm

class VisualExtractor(nn.Module):
    def __init__(self, visual_extractor='resnet101', visual_extractor_pretrained=True):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = visual_extractor
        self.pretrained = visual_extractor_pretrained
        model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)

    def forward(self, images):
        patch_feats = self.model(images)
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        return patch_feats

class VisualExtractorFPN(nn.Module):
    def __init__(self, visual_extractor='resnet152', visual_extractor_pretrained=True, grid_size=None):
        super(VisualExtractorFPN, self).__init__()
        self.visual_extractor = visual_extractor
        self.pretrained = visual_extractor_pretrained
        model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
        modules = list(model.children())[:-2]
        # Main model
        self.model = nn.Sequential(*modules)

        # ------------------------- Top-down layers ------------------------- 
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        # Pooling layers
        if grid_size:
            self.pooling = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        else:
            self.pooling = nn.AdaptiveAvgPool2d((7, 7))

        # Self-Attention
        self.attention = ScaledDotProductAttention(d_model=256, d_k=64, d_v=64, h=8)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=True) + y

    def forward(self, images):
        # Bottom-up
        c1 = self.model[0](images)
        c1 = self.model[1](c1)
        c1 = self.model[2](c1)
        c1 = self.model[3](c1)
        c2 = self.model[4](c1)
        c3 = self.model[5](c2)
        c4 = self.model[6](c3)
        c5 = self.model[7](c4)

        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))

        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        batch_size, feat_size, _, _ = p4.shape
        p4 = self.pooling(p4).reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        p3 = self.pooling(p3).reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        p2 = self.pooling(p2).reshape(batch_size, feat_size, -1).permute(0, 2, 1)

        p432 = torch.cat([p4, p3, p2], dim=1)
        p432 = self.attention(p432, p432, p432)
        # p432 = self.projection(p432)
        # p432 = self.dropout(p432)
        # p432 = self.layer_norm(p432)
        
        return p432[:, :p4.shape[1], :]


class VisualExtractorFPN_from_timm(nn.Module):
    def __init__(self, model):
        super(VisualExtractorFPN_from_timm, self).__init__()
        modules = list(model.children())[:-2]
        # Main model
        self.model = nn.Sequential(*modules)

        # ------------------------- Top-down layers ------------------------- 
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        # Pooling layers
        self.pooling = nn.AdaptiveAvgPool2d((7, 7))

        # Self-Attention
        self.attention = ScaledDotProductAttention(d_model=256, d_k=64, d_v=64, h=8)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=True) + y

    def forward(self, images):
        # Bottom-up
        c1 = self.model[0](images)
        c1 = self.model[1](c1)
        c1 = self.model[2](c1)
        c1 = self.model[3](c1)
        c2 = self.model[4](c1)
        c3 = self.model[5](c2)
        c4 = self.model[6](c3)
        c5 = self.model[7](c4)

        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))

        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        batch_size, feat_size, _, _ = p4.shape
        p4 = self.pooling(p4).reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        p3 = self.pooling(p3).reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        p2 = self.pooling(p2).reshape(batch_size, feat_size, -1).permute(0, 2, 1)

        p432 = torch.cat([p4, p3, p2], dim=1)
        p432 = self.attention(p432, p432, p432)
        return p432[:, :p4.shape[1], :]

class Transformer(CaptioningModel):
    def __init__(self, bos_idx, encoder, decoder, pretrained_backbone=None, dataset=None):
        super(Transformer, self).__init__()
        self.bos_idx = bos_idx
        self.encoder = encoder
        self.decoder = decoder
        self.dataset = dataset
        if pretrained_backbone:
            self.pretrained_backbone = pretrained_backbone
        else:
            self.pretrained_backbone = 'resnet101'
        self.visual_extractor = VisualExtractor(visual_extractor=self.pretrained_backbone, visual_extractor_pretrained=True).cuda()
        self.visual_extractor = VisualExtractorFPN(visual_extractor='resnet101', visual_extractor_pretrained=True).cuda()
        self.register_state('enc_output', None)
        self.register_state('mask_enc', None)
        self.init_weights()

    @property
    def d_model(self):
        return self.decoder.d_model

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, images, seq, *args):
        if self.dataset is not None:
            if self.dataset == 'mimic_cxr':
                att_feats = self.visual_extractor(images)
                enc_output, mask_enc = self.encoder(att_feats)
                dec_output = self.decoder(seq, enc_output, mask_enc)
            else:
                att_feats_0 = self.visual_extractor(images[:, 0])
                att_feats_1 = self.visual_extractor(images[:, 1])
                att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
                enc_output, mask_enc = self.encoder(att_feats)
                dec_output = self.decoder(seq, enc_output, mask_enc)
        else:
            att_feats_0 = self.visual_extractor(images[:, 0])
            att_feats_1 = self.visual_extractor(images[:, 1])
            att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
            enc_output, mask_enc = self.encoder(att_feats)
            dec_output = self.decoder(seq, enc_output, mask_enc)

        return dec_output

    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device),
                None, None]

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        it = None
        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            if t == 0:
                if self.dataset is not None:
                    if self.dataset == 'mimic_cxr':
                        att_feats = self.visual_extractor(visual)
                        self.enc_output, self.mask_enc = self.encoder(att_feats)
                    else:
                        att_feats_0 = self.visual_extractor(visual[:, 0])
                        att_feats_1 = self.visual_extractor(visual[:, 1])
                        att_feats1 = torch.cat((att_feats_0, att_feats_1), dim=1)
                        self.enc_output, self.mask_enc = self.encoder(att_feats1)
                else:
                    att_feats_0 = self.visual_extractor(visual[:, 0])
                    att_feats_1 = self.visual_extractor(visual[:, 1])
                    att_feats1 = torch.cat((att_feats_0, att_feats_1), dim=1)
                    self.enc_output, self.mask_enc = self.encoder(att_feats1)

                if isinstance(visual, torch.Tensor):
                    it = visual.data.new_full((visual.shape[0], 1), self.bos_idx).long()
                else:
                    it = visual[0].data.new_full((visual[0].shape[0], 1), self.bos_idx).long()

            else:
                it = prev_output
        return self.decoder(it, self.enc_output, self.mask_enc)

class TransformerEnsemble(CaptioningModel):
    def __init__(self, model: Transformer, weight_files):
        super(TransformerEnsemble, self).__init__()
        self.n = len(weight_files)
        self.models = ModuleList([copy.deepcopy(model) for _ in range(self.n)])
        for i in range(self.n):
            state_dict_i = torch.load(weight_files[i])['state_dict']
            self.models[i].load_state_dict(state_dict_i)

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        out_ensemble = []
        for i in range(self.n):
            out_i = self.models[i].step(t, prev_output, visual, seq, mode, **kwargs)
            out_ensemble.append(out_i.unsqueeze(0))

        return torch.mean(torch.cat(out_ensemble, 0), dim=0)


class OriginalTransformer(CaptioningModel):
    def __init__(self, bos_idx, encoder, decoder, pretrained_backbone=None):
        super(OriginalTransformer, self).__init__()
        self.bos_idx = bos_idx
        self.encoder = encoder
        self.decoder = decoder
        if pretrained_backbone:
            self.pretrained_backbone = pretrained_backbone
        else:
            self.pretrained_backbone = 'resnet101'
        self.visual_extractor = VisualExtractor(visual_extractor=self.pretrained_backbone, visual_extractor_pretrained=True).cuda()
        self.register_state('enc_output', None)
        self.register_state('mask_enc', None)
        self.init_weights()

    @property
    def d_model(self):
        return self.decoder.d_model

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, images, seq, *args):
        att_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1 = self.visual_extractor(images[:, 1])
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        enc_output, mask_enc = self.encoder(att_feats)
        dec_output = self.decoder(seq, enc_output, mask_enc)
        return dec_output

    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device),
                None, None]

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        it = None
        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            if t == 0:
                att_feats_0 = self.visual_extractor(visual[:, 0])
                att_feats_1 = self.visual_extractor(visual[:, 1])
                
                if isinstance(visual, torch.Tensor):
                    it = visual.data.new_full((visual.shape[0], 1), self.bos_idx).long()
                else:
                    it = visual[0].data.new_full((visual[0].shape[0], 1), self.bos_idx).long()

                att_feats1 = torch.cat((att_feats_0, att_feats_1), dim=1)
                self.enc_output, self.mask_enc = self.encoder(att_feats1)
            else:
                it = prev_output

        return self.decoder(it, self.enc_output, self.mask_enc)

class TransformerWithMasks(CaptioningModelWithMasks):
    def __init__(self, bos_idx, encoder, decoder, pretrained_backbone=None, grid_size=None):
        super(TransformerWithMasks, self).__init__()
        self.bos_idx = bos_idx
        self.encoder = encoder
        self.decoder = decoder
        if pretrained_backbone:
            try:
                self.visual_extractor = VisualExtractorFPN(visual_extractor=pretrained_backbone, visual_extractor_pretrained=True, grid_size=grid_size).cuda()
            except:
                print('Loading from timm')
                model = timm.create_model(pretrained_backbone, pretrained=True)
                self.visual_extractor = VisualExtractorFPN_from_timm(model=model).cuda()
        else:
            self.visual_extractor = VisualExtractorFPN(visual_extractor='resnet101', visual_extractor_pretrained=True).cuda()
        
        if grid_size:
            self.seg_mask_pooling = nn.AdaptiveAvgPool2d((grid_size, grid_size))
            self.grid_size = grid_size
        else:
            self.seg_mask_pooling = nn.AdaptiveAvgPool2d((7, 7))
            self.grid_size = 7
        self.register_state('enc_output', None)
        self.register_state('mask_enc', None)
        self.init_weights()

    @property
    def d_model(self):
        return self.decoder.d_model

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, images, masks, seq, *args):
        bs = images.shape[0]
        mask_0 = self.seg_mask_pooling(masks[:, 0, :, :, :]).view(bs, self.grid_size * self.grid_size, -1)
        mask_1 = self.seg_mask_pooling(masks[:, 1, :, :, :]).view(bs, self.grid_size * self.grid_size, -1)
        masks = torch.cat([mask_0, mask_1], dim=1)

        att_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1 = self.visual_extractor(images[:, 1])
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)

        enc_output, mask_enc = self.encoder(att_feats, masks)
        dec_output = self.decoder(seq, enc_output, mask_enc)
        return dec_output

    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device),
                None, None]

    def step(self, t, prev_output, visual, masks, seq, mode='teacher_forcing', **kwargs):
        it = None
        bs = visual.shape[0]
        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            if t == 0:
                att_feats_0 = self.visual_extractor(visual[:, 0])
                att_feats_1 = self.visual_extractor(visual[:, 1])

                mask_0 = self.seg_mask_pooling(masks[:, 0, :, :, :]).view(bs, self.grid_size * self.grid_size, -1)
                mask_1 = self.seg_mask_pooling(masks[:, 1, :, :, :]).view(bs, self.grid_size * self.grid_size, -1)
                
                if isinstance(visual, torch.Tensor):
                    it = visual.data.new_full((visual.shape[0], 1), self.bos_idx).long()
                else:
                    it = visual[0].data.new_full((visual[0].shape[0], 1), self.bos_idx).long()

                att_feats1 = torch.cat((att_feats_0, att_feats_1), dim=1)
                masks = torch.cat([mask_0, mask_1], dim=1)
                self.enc_output, self.mask_enc = self.encoder(att_feats1, masks)
            else:
                it = prev_output

        return self.decoder(it, self.enc_output, self.mask_enc)

class TransformerEnsemble(CaptioningModel):
    def __init__(self, model: Transformer, weight_files):
        super(TransformerEnsemble, self).__init__()
        self.n = len(weight_files)
        self.models = ModuleList([copy.deepcopy(model) for _ in range(self.n)])
        for i in range(self.n):
            state_dict_i = torch.load(weight_files[i])['state_dict']
            self.models[i].load_state_dict(state_dict_i)

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        out_ensemble = []
        for i in range(self.n):
            out_i = self.models[i].step(t, prev_output, visual, seq, mode, **kwargs)
            out_ensemble.append(out_i.unsqueeze(0))

        return torch.mean(torch.cat(out_ensemble, 0), dim=0)