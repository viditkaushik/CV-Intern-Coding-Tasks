# model.py
import torch
import torch.nn as nn
import torchvision.models as models
import math

class QueryDETR(nn.Module):
    def __init__(self, num_classes, num_queries=20, hidden_dim=256):
        super().__init__()
        
        self.backbone = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-2])
        self.input_proj = nn.Conv2d(512, hidden_dim, kernel_size=1)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.hidden_dim = hidden_dim
        self.class_head = nn.Linear(hidden_dim, num_classes + 1) 
        self.bbox_head = nn.Linear(hidden_dim, 4) # this line predicts (center_x, center_y, width, height)

    def forward(self, x):
        features = self.backbone(x)         # Shape: (B, 512, H/32, W/32)
        proj_features = self.input_proj(features) # Shape: (B, hidden_dim, H/32, W/32)
        
        B, C, H, W = proj_features.shape
        memory = proj_features.flatten(2).permute(0, 2, 1)
        
        pos = self._get_pos_embed(B, H, W, proj_features.device)
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        decoder_output = self.transformer_decoder(tgt=query_embed, memory=memory + pos)
        
        class_logits = self.class_head(decoder_output)
        pred_boxes = self.bbox_head(decoder_output).sigmoid() 

        return {'pred_logits': class_logits, 'pred_boxes': pred_boxes}
    
    def _get_pos_embed(self, B, H, W, device):
        pos = torch.zeros(B, H*W, self.hidden_dim, device=device)
        return pos