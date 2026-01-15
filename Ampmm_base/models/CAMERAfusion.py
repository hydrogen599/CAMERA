import torch
import torch.nn as nn

from .losses import build_loss_evaluator
from .TRMfusion import CrossAttn, SelfAttn

class BiGRU(nn.Module):
    def __init__(self, d_model, emb_size, hidden_dim, num_layers=3, dropout=0.2):
        super(BiGRU, self).__init__()

        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.emb_size = emb_size

        self.gru_struct = nn.GRU(self.d_model, self.hidden_dim, num_layers=self.num_layers, dropout=dropout, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(2*hidden_dim*num_layers, 2*emb_size),
            nn.LeakyReLU(),
            nn.LayerNorm(2*emb_size, eps=1e-6),
            nn.Linear(2*emb_size, emb_size)
        )

    def forward(self, input):
        input = input.permute(1, 0, 2)
        _, struct_hn = self.gru_struct(input)

        struct_hn = struct_hn.permute(1, 0, 2)
        struct_hn = struct_hn.reshape(struct_hn.shape[0], -1)

        emb = self.fc(struct_hn)
        return emb

class DesModel(nn.Module):
    def __init__(self, des_size=676, emb_size=1280):
        super(DesModel,self).__init__()
        self.emb_layer = nn.Sequential(
            nn.Linear(des_size, emb_size),
            )
        
    def forward(self, data):
        emb = self.emb_layer(data)
        return emb

class StcModel(nn.Module):
    def __init__(self, stc_size=128, emb_size=1280, hidden_dim=64, num_layers=3):
        super(StcModel,self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.stc_embedding = nn.Embedding(4, stc_size, padding_idx=0)
        self.encoder_struct = nn.TransformerEncoderLayer(d_model=stc_size, nhead=8)
        self.transformer_struct= nn.TransformerEncoder(self.encoder_struct, num_layers=1)
        self.gru_struct = BiGRU(d_model=stc_size, emb_size=emb_size, hidden_dim=hidden_dim, num_layers=num_layers)
    
    def forward(self, data):
        emb = self.stc_embedding(data)
        struct_output = self.transformer_struct(emb)
        emb = self.gru_struct(struct_output)

        return emb

class FusionBlock(nn.Module):
        def __init__(self, emb_size=1280):
            super(FusionBlock, self).__init__()

            self.fc = nn.Sequential(
                nn.Linear(emb_size * 2, 2048),
                nn.ReLU(),
                nn.LayerNorm(2048, eps=1e-6),
                nn.Linear(2048, emb_size)
            )

            self.fc2 = nn.Sequential(nn.Linear(emb_size*3, emb_size),
                                    nn.ReLU(),
                                    nn.LayerNorm(emb_size, eps=1e-6))

        def forward(self, input1, input2):
            token = torch.cat((input1, input2), dim=1)
            token = self.fc(token)

            token = torch.cat((input1, token, input2), dim=1)
            token = self.fc2(token)
            return token

# TODO 加入GRU
class CAMERAfusion(nn.Module):
    def __init__(self, cfg, des_size=676, stc_size=128, 
                 emb_size=1280, d_inner=2048, 
                 n_gru_layers=3, n_cross_layers=3, n_head=4, d_k=1280, d_v=1280, dropout=0.2):
        super(CAMERAfusion, self).__init__()

        self.seq_feature_extractor = BiGRU(d_model=emb_size, emb_size=emb_size, hidden_dim=d_inner, num_layers=n_gru_layers)
        self.des_feature_extractor = DesModel(des_size=des_size, emb_size=emb_size)
        self.stc_feature_extractor = StcModel(stc_size=stc_size, emb_size=emb_size, num_layers=n_gru_layers)
        self.fusion_des = CrossAttn(d_model=emb_size, d_inner=d_inner, d_k=d_k, d_v=d_v,
                                       dropout=dropout, n_layers=n_cross_layers, n_head=n_head)
        self.fusion_stc = CrossAttn(d_model=emb_size, d_inner=d_inner, d_k=d_k, d_v=d_v,
                                       dropout=dropout, n_layers=n_cross_layers, n_head=n_head)
        
        self.second_fusion = FusionBlock(emb_size=emb_size)

        self.seq_head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_size, 512),
                nn.LeakyReLU(),
                nn.LayerNorm(512, eps=1e-6),

                nn.Linear(512, 64),
                nn.LayerNorm(64, eps=1e-6),
                nn.LeakyReLU(),

                nn.Linear(64,1)
            ) for _ in range(cfg.bac_num)
        ])

        self.des_head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_size, 512),
                nn.ReLU(),
                nn.LayerNorm(512, eps=1e-6),
                nn.Linear(512, 64),
                nn.LayerNorm(64, eps=1e-6),
                nn.ReLU(),
                nn.Linear(64,1)
            ) for _ in range(cfg.bac_num)
        ])

        self.stc_head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_size, 512),
                nn.LeakyReLU(),
                nn.LayerNorm(512, eps=1e-6),

                nn.Linear(512, 64),
                nn.LayerNorm(64, eps=1e-6),
                nn.LeakyReLU(),

                nn.Linear(64,1)
            ) for _ in range(cfg.bac_num)
        ])

        self.final_head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_size, 256),
                nn.ReLU(),
                nn.LayerNorm(256, eps=1e-6),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.LayerNorm(64, eps=1e-6),
                nn.Linear(64, 1)
            ) for _ in range(cfg.bac_num)
        ])

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
        self.loss_evaluator = build_loss_evaluator(cfg)

    def forward(self, input_data):
        esm_emb = input_data['batch_tokens']
        des_info = input_data['des']
        stc_info = input_data['stc']

        esm_emb = self.seq_feature_extractor(esm_emb)
        des_emb = self.des_feature_extractor(des_info)
        stc_emb = self.stc_feature_extractor(stc_info)
        # print(esm_emb.shape, esm_pred.shape)
        # print(des_emb.shape, des_pred.shape)
        # print(stc_emb.shape, stc_pred.shape)

        if self.training:
            esm_pred = self.seq_head[input_data['bacterium_id']](esm_emb)
            des_pred = self.des_head[input_data['bacterium_id']](des_emb)
            stc_pred = self.stc_head[input_data['bacterium_id']](stc_emb)
            esm_pred = esm_pred.squeeze(dim=-1)
            des_pred = des_pred.squeeze(dim=-1)
            stc_pred = stc_pred.squeeze(dim=-1)
        
        esm_emb = esm_emb.unsqueeze(1) #[n, 1, emb_size]
        des_emb = des_emb.unsqueeze(1) #[n, 1, emb_size]
        stc_emb = stc_emb.unsqueeze(1) #[n, 1, emb_size]

        esm_des_fusion = self.fusion_des(esm_emb, des_emb)
        esm_stc_fusion = self.fusion_stc(esm_emb, stc_emb)
        esm_des_token = esm_des_fusion.squeeze(1)
        esm_stc_token = esm_stc_fusion.squeeze(1)

        final_token = self.second_fusion(esm_des_token, esm_stc_token)
        pred = self.final_head[input_data['bacterium_id']](final_token)
        pred = pred.squeeze(dim=-1)
        
        gt = input_data['mic']
        if self.training:
            loss_dict = self.loss_evaluator({
                'final_pred':pred,
                'esm_pred':esm_pred,
                'des_pred':des_pred,
                'stc_pred':stc_pred,
            },gt)
            return loss_dict
        else:
            results = dict(
                model_outputs = pred,
                labels = gt
            )
            return results