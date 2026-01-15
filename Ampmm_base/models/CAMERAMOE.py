import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import build_loss_evaluator
from .TRMfusion import MultiHeadAttention, PositionwiseFeedForward

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

class LinearFusion(nn.Module):
        def __init__(self, emb_size=1280):
            super(LinearFusion, self).__init__()

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
    
class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear = nn.Linear(n_embed, num_experts)  # 噪声层
    
    def forward(self, mh_output):
        logits = self.topkroute_linear(mh_output)
        noise_logits = self.noise_linear(mh_output)
        noise = torch.randn_like(logits) * F.softplus(noise_logits)  # 添加噪声
        noisy_logits = logits + noise
        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices

class SparseMoE(nn.Module):
    def __init__(self, d_in, d_hid, num_experts=2, top_k=1, dropout=0.2):
        super(SparseMoE, self).__init__()
        self.router = NoisyTopkRouter(d_in, num_experts, top_k)
        self.experts = nn.ModuleList([PositionwiseFeedForward(d_in, d_hid, dropout=dropout) for _ in range(num_experts)])
        self.top_k = top_k

    def forward(self, x):
        gating_output, indices = self.router(x)
        final_output = torch.zeros_like(x)  # 初始化输出
        flat_x = x.view(-1, x.size(-1))  # 展平输入
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))  # 展平路由器输出

        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1)  # 选择当前专家处理的tokens
            flat_mask = expert_mask.view(-1)
            if flat_mask.any():
                expert_input = flat_x[flat_mask]  # 获取输入
                expert_output = expert(expert_input)  # 专家处理
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)  # 获取权重
                weighted_output = expert_output * gating_scores  # 加权输出
                final_output[expert_mask] += weighted_output.squeeze(1)  # 累加结果

        return final_output
    
class MOETransformer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, num_experts=2, top_k=1, dropout=0.2):
        super(MOETransformer, self).__init__()

        self.cross_attn = MultiHeadAttention(d_model, d_k, d_v, n_head, dropout=dropout)
        self.smoe = SparseMoE(d_model, d_inner, num_experts, top_k, dropout=dropout)  # 稀疏MOE

    def forward(self, enc_feat, dec_feat, dec_enc_attn_mask=None):
        output, cross_attn = self.cross_attn(enc_feat, dec_feat, dec_feat, dec_enc_attn_mask)
        output = self.smoe(output)
        return output, cross_attn
    
class FusionLayer(nn.Module):
    def __init__(self, d_model, d_inner, d_k, d_v, dropout=0.2, n_head=4):
        super(FusionLayer, self).__init__()
        self.slf_attn_layer = MOETransformer(d_model, d_inner, n_head, d_k, d_v)
        self.cross_attn_layer = MOETransformer(d_model, d_inner, n_head, d_k, d_v)

    def forward(self, emb_feat, dec_feat, slf_attn_mask=None, dec_enc_attn_mask=None):
        slf_attn_output, slf_attn = self.slf_attn_layer(emb_feat,emb_feat,slf_attn_mask)
        output, cross_attn = self.cross_attn_layer(slf_attn_output,dec_feat,dec_enc_attn_mask)
        return output, slf_attn, cross_attn
    
class FusionBlock(nn.Module):
    """
    Fusion module
    """
    def __init__(self, d_model, d_inner, d_k, d_v, dropout=0.2, n_layers=4, n_head=4):
        super(FusionBlock, self).__init__()
        self.fusionlayers = nn.ModuleList([FusionLayer(d_model,d_inner,d_k,d_v,dropout,n_head) 
                                           for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, emb_feat, dec_feat, slf_attn_mask=None, dec_enc_attn_mask=None, return_attns=False):
        slf_attn_list = []
        cross_attn_list = []
        dec_feat = self.layer_norm(dec_feat)
        for fusion_layer in self.fusionlayers:
            emb_feat, slf_attn, cross_attn = fusion_layer(emb_feat, dec_feat, slf_attn_mask, dec_enc_attn_mask)
            slf_attn_list += slf_attn
            cross_attn_list += cross_attn
        if return_attns:
            return emb_feat, slf_attn_list, cross_attn_list
        return emb_feat
    
class CAMERAMOE(nn.Module):
    def __init__(self, cfg, des_size=676, stc_size=128, 
                 emb_size=1280, d_inner=2048, 
                 n_gru_layers=3, n_cross_layers=3, n_head=4, d_k=1280, d_v=1280, dropout=0.2):
        super(CAMERAMOE, self).__init__()

        self.seq_feature_extractor = BiGRU(d_model=emb_size, emb_size=emb_size, hidden_dim=d_inner, num_layers=n_gru_layers)
        self.des_feature_extractor = DesModel(des_size=des_size, emb_size=emb_size)
        self.stc_feature_extractor = StcModel(stc_size=stc_size, emb_size=emb_size, num_layers=n_gru_layers)
        self.fusion_des = FusionBlock(d_model=emb_size, d_inner=d_inner, d_k=d_k, d_v=d_v,
                                       dropout=dropout, n_layers=n_cross_layers, n_head=n_head)
        self.fusion_stc = FusionBlock(d_model=emb_size, d_inner=d_inner, d_k=d_k, d_v=d_v,
                                       dropout=dropout, n_layers=n_cross_layers, n_head=n_head)
        
        self.second_fusion = LinearFusion(emb_size=emb_size)

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