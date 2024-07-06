import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding,DataEmbedding_wo_pos,DataEmbedding_wo_temp,DataEmbedding_wo_pos_temp
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, config):
        super(Model, self).__init__()
        self.pred_len = config['pred_len']
        self.output_attention = config['output_attention']

        # Embedding
        if config['embed_type'] == 0:
            self.enc_embedding = DataEmbedding(config['enc_in'], config['d_model'], config['embed'], config['freq'],
                                            config['dropout'])
            self.dec_embedding = DataEmbedding(config['dec_in'], config['d_model'], config['embed'], config['freq'],
                                           config['dropout'])
        elif config['embed_type'] == 1:
            self.enc_embedding = DataEmbedding(config['enc_in'], config['d_model'], config['embed'], config['freq'],
                                                    config['dropout'])
            self.dec_embedding = DataEmbedding(config['dec_in'], config['d_model'], config['embed'], config['freq'],
                                                    config['dropout'])
        elif config['embed_type'] == 2:
            self.enc_embedding = DataEmbedding_wo_pos(config['enc_in'], config['d_model'], config['embed'], config['freq'],
                                                    config['dropout'])
            self.dec_embedding = DataEmbedding_wo_pos(config['dec_in'], config['d_model'], config['embed'], config['freq'],
                                                    config['dropout'])

        elif config['embed_type'] == 3:
            self.enc_embedding = DataEmbedding_wo_temp(config['enc_in'], config['d_model'], config['embed'], config['freq'],
                                                    config['dropout'])
            self.dec_embedding = DataEmbedding_wo_temp(config['dec_in'], config['d_model'], config['embed'], config['freq'],
                                                    config['dropout'])
        elif config['embed_type'] == 4:
            self.enc_embedding = DataEmbedding_wo_pos_temp(config['enc_in'], config['d_model'], config['embed'], config['freq'],
                                                    config['dropout'])
            self.dec_embedding = DataEmbedding_wo_pos_temp(config['dec_in'], config['d_model'], config['embed'], config['freq'],
                                                    config['dropout'])
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, config['factor'], attention_dropout=config['dropout'],
                                      output_attention=config['output_attention']), config['d_model'], config['n_heads']),
                    config['d_model'],
                    config['d_ff'],
                    dropout=config['dropout'],
                    activation=config['activation']
                ) for l in range(config['e_layers'])
            ],
            norm_layer=torch.nn.LayerNorm(config['d_model'])
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, config['factor'], attention_dropout=config['dropout'], output_attention=False),
                        config['d_model'], config['n_heads']),
                    AttentionLayer(
                        FullAttention(False, config['factor'], attention_dropout=config['dropout'], output_attention=False),
                        config['d_model'], config['n_heads']),
                    config['d_model'],
                    config['d_ff'],
                    dropout=config['dropout'],
                    activation=config['activation'],
                )
                for l in range(config['d_layers'])
            ],
            norm_layer=torch.nn.LayerNorm(config['d_model']),
            projection=nn.Linear(config['d_model'], config['c_out'], bias=True)
        )
# def forward(self, x_enc,x_mark_enc=torch.rand(16,182,4,device=device), x_dec=torch.rand(16,91,24,device=device), x_mark_dec=torch.rand(16,273,4,device=device),
#                 enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
    
    def forward(self, x_enc, x_mark_enc=torch.rand(16,182,4,device=device), x_dec=torch.rand(16,182,24,device=device), x_mark_dec=torch.rand(16,182,4,device=device), 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        b,_,_=x_enc.size()
        x_mark_enc=torch.rand(b,182,4,device=device)
        x_dec=torch.rand(b,182,24,device=device)
        x_dec=x_enc
        x_mark_dec=torch.rand(b,182,4,device=device)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # dec_out = self.dec_embedding(x_dec, x_mark_dec)
        # dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        # if self.output_attention:
        #     return dec_out[:, -self.pred_len:, :], attns
        # else:
        #     return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.output_attention:
            return enc_out[:, -self.pred_len:, :], attns
            # return enc_out
        else:
            return enc_out[:, -self.pred_len:, :]  # [B, L, D]