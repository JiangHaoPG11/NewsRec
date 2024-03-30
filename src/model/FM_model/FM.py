import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import *

class FM(torch.nn.Module):
    def __init__(self, args, device):
        super(FM, self).__init__()
        self.args = args
        self.device = device
        self.user_emb_matrix = nn.Embedding(self.args.user_num, args.embedding_dim)
        self.item_emb_matrix = nn.Embedding(self.args.news_num, args.embedding_dim)
        self.use_pretrain = False
        if self.use_pretrain == False:
            nn.init.xavier_uniform_(self.user_emb_matrix.weight)
            nn.init.xavier_uniform_(self.item_emb_matrix.weight)

    def forward(self, user_index, item_index):
        user_index = user_index.to(self.device)
        item_index = item_index.to(self.device)
        # FM
        user_embed = self.user_emb_matrix(user_index).unsqueeze(1)
        item_embed = self.item_emb_matrix(item_index)
        scores = torch.sigmoid(torch.sum(user_embed * item_embed, dim=-1))
        return scores

    def test(self, user_index, item_index):
        user_index = user_index.to(self.device)
        item_index = item_index.to(self.device)
        # FM
        user_embed = self.user_emb_matrix(user_index).unsqueeze(1)
        item_embed = self.item_emb_matrix(item_index)
        scores = torch.sigmoid(torch.sum(user_embed * item_embed, dim=-1))
        return scores
