import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import *

class news_encoder(torch.nn.Module):
    def __init__(self, word_dim, attention_dim, attention_heads, query_vector_dim):
        super(news_encoder, self).__init__()
        self.multi_dim = attention_dim * attention_heads
        self.multiheadatt = MultiHeadSelfAttention(
            word_dim, 
            self.multi_dim, 
            attention_heads
        )
        self.word_attention = Additive_Attention(query_vector_dim, self.multi_dim)
        self.fc1 = nn.Linear(self.multi_dim, self.multi_dim)
        self.fc2 = nn.Linear(self.multi_dim, self.multi_dim)
        self.dropout_prob = 0.2

    def forward(self, word_embedding):
        word_embedding = self.multiheadatt(word_embedding)
        word_embedding = F.dropout(word_embedding, p=self.dropout_prob, training=self.training)
        news_rep = self.fc2(torch.relu(self.fc1(word_embedding)))
        news_rep = self.word_attention(word_embedding)
        return news_rep

class user_encoder(torch.nn.Module):
    def __init__(self,  word_dim, attention_dim, attention_heads, query_vector_dim):
        super(user_encoder, self).__init__()
        self.news_encoder = news_encoder(
            word_dim, 
            attention_dim, 
            attention_heads, 
            query_vector_dim
        )
        self.multi_dim = attention_dim * attention_heads
        self.user_attention = Additive_Attention(query_vector_dim, self.multi_dim)
        self.norm1 = nn.LayerNorm(self.multi_dim)
        self.norm2 = nn.LayerNorm(self.multi_dim)

        self.dropout_prob = 0.2

    def forward(self, word_embedding):
        
        clicked_news_rep = self.news_encoder(word_embedding).unsqueeze(0)
        clicked_news_rep = F.dropout(clicked_news_rep, p=self.dropout_prob, training=self.training)
        user_seq_rep = self.user_attention(clicked_news_rep)
        user_seq_rep = self.norm1(user_seq_rep)

        # sample num 
        user_clicked_selected = torch.randperm(word_embedding.shape[0])[:40].sort().values
        word_embedding_selected = word_embedding[user_clicked_selected]
        selected_clicked_news_rep = self.news_encoder(word_embedding_selected).unsqueeze(0)
        selected_user_seq_rep = self.user_attention(selected_clicked_news_rep)
        selected_user_seq_rep = self.norm2(selected_user_seq_rep)
        
        return user_seq_rep, selected_user_seq_rep

class MCCM(torch.nn.Module):
    def __init__(self, args, entity_embedding, relation_embedding, news_entity_dict, entity_adj, relation_adj,
                 news_title_word_index, word_embedding, news_category_index, news_subcategory_index, device):
        super(MCCM, self).__init__()
        self.args = args
        self.device = device

        # no_embedding
        self.word_embedding = word_embedding
        # self.entity_embedding = entity_embedding
        # self.relation_embedding = relation_embedding

        # MCCM
        self.news_encoder = news_encoder(
            self.args.word_embedding_dim, 
            self.args.attention_dim, 
            self.args.attention_heads, 
            self.args.query_vector_dim
        ).to(device)
        self.user_encoder = user_encoder(
            self.args.word_embedding_dim, 
            self.args.attention_dim, 
            self.args.attention_heads,
            self.args.query_vector_dim
        ).to(device)

        # dict
        self.news_title_word_dict = news_title_word_index
        self.news_category_dict = news_category_index
        self.news_subcategory_dict = news_subcategory_index
        self.entity_adj = entity_adj
        self.relation_adj = relation_adj
        self.news_entity_dict = news_entity_dict

    def get_user_news_rep(self, candidate_news_index, user_clicked_news_index):
        # 新闻单词
        candidate_news_word_embedding = self.word_embedding[self.news_title_word_dict[candidate_news_index]].to(self.device)
        user_clicked_news_word_embedding = self.word_embedding[self.news_title_word_dict[user_clicked_news_index]].to(self.device)

        ## 新闻编码器
        news_rep = None
        for i in range(self.args.sample_size):
            title_word_embedding_one = candidate_news_word_embedding[:, i, :]
            news_rep_one = self.news_encoder(title_word_embedding_one)
            news_rep_one = news_rep_one.unsqueeze(1)
            if i == 0:
                news_rep = news_rep_one
            else:
                news_rep = torch.cat([news_rep, news_rep_one], dim=1)
        # 用户编码器
        user_rep = None
        selected_user_rep = None
        for i in range(self.args.batch_size):
            clicked_word_embedding_one = user_clicked_news_word_embedding[i, :, :, :]
            clicked_word_embedding_one = clicked_word_embedding_one.squeeze()
            user_rep_one, selected_user_rep_one = self.user_encoder(clicked_word_embedding_one)
            if i == 0:
                user_rep = user_rep_one.unsqueeze(0)
                selected_user_rep = selected_user_rep_one.unsqueeze(0)
            else:
                user_rep = torch.cat([user_rep, user_rep_one.unsqueeze(0)], dim=0)
                selected_user_rep = torch.cat([selected_user_rep, selected_user_rep_one.unsqueeze(0)], dim=0)

        return user_rep, selected_user_rep, news_rep

    def contrastive_loss(self, v_u, v_pos, v_neg, tau=1.0):
        pos_scores = torch.sum(v_u * v_pos, dim=1) / tau
        all_scores = torch.matmul(v_u, v_neg.transpose(0, 1)) / tau
        mask = torch.eye(v_u.size(0)).bool()
        all_scores[mask] = float('-inf')
        loss = torch.mean(
            - torch.log(
                torch.exp(pos_scores) / (torch.sum(torch.exp(all_scores), dim = 1) + torch.exp(pos_scores))
            )
        )
        print(loss)
        return loss.mean()

    def forward(self, candidate_news, user_clicked_news_index):
        # 新闻用户表征
        user_rep, selected_user_rep, news_rep = self.get_user_news_rep(candidate_news, user_clicked_news_index)
        contrast_loss = self.contrastive_loss(
            user_rep.squeeze(), 
            selected_user_rep.squeeze(), 
            user_rep.squeeze(),
            tau=5
        )
        # 预测得分
        score = torch.sum(news_rep * user_rep, dim=-1).view(self.args.batch_size, -1)
        return score, contrast_loss

    def test(self, candidate_news, user_clicked_news_index):
        # 新闻用户表征
        user_rep, _, news_rep = self.get_user_news_rep(candidate_news, user_clicked_news_index)
        # 预测得分
        score = torch.sum(news_rep * user_rep, dim=-1).view(self.args.batch_size, -1)
        # score = torch.sigmoid(score)
        return score, user_rep, news_rep
