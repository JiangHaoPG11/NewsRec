import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import *

class news_encoder(torch.nn.Module):
    def __init__(self, word_dim, attention_dim, attention_heads, query_vector_dim):
        super(news_encoder, self).__init__()
        self.multiheadatt = MultiHeadSelfAttention(word_dim, attention_dim * attention_heads, attention_heads)
        self.multi_dim = attention_dim * attention_heads
        self.norm = nn.LayerNorm(self.multi_dim)
        self.word_attention = Additive_Attention(query_vector_dim, self.multi_dim)
        self.dropout_prob = 0.2

    def forward(self, word_embedding):
        word_embedding = self.multiheadatt(word_embedding)
        word_embedding = self.norm(word_embedding)
        word_embedding = F.dropout(word_embedding, p=self.dropout_prob, training=self.training)
        news_rep = self.word_attention(word_embedding)
        # news_rep = torch.tanh(news_rep)
        #news_rep = F.dropout(news_rep, p=self.dropout_prob, training=self.training)
        #news_rep = self.norm2(news_rep)
        return news_rep

class user_encoder(torch.nn.Module):
    def __init__(self,  word_dim, attention_dim, attention_heads, query_vector_dim):
        super(user_encoder, self).__init__()
        self.news_encoder = news_encoder(word_dim, attention_dim, attention_heads, query_vector_dim)
        self.multiheadatt = MultiHeadSelfAttention(attention_dim * attention_heads, attention_dim * attention_heads, attention_heads)
        self.multi_dim = attention_dim * attention_heads
        self.user_attention = Additive_Attention(query_vector_dim, self.multi_dim)
        self.layer_norm = nn.LayerNorm(self.multi_dim)
        self.dropout_prob = 0.2

    def forward(self, word_embedding):
        clicked_news_rep = self.news_encoder(word_embedding).unsqueeze(0)
        clicked_news_rep = F.dropout(clicked_news_rep, p=self.dropout_prob, training=self.training)
        user_seq_rep = self.multiheadatt(clicked_news_rep)
        user_seq_rep = self.layer_norm(user_seq_rep)
        user_seq_rep = F.dropout(user_seq_rep, p=self.dropout_prob, training=self.training)
        user_rep = self.user_attention(user_seq_rep)
        return user_rep

class NRMS(torch.nn.Module):
    def __init__(self, args, entity_embedding, relation_embedding, news_entity_dict, entity_adj, relation_adj,
                 news_title_word_index, word_embedding, news_category_index, news_subcategory_index, device):
        super(NRMS, self).__init__()
        self.args = args
        self.device = device

        # no_embedding
        self.word_embedding = word_embedding
        # self.entity_embedding = entity_embedding
        # self.relation_embedding = relation_embedding

        # NRMS
        self.news_encoder = news_encoder(self.args.word_embedding_dim, self.args.attention_dim, self.args.attention_heads, self.args.query_vector_dim)
        self.user_encoder = user_encoder(self.args.word_embedding_dim, self.args.attention_dim, self.args.attention_heads, self.args.query_vector_dim)

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
        for i in range(self.args.batch_size):
            clicked_word_embedding_one = user_clicked_news_word_embedding[i, :, :, :]
            clicked_word_embedding_one = clicked_word_embedding_one.squeeze()
            user_rep_one = self.user_encoder(clicked_word_embedding_one).unsqueeze(0)
            if i == 0:
                user_rep = user_rep_one
            else:
                user_rep = torch.cat([user_rep, user_rep_one], dim=0)
        return user_rep, news_rep

    def forward(self, candidate_news, user_clicked_news_index):
        print(candidate_news.shape)
        print(user_clicked_news_index.shape)
        stop
        # 新闻用户表征
        user_rep, news_rep = self.get_user_news_rep(candidate_news, user_clicked_news_index)
        # 预测得分
        score = torch.sum(news_rep * user_rep, dim=-1).view(self.args.batch_size, -1)
        return score

    def test(self, candidate_news, user_clicked_news_index):
        print(candidate_news.shape)
        print(user_clicked_news_index.shape)
        stop
        # 新闻用户表征
        user_rep, news_rep = self.get_user_news_rep(candidate_news, user_clicked_news_index)
        # 预测得分
        score = torch.sum(news_rep * user_rep, dim=-1).view(self.args.batch_size, -1)
        # score = torch.sigmoid(score)
        return score, user_rep, news_rep
