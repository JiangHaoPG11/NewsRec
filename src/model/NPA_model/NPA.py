import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import *

class news_encoder(torch.nn.Module):
    def __init__(self, word_dim, num_filters, window_sizes, query_vector_dim):
        super(news_encoder, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=num_filters,
                              kernel_size=(window_sizes, word_dim),
                              padding=(int((window_sizes - 1) / 2), 0))
        self.news_attention = QueryAttention(query_vector_dim, num_filters)
        self.dropout_prob = 0.3

    def forward(self, word_embedding, user_embedding):
        # 单词表征
        word_embedding = self.conv(word_embedding.unsqueeze(dim=1)).squeeze(dim=3)
        word_embedding = F.dropout(word_embedding, p=self.dropout_prob, training=self.training)
        word_embedding = torch.relu(word_embedding.transpose(2,1))
        # 附加注意力
        if len(user_embedding.shape) == 2:
            user_embedding = user_embedding.unsqueeze(1)
        news_rep = torch.tanh(self.news_attention(user_embedding, word_embedding))
        news_rep = F.dropout(news_rep, p=self.dropout_prob, training=self.training)
        return news_rep

class user_encoder(torch.nn.Module):
    def __init__(self, word_dim, num_filters, window_sizes, query_vector_dim):
        super(user_encoder, self).__init__()
        self.user_attention = QueryAttention(query_vector_dim, num_filters)
        self.dropout_prob = 0.3

    def forward(self, clicked_news_rep, user_embeding):
        user_rep = torch.tanh(self.user_attention(user_embeding.unsqueeze(0).unsqueeze(0), clicked_news_rep.unsqueeze(0)))
        user_rep = F.dropout(user_rep, p=self.dropout_prob, training=self.training)
        return user_rep

class user_embedding_encoder(torch.nn.Module):
    def __init__(self, args):
        super(user_embedding_encoder, self).__init__()
        self.args = args
        self.user_embedding = nn.Embedding(self.args.user_num, self.args.embedding_dim)
        self.fc1 = nn.Linear(self.args.embedding_dim, self.args.query_vector_dim, bias=True)
        self.fc2 = nn.Linear(self.args.embedding_dim, self.args.query_vector_dim, bias=True)

    def forward(self, user_index):
        # 获取用户嵌入
        user_embedding = self.user_embedding(user_index)
        user_vector = torch.relu(self.fc1(user_embedding))
        user_vector_2 = torch.relu(self.fc2(user_embedding))
        return user_vector, user_vector_2


class NPA(torch.nn.Module):
    def __init__(self, args, entity_embedding, relation_embedding, news_entity_dict, entity_adj, relation_adj,
                 news_title_word_index, word_embedding, news_category_index, news_subcategory_index, device):
        super(NPA, self).__init__()
        self.args = args
        self.device = device

        # no_embedding
        self.word_embedding = word_embedding
        # self.entity_embedding = entity_embedding
        # self.relation_embedding = relation_embedding

        self.user_embedding_encoder = user_embedding_encoder(args)

        # NPA
        self.news_encoder = news_encoder(self.args.word_embedding_dim, self.args.cnn_num_filters,
                                         self.args.cnn_window_sizes, self.args.query_vector_dim)
        self.user_encoder = user_encoder(self.args.word_embedding_dim, self.args.cnn_num_filters,
                                         self.args.cnn_window_sizes, self.args.query_vector_dim)

        # dict
        self.news_title_word_dict = news_title_word_index
        self.news_category_dict = news_category_index
        self.news_subcategory_dict = news_subcategory_index
        self.entity_adj = entity_adj
        self.relation_adj = relation_adj
        self.news_entity_dict = news_entity_dict

    def get_user_news_rep(self, user_index,  candidate_news_index, user_clicked_news_index):
        # 新闻单词
        candidate_news_word_embedding = self.word_embedding[self.news_title_word_dict[candidate_news_index]].to(self.device)
        user_clicked_news_word_embedding = self.word_embedding[self.news_title_word_dict[user_clicked_news_index]].to(self.device)
        
        # user query
        user_vector, user_vector_2 = self.user_embedding_encoder(user_index.to(self.device))

        ## 新闻编码器
        news_rep = None
        for i in range(self.args.sample_size):
            title_word_embedding_one = candidate_news_word_embedding[:, i, :]
            news_rep_one = self.news_encoder(title_word_embedding_one, user_vector)
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
            # 用户嵌入
            user_vector_one = user_vector[i, :]
            user_vector_one_2 = user_vector_2[i, :]
            # 点击新闻表示
            clicked_news_rep_one = self.news_encoder(title_word_embedding_one, user_vector_one.unsqueeze(0).repeat(50, 1, 1))
            # 用户表示
            user_rep_one = self.user_encoder(clicked_news_rep_one, user_vector_one_2).unsqueeze(0)
            if i == 0:
                user_rep = user_rep_one
            else:
                user_rep = torch.cat([user_rep, user_rep_one], dim=0)
        return user_rep, news_rep

    def forward(self, user_index, candidate_news, user_clicked_news_index):
        # 新闻用户表征
        user_rep, news_rep = self.get_user_news_rep(user_index,  candidate_news, user_clicked_news_index)
        # 预测得分
        score = torch.sum(news_rep * user_rep, dim=-1).view(self.args.batch_size, -1)
        return score

    def test(self,user_index, candidate_news, user_clicked_news_index):
        # 新闻用户表征
        user_rep, news_rep = self.get_user_news_rep(user_index, candidate_news, user_clicked_news_index)
        # 预测得分
        score = torch.sum(news_rep * user_rep, dim=-1).view(self.args.batch_size, -1)
        score = torch.sigmoid(score)
        return score, news_rep
