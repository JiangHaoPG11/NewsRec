import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import *

class news_encoder(torch.nn.Module):
    def __init__(self, batch_size, word_dim, entity_dim, news_entity_size, num_filters, window_sizes, use_context):
        super(news_encoder, self).__init__()
        self.news_dim = num_filters * len(window_sizes)
        self.KCNN = KCNN(batch_size, news_entity_size, word_dim, entity_dim, num_filters, window_sizes, use_context)
        self.norm1 = nn.LayerNorm(self.news_dim)
        self.dropout_prob = 0.2

    def forward(self, word_embedding, entity_embedding, context_embedding = None):
        news_rep = self.KCNN(word_embedding, entity_embedding, context_embedding)
        news_rep = F.dropout(news_rep, p=self.dropout_prob, training=self.training)
        news_rep = self.norm1(news_rep)
        return news_rep

class user_encoder(torch.nn.Module):
    def __init__(self, user_clicked_news_num, word_dim, entity_dim, news_entity_size, num_filters, window_sizes, use_context):
        super(user_encoder, self).__init__()
        self.news_dim = num_filters * len(window_sizes)
        self.news_encoder = news_encoder(user_clicked_news_num, word_dim, entity_dim, news_entity_size, num_filters, window_sizes, use_context)
        self.norm1 = nn.LayerNorm(self.news_dim)
        self.user_attention = dkn_attention(window_sizes, num_filters, user_clicked_news_num)
        self.norm2 = nn.LayerNorm(self.news_dim)
        self.dropout_prob = 0.2

    def forward(self, word_embedding, entity_embedding, candidate_news_rep, context_embedding=None):
        news_rep = self.news_encoder(word_embedding, entity_embedding, context_embedding)
        news_rep = F.dropout(news_rep, p=self.dropout_prob, training=self.training)
        news_rep = self.norm1(news_rep)
        user_rep = self.user_attention(candidate_news_rep, news_rep)
        user_rep = F.dropout(user_rep, p=self.dropout_prob, training=self.training)
        user_rep = self.norm2(user_rep)
        return user_rep

class DKN(torch.nn.Module):
    def __init__(self, args, entity_embedding, relation_embedding, news_entity_dict, entity_adj, relation_adj,
                 news_title_word_index, word_embedding, news_category_index, news_subcategory_index,
                 device):
        super(DKN, self).__init__()
        self.args = args
        self.device = device

        # no_embedding
        self.word_embedding = word_embedding
        self.entity_embedding = entity_embedding.to(device)
        self.relation_embedding = relation_embedding.to(device)

        # embedding
        self.category_embedding = nn.Embedding(self.args.category_num, self.args.embedding_dim).to(device)
        self.subcategory_embedding = nn.Embedding(self.args.subcategory_num, self.args.embedding_dim).to(device)

        # DKN
        self.news_encoder = news_encoder(
            self.args.batch_size, 
            self.args.word_embedding_dim, 
            self.args.entity_embedding_dim,
            self.args.news_entity_size, 
            self.args.kcnn_num_filters, 
            self.args.kcnn_window_sizes, 
            False
        )
        
        self.user_encoder = user_encoder(
            self.args.user_clicked_num, 
            self.args.word_embedding_dim, 
            self.args.entity_embedding_dim,
            self.args.news_entity_size, 
            self.args.kcnn_num_filters, 
            self.args.kcnn_window_sizes, 
            False
        )

        self.dnn = nn.Sequential(nn.Linear(self.args.word_embedding_dim, int(math.sqrt(self.args.word_embedding_dim))),
                                 nn.ReLU(),
                                 nn.Linear(int(math.sqrt(self.args.word_embedding_dim)), 1))
        # dict
        self.news_title_word_dict = news_title_word_index
        self.news_category_dict = news_category_index
        self.news_subcategory_dict = news_subcategory_index
        self.entity_adj = entity_adj
        self.relation_adj = relation_adj
        self.news_entity_dict = news_entity_dict

    def get_news_entities_batch(self, newsids):
        news_entities = []
        newsids = newsids.unsqueeze(-1)
        for i in range(newsids.shape[0]):
            news_entities.append([])
            for j in range(newsids.shape[1]):
                news_entities[-1].append([])
                news_entities[-1][-1].append(self.news_entity_dict[int(newsids[i, j])][:self.args.news_entity_size])
        return np.array(news_entities)

    def get_user_news_rep(self, candidate_news_index, user_clicked_news_index):
        # 新闻单词
        candidate_news_word_embedding = self.word_embedding[self.news_title_word_dict[candidate_news_index]].to(self.device)
        user_clicked_news_word_embedding = self.word_embedding[self.news_title_word_dict[user_clicked_news_index]].to(self.device)
        # 新闻实体
        candidate_news_entity_embedding = self.entity_embedding[self.get_news_entities_batch(candidate_news_index)].to(self.device).squeeze()
        user_clicked_news_entity_embedding = self.entity_embedding[self.get_news_entities_batch(user_clicked_news_index)].to(self.device).squeeze()
        ## 新闻编码器
        news_rep = None
        for i in range(self.args.sample_size):
            news_word_embedding_one = candidate_news_word_embedding[:, i, :self.args.news_entity_size, :]
            news_entity_embedding_one = candidate_news_entity_embedding[:, i, :, :]
            news_rep_one = self.news_encoder(news_word_embedding_one, news_entity_embedding_one)
            news_rep_one = news_rep_one.unsqueeze(1)
            if i == 0:
                news_rep = news_rep_one
            else:
                news_rep = torch.cat([news_rep, news_rep_one], dim=1)

        # 用户编码器
        user_rep = None
        for i in range(self.args.batch_size):
            clicked_word_embedding_one = user_clicked_news_word_embedding[i, :, :self.args.news_entity_size, :]
            clicked_word_embedding_one = clicked_word_embedding_one.squeeze()
            clicked_entity_embedding_one = user_clicked_news_entity_embedding[i, :, :self.args.news_entity_size, :]
            clicked_entity_embedding_one = clicked_entity_embedding_one.squeeze()
            candidate_news_rep = news_rep[i, :]
            user_rep_one = self.user_encoder(clicked_word_embedding_one, clicked_entity_embedding_one,
                                             candidate_news_rep)
            user_rep_one = user_rep_one.unsqueeze(0)
            if i == 0:
                user_rep = user_rep_one
            else:
                user_rep = torch.cat([user_rep, user_rep_one], dim=0)
        return user_rep, news_rep

    def forward(self, candidate_news, user_clicked_news_index):
        # candidate_news = torch.flatten(candidate_news, 0, 1)
        # 新闻用户表征
        user_rep, news_rep = self.get_user_news_rep(candidate_news, user_clicked_news_index)
        # 预测得分
        score = self.dnn(torch.cat([news_rep, user_rep], dim=2)).squeeze()
        return score

    def test(self, candidate_news, user_clicked_news_index):
        # candidate_news = torch.flatten(candidate_news, 0, 1)
        # 新闻用户表征
        user_rep, news_rep = self.get_user_news_rep(candidate_news, user_clicked_news_index)
        # 预测得分
        score = self.dnn(torch.cat([news_rep, user_rep], dim=2)).squeeze()
        return score
