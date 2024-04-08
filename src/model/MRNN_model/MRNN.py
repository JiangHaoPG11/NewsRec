import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import *

class news_encoder(torch.nn.Module):
    def __init__(self, word_dim, attention_dim, attention_heads, query_vector_dim, entity_size, entity_embedding_dim,
                 category_dim, subcategory_dim, category_size, subcategory_size):
        super(news_encoder, self).__init__()
        self.multi_dim = attention_dim * attention_heads
        self.word_norm = nn.LayerNorm(self.multi_dim)

        # 主题级表征网络
        self.embedding_layer1 = nn.Embedding(category_size, embedding_dim=category_dim)
        self.fc1 = nn.Linear(category_dim, self.multi_dim, bias=True)

        # 副主题级表征网络
        self.embedding_layer2 = nn.Embedding(subcategory_size, embedding_dim=subcategory_dim)
        self.fc2 = nn.Linear(subcategory_dim, self.multi_dim, bias=True)

        # 单词级表征网络
        self.multiheadatt = MultiHeadSelfAttention(word_dim, attention_dim * attention_heads, attention_heads)
        self.word_attention = Additive_Attention(query_vector_dim, self.multi_dim)

        # 实体级表征网络
        self.fc3 = nn.Linear(entity_embedding_dim, entity_embedding_dim, bias=True)
        self.GCN = gcn(entity_size, entity_embedding_dim, self.multi_dim)
        self.entity_attention = Additive_Attention(query_vector_dim, self.multi_dim)
        self.news_attention = Additive_Attention(query_vector_dim, self.multi_dim)
        self.entity_norm = nn.LayerNorm(self.multi_dim)
        self.dropout_prob = 0.2

    def forward(self, word_embedding, entity_embedding, category_index, subcategory_index):
        # 主题级表征
        category_embedding = self.embedding_layer1(category_index.to(torch.int64))
        category_rep = torch.relu(self.fc1(category_embedding))
        category_rep = F.dropout(category_rep, p=self.dropout_prob, training=self.training)

        # 副主题级表征
        subcategory_embedding = self.embedding_layer2(subcategory_index.to(torch.int64))
        subcategory_rep = torch.relu(self.fc2(subcategory_embedding))
        subcategory_rep = F.dropout(subcategory_rep, p=self.dropout_prob, training=self.training)

        # 单词级新闻表征
        word_embedding = self.multiheadatt(word_embedding)
        word_embedding = self.word_norm(word_embedding)
        word_embedding = F.dropout(word_embedding, p=self.dropout_prob, training=self.training)
        # word_rep = self.word_attention(word_embedding)
        word_rep = torch.relu(self.word_attention(word_embedding))
        word_rep = F.dropout(word_rep, p=self.dropout_prob, training=self.training)

        # 实体级新闻表征
        entity_embedding = torch.relu(self.fc3(entity_embedding))
        entity_inter = self.GCN(entity_embedding)
        entity_inter = self.entity_norm(entity_inter)
        entity_inter = F.dropout(entity_inter, p=self.dropout_prob, training=self.training)
        # entity_rep = self.entity_attention(entity_inter)
        entity_rep = torch.relu(self.entity_attention(entity_inter))
        entity_rep = F.dropout(entity_rep, p=self.dropout_prob, training=self.training)

        # 新闻附加注意力
        news_rep = torch.cat(
            [
                word_rep.unsqueeze(1), 
                entity_rep.unsqueeze(1),
                category_rep.unsqueeze(1), 
                subcategory_rep.unsqueeze(1)
            ], dim=1
        )
        news_rep = self.news_attention(news_rep)
        news_rep = news_rep
        # news_rep = F.dropout(news_rep, p=self.dropout_prob, training=self.training)
        return news_rep

class user_encoder(torch.nn.Module):
    def __init__(self, word_dim, attention_dim, attention_heads, query_vector_dim, 
                 entity_size, entity_embedding_dim, category_dim, subcategory_dim, 
                 category_size, subcategory_size):
        super(user_encoder, self).__init__()
        self.multi_dim = attention_dim * attention_heads
        self.user_attention = Additive_Attention(query_vector_dim, self.multi_dim)
        self.norm = nn.LayerNorm(self.multi_dim)

        self.multiheadatt = MultiHeadSelfAttention(self.multi_dim, self.multi_dim, attention_heads)
        self.dropout_prob = 0.2

    def forward(self, clicked_news_rep):
        user_seq_rep = self.multiheadatt(clicked_news_rep)
        user_seq_rep = self.norm(user_seq_rep)
        user_seq_rep = F.dropout(user_seq_rep, p=self.dropout_prob, training=self.training)
        user_rep = self.user_attention(user_seq_rep)
        user_rep = F.dropout(user_rep, p=self.dropout_prob, training=self.training)
        return user_rep

class MRNN(torch.nn.Module):
    def __init__(self, args, entity_embedding, relation_embedding, news_entity_dict, entity_adj, relation_adj,
                 news_title_word_index, word_embedding, news_category_index, news_subcategory_index,  device):
        super(MRNN, self).__init__()
        self.args = args
        self.device = device

        # no_embedding
        self.word_embedding = word_embedding
        self.entity_embedding = entity_embedding.to(device)
        self.relation_embedding = relation_embedding.to(device)

        # embedding
        self.category_embedding = nn.Embedding(self.args.category_num, self.args.embedding_dim).to(device)
        self.subcategory_embedding = nn.Embedding(self.args.subcategory_num, self.args.embedding_dim).to(device)

        # MRNN
        self.news_encoder = news_encoder(self.args.word_embedding_dim, self.args.attention_dim,
                                         self.args.attention_heads, self.args.query_vector_dim,
                                         self.args.news_entity_size, self.args.entity_embedding_dim,
                                         self.args.category_embedding_dim, self.args.subcategory_embedding_dim,
                                         self.args.category_num, self.args.subcategory_num)
        self.user_encoder = user_encoder(self.args.word_embedding_dim, self.args.attention_dim,
                                         self.args.attention_heads, self.args.query_vector_dim,
                                         self.args.news_entity_size, self.args.entity_embedding_dim,
                                         self.args.category_embedding_dim, self.args.subcategory_embedding_dim,
                                         self.args.category_num, self.args.subcategory_num)
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
        # 新闻主题
        candidate_news_category_index = torch.IntTensor(self.news_category_dict[np.array(candidate_news_index)]).to(self.device)
        user_clicked_news_category_index = torch.IntTensor(self.news_category_dict[np.array(user_clicked_news_index.cpu())]).to(self.device)
        # 新闻副主题
        candidate_news_subcategory_index = torch.IntTensor(self.news_subcategory_dict[np.array(candidate_news_index.cpu())]).to(self.device)
        user_clicked_news_subcategory_index = torch.IntTensor(self.news_subcategory_dict[np.array(user_clicked_news_index.cpu())]).to(self.device)

        # 新闻编码器
        news_rep = None
        for i in range(self.args.sample_size):
            news_word_embedding_one = candidate_news_word_embedding[:, i, :, :]
            news_entity_embedding_one = candidate_news_entity_embedding[:, i, :, :]
            news_category_index = candidate_news_category_index[:, i]
            news_subcategory_index = candidate_news_subcategory_index[:, i]
            news_rep_one = self.news_encoder(news_word_embedding_one, news_entity_embedding_one,
                                             news_category_index, news_subcategory_index).unsqueeze(1)
            if i == 0:
                news_rep = news_rep_one
            else:
                news_rep = torch.cat([news_rep, news_rep_one], dim=1)
        # 用户编码器
        user_rep = None
        for i in range(self.args.batch_size):
            # 点击新闻单词嵌入
            clicked_news_word_embedding_one = user_clicked_news_word_embedding[i, :, :, :]
            clicked_news_word_embedding_one = clicked_news_word_embedding_one.squeeze()
            # 点击新闻实体嵌入
            clicked_news_entity_embedding_one = user_clicked_news_entity_embedding[i, :, :, :]
            clicked_news_entity_embedding_one = clicked_news_entity_embedding_one.squeeze()
            # 点击新闻主题index
            clicked_news_category_index = user_clicked_news_category_index[i, :]
            # 点击新闻副主题index
            clicked_news_subcategory_index = user_clicked_news_subcategory_index[i, :]
            # 点击新闻表示
            clicked_news_rep = self.news_encoder(clicked_news_word_embedding_one, clicked_news_entity_embedding_one, 
                                                 clicked_news_category_index, clicked_news_subcategory_index).unsqueeze(0)
            # 用户表征
            user_rep_one = self.user_encoder(clicked_news_rep).unsqueeze(0)
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
        score = torch.sum(news_rep * user_rep, dim=-1).view(self.args.batch_size, -1)
        return score

    def test(self, candidate_news, user_clicked_news_index):
        # candidate_news = torch.flatten(candidate_news, 0, 1)
        # 新闻用户表征
        user_rep, news_rep = self.get_user_news_rep(candidate_news, user_clicked_news_index)
        # 预测得分
        score = torch.sum(news_rep * user_rep, dim=-1).view(self.args.batch_size, -1)
        # score = torch.sigmoid(score)
        return score
