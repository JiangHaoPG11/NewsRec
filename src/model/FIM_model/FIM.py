import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import *

class news_encoder(torch.nn.Module):
    def __init__(self, title_word_size, category_size, subcategory_size):
        super(news_encoder, self).__init__()
        self.news_feature_size = title_word_size + 2
        self.category_embedding_layer = nn.Embedding(category_size, embedding_dim=300)
        self.subcategory_embedding_layer = nn.Embedding(subcategory_size, embedding_dim=300)
        self.HDC_cnn = HDC_CNN_extractor(self.news_feature_size)
        self.dropout_prob = 0.2

    def forward(self, word_embedding, category_index, subcategory_index):
        # 主题嵌入
        category_embedding = self.category_embedding_layer(category_index)
        category_embedding = category_embedding.unsqueeze(1)
        # 副主题嵌入
        subcategory_embedding = self.subcategory_embedding_layer(subcategory_index)
        subcategory_embedding = subcategory_embedding.unsqueeze(1)
        # 空洞卷积
        news_feature_embedding = torch.cat([word_embedding, category_embedding, subcategory_embedding], dim = 1)
        news_vec = self.HDC_cnn(news_feature_embedding)
        return news_vec

class user_encoder(torch.nn.Module):
    def __init__(self, title_word_size, category_size, subcategory_size):
        super(user_encoder, self).__init__()
        self.news_encoder = news_encoder(title_word_size, category_size, subcategory_size)
        self.dropout_prob = 0.2

    def forward(self, word_embedding, category_index, subcategory_index):
        user_rep = self.news_encoder(word_embedding, category_index, subcategory_index).unsqueeze(0)
        return user_rep

class FIM(torch.nn.Module):
    def __init__(self, args, entity_embedding, relation_embedding, news_entity_dict, entity_adj, relation_adj,
                 news_title_word_index, word_embedding, news_category_index, news_subcategory_index,
                 device):
        super(FIM, self).__init__()
        self.args = args
        self.device = device

        # no_embedding
        self.word_embedding = word_embedding
        self.entity_embedding = entity_embedding.to(device)
        self.relation_embedding = relation_embedding.to(device)

        # embedding
        self.category_embedding = nn.Embedding(self.args.category_num, self.args.embedding_dim).to(device)
        self.subcategory_embedding = nn.Embedding(self.args.subcategory_num, self.args.embedding_dim).to(device)

        # FIM
        self.news_encoder = news_encoder(self.args.title_word_size, self.args.category_num, self.args.subcategory_num)
        self.user_encoder = user_encoder(self.args.title_word_size, self.args.category_num, self.args.subcategory_num)
        self.interaction_layer = FIM_interaction_layer(self.args.feature_dim)
        
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

    def get_score(self, candidate_news_index, user_clicked_news_index):
        # 新闻单词
        candidate_news_word_embedding = self.word_embedding[self.news_title_word_dict[candidate_news_index]].to(self.device)
        user_clicked_news_word_embedding = self.word_embedding[self.news_title_word_dict[user_clicked_news_index]].to(self.device)
        # 新闻主题
        candidate_news_category_index = torch.IntTensor(self.news_category_dict[np.array(candidate_news_index)]).to(self.device)
        user_clicked_news_category_index = torch.IntTensor(self.news_category_dict[np.array(user_clicked_news_index.cpu())]).to(self.device)
        # 新闻副主题
        candidate_news_subcategory_index = torch.IntTensor(self.news_subcategory_dict[np.array(candidate_news_index.cpu())]).to(self.device)
        user_clicked_news_subcategory_index = torch.IntTensor(self.news_subcategory_dict[np.array(user_clicked_news_index.cpu())]).to(self.device)

        # 新闻编码器
        news_rep = None
        for i in range(self.args.sample_size):
            news_word_embedding_one = candidate_news_word_embedding[:, i, :]
            news_category_index = candidate_news_category_index[:, i]
            news_subcategory_index = candidate_news_subcategory_index[:, i]
            news_rep_one = self.news_encoder(news_word_embedding_one, news_category_index, news_subcategory_index).unsqueeze(
                1)
            if i == 0:
                news_rep = news_rep_one
            else:
                news_rep = torch.cat([news_rep, news_rep_one], dim=1)

        # 用户编码器
        user_rep = None
        for i in range(self.args.batch_size):
            clicked_news_word_embedding_one = user_clicked_news_word_embedding[i, :, :, :]
            clicked_news_word_embedding_one = clicked_news_word_embedding_one.squeeze()
            # 点击新闻主题index
            clicked_news_category_index = user_clicked_news_category_index[i, :]
            # 点击新闻副主题index
            clicked_news_subcategory_index = user_clicked_news_subcategory_index[i, :]
            # 用户表征
            user_rep_one = self.user_encoder(clicked_news_word_embedding_one,
                                             clicked_news_category_index,
                                             clicked_news_subcategory_index)
            if i == 0:
                user_rep = user_rep_one
            else:
                user_rep = torch.cat([user_rep, user_rep_one], dim=0)
        score = self.interaction_layer(user_rep, news_rep)
        return score

    def forward(self, candidate_news, user_clicked_news_index):
        # candidate_news = torch.flatten(candidate_news, 0, 1)
        score = self.get_score(candidate_news, user_clicked_news_index)
        return score

    def test(self, candidate_news, user_clicked_news_index):
        # candidate_news = torch.flatten(candidate_news, 0, 1)
        score = self.get_score(candidate_news, user_clicked_news_index)
        return score
