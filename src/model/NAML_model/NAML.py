# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from utils.utils import *

# class news_encoder(torch.nn.Module):
#     def __init__(self, word_dim, title_word_size, category_dim, subcategory_dim,
#                  dropout_prob, query_vector_dim, num_filters, window_sizes, category_size, subcategory_size):
#         super(news_encoder, self).__init__()
#         self.category_embedding_layer = nn.Embedding(category_size, embedding_dim=category_dim)
#         self.subcategory_embedding_layer = nn.Embedding(subcategory_size, embedding_dim=subcategory_dim)
#         self.fc1 = nn.Linear(category_dim, num_filters, bias=True)
#         self.fc2 = nn.Linear(subcategory_dim, num_filters, bias=True)
#         self.cnn = cnn(title_word_size, word_dim, dropout_prob, query_vector_dim, num_filters, window_sizes)        
#         self.news_attention = Additive_Attention(query_vector_dim, num_filters)
#         self.dropout_prob = 0.2

#     def forward(self, word_embedding, category_index, subcategory_index):
#         # 主题表征
#         category_embedding = self.category_embedding_layer(category_index.to(torch.int64))
#         category_rep = torch.relu(self.fc1(category_embedding))
#         # 副主题表征
#         subcategory_embedding = self.subcategory_embedding_layer(subcategory_index.to(torch.int64))
#         subcategory_rep = torch.relu(self.fc2(subcategory_embedding))
#         # 单词表征
#         word_rep = torch.relu(self.cnn(word_embedding))
#         word_rep = F.dropout(word_rep, p=self.dropout_prob, training=self.training)
#         # 附加注意力
#         news_rep = torch.cat([word_rep.unsqueeze(1), category_rep.unsqueeze(1), subcategory_rep.unsqueeze(1)], dim=1)
#         news_rep = torch.tanh(self.news_attention(news_rep))
#         return news_rep

# class user_encoder(torch.nn.Module):
#     def __init__(self, word_dim, title_word_size, category_dim, subcategory_dim,
#                  dropout_prob, query_vector_dim, num_filters, window_sizes, category_size,subcategory_size):
#         super(user_encoder, self).__init__()
#         self.user_attention = Additive_Attention(query_vector_dim, num_filters)
#         self.dropout_prob = 0.2

#     def forward(self, clicked_news_rep):
#         user_rep = torch.tanh(self.user_attention(clicked_news_rep.unsqueeze(0)))
#         return user_rep

# class NAML(torch.nn.Module):
#     def __init__(self, args, entity_embedding, relation_embedding, news_entity_dict, entity_adj, relation_adj,
#                  news_title_word_index, word_embedding, news_category_index, news_subcategory_index, device):
#         super(NAML, self).__init__()
#         self.args = args
#         self.device = device

#         # no_embedding
#         self.word_embedding = word_embedding
#         # self.entity_embedding = entity_embedding
#         # self.relation_embedding = relation_embedding

#         # NAML
#         self.news_encoder = news_encoder(self.args.word_embedding_dim, self.args.title_word_size,
#                                         self.args.category_embedding_dim, self.args.subcategory_embedding_dim,
#                                         self.args.drop_prob, self.args.query_vector_dim,
#                                         self.args.cnn_num_filters, self.args.cnn_window_sizes,
#                                         self.args.category_num, self.args.subcategory_num)

#         self.user_encoder = user_encoder(self.args.word_embedding_dim, self.args.title_word_size,
#                                          self.args.category_embedding_dim, self.args.subcategory_embedding_dim,
#                                          self.args.drop_prob, self.args.query_vector_dim,
#                                          self.args.cnn_num_filters, self.args.cnn_window_sizes,
#                                          self.args.category_num, self.args.subcategory_num)

#         # dict
#         self.news_title_word_dict = news_title_word_index
#         self.news_category_dict = news_category_index
#         self.news_subcategory_dict = news_subcategory_index
#         self.entity_adj = entity_adj
#         self.relation_adj = relation_adj
#         self.news_entity_dict = news_entity_dict

#     def get_user_news_rep(self, candidate_news_index, user_clicked_news_index):
#         # 新闻单词
#         candidate_news_word_embedding = self.word_embedding[self.news_title_word_dict[candidate_news_index]].to(self.device)
#         user_clicked_news_word_embedding = self.word_embedding[self.news_title_word_dict[user_clicked_news_index]].to(self.device)

#         # 新闻主题
#         candidate_news_category_index = torch.IntTensor(self.news_category_dict[np.array(candidate_news_index)]).to(self.device)
#         user_clicked_news_category_index = torch.IntTensor(self.news_category_dict[np.array(user_clicked_news_index.cpu())]).to(self.device)
#         # 新闻副主题
#         candidate_news_subcategory_index = torch.IntTensor(self.news_subcategory_dict[np.array(candidate_news_index.cpu())]).to(self.device)
#         user_clicked_news_subcategory_index = torch.IntTensor(self.news_subcategory_dict[np.array(user_clicked_news_index.cpu())]).to(self.device)
        
#         # 新闻编码器
#         news_rep = None
#         for i in range(self.args.sample_size):
#             news_word_embedding_one = candidate_news_word_embedding[:, i, :]
#             news_category_index = candidate_news_category_index[:, i]
#             news_subcategory_index = candidate_news_subcategory_index[:, i]
#             news_rep_one = self.news_encoder(news_word_embedding_one, news_category_index, news_subcategory_index).unsqueeze(1)
#             if i == 0:
#                 news_rep = news_rep_one
#             else:
#                 news_rep = torch.cat([news_rep, news_rep_one], dim=1)
#         # 用户编码器
#         user_rep = None
#         for i in range(self.args.batch_size):
#             # 点击新闻单词嵌入
#             clicked_news_word_embedding_one = user_clicked_news_word_embedding[i, :, :, :]
#             clicked_news_word_embedding_one = clicked_news_word_embedding_one.squeeze()
#             # 点击新闻主题index
#             clicked_news_category_index = user_clicked_news_category_index[i, :]
#             # 点击新闻副主题index
#             clicked_news_subcategory_index = user_clicked_news_subcategory_index[i, :]
#             # 点击新闻表征
#             clicked_news_rep = self.news_encoder(clicked_news_word_embedding_one, clicked_news_category_index, clicked_news_subcategory_index)
#             # 用户表征
#             user_rep_one = self.user_encoder(clicked_news_rep).unsqueeze(0)
#             if i == 0:
#                 user_rep = user_rep_one
#             else:
#                 user_rep = torch.cat([user_rep, user_rep_one], dim=0)
#         return user_rep, news_rep

#     def forward(self, candidate_news, user_clicked_news_index):
#         # 新闻用户表征
#         user_rep, news_rep = self.get_user_news_rep(candidate_news, user_clicked_news_index)
#         # 预测得分
#         score = torch.sum(news_rep * user_rep, dim=-1).view(self.args.batch_size, -1)
#         return score

#     def test(self, candidate_news, user_clicked_news_index):
#         # 新闻用户表征
#         user_rep, news_rep = self.get_user_news_rep(candidate_news, user_clicked_news_index)
#         # 预测得分
#         score = torch.sum(news_rep * user_rep, dim=-1).view(self.args.batch_size, -1)
#         return score, user_rep, news_rep

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import *

class news_encoder(torch.nn.Module):
    def __init__(self, word_dim, title_word_size, category_dim, subcategory_dim,
                 dropout_prob, query_vector_dim, num_filters, window_sizes, category_size, subcategory_size):
        super(news_encoder, self).__init__()
        self.category_embedding_layer = nn.Embedding(category_size, embedding_dim=category_dim)
        self.subcategory_embedding_layer = nn.Embedding(subcategory_size, embedding_dim=subcategory_dim)
        self.fc1 = nn.Linear(category_dim, num_filters, bias=True)
        self.fc2 = nn.Linear(subcategory_dim, num_filters, bias=True)
        self.cnn = cnn(title_word_size, word_dim, dropout_prob, query_vector_dim, num_filters, window_sizes)        
        self.news_attention = Additive_Attention(query_vector_dim, num_filters)
        self.dropout_prob = 0.2

    def forward(self, word_embedding, category_index, subcategory_index):
        # 主题表征
        category_embedding = self.category_embedding_layer(category_index.to(torch.int64))
        category_rep = torch.relu(self.fc1(category_embedding))
        # 副主题表征
        subcategory_embedding = self.subcategory_embedding_layer(subcategory_index.to(torch.int64))
        subcategory_rep = torch.relu(self.fc2(subcategory_embedding))
        # 单词表征
        word_rep = self.cnn(word_embedding)
        word_rep = F.dropout(word_rep, p=self.dropout_prob, training=self.training)
        # 附加注意力
        news_rep = torch.cat([word_rep.unsqueeze(1), category_rep.unsqueeze(1), subcategory_rep.unsqueeze(1)], dim=1)
        news_rep = self.news_attention(news_rep)
        return news_rep

class user_encoder(torch.nn.Module):
    def __init__(self, word_dim, title_word_size, category_dim, subcategory_dim,
                 dropout_prob, query_vector_dim, num_filters, window_sizes, category_size,subcategory_size):
        super(user_encoder, self).__init__()
        self.user_attention = Additive_Attention(query_vector_dim, num_filters)
        self.dropout_prob = 0.2

    def forward(self, clicked_news_rep):
        user_rep = self.user_attention(clicked_news_rep.unsqueeze(0))
        return user_rep

class NAML(torch.nn.Module):
    def __init__(self, args, entity_embedding, relation_embedding, news_entity_dict, entity_adj, relation_adj,
                 news_title_word_index, word_embedding, news_category_index, news_subcategory_index, device):
        super(NAML, self).__init__()
        self.args = args
        self.device = device

        # no_embedding
        self.word_embedding = word_embedding
        # self.entity_embedding = entity_embedding
        # self.relation_embedding = relation_embedding

        # NAML
        self.news_encoder = news_encoder(self.args.word_embedding_dim, self.args.title_word_size,
                                        self.args.category_embedding_dim, self.args.subcategory_embedding_dim,
                                        self.args.drop_prob, self.args.query_vector_dim,
                                        self.args.cnn_num_filters, self.args.cnn_window_sizes,
                                        self.args.category_num, self.args.subcategory_num)

        self.user_encoder = user_encoder(self.args.word_embedding_dim, self.args.title_word_size,
                                         self.args.category_embedding_dim, self.args.subcategory_embedding_dim,
                                         self.args.drop_prob, self.args.query_vector_dim,
                                         self.args.cnn_num_filters, self.args.cnn_window_sizes,
                                         self.args.category_num, self.args.subcategory_num)

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
            news_rep_one = self.news_encoder(news_word_embedding_one, news_category_index, news_subcategory_index).unsqueeze(1)
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
            # 点击新闻主题index
            clicked_news_category_index = user_clicked_news_category_index[i, :]
            # 点击新闻副主题index
            clicked_news_subcategory_index = user_clicked_news_subcategory_index[i, :]
            # 点击新闻表征
            clicked_news_rep = self.news_encoder(clicked_news_word_embedding_one, clicked_news_category_index, clicked_news_subcategory_index)
            # 用户表征
            user_rep_one = self.user_encoder(clicked_news_rep).unsqueeze(0)
            if i == 0:
                user_rep = user_rep_one
            else:
                user_rep = torch.cat([user_rep, user_rep_one], dim=0)
        return user_rep, news_rep

    def forward(self, candidate_news, user_clicked_news_index):
        # 新闻用户表征
        user_rep, news_rep = self.get_user_news_rep(candidate_news, user_clicked_news_index)
        # 预测得分
        score = torch.sum(news_rep * user_rep, dim=-1).view(self.args.batch_size, -1)
        return score

    def test(self, candidate_news, user_clicked_news_index):
        # 新闻用户表征
        user_rep, news_rep = self.get_user_news_rep(candidate_news, user_clicked_news_index)
        # news_rep = news_rep[:,0,:]
        # 预测得分
        score = torch.sum(news_rep * user_rep, dim=-1).view(self.args.batch_size, -1)
        # score = torch.sigmoid(score)
        return score, user_rep, news_rep
