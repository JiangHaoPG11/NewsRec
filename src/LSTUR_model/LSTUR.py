import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import *
from torch.nn.utils.rnn import pack_padded_sequence

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class news_encoder(torch.nn.Module):
    def __init__(self, word_dim, title_word_size, category_dim, subcategory_dim,
                 dropout_prob, query_vector_dim, num_filters, window_sizes, category_size, subcategory_size):
        super(news_encoder, self).__init__()
        self.embedding_layer1 = nn.Embedding(category_size, embedding_dim=category_dim)
        self.embedding_layer2 = nn.Embedding(subcategory_size, embedding_dim=subcategory_dim)
        self.fc1 = nn.Linear(category_dim, num_filters, bias=True)
        self.fc2 = nn.Linear(subcategory_dim, num_filters, bias=True)
        self.cnn = cnn(title_word_size, word_dim, dropout_prob, query_vector_dim, num_filters, window_sizes)
        self.dropout_prob = 0.3
        
    def forward(self, word_embedding, category_index, subcategory_index):
        # 主题表征
        category_embedding = self.embedding_layer1(category_index.to(torch.int64))
        category_rep = self.fc1(category_embedding)
        category_rep = F.dropout(category_rep, p=self.dropout_prob, training=self.training)
        # 副主题表征
        subcategory_embedding = self.embedding_layer2(subcategory_index.to(torch.int64))
        subcategory_rep = self.fc2(subcategory_embedding)
        subcategory_rep = F.dropout(subcategory_rep, p=self.dropout_prob, training=self.training)
        # 单词表征
        # word_embedding = F.dropout(word_embedding, p=self.dropout_prob, training=self.training)
        word_rep = self.cnn(word_embedding)
        word_rep = F.dropout(word_rep, p=self.dropout_prob, training=self.training)
        # 附加注意力
        news_rep = torch.cat([word_rep, category_rep, subcategory_rep], dim=1)
        return news_rep

class user_encoder(torch.nn.Module):
    def __init__(self, word_dim, title_word_size, category_dim, subcategory_dim,
                 dropout_prob, query_vector_dim, num_filters, window_sizes, category_size, subcategory_size,
                 user_size, long_short_term_method, masking_probability):
        super(user_encoder, self).__init__()
        self.masking_probability = masking_probability
        self.long_short_term_method = long_short_term_method
        self.user_embedding = nn.Embedding(user_size,
                                           num_filters * 3 if long_short_term_method == 'ini' else int(num_filters * 1.5),
                                           padding_idx=0)

        self.news_encoder = news_encoder(word_dim, title_word_size, category_dim, subcategory_dim,
                                         dropout_prob, query_vector_dim, num_filters, window_sizes, 
                                         category_size, subcategory_size)

        self.gru = nn.GRU(num_filters * 3,
                          num_filters * 3 if long_short_term_method == 'ini' else int(num_filters * 1.5),
                          num_layers = 1, 
                          batch_first = True )
                          
        self.dropout_prob = 0.3

    def forward(self, user_index, word_embedding, category_index, subcategory_index):
        # 新闻编码
        news_rep = self.news_encoder(word_embedding, category_index, subcategory_index)
        # 用户编码
        user_embedding = F.dropout2d(self.user_embedding(user_index.to(device)).unsqueeze(dim=0),
                                     p=self.masking_probability,
                                     training=self.training)

        if self.long_short_term_method == 'ini':
            # packed_clicked_news_rep = pack_padded_sequence(news_rep.unsqueeze(0), lengths=clicked_news_length, batch_first=True, enforce_sorted=False)
            _, last_hidden = self.gru(news_rep.unsqueeze(0), user_embedding.unsqueeze(0))
            return last_hidden.squeeze(dim=0)
        else:
            # packed_clicked_news_rep = pack_padded_sequence(news_rep,lengths =clicked_news_length, batch_first=True, enforce_sorted=False)
            _, last_hidden = self.gru(news_rep)
            return torch.cat((last_hidden.squeeze(dim=0), user_embedding), dim=1)

class LSTUR(torch.nn.Module):
    def __init__(self, args, entity_embedding, relation_embedding, news_entity_dict, entity_adj, relation_adj,
                 news_title_word_index, word_embedding, news_category_index, news_subcategory_index, device):
        super(LSTUR, self).__init__()
        self.args = args
        self.device = device

        # no_embedding
        self.word_embedding = word_embedding
        # self.entity_embedding = entity_embedding
        # self.relation_embedding = relation_embedding

        # LSTUR
        self.news_encoder = news_encoder(self.args.word_embedding_dim, self.args.title_word_size,
                                        self.args.category_embedding_dim, self.args.subcategory_embedding_dim,
                                        self.args.drop_prob, self.args.query_vector_dim,
                                        self.args.cnn_num_filters, self.args.cnn_window_sizes,
                                        self.args.category_num, self.args.subcategory_num)

        self.user_encoder = user_encoder(self.args.word_embedding_dim, self.args.title_word_size,
                                         self.args.category_embedding_dim, self.args.subcategory_embedding_dim,
                                         self.args.drop_prob, self.args.query_vector_dim,
                                         self.args.cnn_num_filters, self.args.cnn_window_sizes,
                                         self.args.category_num, self.args.subcategory_num,
                                         self.args.user_num,  self.args.long_short_term_method, self.args.masking_probability)

        # dict
        self.news_title_word_dict = news_title_word_index
        self.news_category_dict = news_category_index
        self.news_subcategory_dict = news_subcategory_index
        self.entity_adj = entity_adj
        self.relation_adj = relation_adj
        self.news_entity_dict = news_entity_dict

    def get_user_news_rep(self, user_index, candidate_news_index, user_clicked_news_index):

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
            # 用户嵌入
            user_index_one = user_index[i]
            # 用户表征
            user_rep_one = self.user_encoder(user_index_one, clicked_news_word_embedding_one,
                                             clicked_news_category_index,
                                             clicked_news_subcategory_index).unsqueeze(1)

            if i == 0:
                user_rep = user_rep_one
            else:
                user_rep = torch.cat([user_rep, user_rep_one], dim=0)
        return user_rep, news_rep

    def forward(self, user_index, candidate_news, user_clicked_news_index):
        # 新闻用户表征
        user_rep, news_rep = self.get_user_news_rep(user_index, candidate_news, user_clicked_news_index)
        # 预测得分
        score = torch.sum(news_rep * user_rep, dim=-1).view(self.args.batch_size, -1)
        return score

    def test(self, user_index, candidate_news, user_clicked_news_index):
        # 新闻用户表征
        user_rep, news_rep = self.get_user_news_rep(user_index, candidate_news, user_clicked_news_index)
        # 预测得分
        score = torch.sum(news_rep * user_rep, dim=-1).view(self.args.batch_size, -1)
        return score
