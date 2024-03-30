import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import *

class news_att_encoder(torch.nn.Module):
    def __init__(self, args):
        super(news_att_encoder, self).__init__()
        self.args = args
        self.embedding_layer1 = nn.Embedding(self.args.category_num, embedding_dim=self.args.category_embedding_dim)
        self.embedding_layer2 = nn.Embedding(self.args.subcategory_num, embedding_dim=self.args.subcategory_embedding_dim)
        self.news_embedding = nn.Embedding(self.args.news_num, self.args.embedding_dim)

        self.fc1 = nn.Linear(self.args.category_embedding_dim, self.args.attention_dim * self.args.attention_heads, bias=True)
        self.fc2 = nn.Linear(self.args.word_embedding_dim, self.args.attention_dim * self.args.attention_heads)

        self.attention = Additive_Attention(self.args.query_vector_dim, self.args.attention_dim * self.args.attention_heads)
        self.dropout_prob = 0.2

    def forward(self, candidate_newsindex, category_index, word_embedding):
        # 主题表征
        category_embedding = self.embedding_layer1(category_index.to(torch.int64))
        category_rep = torch.tanh(self.fc1(category_embedding))
        category_rep = F.dropout(category_rep, p=self.dropout_prob, training=self.training)

        # 单词表征
        word_rep = torch.tanh(self.fc2(torch.mean(word_embedding, dim=-2)))
        word_rep = F.dropout(word_rep, p=self.dropout_prob, training=self.training)

        # 副主题表征
        # subcategory_embedding = self.embedding_layer2(subcategory_index.to(torch.int64))
        # subcategory_rep = self.fc2(subcategory_embedding)
        # subcategory_rep = F.dropout(subcategory_rep, p=self.dropout_prob, training=self.training)

        # id表征
        # news_embedding = self.news_embedding(candidate_newsindex)
        # newsId_rep = self.fc3(news_embedding)
        # newsId_rep = F.dropout(newsId_rep, p=self.dropout_prob, training=self.training)

        # 附加注意力
        newsatt_rep = torch.cat([category_rep.unsqueeze(1), word_rep.unsqueeze(1)], dim=1)
        newsatt_rep = torch.tanh(self.attention(newsatt_rep))
        newsatt_rep = F.dropout(newsatt_rep, p=self.dropout_prob, training=self.training)
        return newsatt_rep

class user_att_encoder(torch.nn.Module):
    def __init__(self, args):
        super(user_att_encoder, self).__init__()
        self.args = args
        self.category_dim = self.args.category_embedding_dim
        self.category_num = self.args.category_num
        self.multi_dim  = self.args.attention_dim * self.args.attention_heads

        self.category_embedding = nn.Embedding(self.category_num, self.category_dim)
        self.user_embedding = nn.Embedding(self.args.user_num, self.args.embedding_dim)

        self.fc1 = nn.Linear(self.args.embedding_dim, self.multi_dim)
        self.attention = Additive_Attention(self.args.query_vector_dim, self.multi_dim)

        self.dropout_prob = 0.2
    
    def forward(self, user_interest_index):
        # # id表征
        # user_embedding = self.user_embedding(user_index).unsqueeze(0).unsqueeze(0)
        # userId_rep = torch.tanh(self.fc1(user_embedding))
        # userId_rep = F.dropout(userId_rep, p=self.dropout_prob, training=self.training)

        # 点击新闻兴趣
        user_interest_embedding = self.category_embedding(user_interest_index)
        user_interest_embedding = torch.tanh(self.fc1(user_interest_embedding))
        user_interest_embedding = F.dropout(user_interest_embedding, p=self.dropout_prob, training=self.training)

        # 融合
        user_att_rep = torch.tanh(self.attention(user_interest_embedding.unsqueeze(0)))
        user_att_rep = F.dropout(user_att_rep, p=self.dropout_prob, training=self.training)
        return user_att_rep

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
        user_embedding = self.user_embedding(user_index)
        user_vector = torch.relu(self.fc1(user_embedding))
        user_vector_2 = torch.relu(self.fc2(user_embedding))
        return user_vector, user_vector_2


class SFT_NPA(torch.nn.Module):
    def __init__(self, args, entity_embedding, relation_embedding, news_entity_dict, entity_adj, relation_adj,
                 news_title_word_index, word_embedding, news_category_index, news_subcategory_index, device):
        super(SFT_NPA, self).__init__()
        self.args = args
        self.device = device

        # no_embedding
        self.word_embedding = word_embedding 
        self.entity_embedding = entity_embedding
        self.relation_embedding = relation_embedding

        # 用户ID嵌入编码器
        self.user_embedding_encoder = user_embedding_encoder(args)

        # NPA
        self.news_encoder = news_encoder(self.args.word_embedding_dim, self.args.cnn_num_filters,
                                         self.args.cnn_window_sizes, self.args.query_vector_dim)
        self.user_encoder = user_encoder(self.args.word_embedding_dim, self.args.cnn_num_filters,
                                         self.args.cnn_window_sizes, self.args.query_vector_dim)

        # 属性编码器
        self.news_att_encoder = news_att_encoder(args)
        self.user_att_encoder = user_att_encoder(args)

        # 用户和新闻ID嵌入
        # self.newsId_encoder = newsId_encoder(args)
        # self.userId_encoder = userId_encoder(args)

        # zeroshot学习
        self.zeroshot_news_tower = zeroshot_news_contrastive_tower(args)
        self.zeroshot_user_tower = zeroshot_user_contrastive_tower(args)

        # predict 
        # self.predict_layer = predict_id_layer(args)

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
        return user_rep, news_rep, None

    def get_user_news_att_rep(self, candidate_news_index, user_index, user_clicked_news_index):
        # 单词嵌入
        candidate_news_word_embedding = self.word_embedding[self.news_title_word_dict[candidate_news_index]].to(self.device)

        # 新闻主题
        candidate_news_category_index = torch.LongTensor(self.news_category_dict[np.array(candidate_news_index)]).to(self.device)
        user_clicked_news_category_index = torch.LongTensor(self.news_category_dict[np.array(user_clicked_news_index.cpu())]).to(self.device)

        # 新闻编码器
        news_rep = None
        for i in range(self.args.sample_size):
            news_index = candidate_news_index[:, i].to(self.device)
            news_category_index = candidate_news_category_index[:, i]
            news_word_embedding_one = candidate_news_word_embedding[:, i, :, :]

            news_rep_one = self.news_att_encoder(news_index, news_category_index, news_word_embedding_one)
            if i == 0:
                news_rep = news_rep_one.unsqueeze(1)
            else:
                news_rep = torch.cat([news_rep, news_rep_one.unsqueeze(1)], dim=1)
                
        # 用户编码器
        user_rep = None
        for i in range(self.args.batch_size):
            # user_index_one = user_index[i].to(self.device)
            # 点击新闻主题index
            clicked_news_category_index = user_clicked_news_category_index[i, :50]
            # 用户表征
            user_rep_one = self.user_att_encoder(clicked_news_category_index).unsqueeze(0)

            if i == 0:
                user_rep = user_rep_one
            else:
                user_rep = torch.cat([user_rep, user_rep_one], dim=0)
        return news_rep, user_rep

    def forward(self, candidate_newsindex, user_index, user_clicked_newsindex, user_type_index, news_type_index):
        # 新闻用户表征
        user_rep, news_rep, news_feature_list = self.get_user_news_rep(user_index,  candidate_newsindex, user_clicked_newsindex)

        # userId/ newsId
        # newsId_rep = self.newsId_encoder(candidate_newsindex.to(self.device))
        # userId_rep = self.userId_encoder(user_index.to(self.device))

        # zeroshot学习(Id)
        # loss_zeroshot_news, newsId_rep = self.zeroshot_news_tower(news_rep, newsId_rep, news_feature_list, news_type_index.to(self.device))
        # loss_zeroshot_user, userId_rep = self.zeroshot_user_tower(user_rep, userId_rep, user_type_index.to(self.device))

        # zeroshot学习(Att)
        news_att_rep, user_att_rep = self.get_user_news_att_rep(candidate_newsindex, user_index, user_clicked_newsindex)
        loss_zeroshot_user, user_rep_update, Lp, Ln = self.zeroshot_user_tower(user_rep, user_att_rep.squeeze(), user_type_index.to(self.device))
        loss_zeroshot_news, news_rep_update = self.zeroshot_news_tower(news_rep, news_att_rep, news_feature_list, news_type_index.to(self.device))

        # 暂时不优化冷新闻
        # loss_zeroshot_news = torch.tensor(0)
        # loss_zeroshot_user = torch.tensor(0)
        # La = torch.tensor(0)
        # Lc = torch.tensor(0)
        # Ld = torch.tensor(0)

        #################### 预测得分 ####################
        # DNN
        # score = self.predict_layer(news_rep, user_rep.repeat(1, news_rep.shape[1], 1),
        #                            newsId_rep, userId_rep.unsqueeze(1).repeat(1, newsId_rep.shape[1], 1))

        # DOT
        # score_semantic = torch.sum(news_rep * user_rep, dim=-1).view(self.args.batch_size, -1)
        # score_id = torch.sum(newsId_rep * userId_rep.unsqueeze(1), dim=-1).view(self.args.batch_size, -1)
        # score = score_id + score_semantic

        # DNN + DOT
        # score_id = self.predict_layer(newsId_rep, userId_rep.unsqueeze(1).repeat(1, newsId_rep.shape[1], 1))
        # score_semantic = torch.sum(news_rep * user_rep, dim=-1).view(self.args.batch_size, -1)
        # score = score_id + score_semantic

        # replace
        # user = torch.where(user_type_index.to(self.device).unsqueeze(1) == 0, user_att_rep.squeeze(),
        #                    user_rep.squeeze()).view(self.args.batch_size, 1, -1)
        # news = torch.where(news_type_index.to(self.device).unsqueeze(-1) == 0, newsId_rep, news_rep)
        # score = self.predict_layer(user.repeat(1, news.shape[1], 1), news)
        
        # rec_score = torch.sum(news_rep * user_rep.view(self.args.batch_size, 1, -1), dim=-1).view(self.args.batch_size, -1)
        rec_att_score = torch.sum(news_rep_update * user_rep_update.view(self.args.batch_size, 1, -1), dim=-1).view(self.args.batch_size, -1)

        news = torch.where(news_type_index.to(self.device).unsqueeze(-1) == 0, news_rep_update, news_rep)
        user = torch.where(user_type_index.to(self.device).unsqueeze(1) == 0, user_rep_update.squeeze(), user_rep.squeeze())
        rec_score = torch.sum(news * user.view(self.args.batch_size, 1, -1), dim=-1).view(self.args.batch_size, -1)        

        return rec_score, rec_att_score, loss_zeroshot_news, loss_zeroshot_user, Lp, Ln

    def test(self, candidate_newsindex, user_index, user_clicked_newsindex, user_type_index, news_type_index ):
        # 新闻用户表征
        user_rep, news_rep, news_feature_list = self.get_user_news_rep(user_index,  candidate_newsindex, user_clicked_newsindex)

        # newsId_rep = self.newsId_encoder(candidate_newsindex.to(self.device))
        # userId_rep = self.userId_encoder(user_index.to(self.device))

        # zeroshot学习(Id)
        # _, newsId_rep = self.zeroshot_news_tower(news_rep, newsId_rep, news_feature_list, news_type_index.to(self.device))
        # _, userId_rep = self.zeroshot_user_tower(user_rep, userId_rep, user_type_index.to(self.device))

        # zeroshot学习(Att)
        news_att_rep, user_att_rep = self.get_user_news_att_rep(candidate_newsindex, user_index, user_clicked_newsindex)
        _, user_rep_update, _, _ = self.zeroshot_user_tower(user_rep, user_att_rep.squeeze(), user_type_index.to(self.device))
        _, news_rep_update = self.zeroshot_news_tower(news_rep, news_att_rep, news_feature_list, news_type_index.to(self.device))

        # 预测得分
        # DNN
        # score = self.predict_layer(news_rep, user_rep.repeat(1, news_rep.shape[1], 1),
        #                            newsId_rep, userId_rep.unsqueeze(1).repeat(1, newsId_rep.shape[1], 1))

        # DOT
        # score_semantic = torch.sum(news_rep * user_rep, dim=-1).view(self.args.batch_size, -1)
        # score_id = torch.sum(newsId_rep * userId_rep.unsqueeze(1), dim=-1).view(self.args.batch_size, -1)
        # score = score_id + score_semantic

        # DNN + DOT
        # score_id = self.predict_layer(newsId_rep, userId_rep.unsqueeze(1).repeat(1, newsId_rep.shape[1], 1))
        # score_semantic = torch.sum(news_rep * user_rep, dim=-1).view(self.args.batch_size, -1)
        # score = score_id + score_semantic

        # replace
        # user = torch.where(user_type_index.to(self.device).unsqueeze(1) == 0, user_att_rep.squeeze(), user_rep.squeeze()).view(self.args.batch_size, 1, -1)
        # news = torch.where(news_type_index.to(self.device).unsqueeze(-1) == 0, newsId_rep, news_rep)
        # score = self.predict_layer(user.repeat(1, news.shape[1], 1), news)
        news = torch.where(news_type_index.to(self.device).unsqueeze(-1) == 0, news_rep_update, news_rep)
        user = torch.where(user_type_index.to(self.device).unsqueeze(1) == 0, user_rep_update.squeeze(), user_rep.squeeze())
        # score = torch.sum(news_rep_update * user_rep_update.view(self.args.batch_size, 1, -1), dim=-1).view(self.args.batch_size, -1)        
        score = torch.sum(news * user.view(self.args.batch_size, 1, -1), dim=-1).view(self.args.batch_size, -1)        

        score = torch.sigmoid(score)
        return score
