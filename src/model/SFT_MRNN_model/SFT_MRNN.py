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
        
        self.norm = nn.LayerNorm(self.args.attention_dim * self.args.attention_heads)
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
        newsatt_rep = torch.tanh(self.norm(self.attention(newsatt_rep)))
        newsatt_rep = F.dropout(newsatt_rep, p=self.dropout_prob, training=self.training)
        return newsatt_rep

class user_att_encoder(torch.nn.Module):
    def __init__(self, args):
        super(user_att_encoder, self).__init__()
        self.args = args
        self.multi_dim  = self.args.attention_dim * self.args.attention_heads
        self.category_embedding = nn.Embedding(self.args.category_num,  self.args.category_embedding_dim)
        self.user_embedding = nn.Embedding(self.args.user_num, self.args.embedding_dim)
        self.fc1 = nn.Linear(self.args.embedding_dim, self.multi_dim)
        self.norm = nn.LayerNorm(self.multi_dim)
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
        user_att_rep = torch.tanh(self.norm(self.attention(user_interest_embedding.unsqueeze(0))))
        user_att_rep = F.dropout(user_att_rep, p=self.dropout_prob, training=self.training)
        return user_att_rep

class news_encoder(torch.nn.Module):
    def __init__(self, word_dim, attention_dim, attention_heads, query_vector_dim, entity_size, entity_embedding_dim,
                 category_dim, subcategory_dim, category_size, subcategory_size):
        super(news_encoder, self).__init__()
        self.multi_dim = attention_dim * attention_heads
        self.norm = nn.LayerNorm(self.multi_dim)

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
        self.norm = nn.LayerNorm(self.multi_dim)
        self.dropout_prob = 0.2

    def forward(self, word_embedding, entity_embedding, category_index, subcategory_index):
        # 主题级表征
        category_embedding = self.embedding_layer1(category_index.to(torch.int64))
        category_rep = torch.tanh(self.fc1(category_embedding))
        category_rep = F.dropout(category_rep, p=self.dropout_prob, training=self.training)

        # 副主题级表征
        subcategory_embedding = self.embedding_layer2(subcategory_index.to(torch.int64))
        subcategory_rep = torch.tanh(self.fc2(subcategory_embedding))
        subcategory_rep = F.dropout(subcategory_rep, p=self.dropout_prob, training=self.training)

        # 单词级新闻表征
        word_embedding = self.multiheadatt(word_embedding)
        # word_embedding = self.norm(word_embedding)
        word_embedding = F.dropout(word_embedding, p=self.dropout_prob, training=self.training)
        word_rep = torch.tanh(self.word_attention(word_embedding))
        word_rep = F.dropout(word_rep, p=self.dropout_prob, training=self.training)

        # 实体级新闻表征
        entity_embedding = torch.tanh(self.fc3(entity_embedding))
        entity_inter = self.GCN(entity_embedding)
        # entity_inter = self.norm(entity_inter)
        entity_inter = F.dropout(entity_inter, p=self.dropout_prob, training=self.training)
        entity_rep = torch.tanh(self.entity_attention(entity_inter))
        entity_rep = F.dropout(entity_rep, p=self.dropout_prob, training=self.training)

        # 新闻附加注意力
        news_rep = torch.cat([word_rep.unsqueeze(1), entity_rep.unsqueeze(1),
                              category_rep.unsqueeze(1), subcategory_rep.unsqueeze(1)], dim=1)
        news_rep = self.news_attention(news_rep)
        news_rep = torch.tanh(news_rep)
        news_rep = F.dropout(news_rep, p=self.dropout_prob, training=self.training)
        return news_rep, category_rep, subcategory_rep, word_rep, entity_rep

class user_encoder(torch.nn.Module):
    def __init__(self, word_dim, attention_dim, attention_heads, query_vector_dim, 
                 entity_size, entity_embedding_dim, category_dim, subcategory_dim, 
                 category_size, subcategory_size):
        super(user_encoder, self).__init__()
        self.multi_dim = attention_dim * attention_heads
        self.user_attention = Additive_Attention(query_vector_dim, self.multi_dim)
        self.multiheadatt = MultiHeadSelfAttention(self.multi_dim, self.multi_dim, attention_heads)
        self.dropout_prob = 0.2

    def forward(self, clicked_news_rep):
        user_seq_rep = self.multiheadatt(clicked_news_rep)
        user_seq_rep = F.dropout(user_seq_rep, p=self.dropout_prob, training=self.training)
        user_rep = torch.tanh(self.user_attention(user_seq_rep))
        user_rep = F.dropout(user_rep, p=self.dropout_prob, training=self.training)
        return user_rep


class SFT_MRNN(torch.nn.Module):
    def __init__(self, args, entity_embedding, relation_embedding, news_entity_dict, entity_adj, relation_adj,
                 news_title_word_index, word_embedding, news_category_index, news_subcategory_index, device):
        super(SFT_MRNN, self).__init__()
        self.args = args
        self.device = device

        # no_embedding
        self.word_embedding = word_embedding 
        self.entity_embedding = entity_embedding
        self.relation_embedding = relation_embedding

        # encoder
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

    def get_news_entities(self, newsids):
        news_entities = []
        newsids = newsids.unsqueeze(-1)
        for i in range(newsids.shape[0]):
            news_entities.append([])
            for j in range(newsids.shape[1]):
                # news_entities[-1].append([])
                news_entities[-1].append(self.news_entity_dict[int(newsids[i, j])][:self.args.news_entity_size])
        return np.array(news_entities)
    
    def get_neighor_entities(self, entity, k=5):
        neighor_entity = []
        neighor_relation = []
        if len(entity.shape) == 2:
            for i in range(entity.shape[0]):
                neighor_entity.append([])
                neighor_relation.append([])
                for j in range(entity.shape[1]):
                    if entity[i, j] in self.entity_adj.keys():
                        neighor_entity[-1].append([])
                        neighor_entity[-1][-1].append(self.entity_adj[int(entity[i, j])][:k])
                        neighor_relation[-1].append([])
                        neighor_relation[-1][-1].append(self.relation_adj[int(entity[i, j])][:k])
                    else:
                        neighor_entity[-1].append([])
                        neighor_entity[-1][-1].append([0] * k)
                        neighor_relation[-1].append([])
                        neighor_relation[-1][-1].append([0] * k)
        elif len(entity.shape) == 3:
            for i in range(entity.shape[0]):
                neighor_entity.append([])
                neighor_relation.append([])
                for j in range(entity.shape[1]):
                    neighor_entity[-1].append([])
                    neighor_relation[-1].append([])
                    for m in range(entity.shape[2]):
                        if entity[i, j, m] in self.entity_adj.keys():
                            neighor_entity[-1][-1].append([])
                            neighor_entity[-1][-1][-1].append(self.entity_adj[int(entity[i, j, m])][:k])
                            neighor_relation[-1][-1].append([])
                            neighor_relation[-1][-1][-1].append(self.relation_adj[int(entity[i, j, m])][:k])
                        else:
                            neighor_entity[-1][-1].append([])
                            neighor_entity[-1][-1][-1].append([0] * k)
                            neighor_relation[-1][-1].append([])
                            neighor_relation[-1][-1][-1].append([0] * k)
        return np.array(neighor_entity), np.array(neighor_relation)

    def get_user_news_rep(self, candidate_news_index, user_clicked_news_index):
        # 新闻单词
        candidate_news_word_embedding = self.word_embedding[self.news_title_word_dict[candidate_news_index]].to(self.device)
        user_clicked_news_word_embedding = self.word_embedding[self.news_title_word_dict[user_clicked_news_index]].to(self.device)
        # 新闻实体
        candidate_news_entity_embedding = self.entity_embedding[self.get_news_entities(candidate_news_index)].to(self.device).squeeze()
        user_clicked_news_entity_embedding = self.entity_embedding[self.get_news_entities(user_clicked_news_index)].to(self.device).squeeze()
        # 新闻主题
        candidate_news_category_index = torch.LongTensor(self.news_category_dict[np.array(candidate_news_index)]).to(self.device)
        user_clicked_news_category_index = torch.LongTensor(self.news_category_dict[np.array(user_clicked_news_index.cpu())]).to(self.device)
        # 新闻副主题
        candidate_news_subcategory_index = torch.LongTensor(self.news_subcategory_dict[np.array(candidate_news_index.cpu())]).to(self.device)
        user_clicked_news_subcategory_index = torch.LongTensor(self.news_subcategory_dict[np.array(user_clicked_news_index.cpu())]).to(self.device)

        # 新闻编码器
        news_rep = None
        news_word_rep = None
        news_entity_rep = None
        news_category_rep = None
        news_subcategory_rep = None
        
        for i in range(self.args.sample_size):
            news_word_embedding_one = candidate_news_word_embedding[:, i, :, :]
            news_entity_embedding_one = candidate_news_entity_embedding[:, i, :, :]
            news_category_index = candidate_news_category_index[:, i]
            news_subcategory_index = candidate_news_subcategory_index[:, i]
            news_rep_one, category_rep_one, \
            subcategory_rep_one, word_rep_one, entity_rep_one = self.news_encoder(news_word_embedding_one, news_entity_embedding_one,
                                                                                  news_category_index, news_subcategory_index)
            if i == 0:
                news_rep = news_rep_one.unsqueeze(1)
                news_word_rep = word_rep_one.unsqueeze(1)
                news_entity_rep = entity_rep_one.unsqueeze(1)
                news_category_rep = category_rep_one.unsqueeze(1)
                news_subcategory_rep = subcategory_rep_one.unsqueeze(1)
            else:
                news_rep = torch.cat([news_rep, news_rep_one.unsqueeze(1)], dim=1)
                news_word_rep = torch.cat([news_word_rep, word_rep_one.unsqueeze(1)], dim=1)
                news_entity_rep = torch.cat([news_entity_rep, entity_rep_one.unsqueeze(1)], dim=1)
                news_category_rep = torch.cat([news_category_rep, category_rep_one.unsqueeze(1)], dim=1)
                news_subcategory_rep = torch.cat([news_subcategory_rep, subcategory_rep_one.unsqueeze(1)], dim=1)

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
            clicked_news_rep, _, _, _, _ = self.news_encoder(clicked_news_word_embedding_one, clicked_news_entity_embedding_one, 
                                                 clicked_news_category_index, clicked_news_subcategory_index)
            clicked_news_rep = clicked_news_rep.unsqueeze(0)
            # 用户表征
            user_rep_one = self.user_encoder(clicked_news_rep).unsqueeze(0)
            if i == 0:
                user_rep = user_rep_one
            else:
                user_rep = torch.cat([user_rep, user_rep_one], dim=0)
        return user_rep, news_rep, [news_word_rep, news_entity_rep, news_category_rep, news_subcategory_rep]

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
        user_rep, news_rep, news_feature_list = self.get_user_news_rep(candidate_newsindex, user_clicked_newsindex)

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
        
        rec_score = torch.sum(news_rep * user_rep.view(self.args.batch_size, 1, -1), dim=-1).view(self.args.batch_size, -1)
        rec_att_score = torch.sum(news_rep_update * user_rep_update.view(self.args.batch_size, 1, -1), dim=-1).view(self.args.batch_size, -1)

        #news = torch.where(news_type_index.to(self.device).unsqueeze(-1) == 0, news_rep_update, news_rep)
        #user = torch.where(user_type_index.to(self.device).unsqueeze(1) == 0, user_rep_update.squeeze(), user_rep.squeeze())
        # rec_score = torch.sum(news * user.view(self.args.batch_size, 1, -1), dim=-1).view(self.args.batch_size, -1)        

        return rec_score, rec_att_score, loss_zeroshot_news, loss_zeroshot_user, Lp, Ln

    def test(self, candidate_newsindex, user_index, user_clicked_newsindex, user_type_index, news_type_index ):
        # 新闻用户表征
        user_rep, news_rep, news_feature_list = self.get_user_news_rep(candidate_newsindex, user_clicked_newsindex)

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
        # news = torch.where(news_type_index.to(self.device).unsqueeze(-1) == 0, news_rep_update, news_rep)
        # user = torch.where(user_type_index.to(self.device).unsqueeze(1) == 0, user_rep_update.squeeze(), user_rep.squeeze())
        # score = torch.sum(news * user.view(self.args.batch_size, 1, -1), dim=-1).view(self.args.batch_size, -1)        
        score = torch.sum(news_rep_update * user_rep_update.view(self.args.batch_size, 1, -1), dim=-1).view(self.args.batch_size, -1)        

        score = torch.sigmoid(score)
        return score
