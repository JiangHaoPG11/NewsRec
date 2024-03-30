import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import *

class Knowledge_co_encoder(torch.nn.Module):
    def __init__(self, neigh_num, entity_dim):
        super(Knowledge_co_encoder, self).__init__()
        self.neigh_num = neigh_num
        self.entity_dim = entity_dim
        # gat
        self.gat = GAT(self.entity_dim, self.entity_dim)
        self.gat_fc = nn.Linear(2 * self.entity_dim, self.entity_dim, bias = True)
        # GCAT
        self.gcat = GraphCoAttNet(self.entity_dim, self.entity_dim)
        self.gcat_fc = nn.Linear(2 * self.entity_dim, self.entity_dim, bias = True)
        # co-attention
        self.co_att = GraphCoAttNet(self.entity_dim, self.entity_dim)
        
    def forward(self, clicked_entity, clicked_onehop, news_entity, news_onehop):
        # 模型中第一部分：gat
        news_can = self.gat(news_onehop) # b, 50, max_num, 100
        news_can = torch.cat([news_can,news_entity], dim = -1)
        news_can = self.gat_fc(news_can).unsqueeze(-2).repeat(1, 1, 1,self.neigh_num, 1)
        user_can = self.gat(clicked_onehop) # b, 50, max_num, 100
        user_can = torch.cat([user_can, clicked_entity], dim=-1)
        user_can = self.gat_fc(user_can).unsqueeze(-2).repeat(1, 1, 1,self.neigh_num, 1)
        # 模型中第二部分：GCAT
        news_entity_onehop = self.gcat(news_onehop, user_can)  # b, 50, max_num, 100
        user_entity_onehop = self.gcat(clicked_onehop, news_can)  # b, 50, max_num, 100
        news_entity_vecs = self.gcat_fc(torch.cat([news_entity, news_entity_onehop], dim = -1)) # b, 50, max_num, 100
        user_entity_vecs =self.gcat_fc(torch.cat([clicked_entity, user_entity_onehop], dim=-1)) # b, 50, max_num, 100
        # 模型中第三部分： entity co-attention network
        news_entity_vecs = self.co_att(news_entity_vecs, clicked_entity)
        user_entity_vec = self.co_att(user_entity_vecs, news_entity)
        return news_entity_vecs, user_entity_vec

class Semantic_co_Encoder(torch.nn.Module):
    def __init__(self, word_dim, word_num, attention_dim, attention_heads):
        super(Semantic_co_Encoder, self).__init__()
        self.word_num = word_num
        self.attention_dim = attention_dim * attention_heads
        self.multiheadatt_news = MultiHeadSelfAttention_2(word_dim, self.attention_dim, attention_heads)
        self.multiheadatt_clicked = MultiHeadSelfAttention_3(word_dim, self.attention_dim, attention_heads)
        # self.news_att = Additive_Attention(query_vector_dim, self.attention_dim)
        # self.clicked_att = Additive_Attention(query_vector_dim, self.attention_dim)
        self.news_att_1 = nn.Linear(self.attention_dim, 200, bias = True)
        self.news_att_2 = nn.Linear(200, 1, bias = True)
        self.clicked_att_1 = nn.Linear(self.attention_dim, 200, bias=True)
        self.clicked_att_2 = nn.Linear(200, 1, bias=True)
        self.get_agg = get_agg(self.word_num)
        self.get_context_aggergator = get_context_aggergator(self.attention_dim)

    def forward(self, news_title, clicked_title):
        batch_size = news_title.size(0)
        # 新闻编码器
        news_title = self.multiheadatt_news(news_title) #b, max_word, 300
        clicked_title = self.multiheadatt_clicked(clicked_title)  # b, 50, max_word, 400

        # 计算候选新闻自身注意力
        news_title_att_vecs = torch.tanh(self.news_att_1(news_title)) #b, max_word, 200
        news_title_att0 = self.news_att_2(news_title_att_vecs) #b, max_word, 1
        news_title_att = news_title_att0.squeeze().unsqueeze(1).repeat(1,50,1) #b, 50, max_word

        # 计算点击新闻自身注意力
        clicked_title_att_vecs = torch.tanh(self.clicked_att_1(clicked_title)) #b, 50, max_word, 200
        clicked_title_att = self.clicked_att_2(clicked_title_att_vecs).squeeze()   #b, 50, max_word

        # 计算候选交叉注意力
        clicked_title_att_vecs = torch.flatten(clicked_title_att_vecs, 1, 2)  #b, 50*max_word, 200
        news_title_att_vecs = torch.transpose(news_title_att_vecs, 2, 1)  #b, 200, max_word
        cross_att = torch.matmul(clicked_title_att_vecs, news_title_att_vecs)  #b, 50*max_word, max_word
        cross_att_candi = F.softmax(cross_att, dim=-1)  #b, 50*max_word, max_word
        cross_att_candi = 0.001 * torch.reshape(torch.matmul(cross_att_candi, news_title_att0), (-1, 50, self.word_num))#b, 50, max_word

        # 计算点击注意力（自身注意力加交叉注意力）
        clicked_title_att = torch.add(clicked_title_att, cross_att_candi)
        clicked_title_att = F.softmax(clicked_title_att, dim=-1) #b, 50, max_word

        # 计算点击交叉注意力
        cross_att_click = torch.transpose(torch.reshape(cross_att, (batch_size, 50, self.word_num, self.word_num )), -1, -2) #(bz,#click,#candi_word,#click_word,)
        clicked_title_att_re = clicked_title_att.unsqueeze(2) #(bz,#click,1,#click_word,)
        cross_att_click_vecs = torch.cat([cross_att_click, clicked_title_att_re] , dim = -2) #(bz,#click,#candi_word+1,#click_word,)
        cross_att_click = self.get_agg(cross_att_click_vecs)

        # 计算候选注意力（自身注意力加交叉注意力）
        news_title_att = torch.add(news_title_att, cross_att_click)
        news_title_att = F.softmax(news_title_att, dim = -1) #b, 50, max_word
        news_title_vecs = torch.matmul(news_title_att, news_title) #b, 50, 400

        clicked_title_att = clicked_title_att.unsqueeze(-1)
        clicked_title_word_vecs_att = torch.cat([clicked_title,clicked_title_att], dim =-1)
        clicked_title_vecs = self.get_context_aggergator(clicked_title_word_vecs_att) #b, 50, 400

        return news_title_vecs, clicked_title_vecs

class News_User_co_Encoder(torch.nn.Module):
    def __init__(self, input_dim):
        super(News_User_co_Encoder, self).__init__()
        self.news_att_1 = nn.Linear(input_dim, 100, bias = True)
        self.news_att_2 = nn.Linear(100, 1, bias = True)
        self.user_att_1 = nn.Linear(input_dim, 100, bias = True)
        self.user_att_2 = nn.Linear(100, 1, bias = True)
        self.co_att = nn.Linear(input_dim, 100, bias = True)
        self.norm1 = nn.LayerNorm(400)
        self.norm2 = nn.LayerNorm(400)
    def forward(self, news_vecs, user_vecs):
        # 计算自身注意力
        news_att = self.news_att_2(torch.tanh(self.news_att_1(news_vecs)))
        user_att = self.user_att_2(torch.tanh(self.user_att_1(user_vecs))) #(bz,50,1)
        # 计算交叉注意力
        cross_news_vecs = self.co_att(news_vecs) #(bz,50,100)
        cross_user_vecs = self.co_att(user_vecs) #(bz,50,100)
        cross_att = torch.matmul(cross_news_vecs, torch.transpose(cross_user_vecs, -1, -2))
        # 计算用户交叉注意力（自身注意力加交叉注意力）
        cross_user_att = F.softmax(cross_att, dim = -1) #(bz,50,50)
        cross_user_att = 0.01 * torch.matmul(cross_user_att, news_att) #(bz,50,1)
        user_att = F.softmax(torch.add(user_att,cross_user_att), dim = -1) #(bz,50,1)
        # 计算新闻交叉注意力（自身注意力加交叉注意力）
        cross_news_att = F.softmax(torch.transpose(cross_att,-1,-2) , dim = -1)  # (bz,50,50)
        cross_news_att = 0.01 * torch.matmul(cross_news_att, user_att)  # (bz,50,1)
        news_att = F.softmax(torch.add(news_att, cross_news_att), dim = -1)  # (bz,50,1)
        # 计算新闻向量和用户向量
        news_vec = torch.matmul(torch.transpose(news_att, -1, -2), news_vecs)
        news_vec = self.norm1(news_vec)
        user_vec = torch.matmul(torch.transpose(user_att, -1, -2), user_vecs)
        user_vec = self.norm2(user_vec)

        # 计算得分
        sorce = torch.sum(news_vec * user_vec, dim = 2) # (bz,50,1)
        return sorce

class KIM(torch.nn.Module):
    def __init__(self, args, entity_embedding, relation_embedding, news_entity_dict, entity_adj, relation_adj,
                 news_title_word_index, word_embedding, news_category_index, news_subcategory_index,
                 device):
        super(KIM, self).__init__()
        self.args = args
        self.device = device

        # no_embedding
        self.word_embedding = word_embedding
        self.entity_embedding = entity_embedding.to(device)
        self.relation_embedding = relation_embedding.to(device)

        # embedding
        self.category_embedding = nn.Embedding(self.args.category_num, self.args.embedding_dim).to(device)
        self.subcategory_embedding = nn.Embedding(self.args.subcategory_num, self.args.embedding_dim).to(device)

        self.entity_neigh_num = 5

        # KIM
        self.knowledge_co_encoder = Knowledge_co_encoder(self.entity_neigh_num, self.args.entity_embedding_dim)
        self.Semantic_co_Encoder = Semantic_co_Encoder(self.args.word_embedding_dim, self.args.title_word_size, self.args.attention_dim,
                                                       self.args.attention_heads)
        self.user_news_co_encoder = News_User_co_Encoder(self.args.attention_dim * self.args.attention_heads)

        self.MergeLayer = nn.Linear(self.args.attention_dim * self.args.attention_heads + self.args.entity_embedding_dim,
                                    self.args.attention_dim * self.args.attention_heads, bias=True)

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

    def get_score(self, candidate_news_index, user_clicked_news_index):
        # 新闻单词
        candidate_news_word_embedding = self.word_embedding[
            self.news_title_word_dict
            [
                candidate_news_index
            ]
        ].to(self.device)

        user_clicked_news_word_embedding = self.word_embedding[
            self.news_title_word_dict
            [
                user_clicked_news_index
            ]
        ].to(self.device)

        # 新闻实体
        news_entity = self.get_news_entities(
            candidate_news_index
        ).squeeze()
        user_news_entity = self.get_news_entities(
            user_clicked_news_index
        ).squeeze()

        candidate_news_entity_embedding = self.entity_embedding[news_entity].to(self.device).squeeze()
        user_clicked_news_entity_embedding = self.entity_embedding[user_news_entity].to(self.device).squeeze()
        
        # 采样邻居节点
        neighor_entity, neighor_relation = self.get_neighor_entities(
            news_entity, k=self.entity_neigh_num
        )
        user_neighor_entity, user_neighor_relation = self.get_neighor_entities(
            user_news_entity, k=self.entity_neigh_num
        )
        
        # 新闻邻居实体
        candidate_news_neigh_entity_embedding = self.entity_embedding[neighor_entity].to(self.device).squeeze()
        user_clicked_news_neigh_entity_embedding = self.entity_embedding[user_neighor_entity].to(self.device).squeeze()

        for i in range(self.args.sample_size):
            news_title = candidate_news_word_embedding[:,i,:,:]
            clicked_title = user_clicked_news_word_embedding
            news_title_vecs, clicked_title_vecs = self.Semantic_co_Encoder(
                news_title, clicked_title
            )
            news_entity = candidate_news_entity_embedding[:,i,:,:].unsqueeze(1).repeat(
                1, user_clicked_news_entity_embedding.shape[1], 1, 1
            )
            news_onehop = candidate_news_neigh_entity_embedding[:,i,:,:,:].unsqueeze(1).repeat(
                1, user_clicked_news_neigh_entity_embedding.shape[1], 1, 1, 1
            )

            clicked_onehop = user_clicked_news_neigh_entity_embedding
            clicked_entity = user_clicked_news_entity_embedding

            news_entity_vecs, user_entity_vecs = self.knowledge_co_encoder(
                clicked_entity, clicked_onehop, news_entity, news_onehop
            )
            news_vecs = torch.cat([news_title_vecs,news_entity_vecs], dim = -1)
            news_vecs = torch.tanh(self.MergeLayer(news_vecs))
            user_vecs = torch.cat([clicked_title_vecs, user_entity_vecs], dim=-1)
            user_vecs = torch.tanh(self.MergeLayer(user_vecs))
            score_one = self.user_news_co_encoder(news_vecs, user_vecs)

            if i == 0:
                score = score_one
            else:
                score = torch.cat([score, score_one], dim = -1)
                
        return score

    def forward(self, candidate_news, user_clicked_news_index):
        # candidate_news = torch.flatten(candidate_news, 0, 1)
        score = self.get_score(candidate_news, user_clicked_news_index)
        return score

    def test(self, candidate_news, user_clicked_news_index):
        # candidate_news = torch.flatten(candidate_news, 0, 1)
        score = self.get_score(candidate_news, user_clicked_news_index)
        return score
