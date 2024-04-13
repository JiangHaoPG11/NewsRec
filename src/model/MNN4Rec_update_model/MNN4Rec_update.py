import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from utils.utils import *

class news_encoder(torch.nn.Module):
    def __init__(self, news_num, embedding_dim, word_dim, attention_dim, attention_heads,
                 query_vector_dim, entity_size, entity_embedding_dim,
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
        word_embedding = F.dropout(word_embedding, p=self.dropout_prob, training=self.training)
        word_embedding = self.multiheadatt(word_embedding)
        word_embedding = self.norm(word_embedding)
        word_embedding = F.dropout(word_embedding, p=self.dropout_prob, training=self.training)
        word_rep = torch.tanh(self.word_attention(word_embedding))
        word_rep = F.dropout(word_rep, p=self.dropout_prob, training=self.training)

        # 实体级新闻表征
        entity_embedding = torch.tanh(self.fc3(entity_embedding))
        entity_inter = self.GCN(entity_embedding)
        entity_inter = self.norm(entity_inter)
        entity_inter = F.dropout(entity_inter, p=self.dropout_prob, training=self.training)
        entity_rep = torch.tanh(self.entity_attention(entity_inter))
        entity_rep = F.dropout(entity_rep, p=self.dropout_prob, training=self.training)

        # 新闻附加注意力
        news_rep = torch.cat([word_rep.unsqueeze(1), entity_rep.unsqueeze(1),
                              category_rep.unsqueeze(1), subcategory_rep.unsqueeze(1)], dim=1)
        news_rep = torch.tanh(self.news_attention(news_rep))
        news_rep = F.dropout(news_rep, p=self.dropout_prob, training=self.training)

        return news_rep

class user_encoder(torch.nn.Module):
    def __init__(self, news_num, embedding_dim, word_dim, attention_dim,
                 attention_heads, query_vector_dim, entity_size, entity_embedding_dim,
                 category_dim, subcategory_dim, category_size, subcategory_size):
        super(user_encoder, self).__init__()
        self.news_encoder = news_encoder(news_num, embedding_dim, word_dim, attention_dim,
                                         attention_heads, query_vector_dim, entity_size,
                                         entity_embedding_dim, category_dim, subcategory_dim,
                                         category_size, subcategory_size)

        self.multi_dim = attention_dim * attention_heads
        self.multiheadatt = MultiHeadSelfAttention(self.multi_dim, self.multi_dim, attention_heads)

        self.norm = nn.LayerNorm(self.multi_dim)

        self.user_attention = Additive_Attention(query_vector_dim, self.multi_dim)
        self.dropout_prob = 0.2

    # def forward(self, word_embedding, entity_embedding, category_index, subcategory_index):
    #     # 点击新闻表征
    #     news_rep = self.news_encoder(word_embedding, entity_embedding,
    #                                  category_index, subcategory_index).unsqueeze(0)
    #     news_rep = F.dropout(news_rep, p=self.dropout_prob, training=self.training)
    #     news_rep = self.multiheadatt(news_rep)
    #     news_rep = F.dropout(news_rep, p=self.dropout_prob, training=self.training)
    #     # 用户表征
    #     user_rep = torch.tanh(self.user_attention(news_rep))
    #     user_rep = F.dropout(user_rep, p=self.dropout_prob, training=self.training)
    #     return user_rep

    def forward(self, user_clicked_seq):
        # 点击新闻表征
        # news_rep = F.dropout(news_rep, p=self.dropout_prob, training=self.training)
        news_rep = self.multiheadatt(user_clicked_seq)
        news_rep = F.dropout(news_rep, p=self.dropout_prob, training=self.training)
        # 用户表征
        user_rep = torch.tanh(self.user_attention(news_rep))
        user_rep = F.dropout(user_rep, p=self.dropout_prob, training=self.training)
        return user_rep

class MNN4Rec_update(torch.nn.Module):
    def __init__(self, args, entity_embedding, relation_embedding, news_entity_dict, entity_adj, relation_adj,
                 news_title_word_index, word_embedding, news_category_index, news_subcategory_index, device):
        super(MNN4Rec_update, self).__init__()
        self.args = args
        self.device = device

        self.topk_implicit = self.args.topk_implicit
        self.topk_explicit = self.args.topk_explicit
        self.use_news_relation = self.args.use_news_relation

        # user news id embedding
        # self.user_ID_embedding = nn.Embedding(self.args.user_num, self.args.embedding_dim)
        # self.news_ID_embedding = nn.Embedding(self.args.news_num, self.args.embedding_dim)

        # no_embedding
        self.word_embedding = word_embedding
        self.entity_embedding = entity_embedding.to(device)
        self.relation_embedding = relation_embedding.to(device)

        # MNN4Rec_update
        self.news_encoder = news_encoder(self.args.news_num, self.args.embedding_dim,
                                         self.args.word_embedding_dim, self.args.attention_dim,
                                         self.args.attention_heads, self.args.query_vector_dim,
                                         self.args.news_entity_size, self.args.entity_embedding_dim,
                                         self.args.category_embedding_dim, self.args.subcategory_embedding_dim,
                                         self.args.category_num, self.args.subcategory_num)
        self.user_encoder = user_encoder(self.args.news_num, self.args.embedding_dim,
                                         self.args.word_embedding_dim, self.args.attention_dim,
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

        # reverse
        self.category_news_dict = self._construct_category_news_dict()
        self.subcategory_news_dict = self._construct_subcategory_news_dict()
        self.entity_news_dict = self._construct_entity_news_dict()

        # 邻居聚合
        self.KGAT = KGAT(self.args.entity_embedding_dim, self.args.entity_embedding_dim)
        self.relation_transfor_layer = nn.Linear(2 * self.args.entity_embedding_dim, self.args.embedding_dim, bias=True)
        self.relation_attention = Additive_Attention(self.args.query_vector_dim, self.args.embedding_dim)
        self.neighbor_agg = Additive_Attention(self.args.query_vector_dim,
                                               self.args.attention_heads * self.args.attention_dim)
        # self.neighbor_agg = SimilarityAttention()

    def _construct_category_news_dict(self):
        category_news_dict = {}
        for i in range(self.news_category_dict.shape[0]):
            if self.news_category_dict[i] not in category_news_dict.keys():
                category_news_dict[self.news_category_dict[i]] = []
            if i != self.args.news_num - 1:
                category_news_dict[self.news_category_dict[i]].append(i)
        return category_news_dict

    def _construct_subcategory_news_dict(self):
        subcategory_news_dict = {}
        for i in range(self.news_subcategory_dict.shape[0]):
            if self.news_subcategory_dict[i] not in subcategory_news_dict.keys():
                subcategory_news_dict[self.news_subcategory_dict[i]] = []
            if i != self.args.news_num - 1:
                subcategory_news_dict[self.news_subcategory_dict[i]].append(i)
        return subcategory_news_dict

    def _construct_entity_news_dict(self):
        entity_news_dict = {}
        for key, value in self.news_entity_dict.items():
            for entity_index in value:
                if entity_index not in entity_news_dict.keys():
                    entity_news_dict[entity_index] = []
                if key != self.args.news_num - 1:
                    entity_news_dict[entity_index].append(key)
        return entity_news_dict

    def get_news_entities(self, newsids):
        news_entities = []
        newsids = newsids.unsqueeze(-1)
        for i in range(newsids.shape[0]):
            news_entities.append([])
            for j in range(newsids.shape[1]):
                # news_entities[-1].append([])
                news_entities[-1].append(self.news_entity_dict[int(newsids[i, j])][:self.args.news_entity_size])
        return np.array(news_entities)
    
    def get_neighbor_entities(self, entity, k=5):
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

    def sample_news_implicit_neighbor(self, news_category_index, news_subcategory_index, topk):
        k_category = int(topk/2)
        k_subcategory = topk - k_category
        news_category_index_numpy = news_category_index.cpu().numpy()
        news_subcategory_index_numpy = news_subcategory_index.cpu().numpy()
        news_implicit_neighbor = []
        for i in range(news_category_index_numpy.shape[0]):
            if len(self.category_news_dict[news_category_index_numpy[i]]) == 0:
                select_implicit_neighbor1 = [self.args.news_num - 1] * k_category
            else:
                select_implicit_neighbor1 = np.random.choice(self.category_news_dict[news_category_index_numpy[i]], size=k_category)
            if len(self.subcategory_news_dict[news_subcategory_index_numpy[i]]) == 0:
                select_implicit_neighbor2 = [self.args.news_num - 1] * k_subcategory
            else:
                select_implicit_neighbor2 = np.random.choice(self.subcategory_news_dict[news_subcategory_index_numpy[i]], size=k_subcategory)
            select_implicit_neighbor = list(select_implicit_neighbor1) + list(select_implicit_neighbor2)
            news_implicit_neighbor.append(select_implicit_neighbor)
        return torch.LongTensor(news_implicit_neighbor).to(device)

    def sample_news_explicit_neighbor(self, news_entity_index, topk):
        def _cal_c_ex_vector(news_entity):
            # 计算新闻感知向量
            neighor_entity, neighor_relation = self.get_neighbor_entities((news_entity), k = 5)
            entity_embedding = self.entity_embedding[news_entity].to(self.device).squeeze()
            neigh_entity_embedding = self.entity_embedding[neighor_entity].to(self.device).squeeze()
            neigh_relation_embedding = self.relation_embedding[neighor_relation].to(self.device).squeeze()
            # 新闻感知向量
            c_ex = self.KGAT(entity_embedding, neigh_entity_embedding, neigh_relation_embedding)
            c_ex = torch.tanh(self.relation_transfor_layer(c_ex))
            return c_ex

        def _select_topk_sim_explicit_neighbor(c_ex_origin, c_ex_neighbor, explicit_neighbor, topk, device):
            weights = F.softmax(torch.matmul(c_ex_neighbor, torch.transpose(c_ex_origin, -1, -2)).squeeze(), dim=-1)
            weights = weights.squeeze(-1)
            m = Categorical(weights)
            acts_idx = torch.transpose(m.sample(sample_shape=torch.Size([topk])), -2, -1).to(device)
            explicit_neighbor_selected = explicit_neighbor.gather(1, acts_idx)
            return explicit_neighbor_selected

        news_explicit_neighbor = []
        for i in range(news_entity_index.shape[0]):
            explicit_neighbor = []
            news_entity_list = news_entity_index[i]
            for entity in news_entity_list:
                if entity != 0:
                    explicit_neighbor.extend(self.entity_news_dict[entity])
            if len(explicit_neighbor) < topk:
                select_explicit_neighbor = explicit_neighbor + [self.args.news_num - 1] * (topk * 2 - len(explicit_neighbor))
            else:
                select_explicit_neighbor = np.random.choice(explicit_neighbor, size=topk * 2)
            news_explicit_neighbor.append(list(select_explicit_neighbor))

        news_explicit_neighbor = np.array(news_explicit_neighbor)
        c_ex_neighbor = _cal_c_ex_vector(news_explicit_neighbor)
        c_ex_origin = _cal_c_ex_vector(news_entity_index)
        c_ex_origin = torch.tanh(self.relation_attention(c_ex_origin)).unsqueeze(1)
        news_explicit_neighbor = _select_topk_sim_explicit_neighbor(c_ex_origin, c_ex_neighbor,
                                                                    torch.LongTensor(news_explicit_neighbor).to(device),
                                                                    self.topk_explicit, self.device)
        return news_explicit_neighbor

    def get_news_feature(self, news_index):
        # 新闻单词
        news_word_embedding = self.word_embedding[self.news_title_word_dict[news_index]].to(self.device)
        # 新闻实体
        news_entity = self.get_news_entities(news_index).squeeze()
        news_entity_embedding = self.entity_embedding[news_entity].to(self.device).squeeze()
        # 新闻主题
        news_category_index = torch.IntTensor(self.news_category_dict[np.array(news_index)]).to(self.device)
        # 新闻副主题
        news_subcategory_index = torch.IntTensor(self.news_subcategory_dict[np.array(news_index)]).to(self.device)
        # 采样邻居节点
        neighor_entity, neighor_relation = self.get_neighbor_entities(news_entity, k=5)
        # 新闻邻居实体
        news_neigh_entity_embedding = self.entity_embedding[neighor_entity].to(self.device).squeeze()
        # 新闻邻居实体关系
        news_neigh_relation_embedding = self.relation_embedding[neighor_relation].to(self.device).squeeze()
        return news_word_embedding, news_entity_embedding, news_category_index, news_subcategory_index, \
               news_neigh_entity_embedding, news_neigh_relation_embedding, news_entity

    def compute_news_rep(self, news_word_embedding, news_entity_embedding,
                         news_category_index, news_subcategory_index, news_entity_index):
        # 新闻语义表征
        semantic_news_rep = self.news_encoder(news_word_embedding, news_entity_embedding,
                                              news_category_index, news_subcategory_index)
        if self.use_news_relation == True:
            # 构建新闻隐式邻居节点
            news_explicit_neighbor = self.sample_news_explicit_neighbor(news_entity_index,
                                                                        self.topk_explicit)
            # 构建新闻隐式邻居节点
            news_implicit_neighbor = self.sample_news_implicit_neighbor(news_category_index,
                                                                        news_subcategory_index,
                                                                        self.topk_implicit)
            news_neighbor = torch.cat([news_explicit_neighbor, news_implicit_neighbor], dim=-1)
            # news_neighbor = news_implicit_neighbor

            # 邻居新闻特征
            neighbor_news_word_embedding, neighbor_news_entity_embedding, \
            neighbor_news_category_index, neighbor_news_subcategory_index, \
            neighbor_news_neigh_entity_embedding, neighbor_news_neigh_relation_embedding, implicit_news_entity = self.get_news_feature(torch.flatten(news_neighbor.cpu(), 0, 1))

            # 邻居新闻编码
            neighbor_news_rep = self.news_encoder(neighbor_news_word_embedding, neighbor_news_entity_embedding,
                                                  neighbor_news_category_index, neighbor_news_subcategory_index)
            neighbor_news_rep = neighbor_news_rep.view(-1,
                                                       self.topk_explicit + self.topk_implicit,
                                                       semantic_news_rep.shape[-1])
            # 邻居聚合
            neighbor_news_rep = torch.tanh(self.neighbor_agg(neighbor_news_rep))
            neighbor_news_rep = F.dropout(neighbor_news_rep, p=self.args.drop_prob, training=self.training)

            return [semantic_news_rep.view(self.args.batch_size, 1, -1) +
                    neighbor_news_rep.view(self.args.batch_size, 1, -1)]
        else:
            return [semantic_news_rep.view(self.args.batch_size, 1, -1)]


    def get_user_news_rep(self, user_index, candidate_news_index, user_clicked_news_index, test_flag = False):
        # 候选新闻特征
        candidate_news_word_embedding, candidate_news_entity_embedding, \
        candidate_news_category_index,  candidate_news_subcategory_index, \
        candidate_news_neigh_entity_embedding, candidate_news_neigh_relation_embedding, candidate_news_entity = self.get_news_feature(candidate_news_index)

        # 新闻编码器
        news_rep = None
        
        if test_flag:
            candidate_news_num = 1
        else:
            candidate_news_num = self.args.sample_size

        for i in range(candidate_news_num):
            news_word_embedding_one = candidate_news_word_embedding[:, i, :, :]
            news_entity_embedding_one = candidate_news_entity_embedding[:, i, :, :]
            news_category_index = candidate_news_category_index[:, i]
            news_subcategory_index = candidate_news_subcategory_index[:, i]
            news_entity_index = candidate_news_entity[:, i, :]
            news_rep_one = self.compute_news_rep(news_word_embedding_one, news_entity_embedding_one,
                                                 news_category_index, news_subcategory_index, news_entity_index)
            if i == 0:
                news_rep = news_rep_one[0]
            else:
                news_rep = torch.cat([news_rep, news_rep_one[0]], dim=1)
            # if self.use_news_relation == False:
            #     if i == 0:
            #         news_rep = news_rep_one[0]
            #     else:
            #         news_rep = torch.cat([news_rep, news_rep_one[0]], dim=1)
            # else:
            #     news_rep_cat = torch.cat([news_rep_one[0], news_rep_one[1]], dim=-1)
            #     if i == 0:
            #         news_rep = news_rep_cat
            #     else:
            #         news_rep = torch.cat([news_rep, news_rep_cat], dim=1)


        # 用户点击新闻特征
        user_clicked_news_word_embedding, user_clicked_news_entity_embedding, \
        user_clicked_news_category_index, user_clicked_news_subcategory_index, \
        user_clicked_news_neigh_entity_embedding, user_clicked_news_neigh_relation_embedding,  user_clicked_news_entity = self.get_news_feature(user_clicked_news_index)

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

            clicked_news_rep = self.news_encoder(
                clicked_news_word_embedding_one, clicked_news_entity_embedding_one,
                clicked_news_category_index, clicked_news_subcategory_index
            ).unsqueeze(0)
            # 用户表征
            user_rep_one = self.user_encoder(clicked_news_rep).unsqueeze(0)
            if i == 0:
                user_rep = user_rep_one
            else:
                user_rep = torch.cat([user_rep, user_rep_one], dim=0)
        return user_rep, news_rep

    def forward(self, user_index, candidate_news, user_clicked_news_index):
        #candidate_news = torch.flatten(candidate_news, 0, 1)
        # 新闻用户表征
        user_rep, news_rep = self.get_user_news_rep(user_index, candidate_news, user_clicked_news_index)
        score = torch.sum(news_rep * user_rep, dim=-1).view(self.args.batch_size, -1)
        # 预测得分
        # if self.use_news_relation == False:
        #     score = torch.sum(news_rep * user_rep, dim=-1).view(self.args.batch_size, -1)
        # else:
        #     user_rep = torch.cat([user_rep, user_rep], dim=-1)
        #     score = torch.sum(news_rep * user_rep, dim=-1).view(self.args.batch_size, -1)
        return score

    def test(self, user_index, candidate_news, user_clicked_news_index):
        #candidate_news = torch.flatten(candidate_news, 0, 1)
        # 新闻用户表征
        user_rep, news_rep = self.get_user_news_rep(user_index, candidate_news, user_clicked_news_index, True)
        score = torch.sum(news_rep * user_rep, dim=-1).view(self.args.batch_size, -1)
        # 预测得分
        # if self.use_news_relation == False:
        #     score = torch.sum(news_rep * user_rep, dim=-1).view(self.args.batch_size, -1)
        # else:
        #     user_rep = torch.cat([user_rep, user_rep], dim=-1)
        #     score = torch.sum(news_rep * user_rep, dim=-1).view(self.args.batch_size, -1)
        return score
