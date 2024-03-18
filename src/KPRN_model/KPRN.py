import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

class KPRN(nn.Module):
    def __init__(self, args, news_title_embedding, entity_embedding, relation_embedding,
                 entity_adj, relation_adj, news_entity_dict, entity_news_dict,
                 total_paths_index, total_relations_index, total_type_index):
        super(KPRN, self).__init__()
        self._parse_args(args)
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.entity_embedding = nn.Embedding.from_pretrained(entity_embedding)
        self.relation_embedding = nn.Embedding.from_pretrained(relation_embedding)
        self.user_embedding = nn.Embedding(self.args.user_size, self.embedding_dim)
        self.news_embedding = nn.Embedding(self.args.title_num, self.embedding_dim)

        self.entity_adj = entity_adj
        self.relation_adj = relation_adj
        self.news_entity_dict = news_entity_dict
        self.entity_news_dict = entity_news_dict
        self.total_paths_index = total_paths_index
        self.total_relations_index = total_relations_index
        self.total_type_index = total_type_index

        self.transform_matrix = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.news_to_entity = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.type_embedding = nn.Embedding(3, self.embedding_dim) # user, news, entity

        self.elu = nn.ELU()

        self.path_score_l1 = nn.Linear(self.args.embedding_size, 16)
        self.path_score_l2 = nn.Linear(16, 1)

        self.lstm = {}
        for i in range(self.kprn_max_path):
            self.lstm[i] = nn.LSTM(self.embedding_dim * 3, self.embedding_dim, self.kprn_path_long, batch_first = True).to(self.device)

    
    def _parse_args(self, args):
        self.embedding_dim = args.embedding_size
        self.title_embedding_dim = args.title_embedding_dim
        self.kprn_path_long = args.kprn_path_long
        self.kprn_max_path = args.kprn_max_path

    def trans_news_embedding(self, news_index):
        trans_news_embedding = self.news_embedding(news_index)
        return trans_news_embedding

    def get_news_entities_batch(self, newsids):
        news_entities = []
        news_relations = []
        for i in range(len(newsids)):
            news_entities.append(self.news_entity_dict[int(newsids[i])])
            news_relations.append([0 for k in range(len(self.news_entity_dict[int(newsids[i])]))])
        news_entities = torch.tensor(news_entities).to(self.device)
        news_relations = torch.tensor(news_relations).to(self.device)# bz, news_entity_num
        return news_entities, news_relations

    def get_batch_path(self, user_index):
        batch_path_index = self.total_paths_index[user_index]
        batch_type_index = self.total_type_index[user_index]
        batch_relations_index = self.total_relations_index[user_index]
        return batch_path_index, batch_type_index, batch_relations_index

    def get_batch_embedding(self, batch_path_index, batch_type_index, batch_relations_index):
        node_embedding_list = None
        for i in range(batch_path_index.shape[0]):
            single_path_index = batch_path_index[i, :]
            single_type_index = batch_type_index[i, :]
            single_node_embedding_list = None
            for j in range(single_type_index.shape[0]):
                path_type = single_type_index[j]
                if path_type == 0:
                    node_embedding = self.user_embedding(single_path_index[j])
                    node_embedding = node_embedding.unsqueeze(0)
                elif path_type == 1:
                    node_embedding = self.trans_news_embedding(single_path_index[j])
                    node_embedding = node_embedding.unsqueeze(0)
                else:
                    node_embedding = self.entity_embedding(single_path_index[j])
                    node_embedding = node_embedding.unsqueeze(0)
                if j == 0:
                    single_node_embedding_list = node_embedding
                else:
                    single_node_embedding_list = torch.cat([single_node_embedding_list, node_embedding], dim = 0)
            single_node_embedding_list = single_node_embedding_list.unsqueeze(0)
            if i == 0:
                node_embedding_list = single_node_embedding_list
            else:
                node_embedding_list = torch.cat([node_embedding_list, single_node_embedding_list], dim=0)
        type_embedding_list = self.type_embedding(batch_type_index)
        relations_embedding_list = self.relation_embedding(batch_relations_index)
        return node_embedding_list, type_embedding_list, relations_embedding_list

    def forward(self, candidate_newsindex, user_index):
        batch_path_index, batch_type_index, batch_relations_index = self.get_batch_path(user_index)
        batch_path_index = batch_path_index.to(self.device)
        batch_type_index = batch_type_index.to(self.device)
        batch_relations_index = batch_relations_index.to(self.device)

        news_embedding = self.news_embedding(candidate_newsindex.to(self.device))
        user_embedding = self.user_embedding(user_index.to(self.device)).unsqueeze(1)

        total_output = None
        total_score = None
        for j in range(candidate_newsindex.shape[1]):
            batch_path_index_select = batch_path_index[:, j, :, :]
            batch_type_index_select = batch_type_index[:, j, :, :]
            batch_relations_index_select = batch_relations_index[:, j, :, :]
            for i in range(self.kprn_max_path):
                node_embedding, type_embedding, relation_embedding = self.get_batch_embedding(
                    batch_path_index_select[:, i, ],
                    batch_type_index_select[:, i, ],
                    batch_relations_index_select[:, i, ])
                input1 = torch.cat([node_embedding, type_embedding, relation_embedding], dim=-1).to(self.device)
                output, (_, _) = self.lstm[i](input1)
                output = output[:, i, :].unsqueeze(1)
                if total_output == None:
                    total_output = output
                else:
                    total_output = torch.cat([total_output, output], dim=1)
            path_score = torch.sigmoid(self.path_score_l2(torch.relu(self.path_score_l1(total_output))))
            score = torch.mean(path_score, dim=1)
            if total_score == None:
                total_score = score
            else:
                total_score = torch.cat([total_score, score], dim=-1)

        emb_score = torch.sigmoid(torch.sum(news_embedding * user_embedding, dim=-1))
        score = emb_score + total_score
        return score
