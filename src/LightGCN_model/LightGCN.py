import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import *
import scipy.sparse as sp

class LightGCN(nn.Module):
    def __init__(self, args, device, user_click_dict):
        super(LightGCN, self).__init__()
        self.args = args
        self.device = device
        self.__init_weight()
        self.Graph = self._convert_sp_mat_to_sp_tensor(user_click_dict)

    def __init_weight(self):
        self.num_users = self.args.user_num
        self.num_items = self.args.news_num
        self.latent_dim = self.args.embedding_dim
        self.n_layers = self.args.lgn_layers
        self.keep_prob = self.args.keep_prob
        self.A_split = False
        self.user_emds = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.args.embedding_dim)
        self.item_emds = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.args.embedding_dim)

        nn.init.normal_(self.user_emds.weight, std=0.1)
        nn.init.normal_(self.item_emds.weight, std=0.1)

    def _convert_sp_mat_to_sp_tensor(self, user_click_dict):
        val_list = []
        shape1_list = []
        shape2_list = []
        for i in range(len(user_click_dict)):
            news_index = user_click_dict[i]
            for j in news_index:
                if j != self.num_items - 1:
                    val_list.append(1)
                    val_list.append(1)
                    shape1_list.append(i)
                    shape2_list.append(j + self.num_users)
                    shape1_list.append(j + self.num_users)
                    shape2_list.append(i)
        adj = sp.coo_matrix((val_list,
                             (shape1_list, shape2_list)), shape=(self.num_users + self.num_items,
                                                                 self.num_users + self.num_items))
        i = torch.LongTensor([adj.row, adj.col])
        v = torch.from_numpy(adj.data).float()
        return torch.sparse.FloatTensor(i, v, adj.shape).to(self.device)

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def LightGCN_forward(self):
        user_emds = self.user_emds.weight
        item_emds = self.item_emds.weight
        all_emds = torch.cat([user_emds, item_emds])
        embs = [all_emds]
        if self.args.dropout:
            if self.training:
                g_dropped = self.__dropout(self.keep_prob)
            else:
                g_dropped = self.Graph
        else:
            g_dropped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_dropped)):
                    temp_emb.append(torch.sparse.mm(g_dropped[f], all_emds))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emds = side_emb
            else:
                all_emds = torch.sparse.mm(g_dropped, all_emds)
            embs.append(all_emds)
        embs = torch.stack(embs, dim=1)
        lgn_out = torch.mean(embs, dim=1)
        users, items = torch.split(lgn_out, [self.num_users, self.num_items])
        return users, items

    def create_bpr_loss(self, users, items, labels):
        batch_size = users.shape[0]
        scores = (items * users.unsqueeze(1)).sum(dim=-1)
        scores = torch.sigmoid(scores)
        rec_loss = F.cross_entropy(F.softmax(scores, dim = -1), torch.argmax(labels.to(self.device), dim=1))
        regularizer = (torch.norm(users) ** 2 + torch.norm(items) ** 2) / 2
        emb_loss = self.args.l2 * regularizer / batch_size
        return rec_loss + emb_loss, scores, rec_loss, emb_loss

    def forward(self,  user_index, candidate_news_index, label):
        all_users_embedding, all_news_embedding = self.LightGCN_forward()
        users_ID_embs = all_users_embedding[user_index.long()]
        news_ID_embs = all_news_embedding[candidate_news_index.long()]
        return self.create_bpr_loss(users_ID_embs, news_ID_embs, label)

    def test(self, user_index, candidate_news_index):
        all_users_embedding, all_news_embedding = self.LightGCN_forward()
        users_ID_embs = all_users_embedding[user_index.long()]
        items_ID_embs = all_news_embedding[candidate_news_index.long()]
        scores = torch.sigmoid((items_ID_embs * users_ID_embs.unsqueeze(1)).sum(dim=-1))
        return scores
