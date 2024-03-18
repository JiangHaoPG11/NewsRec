import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp

class Aggregator(nn.Module):
    def __init__(self, device, news_entity_dict, entity_adj, relation_adj):
        super(Aggregator, self).__init__()
        self.news_entity_dict = news_entity_dict
        self.entity_adj = entity_adj
        self.relation_adj = relation_adj
        self.device = device
        self.news_atten = nn.Linear(100, 1)
        self.entity_atten = nn.Linear(100, 1)

    def get_news_entities_batch(self):
        news_entities = []
        news_relations = []
        news = []
        for key, value in self.news_entity_dict.items():
            news.append(key)
            news_entities.append(value)
            news_relations.append([0 for k in range(len(value))])
        news = torch.tensor(news).to(self.device)
        news_entities = torch.tensor(news_entities).to(self.device)
        news_relations = torch.tensor(news_relations).to(self.device)# bz, news_entity_num
        return news, news_entities, news_relations

    def get_entities_neigh_batch(self, n_entity):
        neigh_entities = []
        neigh_relations = []
        entities = []
        for i in range(n_entity):
            if i in self.entity_adj.keys():
                entities.append(i)
                neigh_entities.append(self.entity_adj[i])
                neigh_relations.append(self.relation_adj[i])
            else:
                entities.append(i)
                neigh_entities.append([0 for k in range(20)])
                neigh_relations.append([0 for k in range(20)])
        entities = torch.tensor(entities).to(self.device)
        neigh_entities = torch.tensor(neigh_entities).to(self.device)
        neigh_relations = torch.tensor(neigh_relations).to(self.device)# bz, news_entity_num
        return entities, neigh_entities, neigh_relations

    def forward(self, user_emb, all_embedding, entity_emb, relation_emb, interact_mat):

        newsid, news_entities, news_relations = self.get_news_entities_batch()
        news_emb = all_embedding[newsid]
        news_neigh_entities_embedding = entity_emb[news_entities]
        news_neigh_relation_embedding = relation_emb[news_relations]
        news_weight = F.softmax(self.news_atten(news_neigh_entities_embedding + news_neigh_relation_embedding), dim = -1)
        news_agg = torch.matmul(torch.transpose(news_weight, -1, -2), news_neigh_entities_embedding).squeeze()

        entities, neigh_entities, neigh_relations = self.get_entities_neigh_batch(n_entity = len(entity_emb))
        entity_emb = all_embedding[entities]
        neigh_entities_embedding = entity_emb[neigh_entities]
        neigh_relation_embedding = relation_emb[neigh_relations]
        entity_weight = F.softmax(self.entity_atten(neigh_relation_embedding + neigh_entities_embedding), dim = -1)
        entity_agg = torch.matmul(torch.transpose(entity_weight, -1, -2), neigh_entities_embedding).squeeze()

        node_emb = torch.cat([news_agg + news_emb, entity_agg + entity_emb])
        user_agg = torch.sparse.mm(interact_mat, node_emb)
        user_agg = user_emb + user_agg  # [n_users, channel]
        return node_emb, user_agg


class KGAT(nn.Module):
    def __init__(self, device,  n_hops, n_users, n_relations, interact_mat,
                 news_entity_dict,  entity_adj, relation_adj,
                 ind, node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(KGAT, self).__init__()

        self.aggs = nn.ModuleList()
        self.news_entity_dict = news_entity_dict
        self.entity_adj = entity_adj
        self.relation_adj = relation_adj
        self.interact_mat = interact_mat
        self.n_relations = n_relations
        self.n_users = n_users
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.ind = ind
        self.lambda_coeff = 0.5
        self.temperature = 0.2
        self.device = device

        for i in range(n_hops):
            self.aggs.append(Aggregator(self.device, news_entity_dict, entity_adj, relation_adj))
        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    # def _edge_sampling(self, edge_index, edge_type, rate=0.5):
    #     # edge_index: [2, -1]
    #     # edge_type: [-1]
    #     n_edges = edge_index.shape[1]
    #     random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
    #     return edge_index[:, random_indices], edge_type[random_indices]

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()
        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()
        i = i[:, dropout_mask]
        v = v[dropout_mask]
        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def forward(self, user_embedding, all_embedding, entity_embedding, relation_embedding, mess_dropout=True, node_dropout=False):
        # """node dropout"""
        # if node_dropout:
        #     edge_index, edge_type = self._edge_sampling(edge_index, edge_type, self.node_dropout_rate)

        node_res_emb = all_embedding  # [n_node, channel]
        user_res_emb = user_embedding  # [n_users, channel]

        for i in range(len(self.aggs)):
            node_emb, user_emb = self.aggs[i](user_embedding, all_embedding, entity_embedding, relation_embedding, self.interact_mat)
            if mess_dropout:
                node_emb = self.dropout(node_emb)
                user_emb = self.dropout(user_emb)

            node_emb = F.normalize(node_emb)
            user_emb = F.normalize(user_emb)

            node_res_emb = torch.add(node_res_emb, node_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)

        return user_res_emb, node_res_emb

class KGAT_model(nn.Module):
    def __init__(self, args, news_title_embedding, entity_embedding, relation_embedding, news_entity_dict,  entity_adj, relation_adj, user_click_dict):
        super(KGAT_model, self).__init__()

        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_users = args.user_num
        self.n_items = args.news_num
        self.n_relations = len(relation_embedding)
        self.n_entities = len(entity_embedding)

        self.user_embedding = nn.Embedding(self.n_users, self.args.embedding_dim)
        self.news_embedding = nn.Embedding(self.n_items, self.args.embedding_dim)
        self.entity_embedding = nn.Embedding.from_pretrained(entity_embedding)
        self.relation_embedding = nn.Embedding.from_pretrained(relation_embedding)
        self.news_entity_dict = news_entity_dict
        self.entity_adj = entity_adj
        self.relation_adj = relation_adj
        self.user_click_dict = user_click_dict

        self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.user_click_dict, self.news_entity_dict).to(self.device)

        self.decay = args.l2
        self.sim_decay = args.sim_regularity
        self.embedding_dim = args.embedding_dim
        self.context_hops = args.context_hops
        self.node_dropout = args.node_dropout
        self.node_dropout_rate = args.node_dropout_rate
        self.mess_dropout = args.mess_dropout
        self.mess_dropout_rate = args.mess_dropout_rate
        self.ind = args.ind

        self.kgat = self._init_model()
        self.lightgcn_layer = 2
        self.n_item_layer = 2

    def _init_model(self):
        return KGAT(device=self.device,
                    n_hops=self.context_hops,
                    n_users=self.n_users,
                    n_relations=self.n_relations,
                    interact_mat=self.interact_mat,
                    news_entity_dict=self.news_entity_dict,
                    entity_adj=self.entity_adj,
                    relation_adj=self.relation_adj,
                    ind=self.ind,
                    node_dropout_rate=self.node_dropout_rate,
                    mess_dropout_rate=self.mess_dropout_rate)

    # update
    def _convert_sp_mat_to_sp_tensor(self, user_click_dict, news_entity_dict):
        val_list = []
        shape1_list = []
        shape2_list = []
        for i in range(len(user_click_dict)):
            news_index = user_click_dict[i]
            for j in news_index:
                if j != self.n_items - 1:
                    val_list.append(1)
                    shape1_list.append(i)
                    shape2_list.append(j)
                    entity_list = news_entity_dict[j]
                    for m in entity_list:
                        if m != 0:
                            val_list.append(1)
                            shape1_list.append(i)
                            shape2_list.append(m + + self.n_items)

        adj = sp.coo_matrix((val_list,
                             (shape1_list, shape2_list)), shape=(self.n_users,
                                                                 self.n_items + self.n_entities))
        i = torch.LongTensor([adj.row, adj.col])
        v = torch.from_numpy(adj.data).float()

        return torch.sparse.FloatTensor(i, v, adj.shape)

    def _concat_all_embedding(self):

        user_embeddings = self.user_embedding.weight
        news_embeddings = self.news_embedding.weight
        entity_embeddings = self.entity_embedding.weight
        relation_embeddings = self.relation_embedding.weight

        all_embedding = torch.cat([news_embeddings, entity_embeddings], dim = 0)
        return user_embeddings, all_embedding, entity_embeddings, relation_embeddings


    def forward(self, user_index, candidate_newsindex, labels):
        candidate_newsindex = candidate_newsindex.to(self.device)
        user_index = user_index.unsqueeze(1).repeat(1, 5).to(self.device)
        user_index = torch.flatten(user_index, 0, 1)
        candidate_newsindex = torch.flatten(candidate_newsindex, 0, 1)


        user_embeddings, all_embedding, \
        entity_embeddings, relation_embeddings = self._concat_all_embedding()

        user_kgat_emb, node_kgat_emb = self.kgat(user_embeddings, all_embedding,
                                                 entity_embeddings, relation_embeddings,
                                                 mess_dropout=self.mess_dropout)
        u_e = user_kgat_emb[user_index]
        i_e = node_kgat_emb[candidate_newsindex]
        
        return self.create_bpr_loss(u_e, i_e, labels)


    def create_bpr_loss(self, users, items, labels):
        batch_size = users.shape[0]
        scores = (items * users).sum(dim=1)
        rec_loss = F.cross_entropy(F.softmax(scores.view(-1, 5), dim=-1),
                                   torch.argmax(labels.to(self.device), dim=1))
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size
        return rec_loss + emb_loss, scores.view(-1, 5), rec_loss, emb_loss

    def test(self, user_index,  candidate_newsindex):
        candidate_newsindex = candidate_newsindex.to(self.device)
        user_index = user_index.unsqueeze(1).repeat(1, 5).to(self.device)
        user_index = torch.flatten(user_index, 0, 1)
        candidate_newsindex = torch.flatten(candidate_newsindex, 0, 1)

        user_embeddings, all_embedding, entity_embeddings, relation_embeddings = self._concat_all_embedding()

        user_kgat_emb, node_kgat_emb = self.kgat(
            user_embeddings, all_embedding,
            entity_embeddings, relation_embeddings,
            mess_dropout=self.mess_dropout
        )

        u_e = user_kgat_emb[user_index]
        i_e = node_kgat_emb[candidate_newsindex]

        scores = (i_e * u_e).sum(dim=1).view(-1, 5)
        return scores
