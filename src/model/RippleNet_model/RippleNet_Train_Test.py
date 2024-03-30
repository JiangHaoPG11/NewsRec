from model.RippleNet_model.RippleNet import RippleNet
from model.RippleNet_model.RippleNet_Trainer import Trainer
import torch

class RippleNet_Train_Test():
    def __init__(self, args, data, device):
        train_dataloader, test_dataloader, vaild_dataloader, \
        news_title_embedding, entity_adj, relation_adj, entity_dict, kg_env, news_entity_dict, entity_news_dict, user_click_dict, \
        news_title_word_index, news_category_index, news_subcategory_index, category_news_dict, subcategory_news_dict, word_embedding, \
        neibor_embedding, neibor_num, entity_embedding, relation_embedding, user_clicked_newsindex, \
        vailddata_size, traindata_size, testdata_size, label_test, bound_test = data

        user_clicked_newsindex = self.get_user_clicked_newsindex(
            args, entity_adj, relation_adj, news_entity_dict, entity_news_dict, user_clicked_newsindex
        )

        ripplenet = RippleNet(args, news_title_embedding, entity_embedding, relation_embedding, user_clicked_newsindex).to(device)
        optimizer_ripplenet = torch.optim.Adam(ripplenet.parameters(), lr= 0.0005)
        self.trainer = Trainer(args, ripplenet, optimizer_ripplenet, data)


    def get_user_clicked_newsindex(args, entity_adj, relation_adj, news_entity_dict, entity_news_dict, user_clicked_newsindex):
        print('constructing ripple set ...')
        # user -> [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]
        user_clicked_newsindex = []
        for i in range(len(user_clicked_newsindex)):
            user_clicked_newsindex.append([])
            for h in range(args.ripplenet_n_hop + 1):
                memories_h = []
                memories_r = []
                memories_t = []
                if h == 0:
                    tails_of_last_hop = user_clicked_newsindex[i]
                    for news in tails_of_last_hop:
                        for news_entity in news_entity_dict[news]:
                            memories_h.append(news)
                            memories_t.append(news_entity)
                            memories_r.append(0)
                elif h == args.ripplenet_n_hop:
                    tails_of_last_hop = user_clicked_newsindex[-1][-1][2]
                    for entity in tails_of_last_hop:
                        for entity_news in entity_news_dict[entity]:
                            memories_h.append(entity)
                            memories_t.append(entity_news)
                            memories_r.append(0)
                else:
                    tails_of_last_hop = user_clicked_newsindex[-1][-1][2]
                    for entity in tails_of_last_hop:
                        if entity in entity_adj.keys():
                            for i in range(len(entity_adj[entity])):
                                memories_h.append(entity)
                                memories_t.append(entity_adj[entity][i])
                                memories_r.append(relation_adj[entity][i])
                        else:
                            memories_h.append(entity)
                            memories_t.append(0)
                            memories_r.append(0)
                if len(memories_h) == 0:
                    user_clicked_newsindex[-1].append(user_clicked_newsindex[i][-1])
                else:
                    # sample a fixed-size 1-hop memory for each user
                    replace = len(memories_h) < args.ripplenet_n_memory
                    indices = np.random.choice(len(memories_h), size=args.ripplenet_n_memory, replace=replace)
                    memories_h = [memories_h[i] for i in indices]
                    memories_r = [memories_r[i] for i in indices]
                    memories_t = [memories_t[i] for i in indices]
                    user_clicked_newsindex[-1].append((memories_h, memories_r, memories_t))
        return torch.IntTensor(user_clicked_newsindex)


    def Train(self):
        print('training begining ...')
        self.trainer.train()

    def Test(self):
        print('testing begining ...')
        self.trainer.test()
