from SFT_NPA_model.SFT_NPA import SFT_NPA
from SFT_NPA_model.SFT_NPA_Trainer import Trainer
import torch

class SFT_NPA_Train_Test():
    def __init__(self, args, data, device):
        train_dataloader, test_dataloader, vaild_dataloader, \
        news_title_embedding, entity_adj, relation_adj, entity_dict, kg_env, news_entity_dict, entity_news_dict, user_click_dict, \
        news_title_word_index, news_category_index, news_subcategory_index, category_news_dict, subcategory_news_dict, word_embedding, \
        neibor_embedding, neibor_num, entity_embedding, relation_embedding, user_clicked_newsindex, \
        vailddata_size, traindata_size, testdata_size, label_test, bound_test = data

        SFT_NPA_model = SFT_NPA(args, entity_embedding, relation_embedding,
                                            news_entity_dict, entity_adj, relation_adj, news_title_word_index,
                                            word_embedding, news_category_index, news_subcategory_index, device).to(device)

        optimizer_Zeroshot_news = torch.optim.Adam(SFT_NPA_model.zeroshot_news_tower.parameters(), lr=2e-5)
        optimizer_Zeroshot_user = torch.optim.Adam(SFT_NPA_model.zeroshot_user_tower.parameters(), lr=2e-5)


        optimizer_att = torch.optim.Adam([{"params": SFT_NPA_model.news_att_encoder.parameters()},
                                          {"params": SFT_NPA_model.user_att_encoder.parameters()},
                                           ], lr=2e-5)
            
        # optimizer_base = torch.optim.Adam([{"params": SFT_NPA_model.news_encoder.parameters()},
        #                                    {"params": SFT_NPA_model.user_encoder.parameters()}
        #                                    ], lr=0.0001)
            
        optimizer_base = torch.optim.Adam([{"params": SFT_NPA_model.news_encoder.parameters()},
                                           {"params": SFT_NPA_model.user_encoder.parameters()},
                                           {"params": SFT_NPA_model.news_att_encoder.parameters()},
                                           {"params": SFT_NPA_model.user_att_encoder.parameters()},
                                           {"params": SFT_NPA_model.zeroshot_news_tower.parameters()},
                                           {"params": SFT_NPA_model.zeroshot_user_tower.parameters()}
                                           ], lr=2e-5)

        for para in SFT_NPA_model.named_parameters():
            print(para[0])
        self.trainer = Trainer(args, SFT_NPA_model, optimizer_att, optimizer_base, optimizer_Zeroshot_news, optimizer_Zeroshot_user, data)
        self.Zeroshot_model = SFT_NPA_model
        self.args = args

    def Train(self):
        print('training begining ...')
        # AnchorKG_model.train()
        self.trainer.train()

    def Test(self):
        self.trainer.test()

    def Test_load(self):
        self.Zeroshot_model.load_state_dict(torch.load(self.args.checkpoint_dir + 'checkpoint-' + self.args.mode + '-epochfinal.pth'))
        self.trainer.test()
