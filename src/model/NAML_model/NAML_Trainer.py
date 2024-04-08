import os
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import no_grad
from utils.measure import *
from numpy import *
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class Trainer():
    def __init__(self, args, NAML_model, optimizer_NAML, data):
        self.args = args
        self.NAML_model = NAML_model
        self.optimizer_NAML = optimizer_NAML
        self.save_period = 1
        self.vaild_period = 1
        self.train_dataloader = data[0]
        self.test_dataloader = data[1]
        self.vaild_dataloader = data[2]
        self.vailddata_size = data[-5]
        self.traindata_size = data[-4]
        self.testdata_size = data[-3]
        self.label_test = data[-2]
        self.bound_test = data[-1]
        self.news_category_index = data[12]


        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        #self.device = torch.device("cpu")

    def cal_auc(self, label, rec_score):
        try:
            auc = roc_auc_score(label.cpu().numpy(), F.softmax(rec_score.cpu(), dim=1).detach().numpy())
        except ValueError:
            auc = 0.5
        return auc

    def optimize_model(self, rec_score, label):
        rec_loss = self.criterion(rec_score, torch.argmax(label, dim=1).to(self.device))
        self.optimizer_NAML.zero_grad()
        rec_loss.backward()
        #for name, param in self.model_recommender.named_parameters():
        #    print('%14s : %s' % (name, param.grad))
        self.optimizer_NAML.step()
        return rec_loss

    def _train_epoch(self, epoch):
        self.NAML_model.train()
        rec_all_loss_list = []
        auc_list = []
        rec_all_loss = 0

        pbar = tqdm(total=self.traindata_size,  desc=f"Epoch {epoch}", ncols=100, leave=True, position=0)
        for data in self.train_dataloader:
            candidate_newsindex, user_index, user_clicked_newsindex, label, user_type_index, news_type_index = data
            rec_score = self.NAML_model(candidate_newsindex, user_clicked_newsindex)
            rec_loss = self.optimize_model(rec_score, label.to(self.device))
            rec_auc = self.cal_auc(label, rec_score)
            rec_all_loss = rec_all_loss + rec_loss.data
            rec_all_loss_list.append(torch.mean(rec_loss).cpu().item())
            auc_list.append(rec_auc)
            pbar.update(self.args.batch_size)

        pbar.close()
        return  mean(rec_all_loss_list), mean(auc_list)

    def _vaild_epoch(self):
        self.NAML_model.eval()
        rec_auc_list = []
        with no_grad():
            pbar = tqdm(total=self.vailddata_size)
            for data in self.vaild_dataloader:
                candidate_newsindex, user_index, user_clicked_newsindex, label, user_type_index, news_type_index = data
                rec_score = self.NAML_model.test(candidate_newsindex, user_clicked_newsindex)
                rec_auc = self.cal_auc(label, rec_score)
                rec_auc_list.append(rec_auc)
                pbar.update(self.args.batch_size)
            pbar.close()
        return mean(rec_auc_list)

    def _save_checkpoint(self, epoch):
        state_NAML_model = self.NAML_model.state_dict()
        filename_NAML = self.args.checkpoint_dir + ('checkpoint-NAML-epoch{}.pth'.format(epoch))
        torch.save(state_NAML_model, filename_NAML)

    def train(self):
        for epoch in range(1, self.args.epoch+1):
            rec_loss, rec_auc = self._train_epoch(epoch)
            print("epoch:{}----recommend loss:{}------rec auc:{}-------".
                  format(epoch, str(rec_loss), str(rec_auc)))

            if epoch % self.vaild_period == 10:
                print('start vaild ...')
                rec_auc = self._vaild_epoch()
                print("epoch:{}---vaild auc:{} ".format(epoch, str(rec_auc)))

            if epoch % self.save_period == 60:
                self._save_checkpoint(epoch)
        self._save_checkpoint('final')

    def test(self):
        print('start testing...')
        pbar = tqdm(total= self.testdata_size)
        self.NAML_model.eval()
        pred_label_list = []
        news_index_list = []
        user_index_list = []
        user_type_list = []
        news_type_list = []
        user_rep_list = []
        news_rep_list = []
        candidate_newsindex_list = []
        candidate_newscategory_list = []

        with no_grad():
            for data in self.test_dataloader:
                candidate_newsindex, user_index, user_clicked_newsindex, user_type_index, news_type_index = data
                rec_score, user_rep, news_rep = self.NAML_model.test(candidate_newsindex, user_clicked_newsindex)
                score = rec_score
                candidate_newscategory_list.extend(self.news_category_index[candidate_newsindex.cpu().numpy()])
                user_rep_list.extend(user_rep.cpu().numpy()[:, 0])
                news_rep_list.extend(news_rep.cpu().numpy()[:, 0])
                pred_label_list.extend(score.cpu().numpy())
                user_index_list.extend(user_index.cpu().numpy())
                news_index_list.extend(candidate_newsindex.cpu().numpy()[:, 0])
                user_type_list.extend(user_type_index.cpu().numpy())
                news_type_list.extend(news_type_index.cpu().numpy())
                candidate_newsindex_list.extend(candidate_newsindex.cpu().numpy()[:,0])
                pbar.update(self.args.batch_size)
            pred_label_list = np.vstack(pred_label_list)
            candidate_newscategory_list = np.vstack(candidate_newscategory_list)
            pbar.close()

        # user_type_list1 = np.array(user_type_list)
        # pca = PCA(n_components=2)
        # user_rep = np.array(user_rep_list)
        # # norm
        # for j in range(user_rep.shape[1]):
        #     user_rep[:, j] = (user_rep[:, j] - np.mean(user_rep[:, j]))/ np.std(user_rep[:, j])  # 第1列均值方差归一化
        # pca = pca.fit(user_rep)
        # np.save('./NAML_user_rep.npy', np.array(user_rep_list))
        # x_dr = pca.transform(user_rep)
        # plt.scatter(x_dr[user_type_list1==0,0],x_dr[user_type_list1==0,1],c='red', label="Cold user")
        # plt.scatter(x_dr[user_type_list1==1,0],x_dr[user_type_list1==1,1],c='green', label="Warm user")
        # plt.scatter(x_dr[user_type_list1==2,0],x_dr[user_type_list1==2,1],c='blue', label="Normal user")
        # plt.legend()
        # plt.savefig("NAML_user_rep.png")
        # plt.show()
        # plt.close()

        # news_type_list1 = np.array(news_type_list)[:, 0]
        # pca = PCA(n_components=2)
        # news_rep = np.array(news_rep_list)
        # # norm
        # for j in range(news_rep.shape[1]):
        #     news_rep[:, j] = (news_rep[:, j] - np.mean(news_rep[:, j]))/ np.std(news_rep[:, j])  # 第1列均值方差归一化
        # pca = pca.fit(news_rep)
        # np.save('./NAML_news_rep.npy', np.array(news_rep_list))
        # x_dr = pca.transform(news_rep)
        # plt.scatter(x_dr[news_type_list1==0,0],x_dr[news_type_list1==0,1],c='red', label="Cold news")
        # plt.scatter(x_dr[news_type_list1==1,0],x_dr[news_type_list1==1,1],c='green', label="Warm news")
        # plt.scatter(x_dr[news_type_list1==2,0],x_dr[news_type_list1==2,1],c='blue', label="Normal news")
        # plt.legend()
        # plt.savefig("NAML_news_rep.png")
        # plt.show()
        # plt.close()

        # 存储预测结果
        folder_path = '../predict/NAML/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        predict_df = pd.DataFrame()
        predict_df['user'] = user_index_list
        predict_df['user_type'] = user_type_list
        predict_df['candidate_news'] = candidate_newsindex_list
        predict_df['candidate_news_type'] = news_type_list
        predict_df['score'] = pred_label_list[:, 0]
        predict_df['label'] = self.label_test[:len(user_index_list)]
        predict_df.to_csv(f'{folder_path}/NAML_predict.csv', index = False)

        test_AUC, test_MRR, test_nDCG5, test_nDCG10 = evaluate(pred_label_list, self.label_test, self.bound_test)
        print("test_AUC = %.4lf, test_MRR = %.4lf, test_nDCG5 = %.4lf, test_nDCG10 = %.4lf" %
              (test_AUC, test_MRR, test_nDCG5, test_nDCG10))
        print('================user====================')
        c_AUC, c_MRR, c_nDCG5, c_nDCG10, c_len, \
        w_AUC, w_MRR, w_nDCG5, w_nDCG10, w_len, = evaluate_warm_cold_u(pred_label_list, user_type_list, news_type_list,
                                                                       self.label_test, self.bound_test)
        print("c_AUC = %.4lf, c_MRR = %.4lf, c_nDCG5 = %.4lf, c_nDCG10 = %.4lf, c_len = %.4lf" %
              (c_AUC, c_MRR, c_nDCG5, c_nDCG10, c_len))
        print("w_AUC = %.4lf, w_MRR = %.4lf, w_nDCG5 = %.4lf, w_nDCG10 = %.4lf, w_len = %.4lf" %
              (w_AUC, w_MRR, w_nDCG5, w_nDCG10, w_len))
        print('================news====================')
        c_AUC, c_MRR, c_nDCG5, c_nDCG10, c_len, \
        w_AUC, w_MRR, w_nDCG5, w_nDCG10, w_len = evaluate_warm_cold_n(pred_label_list, user_type_list, news_type_list, self.label_test, self.bound_test)
        
        print("c_AUC = %.4lf, c_MRR = %.4lf, c_nDCG5 = %.4lf, c_nDCG10 = %.4lf, c_len = %.4lf" %
              (c_AUC, c_MRR, c_nDCG5, c_nDCG10, c_len))
        print("w_AUC = %.4lf, w_MRR = %.4lf, w_nDCG5 = %.4lf, w_nDCG10 = %.4lf, w_len = %.4lf" %
              (w_AUC, w_MRR, w_nDCG5, w_nDCG10, w_len))
        print('================news-user===============')
        cc_AUC, cc_MRR, cc_nDCG5, cc_nDCG10, cc_len, \
        cw_AUC, cw_MRR, cw_nDCG5, cw_nDCG10, cw_len, \
        wc_AUC, wc_MRR, wc_nDCG5, wc_nDCG10, wc_len, \
        ww_AUC, ww_MRR, ww_nDCG5, ww_nDCG10, ww_len = evaluate_warm_cold(pred_label_list, user_type_list, news_type_list, self.label_test, self.bound_test)

        print("cc_AUC = %.4lf, cc_MRR = %.4lf, cc_nDCG5 = %.4lf, cc_nDCG10 = %.4lf, cc_len = %.4lf" %
              (cc_AUC, cc_MRR, cc_nDCG5, cc_nDCG10, cc_len))
        print("cw_AUC = %.4lf, cw_MRR = %.4lf, cw_nDCG5 = %.4lf, cw_nDCG10 = %.4lf, cw_len = %.4lf" %
              (cw_AUC, cw_MRR, cw_nDCG5, cw_nDCG10, cw_len))
        print("wc_AUC = %.4lf, wc_MRR = %.4lf, wc_nDCG5 = %.4lf, wc_nDCG10 = %.4lf, wc_len = %.4lf" %
              (wc_AUC, wc_MRR, wc_nDCG5, wc_nDCG10, wc_len))
        print("ww_AUC = %.4lf, ww_MRR = %.4lf, ww_nDCG5 = %.4lf, w_nDCG10 = %.4lf, ww_len = %.4lf" %
              (ww_AUC, ww_MRR, ww_nDCG5, ww_nDCG10, ww_len))

