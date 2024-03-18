import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import no_grad
from utils.measure import *
from numpy import *
import os

class Trainer():
    def __init__(self, args, KGAT_model, optimizer_KGAT, data):
        self.args = args
        self.KGAT_model = KGAT_model
        self.optimizer_KGAT = optimizer_KGAT
        self.save_period = 100
        self.vaild_period = 40
        self.train_dataloader = data[0]
        self.test_dataloader = data[1]
        self.vaild_dataloader = data[2]
        self.vailddata_size = data[-5]
        self.traindata_size = data[-4]
        self.testdata_size = data[-3]
        self.label_test = data[-2]
        self.bound_test = data[-1]

    def cal_auc(self, score, label):
        try:
            rec_auc = roc_auc_score(label.cpu().numpy(), F.softmax(score.cpu(), dim=1).detach().numpy())
        except ValueError:
            rec_auc = 0.5
        return rec_auc

    def optimize_KGAT(self, loss,):
        self.optimizer_KGAT.zero_grad()
        loss.backward()
        self.optimizer_KGAT.step()

    def _train_epoch(self):
        self.KGAT_model.train()
        rec_all_loss_list = []
        auc_list = []
        pbar = tqdm(total=self.traindata_size)
        for data in self.train_dataloader:
            candidate_newsindex, user_index, user_clicked_newsindex, label, user_type_index, news_type_index = data
            loss, scores, rec_loss, emb_loss = self.KGAT_model(user_index, candidate_newsindex, label)
            self.optimize_KGAT(loss)
            rec_auc = self.cal_auc(scores, label)
            rec_all_loss_list.append(rec_loss.cpu().item())
            auc_list.append(rec_auc)
            pbar.update(self.args.batch_size)
        pbar.close()
        return mean(rec_all_loss_list), mean(auc_list)

    def _vaild_epoch(self):
        pbar = tqdm(total=self.vailddata_size)
        self.KGAT_model.eval()
        rec_auc_list = []
        with no_grad():
            for data in self.vaild_dataloader:
                candidate_newsindex, user_index, user_clicked_newsindex, label, user_type_index, news_type_index = data
                loss, scores, rec_loss, emb_loss = self.KGAT_model(user_index, candidate_newsindex, label)
                rec_auc = self.cal_auc(scores, label)
                rec_auc_list.append(rec_auc)
                pbar.update(self.args.batch_size)
            pbar.close()
        return mean(rec_auc_list)

    def _save_checkpoint(self, epoch):
        state_kgat = self.KGAT_model.state_dict()
        filename = self.args.checkpoint_dir + ('checkpoint-KGAT-epoch{}.pth'.format(epoch))
        torch.save(state_kgat, filename)

    def train(self):
        for epoch in range(1, self.args.epoch+1):
            loss, auc= self._train_epoch()
            print("epoch:{}--- loss:{}------auc:{}------".format(epoch, str(loss), str(auc)))
            if epoch % self.vaild_period == 0:
                print('start vaild ...')
                rec_auc = self._vaild_epoch()
                print("epoch:{}---vaild auc:{} ".format(epoch, str(rec_auc)))
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)
        self._save_checkpoint('final')

    def test(self):
        pbar = tqdm(total= self.testdata_size)
        self.KGAT_model.eval()
        pred_label_list = []

        pred_label_list = []
        news_index_list = []
        user_index_list = []
        user_type_list = []
        news_type_list = []
        candidate_newsindex_list = []
        candidate_newscategory_list = []

        with no_grad():
            for data in self.test_dataloader:
                candidate_newsindex, user_index, user_clicked_newsindex, user_type_index, news_type_index = data
                scores = self.KGAT_model.test(user_index, candidate_newsindex)
                pred_label_list.extend(scores.cpu().numpy())

                candidate_newscategory_list.extend(self.news_category_index[candidate_newsindex.cpu().numpy()])
                news_index_list.extend(candidate_newsindex.cpu().numpy()[:, 0])
                user_index_list.extend(user_index.cpu().numpy())
                user_type_list.extend(user_type_index.cpu().numpy())
                news_type_list.extend(news_type_index.cpu().numpy())
                candidate_newsindex_list.extend(candidate_newsindex.cpu().numpy()[:,0])

                pbar.update(self.args.batch_size)
            pred_label_list = np.vstack(pred_label_list)
            pbar.close()

        # 存储预测结果
        folder_path = '../predict/KGAT/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        predict_df = pd.DataFrame()
        predict_df['user'] = user_index_list
        predict_df['user_type'] = user_type_list
        predict_df['candidate_news'] = candidate_newsindex_list
        predict_df['candidate_news_type'] = news_type_list
        predict_df['score'] = pred_label_list[:, 0]
        predict_df['label'] = self.label_test[:len(user_index_list)]
        predict_df.to_csv('KGAT_predict.csv', index = False)


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
        w_AUC, w_MRR, w_nDCG5, w_nDCG10, w_len = evaluate_warm_cold_n_update(pred_label_list, user_index_list, news_index_list, 
                                                                             user_type_list, news_type_list,
                                                                             self.label_test, self.bound_test)
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





