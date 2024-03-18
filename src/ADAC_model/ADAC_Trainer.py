import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import no_grad
from utils.measure import *
from numpy import *
import os


class Trainer():
    def __init__(self, args, ADAC_model, optimizer_agent, data, device):
        self.args = args
        self.ADAC_model = ADAC_model
        self.optimizer_agent = optimizer_agent

        self.save_period = 100
        self.vaild_period = 30
        self.train_dataloader = data[0]
        self.test_dataloader = data[1]
        self.vaild_dataloader = data[2]
        self.news_embedding = data[3]
        self.entity_dict = data[6]
        self.entity_embedding = data[12]
        self.vailddata_size = data[-5]
        self.traindata_size = data[-4]
        self.testdata_size = data[-3]
        self.label_test = data[-2]
        self.bound_test = data[-1]

        self.device = device

    def cal_auc(self, score, label):
        rec_loss = F.cross_entropy(score, torch.argmax(label.to(self.device), dim=1))
        try:
            rec_auc = roc_auc_score(label.cpu().numpy(), F.softmax(score.cpu(), dim=1).detach().numpy())
        except ValueError:
            rec_auc = 0.5
        return rec_loss, rec_auc

    def optimize_agent(self, batch_rewards, q_values_steps, act_probs_steps, path_step_loss, meta_step_loss, rec_loss):
        all_loss_list = []
        demp_loss_list = []
        for i in range(len(batch_rewards)):
            batch_reward = batch_rewards[i]
            q_values_step = q_values_steps[i]
            act_probs_step = act_probs_steps[i]
            critic_loss, actor_loss = self.ADAC_model.step_update(
                act_probs_step, q_values_step, batch_reward
            )

            print('actor_loss:{}'.format(actor_loss.mean()))
            print('critic_loss:{}'.format(critic_loss.mean()))
            print('path_step_loss:{}'.format(path_step_loss[i].mean()))
            print('meta_step_loss:{}'.format(meta_step_loss[i].mean()))
            print('rec_loss:{}'.format(rec_loss))
            print('---------')
            all_loss_list.append(actor_loss.mean())
            all_loss_list.append(critic_loss.mean())
            demp_loss_list.append(path_step_loss[i].mean())
            demp_loss_list.append(meta_step_loss[i].mean())

        all_loss_list.append(rec_loss)
        all_loss_list.append(torch.stack(demp_loss_list).mean())
        self.optimizer_agent.zero_grad()
        if all_loss_list != []:
            loss = torch.stack(all_loss_list).sum()  # sum up all the loss
            loss.backward()
            self.optimizer_agent.step()

        #all_loss = all_loss + loss.data
        return loss

    def _train_epoch(self):
        self.ADAC_model.train()
        all_loss_list = []

        auc_list = []
        all_loss = 0

        pbar = tqdm(total=self.traindata_size)
        for data in self.train_dataloader:
            candidate_newsindex, user_index, user_clicked_newsindex, label, user_type_index, news_type_index = data
            act_probs_steps, q_values_steps, total_rewards, \
            path_nodes, path_relations, score, path_step_loss, meta_step_loss = self.ADAC_model(
                user_index, candidate_newsindex, user_clicked_newsindex
            )
            score = score.view(self.args.batch_size, -1)
            rec_loss, rec_auc = self.cal_auc(score, label)
            agent_loss = self.optimize_agent(total_rewards, q_values_steps, act_probs_steps, path_step_loss, meta_step_loss, rec_loss )
            all_loss_list.append(agent_loss.cpu().item())
            auc_list.append(rec_auc)
            pbar.update(self.args.batch_size)
            # torch.cuda.empty_cache()
        pbar.close()
        return mean(all_loss_list), mean(auc_list)

    def _vaild_epoch(self):
        pbar = tqdm(total=self.vailddata_size)
        self.ADAC_model.eval()
        rec_auc_list = []
        with no_grad():
            for data in self.vaild_dataloader:
                candidate_newsindex, user_index, user_clicked_newsindex, label, user_type_index, news_type_index = data
                act_probs_steps, q_values_steps, total_rewards, \
                path_nodes, path_relations, score, path_step_loss, meta_step_loss = self.ADAC_model(
                    user_index, candidate_newsindex, user_clicked_newsindex
                )

                score = score.view(self.args.batch_size, -1)
                _, rec_auc = self.cal_auc(score, label)
                rec_auc_list.append(rec_auc)
                pbar.update(self.args.batch_size)
            pbar.close()
        return mean(rec_auc_list)

    def _save_checkpoint(self, epoch):
        state = self.ADAC_model.state_dict()
        filename = self.args.checkpoint_dir + ('checkpoint-ADAC-epoch{}.pth'.format(epoch))
        torch.save(state, filename)

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

            predict_path = False
            if predict_path:
                path_nodes = []
                for data in self.train_dataloader:
                    candidate_newsindex, user_index, user_clicked_newsindex, label, user_type_index, news_type_index = data
                    
                    path_nodes.extend(
                        self.ADAC_model.get_path_list(
                            self.ADAC_model(
                                user_index, candidate_newsindex
                            )[3], self.args.batch_size
                        )
                    )
                    
                    if not os.path.exists("./ADAC_out"):
                        os.makedirs('./ADAC_out')
                    
                    fp_anchor_file = open("./ADAC_out/cand_file_" + str(epoch) + ".tsv", 'w', encoding='utf-8')
                    for i in range(self.args.batch_size):
                        fp_anchor_file.write(candidate_newsindex[i] + '\t' + ' '.join(list(set(path_nodes[i]))) + '\n')
        
        self._save_checkpoint('final')

    def test(self):
        pbar = tqdm(total= self.testdata_size)
        self.ADAC_model.eval()

        pred_label_list = []
        pred_label_list = []
        news_index_list = []
        user_index_list = []
        user_type_list = []
        news_type_list = []
        candidate_newsindex_list = []
        candidate_newscategory_list = []

        total_path_num = 0
        with no_grad():
            for data in self.test_dataloader:
                
                candidate_newsindex, user_index, user_clicked_newsindex, user_type_index, news_type_index = data
                _, path_num, _, best_score, best_path = self.ADAC_model.test(
                    user_index, candidate_newsindex, user_clicked_newsindex
                )

                best_score = best_score.view(self.args.batch_size, -1)

                total_path_num += path_num
                pred_label_list.extend(best_score.cpu().numpy())

                candidate_newscategory_list.extend(self.news_category_index[candidate_newsindex.cpu().numpy()])
                pred_label_list.extend(best_score.cpu().numpy())
                news_index_list.extend(candidate_newsindex.cpu().numpy()[:, 0])
                user_index_list.extend(user_index.cpu().numpy())
                user_type_list.extend(user_type_index.cpu().numpy())
                news_type_list.extend(news_type_index.cpu().numpy())
                candidate_newsindex_list.extend(candidate_newsindex.cpu().numpy()[:,0])

                pbar.update(self.args.batch_size)

            pred_label_list = np.vstack(pred_label_list)
            pbar.close()

        # 存储预测结果
        folder_path = '../predict/ADAC/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        predict_df = pd.DataFrame()
        predict_df['user'] = user_index_list
        predict_df['user_type'] = user_type_list
        predict_df['candidate_news'] = candidate_newsindex_list
        predict_df['candidate_news_type'] = news_type_list
        predict_df['score'] = pred_label_list[:, 0]
        predict_df['label'] = self.label_test[:len(user_index_list)]
        predict_df.to_csv('ADAC_predict.csv', index = False)

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



