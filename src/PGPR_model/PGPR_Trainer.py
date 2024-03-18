import torch
import torch.nn.functional as F
from AnchorKG_model.AnchorKG import *
from tqdm import tqdm
from torch.autograd import no_grad
from utils.measure import *
from numpy import *
import os
import pandas as pd

class Trainer():
    def __init__(self, args, Agent_model, optimizer_agent, data):
        self.args = args
        self.Agent_model = Agent_model
        self.optimizer_agent = optimizer_agent

        self.save_period = 100
        self.vaild_period = 40
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


    def get_batch_reward(self, step_reward):
        num_steps = len(step_reward)
        batch_rewards = None
        for i in range(num_steps):
            temp = torch.FloatTensor(step_reward[i]).unsqueeze(1)
            if i == 0:
                batch_rewards = temp
            else:
                batch_rewards = torch.cat([batch_rewards, temp], dim = -1)
        # batch_rewards = batch_rewards.to(self.device)
        for i in range(1, num_steps):
            batch_rewards[:, num_steps - i - 1] += 0.99 * batch_rewards[:, num_steps - i]
        news_batch_rewards = []
        for i in range(batch_rewards.shape[1]):
            news_batch_rewards.append(batch_rewards[: , i])
        return news_batch_rewards

    def cal_auc(self, score, label):
        rec_loss = F.cross_entropy(score.cpu(), torch.argmax(label, dim=1))
        try:
            rec_auc = roc_auc_score(label.cpu().numpy(), F.softmax(score.cpu(), dim=1).detach().numpy())
        except ValueError:
            rec_auc = 0.5
        return rec_loss, rec_auc

    def optimize_agent(self, batch_rewards, q_values_steps, act_probs_steps, rec_loss, all_loss):
        all_loss_list = []
        all_actor_loss = []
        all_critic_loss = []
        for i in range(len(batch_rewards)):
            batch_reward = batch_rewards[i]
            q_values_step = q_values_steps[i]
            act_probs_step = act_probs_steps[i]
            critic_loss, actor_loss = self.Agent_model.step_update(act_probs_step, q_values_step, batch_reward)

            all_actor_loss.append(actor_loss.cpu().mean())
            all_critic_loss.append(critic_loss.cpu().mean())
            all_loss_list.append(actor_loss.cpu().mean())
            all_loss_list.append(critic_loss.cpu().mean())

        all_loss_list.append(rec_loss)
        self.optimizer_agent.zero_grad()
        if all_loss_list != []:
            loss = torch.stack(all_loss_list).sum()  # sum up all the loss
            loss.backward()
            self.optimizer_agent.step()
        return loss

    def _train_epoch(self):
        self.Agent_model.train()
        all_loss_list = []

        auc_list = []
        all_loss = 0

        pbar = tqdm(total=self.traindata_size)
        for data in self.train_dataloader:
            candidate_newsindex, user_index, user_clicked_newsindex, label, user_type_index, news_type_index = data
            
            act_probs_steps, q_values_steps, step_rewards, \
            path_node, path_relation, score = self.Agent_model(
                user_index, candidate_newsindex, user_clicked_newsindex
            )

            score = score.view(self.args.batch_size, -1)

            rec_loss, rec_auc = self.cal_auc(score, label)

            batch_rewards = self.get_batch_reward(step_rewards)
            
            agent_loss = self.optimize_agent(
                batch_rewards, q_values_steps, act_probs_steps, 
                rec_loss, all_loss
            )
            
            all_loss = agent_loss
            all_loss_list.append(all_loss.cpu().item())
            auc_list.append(rec_auc)
            pbar.update(self.args.batch_size)
        pbar.close()
        return mean(all_loss_list), mean(auc_list)

    def _vaild_epoch(self):
        pbar = tqdm(total=self.vailddata_size)
        self.Agent_model.eval()
        rec_auc_list = []
        with no_grad():
            for data in self.vaild_dataloader:
                candidate_newsindex, user_index, user_clicked_newsindex, label, user_type_index, news_type_index = data
                
                act_probs_steps, q_values_steps, step_rewards, \
                anchor_graph, anchor_relation, score = self.Agent_model(
                    user_index, candidate_newsindex, user_clicked_newsindex
                )
                score = score.view(self.args.batch_size, -1)
                rec_loss, rec_auc = self.cal_auc(score, label)
                rec_auc_list.append(rec_auc)

                pbar.update(self.args.batch_size)
            pbar.close()
        return mean(rec_auc_list)

    def _save_checkpoint(self, epoch):
        state = self.Agent_model.state_dict()
        filename = self.args.checkpoint_dir + ('checkpoint-PGPR-epoch{}.pth'.format(epoch))
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
                        self.Agent_model.get_path_list(
                            self.Agent_model(
                                user_index, candidate_newsindex
                            )[3], self.args.batch_size
                        )
                    )

                    if os.path.exists("./PGPR_out"):
                        os.makedirs("./PGPR_out")

                    fp_anchor_file = open("./PGPR_out/cand_anchor_file_" + str(epoch) + ".tsv", 'w', encoding='utf-8')
                    for i in range(self.args.batch_size):
                        fp_anchor_file.write(candidate_newsindex[i] + '\t' + ' '.join(list(set(path_nodes[i]))) + '\n')
        self._save_checkpoint('final')

    def test(self):
        pbar = tqdm(total= self.testdata_size)
        self.Agent_model.eval()
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

                path_num, path_nodes, path_relations, \
                best_score, best_path = self.Agent_model.test(
                    user_index, candidate_newsindex, user_clicked_newsindex
                )

                total_path_num += path_num

                best_score = best_score.view(self.args.batch_size, -1)
                pred_label_list.extend(best_score.cpu().numpy())

                news_index_list.extend(candidate_newsindex.cpu().numpy()[:, 0])
                user_index_list.extend(user_index.cpu().numpy())
                user_type_list.extend(user_type_index.cpu().numpy())
                news_type_list.extend(news_type_index.cpu().numpy())
                candidate_newsindex_list.extend(candidate_newsindex.cpu().numpy()[:,0])
                candidate_newscategory_list.extend(self.news_category_index[candidate_newsindex.cpu().numpy()])

                pbar.update(self.args.batch_size)
            pred_label_list = np.vstack(pred_label_list)
            pbar.close()


        # 存储预测结果
        folder_path = '../predict/PGPR/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        predict_df = pd.DataFrame()
        predict_df['user'] = user_index_list
        predict_df['user_type'] = user_type_list
        predict_df['candidate_news'] = candidate_newsindex_list
        predict_df['candidate_news_type'] = news_type_list
        predict_df['score'] = pred_label_list[:, 0]
        predict_df['label'] = self.label_test[:len(user_index_list)]
        predict_df.to_csv('PGPR_predict.csv', index = False)


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



