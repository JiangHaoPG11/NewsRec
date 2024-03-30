import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


class Baseline_model(nn.Module):
    def __init__(self, args, news_title_embedding):
        super(Baseline_model, self).__init__()
        self.args = args
        #news_title_embedding = news_title_embedding.tolist()
        #news_title_embedding.append(np.random.normal(-0.1, 0.1, 768))
        self.news_title_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(news_title_embedding))

        self.news_fc1 = nn.Linear(400, self.args.embedding_dim)
        self.news_fc2 = nn.Linear(self.args.embedding_dim, self.args.embedding_dim)

        self.user_fc1 = nn.Linear(400, self.args.embedding_dim)
        self.user_fc2 = nn.Linear(self.args.embedding_dim, self.args.embedding_dim)

        self.predict = nn.Linear(2 * self.args.embedding_dim, 1)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, user_index, user_clicked_newindex, candidate_newsindex):
        candidate_newsindex = candidate_newsindex.to(self.device)
        user_clicked_newindex = user_clicked_newindex.to(self.device)
        user_clicked_newindex = user_clicked_newindex.unsqueeze(1).repeat(1, 5, 1)

        user_embedding = self.news_title_embedding(user_clicked_newindex)
        user_embedding = torch.div(torch.sum(user_embedding, dim=2), 10)
        user_embedding = torch.tanh(self.user_fc1(user_embedding))
        user_embedding = self.user_fc2(user_embedding)

        news_embedding = self.news_title_embedding(candidate_newsindex)
        news_embedding = torch.tanh(self.news_fc1(news_embedding))
        news_embedding = self.news_fc2(news_embedding)
        
        score = torch.sum(user_embedding * news_embedding, dim = -1)
        return score

    def test(self, user_index, user_clicked_newindex,  candidate_newsindex):
        candidate_newsindex = candidate_newsindex[:,0].to(self.device)
        user_clicked_newindex = user_clicked_newindex.to(self.device)
        user_embedding = self.news_title_embedding(user_clicked_newindex)
        user_embedding = torch.div(torch.sum(user_embedding, dim=1), 10)
        user_embedding = torch.tanh(self.user_fc1(user_embedding))
        user_embedding = self.user_fc2(user_embedding)

        news_embedding = self.news_title_embedding(candidate_newsindex)
        news_embedding = torch.tanh(self.news_fc1(news_embedding))
        news_embedding = self.news_fc2(news_embedding)

        score = torch.sum(user_embedding * news_embedding, dim = -1)

        return score

