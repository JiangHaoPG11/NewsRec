import pandas as pd
import numpy as np
import random
from nltk import word_tokenize
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from tqdm import tqdm

from sklearn.feature_extraction.text import CountVectorizer
import nltk
import seaborn as sns
from numpy.random import randn
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')

#nltk.download('stopwords')
### 读取用户历史交互记录
def load_data_behaviors():
    '''
    读取用户行为信息,新闻信息和新闻实体向量
    :return: DataFrame(user_behaviors_df),DataFrame(news_Info_df),DataFrame(entity_embedding_df)
    '''
    # user_behaviors_df = pd.read_table('../../database/MIND_small/MINDsmall_train/behaviors.tsv',
    #                                   header=None,
    #                                   names=['impression_id', 'user_id', 'time', 'history', 'impressions'],
    #                                   sep="\t", nrows = 20000 )
    user_behaviors_df = pd.read_table('../../database/MIND_small/MINDsmall_train/behaviors.tsv',
                                      header=None,
                                      names=['impression_id', 'user_id', 'time', 'history', 'impressions'],
                                      sep="\t")
    news_Info_df = pd.read_table('../../database/MIND_small/MINDsmall_train/news.tsv',
                                header=None,
                                names=['id', 'category', 'subcategory', 'title', 'abstract', 'url',
                                       'title_entities', 'abstract_entities'])
    entity_embedding_df = pd.read_table('../../database/MIND_small/MINDsmall_train/entity_embedding.vec', header=None)

    return user_behaviors_df, news_Info_df, entity_embedding_df

### 获取用户id和点击标签
def get_total_user_id_and_label(user_behaviors_df):
    '''
    针对用户印象日志,对于user_behaviors_df中的每一个用户提取正样本和负样本
    :param user_behaviors_df:
    :return: DataFrame(user_pnewsid_nnews_id_df): 包含用户对应的正样本和负样本新闻id
             List(total_newsid_list):所有新闻id,用于构建index
    '''
    impress_log = user_behaviors_df['impressions'].values.tolist()
    user_list = user_behaviors_df['user_id'].values.tolist()
    user_id_list = []
    # 获得用户的正新闻和负新闻,用于负采样
    user_pnews = []
    user_nnews = []
    user_pnews_list = []
    user_nnews_list = []
    ## 拆解新闻ID,用于获得新闻的index
    total_newsid_list = []
    ## 循环印象日志,得到正新闻id和负新闻id
    for i in range(len(impress_log)):
        line = impress_log[i]
        split = line.split(' ')
        for j in range(len(split)):
            temp = split[j]
            for z in range(len(temp)):
                if temp[z] != '-':
                    continue
                else:
                    break
            New_id = temp[0:z]
            Label = temp[-1]
            if Label == '1':
                user_pnews.append(New_id)
            if Label == '0':
                user_nnews.append(New_id)
            total_newsid_list.append(New_id)
        user_pnews_list.append(user_pnews)
        user_nnews_list.append(user_nnews)
        user_id_list.append(user_list[i])
        # 清空
        user_pnews = []
        user_nnews = []
    user_pnewsid_nnews_id_df = pd.DataFrame(data=None, columns= ['user_id', 'pnews_id', 'nnews_id'])
    user_pnewsid_nnews_id_df['user_id'] = user_id_list
    user_pnewsid_nnews_id_df['pnews_id'] = user_pnews_list
    user_pnewsid_nnews_id_df['nnews_id'] = user_nnews_list
    # print('---------total_newsid_list-------------')
    # print(total_newsid_list)
    return user_pnewsid_nnews_id_df, total_newsid_list

### 获取用户index和点击新闻id
def get_user_index( user_behaviors_df, user_pnewsid_nnews_id_df):
    '''
    构建用户字典,得到用户历史点击新闻id,并进行采样
    :param user_behaviors_df:
    :param user_pnewsid_nnews_id_df:
    :return:  DataFrame(user_pnewsid_nnews_id_df): 对应用户的正样本新闻和负样本新闻
              DataFrame(user_clicked_df):对应用户的点击新闻
              List(user_clicked_newsid_list):用户点击新闻list
              Dict(user_dict):用户字典
    '''
    # 获取用户id,得到用户index
    user_id_list = user_pnewsid_nnews_id_df['user_id'].values.tolist()
    user_index_list = []
    total_user_index_list = []
    temp = []
    user_clicked_newsid_list = []
    # 创建用户字典
    user_dict = {}
    # 获取用户index和对应的点击新闻
    user_id_set_list = list(set(user_id_list))
    index = 0
    for i in range(len(user_id_set_list)):
        if (user_id_set_list[i] not in user_dict.keys()) == True :
            user_dict[user_id_set_list[i]] = index
            user_index_list.append(index)
            index += 1
        temp1 = user_behaviors_df[user_behaviors_df['user_id'] == user_id_set_list[i]]
        state = temp1['history'].values.tolist()[0]
        state = str(state)
        if state != 'nan':
            line = temp1['history'].values.tolist()[0]
        else:
            print(1)
        split = line.split(' ')
        for j in range(len(split)):
            temp.append(split[j])
        user_clicked_newsid_list.append(temp)
        temp = []

    # 将映射字典存储在df中
    user_index_df = pd.DataFrame.from_dict(user_dict, orient='index', columns=['user_index'])
    user_index_df = user_index_df.reset_index()
    print('总共的用户数量{}'.format(user_index_df.shape[0]))
    # print(news_index_df)
    user_index_df.to_csv('../../data/df/user_index_df.csv', index=False)

    ## 获得用户点击新闻id和对应用户表示的df
    user_clicked_df = pd.DataFrame()
    user_clicked_df['user_id'] = user_id_set_list
    user_clicked_df['user_index'] = user_index_list
    user_clicked_df['user_clicked_newsid'] = user_clicked_newsid_list
    # print('--------------user_clicked_df-------------')
    # print(user_clicked_df)

    ## 映射用户index到预测df
    for i in range(len(user_id_list)):
        total_user_index_list.append(user_dict[user_id_list[i]])
    user_pnewsid_nnews_id_df['user_index'] = total_user_index_list
    user_pnewsid_nnews_id_df.to_csv('../../data/df/user_pnewsid_nnews_id_df.csv')
    # print('--------------user_pnewsid_nnews_id_df-------------')
    # print(user_pnewsid_nnews_id_df)
    print('总的用户数{}'.format(len(user_index_list)))
    return user_pnewsid_nnews_id_df, user_clicked_df, user_clicked_newsid_list, user_dict

# 计算冷热用户
def cal_user_clicked_news_ratio(user_dict, user_pnewsid_nnews_id_df, user_clicked_newid_list, new_dict):
    # 获取冷热用户（正确版）
    user_clicked_news_num_list = []
    for i in range(len(user_clicked_newid_list)):
        user_clicked_news_num_list.append(len(user_clicked_newid_list[i]))

    user_clicked_newid_list_origin = user_clicked_newid_list
    # 获取冷热用户（测试版）
    # user_clicked_newid_list = user_pnewsid_nnews_id_df['pnews_id'].values.tolist()
    # user_clicked_newid_list = user_clicked_newid_list[:int(len(user_clicked_newid_list) * 0.7)]
    ############

    user_clicked_news_num_list = []
    for i in range(len(user_clicked_newid_list)):
        user_clicked_news_num_list.append(len(user_clicked_newid_list[i]))

    # 计算历史点击新闻个数
    user_clicked_news_num_list_sort = list(sorted(user_clicked_news_num_list))

    # 画图
    c1, c2 = sns.color_palette('Set1', 2)
    plt.figure(dpi=80,figsize=(12.5, 10))
    plt.rcParams['pdf.fonttype'] = 42
    sns.kdeplot(user_clicked_news_num_list_sort, shade=True, color = c2,label = 'user clicked nums')
    # plt.legend(loc='best', fontsize=10)  # 让图例生效
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.xlabel("clicks num", fontsize=40)  # X轴标签
    plt.legend(fontsize=30)
    plt.title("User Clicks Distribution", fontsize=50)  # 标题
    plt.grid()  # 添加网格
    # plt.show()
    plt.savefig("User_cold_stastic.png")
    plt.close()

    # 获取冷热用户：
    user_indictor = []
    user_clicked_news_index = []
    for i in range(len(user_clicked_newid_list_origin)):
        # if i < len(user_clicked_newid_list):
        if len(user_clicked_newid_list[i]) >= 10:
            user_indictor.append('w')
        elif len(user_clicked_newid_list[i]) <= 1:
            user_indictor.append('c')
        else:
            user_indictor.append('n')
        temp = []
        for news_id in user_clicked_newid_list[i]:
            temp.append(new_dict[news_id])
        user_clicked_news_index.append(temp)
        # 测试版
        # else:
        #     user_indictor.append('c')
        #     user_clicked_news_index.append([])

    # 将映射字典存储在df中
    user_index_df = pd.DataFrame.from_dict(user_dict, orient='index', columns=['user_index'])
    user_index_df['click_news'] = user_clicked_news_index
    user_index_df['w/c'] = user_indictor
    user_index_df = user_index_df.reset_index()
    print('总共的用户数量{}'.format(user_index_df.shape[0]))
    user_index_df.to_csv('../../data/df/user_index_df.csv', index=False)
    # 构建用户类型np 0表示新用户 1 表示老用户 2表示正常用户
    user_type = []
    cold_user_num = 0
    warm_user_num = 0
    norm_user_num = 0
    for type in user_indictor:
        if type == 'c':
            user_type.append(0)
            cold_user_num += 1
        elif type == 'w':
            user_type.append(1)
            warm_user_num += 1
        else:
            user_type.append(2)
            norm_user_num += 1
    np.save('../../data/metadata/user_type', np.array(user_type))
    print('热用户数量{}'.format(warm_user_num))
    print('冷用户数量{}'.format(cold_user_num))
    print('正常用户数量{}'.format(norm_user_num))

### 计算新闻点击率
def cal_news_clicked_user_ratio(user_pnewsid_nnews_id_df, user_clicked_df, candidate_newsid_list, news_dict):

    candidate_newsindex = np.load('../../data/metadata/candidate_newsindex.npy')

    user_index_list = user_pnewsid_nnews_id_df['user_id'].tolist()
    pnews_list = user_pnewsid_nnews_id_df['pnews_id'].tolist()
    nnews_list = user_pnewsid_nnews_id_df['nnews_id'].tolist()

    # 只取训练级的样本
    pnews_list = pnews_list[:int(len(user_index_list) * 0.7)]
    nnews_list = nnews_list[:int(len(user_index_list) * 0.7)]
    exposure_newsindex = candidate_newsindex[:int(len(user_index_list) * 0.7)]
    user_index_list = user_index_list[:int(len(user_index_list) * 0.7)]
    print(len(exposure_newsindex))
    print(len(user_index_list))
    
    # 初始化
    news_exposure_dict = {}
    for i in range(len(news_dict.values())):
        news_exposure_dict[i] = []
    
    for i in range(len(exposure_newsindex)):
        exposure_newsindex_one = exposure_newsindex[i]
        for news in exposure_newsindex_one:
            if news in news_exposure_dict.keys():
                news_exposure_dict[news].append(user_index_list[i])

    news_exposure_num_list = []
    for key, value in news_exposure_dict.items():
        news_exposure_num_list.append(len(value))

    margin = 0
    news_cold_size = int(len(news_dict))
    print('冷新闻最大曝光数：{}'.format(margin))
    print(news_cold_size)

    # 画图
    c1, c2 = sns.color_palette('Set1', 2)
    # plt.legend(loc='best', fontsize=10)  # 让图例生效
    plt.figure(dpi=80,figsize=(12.5,10))
    plt.rcParams['pdf.fonttype'] = 42
    sns.kdeplot(news_exposure_num_list, shade=True, color = c1, label = 'news clicked nums')
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.xlabel("clicked num", fontsize=40)  # X轴标签
    #plt.ylabel('Density', fontsize=20)  # Y轴标签
    plt.title("News Clicked Distribution", fontsize=50)  # 标题
    plt.grid()  # 添加网格
    plt.show()
    plt.legend(fontsize=30)
    plt.savefig("News_cold_stastic.png")
    plt.close()

    # 计算冷热新闻：
    news_index_list = []
    news_user_index_list =[]
    news_indictor = []
    for key, value in news_exposure_dict.items():
        news_index_list.append(key)
        news_user_index_list.append(value)

    for key, value in news_exposure_dict.items():
        if len(value) == 0:
            news_indictor.append('c')
        elif len(value) >= 10:
            news_indictor.append('w')
        else:
            news_indictor.append('n')

    # 将映射字典存储在df中
    news_index_df = pd.DataFrame.from_dict(news_dict,  orient='index', columns=['news_index'])
    news_index_df['news'] = news_user_index_list
    news_index_df['w/c'] = news_indictor
    news_index_df = news_index_df.reset_index()
    print('总共的新闻标题数量{}'.format(news_index_df.shape[0]))
    # print(new_index_df)
    news_index_df.to_csv('../../data/df/news_index_df.csv', index=False)

    # 构建新闻类型np 0表示新新闻 1 表示老新闻
    news_type = []
    warm_news_num = 0
    cold_news_num = 0
    normal_news_num = 0
    for type in news_indictor:
        if type == 'c':
            news_type.append(0)
            cold_news_num += 1
        elif type == 'w':
            news_type.append(1)
            warm_news_num += 1
        else:
            news_type.append(2)
            normal_news_num += 1
    np.save('../../data/metadata/news_type', np.array(news_type))
    print('热新闻数量{}'.format(warm_news_num))
    print('冷新闻数量{}'.format(cold_news_num))
    print('正常新闻数量{}'.format(normal_news_num))

### 计算新闻的流行度
def cal_news_popularity(user_clicked_df, user_pnewsid_nnews_id_df, new_dict):
    news_id_list = new_dict.keys()
    # 合并列表
    total_newid_list = []
    user_clicked_newid_list = user_clicked_df['user_clicked_newsid'].tolist()
    for id_list in user_clicked_newid_list:
        total_newid_list.extend(id_list)
    user_pnewid_list = user_pnewsid_nnews_id_df['pnews_id'].tolist()
    for id_list in user_pnewid_list:
        total_newid_list.extend(id_list)
    user_nnewid_list = user_pnewsid_nnews_id_df['nnews_id'].tolist()
    for id_list in user_nnewid_list:
        total_newid_list.extend(id_list)
    # 计算流行度
    max = 0
    pop_list = []
    for id in news_id_list:
        pop = total_newid_list.count(id)
        if pop > max:
            max = pop
        pop_list.append(pop)
    # 流行度归一化
    pop_list = [x / max for x in pop_list]
    np.save('../../data/metadata/news_popularity', np.array(pop_list))

### 获取用户点击新闻id和候选新闻id映射得到index
def get_index_clicked_newsid_cand_newsid( user_clicked_newsid_list, total_newsid_list):
    '''
    合并用户点击新闻和正负样本新闻list,然后映射得到对应的新闻index
    :param user_clicked_newsid_list:
    :param total_newsid_list:
    :return: dict(news_dict): 新闻ID和新闻index对应字典
    '''
    temp1 = []
    temp2 = []
    for i in range(len(user_clicked_newsid_list)):
        for item in user_clicked_newsid_list[i]:
            temp1.append(item)
    for i in range(len(total_newsid_list)):
        temp2.append(total_newsid_list[i])
    news_id_list = temp1 + temp2

    # 获取候选新闻和用户点击新闻对应的映射字典
    news_dict = {}
    index = 0
    for news_id in news_id_list:
        if (news_id not in news_dict.keys()) == True:
            news_dict[news_id] = index
            index += 1
    news_dict['padding'] = index

    # 将映射字典存储在df中
    news_index_df = pd.DataFrame.from_dict(news_dict, orient='index', columns=['news_index'])
    news_index_df = news_index_df.reset_index()
    print('总共的新闻标题数量{}'.format(news_index_df.shape[0]))
    # print(news_index_df)
    news_index_df.to_csv('../../data/df/news_index_df.csv',index= False)
    return news_dict

### 获取测试集数据
def get_test_news_user(news_dict, user_pnewsid_nnews_id_df):
    '''
    构建用于测试模型的数据:test_user_index,test_candidate_newsindex,test_label,test_bound
    :param news_dict:
    :param user_pnewsid_nnews_id_df:
    :return:
    '''
    pnewsid_list = user_pnewsid_nnews_id_df['pnews_id'].values.tolist()
    nnewsid_list = user_pnewsid_nnews_id_df['nnews_id'].values.tolist()
    userindex_list = user_pnewsid_nnews_id_df['user_index'].values.tolist()

    newsindex_list = []
    userindex = []
    label_list = []
    Bound = []
    index = 0
    for i in range(user_pnewsid_nnews_id_df.shape[0]):
        start = index
        for pnewsid in pnewsid_list[i]:
            temp = [news_dict[pnewsid], 0, 0, 0, 0]
            newsindex_list.append(temp)
            userindex.append(userindex_list[i])
            label_list.append(1)
            index += 1
        for nnewsid in nnewsid_list[i]:
            temp = [news_dict[nnewsid], 0, 0, 0, 0]
            newsindex_list.append(temp)
            userindex.append(userindex_list[i])
            label_list.append(0)
            index += 1
        Bound.append([start,index])
    np.save('../../data/test/test_user_index.npy', np.array(userindex))
    np.save('../../data/test/test_candidate_newsindex.npy', np.array(newsindex_list))
    np.save('../../data/test/test_label.npy', np.array(label_list))
    np.save('../../data/test/test_bound.npy', np.array(Bound))

### 负采样
def negtivate_sample(user_pnewsid_nnews_id_df, news_dict):
    '''
    进行负采样,采样印象日志当中出现但是用户没有点击的样本作为负样本
    :param user_pnewsid_nnews_id_df:
    :param news_dict:
    :return: DataFrame(newsindex_label_df):用户对应的采样正负样本以及label
    '''
    ## 映射正样本新闻id
    pnewsid_list = user_pnewsid_nnews_id_df['pnews_id']
    pnews_index_list_one = []
    pnews_index_list = []
    for pnewsid in pnewsid_list:
        for pnews in pnewsid:
            pnews_index = news_dict[pnews]
            pnews_index_list_one.append(pnews_index)
        pnews_index_list.append(pnews_index_list_one)
        pnews_index_list_one=[]

    ## 映射负样本新闻id
    nnews_index_list_one = []
    nnews_index_list = []
    nnewsid_list = user_pnewsid_nnews_id_df['nnews_id']
    for nnewsid in nnewsid_list:
        for nnews in nnewsid:
            nnews_index = news_dict[nnews]
            nnews_index_list_one.append(nnews_index)
        nnews_index_list.append(nnews_index_list_one)
        nnews_index_list_one=[]

    ## 填充负样本
    for nnewsindex in nnews_index_list:
        if len(nnewsindex) < 4:
            need_len = 4 - len(nnewsindex)
            for i in range(need_len):
                nnewsindex.append(random.sample(nnewsindex,1)[0])

    ## 选取用户
    user_index = []
    candidate_newsindex_list = []
    label = [1, 0 , 0 , 0, 0]
    label_list = []

    user_index_list = user_pnewsid_nnews_id_df['user_index'].values.tolist()
    for i in range(len(user_index_list)):
        user_index_temp = user_index_list[i]
        for pnews_index  in pnews_index_list[i]:
            candidate_newsindex = []
            candidate_newsindex.append(pnews_index)
            candidate_newsindex = candidate_newsindex + random.sample(nnews_index_list[i], 4)
            ## shuffle新闻集
            candidate_order = list(range(4 + 1))
            random.shuffle(candidate_order)
            ## shuffle列表
            candidate_shuffle = []
            label_list_shuffle = []
            for i in candidate_order:
                candidate_shuffle.append(candidate_newsindex[i])
                label_list_shuffle.append(label[i])
            # 标签
            label_list.append(label_list_shuffle)
            # 候选新闻index
            candidate_newsindex_list.append(candidate_shuffle)
            # 用户index
            user_index.append(user_index_temp)
    newsindex_label_df = pd.DataFrame()
    newsindex_label_df['user_index'] = user_index
    newsindex_label_df['candidate_newsindex'] = candidate_newsindex_list
    newsindex_label_df['label'] = label_list
    newsindex_label_df.to_csv('../../data/df/newsindex_label_df.csv')

    ## 保存用户index列表
    user_index = np.array(user_index)
    # print('-----------user_index.npy--------------')
    # print(user_index)
    np.save('../../data/metadata/user_index.npy', user_index)

    ## 保存候选新闻index列表
    candidate_newsindex = np.array(candidate_newsindex_list)
    # print('-----------candidate_newsindex.npy--------------')
    # print(candidate_newsindex)
    np.save('../../data/metadata/candidate_newsindex.npy', candidate_newsindex)

    ## 保存label列表
    label = np.array(label_list)
    # print('-----------label.npy--------------')
    # print(label)
    np.save('../../data/metadata/label.npy', label)

    return newsindex_label_df

### 构建用户点击新闻和候选新闻index
def construct_clicked_newsindex_cand_newsindex( user_clicked_df,  user_clicked_newsid_list, news_dict, maxlen_clicked ):
    '''
    根据新闻字典,将用户点击新闻ID映射成index
    :param user_clicked_df:
    :param user_clicked_newsid_list:
    :param news_dict:
    :return: DataFrame(newsid_Label_df):
    '''
    user_clicked_newsindex_in = []
    user_clicked_newsindex_list = []

    for i in range(len(user_clicked_newsid_list)):
        user_clicked_newsid_list[i] = list(set(user_clicked_newsid_list[i]))
        for item in user_clicked_newsid_list[i]:
            temp = news_dict[item]
            user_clicked_newsindex_in.append(temp)
        user_clicked_newsindex_list.append(user_clicked_newsindex_in)
        user_clicked_newsindex_in = []

    # 记录用户实际点击数
    temp = []
    for i in range(len(user_clicked_newsindex_list)):
        if len(user_clicked_newsindex_list[i]) >= maxlen_clicked:
            temp.append(maxlen_clicked)
        else:
            temp.append(len(user_clicked_newsindex_list[i]))
    # 用户点击padding
    print('最大点击数{}'.format(maxlen_clicked))
    padding = news_dict['padding']
    for i in range(len(user_clicked_newsindex_list)):
        if len(user_clicked_newsindex_list[i]) >= maxlen_clicked:
            user_clicked_newsindex_list[i] = random.sample(user_clicked_newsindex_list[i] ,maxlen_clicked)
        if len(user_clicked_newsindex_list[i]) < maxlen_clicked:
            need_len = maxlen_clicked - len(user_clicked_newsindex_list[i])
            for j in range(need_len):
                user_clicked_newsindex_list[i].append(padding)

    user_clicked_newsindex = np.array(user_clicked_newsindex_list)
    print(len(user_clicked_newsindex_list))
    # print('--------------user_clicked_newsindex---------------')
    # print(user_clicked_newsindex)
    np.save('../../data/metadata/user_clicked_newsindex', user_clicked_newsindex)

    user_clicked_df['user_clicked_newsindex'] = user_clicked_newsindex_list
    user_clicked_df.to_csv('../../data/df/user_clicked_df.csv',index= False)

    # 构建用户邻居字典
    user_index_list = user_clicked_df['user_index'].tolist()
    user_clicked_newsindex_list = user_clicked_df['user_clicked_newsindex'].tolist()
    user_one_hop_dict = {}
    for i in range(len(user_index_list)):
        user_index = user_index_list[i]
        user_one_hop_dict[user_index] = user_clicked_newsindex_list[i]
    user_one_hop_dict[len(user_index_list)] = [padding] * maxlen_clicked

    return user_clicked_df, user_one_hop_dict

### 获取标题文字
def get_title_word(news_dict, news_Info_df):
    '''
    获取新闻标题文字
    :param news_dict:
    :param news_Info_df:
    :return: DataFrame(news_Info_df_select):根据新闻id选取的新闻
             List(word_total_list_t):所有新闻标题的单词List
    '''
    New_id_list = list(news_dict.keys())
    news_Info_df_select = pd.DataFrame(data=None, columns=['id', 'category', 'subcategory', 'title', 'abstract', 'url',
                                                          'title_entities', 'abstract_entities'])
    for New_id in New_id_list:
        df_select = news_Info_df[news_Info_df['id'] == New_id]
        news_Info_df_select = pd.concat([news_Info_df_select, df_select], axis=0)
    word_list = []
    word_total_list_t = []
    title_word_list = news_Info_df_select['title'].tolist()

    for title_word in title_word_list:
        words = word_tokenize(title_word)
        interpunctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', "'"]
        cutwords2 = [word for word in words if word not in interpunctuations]
        stops = set(stopwords.words("english"))
        cutwords3 = [word for word in cutwords2 if word not in stops]
        for word in cutwords3:
            word_list.append(word)
        word_total_list_t.append(word_list)
        word_list = []
    word_total_list_t.append(['padding'])

    # for title_word in title_word_list:
    #     split = title_word.split(' ')
    #     for word in split:
    #         word_list.append(word)
    #     word_total_list_t.append(word_list)
    #     word_list = []
    # word_total_list_t.append(['padding'])

    news_title = news_Info_df_select['title'].values.tolist()
    news_abstract = news_Info_df_select['abstract'].values.tolist()
    news_title.append('padding')
    news_abstract.append('padding')
    np.save('../../data/pretrain-data/news_title.npy', np.array(news_title))
    np.save('../../data/pretrain-data/news_abstract.npy', np.array(news_abstract))
    return news_Info_df_select, word_total_list_t, np.array(news_title)

## 获取标题嵌入
def get_news_embedding (news_title):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained("bert-base-uncased")

    title_text = news_title.tolist() 
    batch_size = 50

    news_title_embedding = []
    for i in tqdm(range( int(len(title_text) / batch_size) + 1 ), desc="Processing News Embedding"):
        if (i + 1) * batch_size > len(title_text):
            batch_title_text = title_text[ i * batch_size: ]
        else:
            batch_title_text = title_text[ i * batch_size: (i + 1) * batch_size ]

        batch_encoded_input = tokenizer(
            batch_title_text, padding=True, truncation=True, return_tensors='pt'
        )

        output = model(
            input_ids = batch_encoded_input["input_ids"], 
            attention_mask = batch_encoded_input["attention_mask"], 
            token_type_ids = batch_encoded_input["token_type_ids"] 
        )

        news_title_embedding.extend(output[1].detach().numpy())

    # add for padding
    assert len(news_title_embedding) == len(title_text)
    news_title_embedding.append(np.random.rand(len(news_title_embedding[-1])))
    # news_title_embedding = []
    # for i in range(len(title_text)+1):
    #     news_title_embedding.append(np.random.rand(400))
    np.save('../../data/metadata/news_title_embedding.npy', np.array(news_title_embedding))

### 获取标题文字index
def get_title_word_index (word_total_list_t, news_Info_df_select):
    '''
    将新闻单词映射到index
    :param word_total_list_t:
    :param news_Info_df_select:
    :return:
            DataFrame(news_Info_df_select):选择的新闻信息
            List(title_word_index):单词对应的index
            max_len:新闻标题的最大长度（用于对齐）
    '''
    word_dict = {}
    index = 0
    word_index_list_one = []
    word_index_list = []
    for word_total_list_t_one in word_total_list_t:
        for word in word_total_list_t_one:
            if (word not in word_dict.keys()) == True:
                word_dict[word] = index
                index += 1
    for word_total_list_t_one in word_total_list_t:
        for word in word_total_list_t_one:
            word_index_list_one.append(word_dict[word])
        word_index_list.append(word_index_list_one)
        word_index_list_one = []
    word_dict['padding'] = index
    ## 构建映射矩阵
    word_index_df = pd.DataFrame(pd.Series(word_dict), columns=['word_id'])
    word_index_df = word_index_df.reset_index().rename(columns={'index': 'word'})
    word_index_df.to_csv('../../data/df/word_index_df.csv', index= False)
    news_Info_df_select['title_word_id'] = word_index_list[:len(word_index_list)-1]
    title_word_index = word_index_list
    ## 记录单词长度
    news_title_length = []
    for i in range(len(title_word_index)):
        news_title_length.append(len(title_word_index[i]))
    news_title_length = np.array(news_title_length)
    np.save('../../data/metadata/news_title_length.npy', news_title_length)
    ## 求最大长度
    max_len = 0
    for i in range(len(title_word_index)):
        if len(title_word_index[i]) > max_len:
            max_len = len(title_word_index[i])

    ## 按照最大长度补全
    empty_word= [word_index_df.shape[0]-1] * max_len
    for i in range(len(title_word_index)):
        if len(title_word_index [i]) == 0:
            title_word_index [i].extend(empty_word)
        elif len(title_word_index[i]) < max_len:
            need_len = max_len - len(title_word_index[i])
            title_word_index[i].extend([word_index_df.shape[0]-1] * need_len)
    title_word_index = np.array(title_word_index)
    np.save('../../data/metadata/news_title_word_index.npy', title_word_index)

    # print('-----------word_index_df------------')
    # print(word_index_df)
    # print('------------title_word_index-------------')
    # print(title_word_index)
    print('词袋数 {}'.format(word_index_df.shape[0]))
    print('新闻标题最大长度 {}'.format(max_len))

    return news_Info_df_select, title_word_index, max_len

### 读取新闻信息,提取每一个新闻的标题实体ID和摘要实体ID
def get_title_and_abstract_entity_id(news_Info_df_select):
    '''
    读取新闻信息,提取每一个新闻的标题实体ID和摘要实体ID
    :param news_Info_df_select:
    :return: DataFrame(news_Info_df_select) :加入提取标题实体ID和摘要实体ID的新闻信息
             List(entity_id_total_list_a):摘要实体ID
             List(entity_id_total_list_t):标题实体ID
    '''
    title_entities_list = news_Info_df_select['title_entities'].values.tolist()
    abstract_entities_list = news_Info_df_select['abstract_entities'].values.tolist()
    ## 提取标题实体ID
    entity_id_total_list_t = []
    for title_entities in title_entities_list:
        entity_id_list_t = []
        if type(title_entities) == float:
            entity_id_list_t.append('padding')
        else:
            if len(title_entities) == 2:
                entity_id_list_t.append('padding')
            else:
                temp = eval(title_entities)
                for i in range(len(temp)):
                    entity_id = temp[i]['WikidataId']
                    entity_id_list_t.append(entity_id)

        entity_id_total_list_t.append(entity_id_list_t)

    entity_id_total_list_a = []
    for abstract_entities in abstract_entities_list:
        entity_id_list_a = []
        if type(abstract_entities) == float:
            entity_id_list_a.append('padding')
        else:
            if len(abstract_entities) == 2:
                entity_id_list_a.append('padding')
            else:
                temp = eval(abstract_entities)
                for i in range(len(temp)):
                    entity_id = temp[i]['WikidataId']
                    entity_id_list_a.append(entity_id)
        entity_id_total_list_a.append(entity_id_list_a)

    return news_Info_df_select, entity_id_total_list_t, entity_id_total_list_a

### 将标题实体ID和摘要实体ID列表合并
def merge_title_and_abstract_list(entity_information_t ,entity_information_a, news_Info_df_select):
    '''
    合并实体ID和摘要实体ID
    :param entity_id_total_list_a:
    :param entity_id_total_list_t:
    :param news_Info_df_select:
    :return:
        DataFrame(news_Info_df_select):加入合并实体的新闻信息
         List(entity_id_list):合并后的实体IDlist
    '''
    entity_id_list = []
    for i in range(len(entity_information_a)):
        temp = entity_information_a[i]
        for entity_id in entity_information_t[i]:
            temp.append(entity_id)
        entity_id_list.append(temp)
    news_Info_df_select['entity_id'] = entity_id_list
    entity_index_df = pd.DataFrame()
    entity_index_df['entity_id'] = entity_id_list
    entity_index_df.to_csv('../../data/df/entity_index_df.csv')
    return news_Info_df_select, entity_id_list

### 获取实体id映射,方便提取实体嵌入
def map_entity_information(entity_id_list, news_Info_df_select, max_len):
    '''
    映射新闻实体ID
    :param entity_id_list:
    :param news_Info_df_select:
    :param max_len:
    :return: DataFrame(entity_index_df):实体ID和实体index df
    '''
    # 映射id
    id_dict = {}
    id_dict['padding'] = 0
    index = 1
    entity_index_list = []
    for entity_id_list_one in entity_id_list:
        entity_index_list_one = []
        for entity_id in entity_id_list_one:
            if (entity_id not in id_dict.keys()) == True :
                id_dict[entity_id] = index
                index = index + 1
            entity_index = id_dict[entity_id]
            entity_index_list_one.append(entity_index)
        entity_index_list.append(entity_index_list_one)
    news_Info_df_select['entity_index'] = entity_index_list
    entity_index_df = pd.DataFrame(pd.Series(id_dict))
    entity_index_df = entity_index_df.reset_index()
    entity_index_df.columns = ['id','index']
    entity_index_df.to_csv('../../data/df/entity_index_df.csv')
    entity_index = news_Info_df_select['entity_index'].values.tolist()
    print('单个新闻最大实体个数{}'.format(max_len))
    empty_entity = [id_dict['padding']] * max_len
    for i in range(len(entity_index)):
        if len(entity_index[i]) == 0:
            entity_index[i].extend(empty_entity)
        elif len(entity_index[i])  < max_len:
            need_len = max_len - len(entity_index[i])
            entity_index[i].extend([id_dict['padding']] * need_len)
        entity_index[i] = [int(entity_index[i][a]) for a in range(max_len)]
        entity_index[i] = np.array(entity_index[i])
        if len(entity_index[i]) != max_len:
            print('error')

    entity_index.append(empty_entity)
    entity_index = np.array(entity_index)
    np.save('../../data/metadata/news_entity_index.npy', entity_index)
    return entity_index_df, id_dict

### 提取实体嵌入向量
def extract_entity_vector(entity_index_df, entity_embedding_df):
    '''
    提取实体嵌入向量
    :param entity_index_df:
    :param entity_embedding_df:
    :return:
        DataFrame(entity_index_df):加入实体向量之后的entity_index_df
        List(embedding_list)
    '''
    ## 加载实体嵌入
    entity_embedding_df['vector'] = entity_embedding_df.iloc[:, 1:101].values.tolist()
    entity_embedding_df = entity_embedding_df[[0,'vector']].rename(columns={0: "entity"})
    ## 加载映射dataframe,按照映射id提取实体嵌入
    entity_id_list = entity_index_df['id'].values.tolist()
    entity_embedding_list = []
    empty_id = 100 * [0]
    for entity_id in entity_id_list:
        entity_embedding = entity_embedding_df[entity_embedding_df['entity'] == entity_id]['vector'].values
        if len(entity_embedding) == 0:
            entity_embedding_list.append(empty_id)
        else:
            entity_embedding_list.append(entity_embedding[0])
    entity_index_df ['entity_embedding'] = entity_embedding_list
    entity_index_df = entity_index_df.fillna(method='ffill')
    embedding_list = entity_index_df['entity_embedding'].values.tolist()
    # embedding_list.append(empty_id)
    entity_embedding_index = np.array(embedding_list)
    np.save('../../data/metadata/news_entity_embedding.npy', entity_embedding_index)

    return entity_index_df

### 获取主题
def get_category_index(news_Info_df_select):
    '''
    获取新闻主题
    :param news_Info_df_select:
    :return:
        DataFrame(news_Info_df_select):加入新闻主题index的新闻信息df
    '''
    index = 0
    category_dict = {}
    category_list = news_Info_df_select['category'].values.tolist()
    for category in category_list:
        if (category not in category_dict.keys()) == True:
            category_dict[category] = index
            index += 1
    category_dict['padding'] = index
    category_index_list = []
    for i in range(len(category_list)):
        category = category_list[i]
        category_index_list.append(category_dict[category])

    news_Info_df_select['category_index'] = category_index_list
    category_index_list.append(category_dict['padding'])
    category_index_df = pd.DataFrame(pd.Series(category_dict), columns=['category_index'])
    category_index_df = category_index_df.reset_index().rename(columns={'index': 'category'})
    print('主题个数{}'.format(category_index_df.shape[0]))

    np.save('../../data/metadata/news_category_index.npy', np.array(category_index_list))
    np.save('../../data/metadata/category_index.npy', np.array(category_index_df['category_index'].values.tolist()))
    category_index_df.to_csv('../../data/df/category_index_df.csv', index=False)

    return news_Info_df_select, category_dict

### 获取子主题
def get_subcategory_index(news_Info_df_select):
    '''
        获取新闻副主题
        :param news_Info_df_select:
        :return:
            DataFrame(news_Info_df_select):加入新闻副主题index的新闻信息df
    '''
    index = 0
    subcategory_dict = {}
    subcategory_list = news_Info_df_select['subcategory'].values.tolist()
    for subcategory in subcategory_list:
        if (subcategory not in subcategory_dict.keys()) == True:
            subcategory_dict[subcategory] = index
            index += 1
    subcategory_dict['padding'] = index
    subcategory_index_list = []

    for i in range(len(subcategory_list)):
        subcategory = subcategory_list[i]
        subcategory_index_list.append(subcategory_dict[subcategory])

    news_Info_df_select['subcategory_index'] = subcategory_index_list
    subcategory_index_list.append(subcategory_dict['padding'])
    subcategory_index_df = pd.DataFrame(pd.Series(subcategory_dict), columns=['subcategory_index'])
    subcategory_index_df = subcategory_index_df.reset_index().rename(columns={'index': 'subcategory'})
    subcategory_index_df.to_csv('../../data/df/subcategory_index_df.csv', index=False)
    print('子主题个数{}'.format(subcategory_index_df.shape[0]))

    np.save('../../data/metadata/news_subcategory_index.npy', np.array(subcategory_index_list))
    np.save('../../data/metadata/subcategory_index', np.array(subcategory_index_df['subcategory_index'].values.tolist()))
    news_Info_df_select.to_csv('../../data/df/news_Info_df.csv')
    return news_Info_df_select, subcategory_dict

if __name__ == "__main__":
    ### 读取用户历史交互记录
    user_behaviors_df, news_Info_df, entity_embedding_df = load_data_behaviors()
    ### 获取用户id和点击新闻id
    user_pnewsid_nnews_id_df, total_newsid_list = get_total_user_id_and_label(user_behaviors_df)
    ### 获取用户index
    newsid_Label_df, user_clicked_df, user_clicked_newsid_list, user_dict = get_user_index( user_behaviors_df, user_pnewsid_nnews_id_df)
    ## 获取用户点击新闻id和候选新闻id映射
    news_dict = get_index_clicked_newsid_cand_newsid(user_clicked_newsid_list, total_newsid_list)
    ## split冷热用户
    cal_user_clicked_news_ratio(user_dict, user_pnewsid_nnews_id_df, user_clicked_newsid_list, news_dict)
    ## split冷热新闻
    cal_news_clicked_user_ratio(user_pnewsid_nnews_id_df, user_clicked_df, total_newsid_list, news_dict)
    print(1)
    # # ## 获取新闻流行度
    # cal_news_popularity(user_clicked_df, user_pnewsid_nnews_id_df, news_dict)
    print(2)
    ## 获取测试集
    get_test_news_user(news_dict, user_pnewsid_nnews_id_df)
    print(3)
    ### 负采样
    negtivate_sample(user_pnewsid_nnews_id_df, news_dict)
    print(4)
    ### 构建用户点击新闻和候选新闻index
    user_clicked_df, user_one_hop_dict = construct_clicked_newsindex_cand_newsindex( user_clicked_df,  user_clicked_newsid_list, news_dict, maxlen_clicked = 50)
    print(5)
    ### 获取标题文字
    news_Info_df_select, word_total_list_t, news_title = get_title_word(news_dict, news_Info_df)
    print(6)
    ### 获取文章嵌入
    get_news_embedding(news_title)
    print(7)
    ### 获取文字index
    news_Info_df_select, word_id_list, max_len = get_title_word_index (word_total_list_t, news_Info_df_select)
    print(8)
    ### 读取新闻信息,提取每一个新闻的标题实体ID和摘要实体ID
    news_Info_df_select, entity_information_t, entity_information_a = get_title_and_abstract_entity_id( news_Info_df_select)
    print(9)
    ### 将标题实体ID和摘要实体ID列表合并
    news_Info_df_select, entity_id_list = merge_title_and_abstract_list( entity_information_a, entity_information_t, news_Info_df_select)
    print(10)
    ### 映射新闻实体index
    entity_index_df, id_dict = map_entity_information(entity_id_list, news_Info_df_select, max_len = 20)
    print(11)
    ### 获取新闻实体向量
    entity_index_df = extract_entity_vector(entity_index_df, entity_embedding_df)
    print(12)
    ### 获取新闻主题
    news_Info_df_select, category_dict = get_category_index(news_Info_df_select)
    print(13)
    ### 获取副新闻主题
    news_Info_df_select, subcategory_dict = get_subcategory_index(news_Info_df_select)
