from sklearn.metrics import roc_auc_score
import numpy as np

def dcg_score(y_true, y_score, k = 10):
    order = np.argsort(y_score)[ :: -1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / (discounts + 1e-16))

def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / (best + 1e-16)

def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / (np.sum(y_true) + + 1e-16)


def evaluate_div(predicted_category, predicted_label, bound):
    predicted_label = predicted_label[:, 0]
    predicted_category = predicted_category[:, 0]
    diversity_entropy_top5_list = []
    diversity_entropy_top10_list = []
    for i in range(len(bound)):
        start, ed = bound[i]
        if ed > len(predicted_label):
            break
        score = predicted_label[start:ed]
        category = predicted_category[start:ed]
        score_sort, category_sort = (list(t) for t in zip(*sorted(zip(score, category), reverse=True)))

        # top5多样性
        diversity_entropy_top5 = 0
        category_sort_top5 = category_sort[:5]
        category_temp_dict = {}
        for category in category_sort_top5:
            if category not in category_temp_dict.keys():
                category_temp_dict[category] = 1
            else:
                category_temp_dict[category] += 1

        for k, v in category_temp_dict.items():
            pi = int(v) / len(category_sort_top5)
            diversity_entropy_top5 += - np.log(pi) * pi
        diversity_entropy_top5 = diversity_entropy_top5 / len(category_temp_dict)

        # top10多样性
        diversity_entropy_top10 = 0
        category_sort_top10 = category_sort[:10]
        category_temp_dict = {}
        for category in category_sort_top10:
            if category not in category_temp_dict.keys():
                category_temp_dict[category] = 1
            else:
                category_temp_dict[category] += 1

        for k, v in category_temp_dict.items():
            pi = int(v) / len(category_sort_top10)
            diversity_entropy_top10 += - np.log(pi) * pi
        diversity_entropy_top10 = diversity_entropy_top10 / len(category_temp_dict)

        diversity_entropy_top5_list.append(diversity_entropy_top5)
        diversity_entropy_top10_list.append(diversity_entropy_top10)

    diversity_entropy_top10_list = np.array(diversity_entropy_top10_list)
    diversity_entropy_top5_list = np.array(diversity_entropy_top5_list)

    diversity_entropy_top5 = diversity_entropy_top5_list.mean()
    diversity_entropy_top10 = diversity_entropy_top10_list.mean()
    return diversity_entropy_top5, diversity_entropy_top10

def evaluate(predicted_label, label, bound):
    predicted_label = predicted_label[:, 0]
    AUC = []
    MRR = []
    nDCG5 = []
    nDCG10 = []
    for i in range(len(bound)):
        start, ed = bound[i]
        if ed > len(predicted_label):
            break
        score = predicted_label[start:ed]
        labels = label[start:ed]
        if len(labels) == 1:
            continue
        try:
            auc = roc_auc_score(labels, score)
        except ValueError:
            continue
        mrr = mrr_score(labels, score)
        ndcg5 = ndcg_score(labels, score, k=5)
        ndcg10 = ndcg_score(labels, score, k=10)

        AUC.append(auc)
        MRR.append(mrr)
        nDCG5.append(ndcg5)
        nDCG10.append(ndcg10)

    AUC = np.array(AUC)
    MRR = np.array(MRR)
    nDCG5 = np.array(nDCG5)
    nDCG10 = np.array(nDCG10)

    AUC = AUC.mean()
    MRR = MRR.mean()
    nDCG5 = nDCG5.mean()
    nDCG10 = nDCG10.mean()

    return AUC, MRR, nDCG5, nDCG10

def evaluate_warm_cold(predicted_label, user_type_list, news_type_list, label, bound):
    def _cal_metric(predicted_label, label, bound):
        if len(predicted_label) == 0:
            return 0, 0, 0, 0
        AUC = []
        MRR = []
        nDCG5 = []
        nDCG10 = []
        for i in range(len(bound)):
            start, ed = bound[i]
            if ed > len(predicted_label):
                break
            score = predicted_label[start:ed]
            labels = label[start:ed]
            if len(labels) == 1:
                continue
            try:
                auc = roc_auc_score(labels, score)
            except ValueError:
                continue
            mrr = mrr_score(labels, score)
            ndcg5 = ndcg_score(labels, score, k=5)
            ndcg10 = ndcg_score(labels, score, k=10)

            AUC.append(auc)
            MRR.append(mrr)
            nDCG5.append(ndcg5)
            nDCG10.append(ndcg10)

        AUC = np.array(AUC)
        MRR = np.array(MRR)
        nDCG5 = np.array(nDCG5)
        nDCG10 = np.array(nDCG10)

        AUC = AUC.mean()
        MRR = MRR.mean()
        nDCG5 = nDCG5.mean()
        nDCG10 = nDCG10.mean()
        return AUC, MRR, nDCG5, nDCG10

    predicted_label = predicted_label[:, 0]

    # split cold warm user/ cold warm news
    cold_user_warm_news_label_list = []
    cold_user_warm_news_bound_list = []
    cold_user_warm_news_pred_label_list = []

    cold_user_cold_news_label_list = []
    cold_user_cold_news_bound_list = []
    cold_user_cold_news_pred_label_list = []

    warm_user_warm_news_label_list = []
    warm_user_warm_news_bound_list = []
    warm_user_warm_news_pred_label_list = []

    warm_user_cold_news_label_list = []
    warm_user_cold_news_bound_list = []
    warm_user_cold_news_pred_label_list = []

    for i in range(len(bound)):
        st, ed = bound[i]
        if ed > len(predicted_label):
            break
        user_type = user_type_list[st]
        if user_type == 0:
            start_c = len(cold_user_cold_news_label_list)
            start_w = len(cold_user_warm_news_label_list)
            for j in range(st, ed):
                if news_type_list[j][0] == 0:
                    cold_user_cold_news_label_list.append(label[j])
                    cold_user_cold_news_pred_label_list.append(predicted_label[j])
                elif news_type_list[j][0] == 1:
                    cold_user_warm_news_label_list.append(label[j])
                    cold_user_warm_news_pred_label_list.append(predicted_label[j])

            end_c = len(cold_user_cold_news_label_list)
            end_w = len(cold_user_warm_news_label_list)
            if start_c != end_c:
                cold_user_cold_news_bound_list.append((start_c, end_c))
            if start_w != end_w:
                cold_user_warm_news_bound_list.append((start_w, end_w))
        elif user_type == 1:
            start_c = len(warm_user_cold_news_label_list)
            start_w = len(warm_user_warm_news_label_list)
            for j in range(st, ed):
                if news_type_list[j][0] == 0:
                    warm_user_cold_news_label_list.append(label[j])
                    warm_user_cold_news_pred_label_list.append(predicted_label[j])
                elif news_type_list[j][0] == 1:
                    warm_user_warm_news_label_list.append(label[j])
                    warm_user_warm_news_pred_label_list.append(predicted_label[j])

            end_c = len(warm_user_cold_news_label_list)
            end_w = len(warm_user_warm_news_label_list)
            if start_c != end_c:
                warm_user_cold_news_bound_list.append((start_c, end_c))
            if start_w != end_w:
                warm_user_warm_news_bound_list.append((start_w, end_w))

    cc_AUC, cc_MRR, cc_nDCG5, cc_nDCG10 = _cal_metric(cold_user_cold_news_pred_label_list, cold_user_cold_news_label_list, cold_user_cold_news_bound_list)
    cw_AUC, cw_MRR, cw_nDCG5, cw_nDCG10 = _cal_metric(cold_user_warm_news_pred_label_list, cold_user_warm_news_label_list, cold_user_warm_news_bound_list)
    wc_AUC, wc_MRR, wc_nDCG5, wc_nDCG10 = _cal_metric(warm_user_cold_news_pred_label_list, warm_user_cold_news_label_list, warm_user_cold_news_bound_list)
    ww_AUC, ww_MRR, ww_nDCG5, ww_nDCG10 = _cal_metric(warm_user_warm_news_pred_label_list, warm_user_warm_news_label_list, warm_user_warm_news_bound_list)

    return cc_AUC, cc_MRR, cc_nDCG5, cc_nDCG10, len(cold_user_cold_news_label_list), \
           cw_AUC, cw_MRR, cw_nDCG5, cw_nDCG10, len(cold_user_warm_news_label_list), \
           wc_AUC, wc_MRR, wc_nDCG5, wc_nDCG10, len(warm_user_cold_news_label_list), \
           ww_AUC, ww_MRR, ww_nDCG5, ww_nDCG10, len(warm_user_warm_news_label_list)


def evaluate_warm_cold_u(predicted_label, user_type_list, news_type_list, label, bound):
    def _cal_metric(predicted_label, label, bound):
        if len(predicted_label) == 0:
            return 0, 0, 0, 0
        AUC = []
        MRR = []
        nDCG5 = []
        nDCG10 = []
        for i in range(len(bound)):
            start, ed = bound[i]
            if ed > len(predicted_label):
                break
            score = predicted_label[start:ed]
            labels = label[start:ed]
            if len(labels) == 1:
                continue
            try:
                auc = roc_auc_score(labels, score)
            except ValueError:
                continue
            mrr = mrr_score(labels, score)
            ndcg5 = ndcg_score(labels, score, k=5)
            ndcg10 = ndcg_score(labels, score, k=10)

            AUC.append(auc)
            MRR.append(mrr)
            nDCG5.append(ndcg5)
            nDCG10.append(ndcg10)

        AUC = np.array(AUC)
        MRR = np.array(MRR)
        nDCG5 = np.array(nDCG5)
        nDCG10 = np.array(nDCG10)

        AUC = AUC.mean()
        MRR = MRR.mean()
        nDCG5 = nDCG5.mean()
        nDCG10 = nDCG10.mean()
        return AUC, MRR, nDCG5, nDCG10

    predicted_label = predicted_label[:, 0]

    # split cold warm user/ cold warm news
    cold_user_label_list = []
    cold_user_bound_list = []
    cold_user_pred_label_list = []

    warm_user_label_list = []
    warm_user_bound_list = []
    warm_user_pred_label_list = []


    for i in range(len(bound)):
        st, ed = bound[i]
        if ed > len(predicted_label):
            break
        user_type = user_type_list[st]
        if user_type == 0:
            start = len(cold_user_label_list)
            cold_user_label_list.extend(label[st:ed])
            cold_user_pred_label_list.extend(predicted_label[st:ed])
            end = len(cold_user_label_list)
            cold_user_bound_list.append((start, end))
        elif user_type == 1:
            start = len(warm_user_label_list)
            warm_user_label_list.extend(label[st:ed])
            warm_user_pred_label_list.extend(predicted_label[st:ed])
            end = len(warm_user_label_list)
            warm_user_bound_list.append((start, end))

    c_AUC, c_MRR, c_nDCG5, c_nDCG10 = _cal_metric(cold_user_pred_label_list, cold_user_label_list, cold_user_bound_list)
    w_AUC, w_MRR, w_nDCG5, w_nDCG10 = _cal_metric(warm_user_pred_label_list, warm_user_label_list, warm_user_bound_list)

    return c_AUC, c_MRR, c_nDCG5, c_nDCG10, len(cold_user_label_list), \
           w_AUC, w_MRR, w_nDCG5, w_nDCG10, len(warm_user_label_list),


def evaluate_warm_cold_n(predicted_label, user_type_list, news_type_list, label, bound):
    def _cal_metric(predicted_label, label, bound):
        if len(predicted_label) == 0:
            return 0, 0, 0, 0
        AUC = []
        MRR = []
        nDCG5 = []
        nDCG10 = []
        for i in range(len(bound)):
            start, ed = bound[i]
            if ed > len(predicted_label):
                break
            score = predicted_label[start:ed]
            labels = label[start:ed]
            
            if len(labels) == 1:
                continue
            try:
                auc = roc_auc_score(labels, score)
            except ValueError:
                continue

            mrr = mrr_score(labels, score)
            ndcg5 = ndcg_score(labels, score, k=5)
            ndcg10 = ndcg_score(labels, score, k=10)

            AUC.append(auc)
            MRR.append(mrr)
            nDCG5.append(ndcg5)
            nDCG10.append(ndcg10)

        AUC = np.array(AUC)
        MRR = np.array(MRR)
        nDCG5 = np.array(nDCG5)
        nDCG10 = np.array(nDCG10)

        AUC = AUC.mean()
        MRR = MRR.mean()
        nDCG5 = nDCG5.mean()
        nDCG10 = nDCG10.mean()
        return AUC, MRR, nDCG5, nDCG10

    predicted_label = predicted_label[:, 0]

    # split cold warm user/ cold warm news
    cold_news_label_list = []
    cold_news_bound_list = []
    cold_news_pred_label_list = []

    warm_news_label_list = []
    warm_news_bound_list = []
    warm_news_pred_label_list = []


    for i in range(len(bound)):
        st, ed = bound[i]
        if ed > len(predicted_label):
            break
        start_c = len(cold_news_label_list)
        start_w = len(warm_news_label_list)
        for j in range(st, ed):
            if news_type_list[j][0] == 0:
                cold_news_label_list.append(label[j])
                cold_news_pred_label_list.append(predicted_label[j])
            elif news_type_list[j][0] == 1:
                warm_news_label_list.append(label[j])
                warm_news_pred_label_list.append(predicted_label[j])
        end_c = len(cold_news_label_list)
        end_w = len(warm_news_label_list)
        if start_c != end_c:
            cold_news_bound_list.append((start_c, end_c))
        if start_w != end_w:
            warm_news_bound_list.append((start_w, end_w))
            
    # cold news
    auc = roc_auc_score(cold_news_label_list, cold_news_pred_label_list)
    mrr = mrr_score(cold_news_label_list, cold_news_pred_label_list)
    ndcg5 = ndcg_score(cold_news_label_list, cold_news_pred_label_list, k=5)
    ndcg10 = ndcg_score(cold_news_label_list, cold_news_pred_label_list, k=10)
    print(auc, mrr, ndcg5, ndcg10)

    c_AUC, c_MRR, c_nDCG5, c_nDCG10 = _cal_metric(cold_news_pred_label_list, cold_news_label_list, cold_news_bound_list)
    w_AUC, w_MRR, w_nDCG5, w_nDCG10 = _cal_metric(warm_news_pred_label_list, warm_news_label_list, warm_news_bound_list)
    return c_AUC, c_MRR, c_nDCG5, c_nDCG10, len(cold_news_label_list), \
           w_AUC, w_MRR, w_nDCG5, w_nDCG10, len(warm_news_label_list),


def evaluate_warm_cold_n_update(predicted_label, user_index, news_index, user_type_list, news_type_list, label, bound):


    def _cal_metric(predicted_label, label, bound):
        if len(predicted_label) == 0:
            return 0, 0, 0, 0
        AUC = []
        MRR = []
        nDCG5 = []
        nDCG10 = []
        for i in range(len(bound)):
            start, ed = bound[i]
            if ed > len(predicted_label):
                break
            score = predicted_label[start:ed]
            labels = label[start:ed]

            if len(labels) == 1:
                continue
            try:
                auc = roc_auc_score(labels, score)
            except ValueError:
                continue
            
            mrr = mrr_score(labels, score)
            ndcg5 = ndcg_score(labels, score, k=5)
            ndcg10 = ndcg_score(labels, score, k=10)

            AUC.append(auc)
            MRR.append(mrr)
            nDCG5.append(ndcg5)
            nDCG10.append(ndcg10)

        AUC = np.array(AUC)
        MRR = np.array(MRR)
        nDCG5 = np.array(nDCG5)
        nDCG10 = np.array(nDCG10)

        AUC = AUC.mean()
        MRR = MRR.mean()
        nDCG5 = nDCG5.mean()
        nDCG10 = nDCG10.mean()
        return AUC, MRR, nDCG5, nDCG10

    predicted_label = predicted_label[:, 0]
    
    news_user_dict = {}
    for i in range(len(news_index)):
        if news_index[i] not in news_user_dict.keys():
            news_user_dict[news_index[i]] = []
        news_user_dict[news_index[i]].append((user_index[i], label[i], predicted_label[i]))

    news_predicted_label = []
    news_label = []
    news_bound = []
    news = []
    user = []
    startIndex = 0
    for k, v in news_user_dict.items():
        for temp in v:
            news.append(k)
            user.append(temp[0])
            news_predicted_label.append(temp[2])
            news_label.append(temp[1])
        news_bound.append((startIndex, startIndex + len(v)))    
        startIndex = startIndex + len(v)

    cold_news_label_list = []
    cold_news_bound_list = []
    cold_news_pred_label_list = []

    warm_news_label_list = []
    warm_news_bound_list = []
    warm_news_pred_label_list = []

    for i in range(len(news_bound)):
        st, ed = news_bound[i]
        if ed > len(news_predicted_label):
            break
        news_type = news_type_list[st][0]
        if news_type == 0:
            start = len(cold_news_label_list)
            cold_news_label_list.extend(news_label[st:ed])
            cold_news_pred_label_list.extend(news_predicted_label[st:ed])
            end = len(cold_news_label_list)
            cold_news_bound_list.append((start, end))
        elif news_type == 1:
            start = len(warm_news_label_list)
            warm_news_label_list.extend(news_label[st:ed])
            warm_news_pred_label_list.extend(news_predicted_label[st:ed])
            end = len(warm_news_label_list)
            warm_news_bound_list.append((start, end))

    c_AUC, c_MRR, c_nDCG5, c_nDCG10 = _cal_metric(cold_news_pred_label_list, cold_news_label_list, cold_news_bound_list)
    w_AUC, w_MRR, w_nDCG5, w_nDCG10 = _cal_metric(warm_news_pred_label_list, warm_news_label_list, warm_news_bound_list)




    # split cold warm user/ cold warm news
    # cold_news_label_list = []
    # cold_news_bound_list = []
    # cold_news_pred_label_list = []

    # warm_news_label_list = []
    # warm_news_bound_list = []
    # warm_news_pred_label_list = []


    # for i in range(len(bound)):
    #     st, ed = bound[i]
    #     if ed > len(predicted_label):
    #         break
    #     start_c = len(cold_news_label_list)
    #     start_w = len(warm_news_label_list)
    #     for j in range(st, ed):
    #         if news_type_list[j][0] == 0:
    #             cold_news_label_list.append(label[j])
    #             cold_news_pred_label_list.append(predicted_label[j])
    #         elif news_type_list[j][0] == 1:
    #             warm_news_label_list.append(label[j])
    #             warm_news_pred_label_list.append(predicted_label[j])
    #     end_c = len(cold_news_label_list)
    #     end_w = len(warm_news_label_list)
    #     if start_c != end_c:
    #         cold_news_bound_list.append((start_c, end_c))
    #     if start_w != end_w:
    #         warm_news_bound_list.append((start_w, end_w))
            
    # # cold news
    # auc = roc_auc_score(cold_news_label_list, cold_news_pred_label_list)
    # mrr = mrr_score(cold_news_label_list, cold_news_pred_label_list)
    # ndcg5 = ndcg_score(cold_news_label_list, cold_news_pred_label_list, k=5)
    # ndcg10 = ndcg_score(cold_news_label_list, cold_news_pred_label_list, k=10)
    # print(auc, mrr, ndcg5, ndcg10)

    # c_AUC, c_MRR, c_nDCG5, c_nDCG10 = _cal_metric(cold_news_pred_label_list, cold_news_label_list, cold_news_bound_list)
    # w_AUC, w_MRR, w_nDCG5, w_nDCG10 = _cal_metric(warm_news_pred_label_list, warm_news_label_list, warm_news_bound_list)
    return c_AUC, c_MRR, c_nDCG5, c_nDCG10, len(cold_news_label_list), \
           w_AUC, w_MRR, w_nDCG5, w_nDCG10, len(warm_news_label_list),