from model.AnchorKG_model.AnchorKG_Train_Test import *
from model.PGPR_model.PGPR_Train_Test import *
from model.RippleNet_model.RippleNet_Train_Test import *
from model.ADAC_model.ADAC_Train_Test import *
from model.KPRN_model.KPRN_Train_Test import *
from model.MRNN_model.MRNN_Train_Test import *
from model.Baseline_model.Baseline_Train_Test import *
from model.KGIN_model.KGIN_Train_Test import *
from model.KGCN_model.KGCN_Train_Test import *
from model.KGAT_model.KGAT_Train_Test import *
from model.LightGCN_model.LightGCN_Train_Test import *
from model.RCENR_model.RCENR_Train_Test import *
from model.MRNN_model.MRNN_Train_Test import *
from model.NRMS_model.NRMS_Train_Test import *
from model.NAML_model.NAML_Train_Test import *
from model.MNN4Rec_model.MNN4Rec_Train_Test import *
from model.LSTUR_model.LSTUR_Train_Test import *
from model.DKN_model.DKN_Train_Test import *
from model.FM_model.FM_Train_Test import *
from model.MNN4Rec_update_model.MNN4Rec_update_Train_Test import *
from model.NPA_model.NPA_Train_Test import *
from model.KIM_model.KIM_Train_Test import *
from model.FIM_model.FIM_Train_Test import *
from model.SFT_NPA_model.SFT_NPA_Train_Test import *
from model.SFT_NRMS_model.SFT_NRMS_Train_Test import *
from model.SFT_NAML_model.SFT_NAML_Train_Test import *
from model.SFT_MRNN_model.SFT_MRNN_Train_Test import *
from DataLoad import load_data
import argparse
import os

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def parse_args_2w():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_flag', type=str2bool, default=True)
    parser.add_argument('--test_flag', type=str2bool, default=True)

    parser.add_argument('--user_data_mode', type=int, default=3)
    parser.add_argument('--news_data_mode', type=int, default=3)
    parser.add_argument('--mode', type=str, default='RCENR')
    parser.add_argument('--epoch', type=int, default= 60)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--sample_size', type=int, default=5)
    parser.add_argument('--optimizer_type', type=str, default='adam')
    parser.add_argument('--loss_type', type=str, default='cross_entropy')
    parser.add_argument('--checkpoint_dir', type=str, default='../save_model/', help='模型保留位置')
    parser.add_argument('--train_ratio', type=float, default= 1)

    parser.add_argument('--user_num', type = int, default=15427)
    parser.add_argument('--sample_num', type = int, default=5)
    parser.add_argument('--user_clicked_num', type=int, default=50)
    parser.add_argument('--warm_user_num', type=int, default=9925, help='热用户数')
    parser.add_argument('--cold_user_num', type=int, default=5502, help='冷用户数')
    parser.add_argument('--news_num', type=int, default=35854, help='新闻总数')
    parser.add_argument('--warm_news_num', type=int, default=6174, help='冷新闻数')
    parser.add_argument('--cold_news_num', type=int, default=29681, help='冷新闻数')
    parser.add_argument('--category_num', type=int, default=18, help='类别向量总数')
    parser.add_argument('--subcategory_num', type=int, default=251, help='子类别向量总数')
    parser.add_argument('--word_num', type=int, default=40300, help='单词总数')
    parser.add_argument('--news_entity_num', type=int, default=21343, help='新闻实体特征个数')
    parser.add_argument('--total_entity_num', type=int, default=111979, help='总实体特征个数')
    parser.add_argument('--total_relation_num', type=int, default=405, help='总关系特征个数')
    parser.add_argument('--news_entity_size', type=int, default=20, help='单个新闻最大实体个数')
    parser.add_argument('--title_word_size', type=int, default=39, help='每个新闻标题中的单词数量')
    parser.add_argument('--entity_neigh_num', type=int, default=5, help='邻居节点个数')
    # MRNN
    parser.add_argument('--attention_heads', type=int, default=20, help='多头注意力的头数')
    parser.add_argument('--num_units', type=int, default=20, help='多头注意力输出维数')
    parser.add_argument('--attention_dim', type=int, default=20, help='注意力层的维数')
    parser.add_argument('--embedding_dim', type=int, default=100, help='新闻和用户向量')
    parser.add_argument('--title_embedding_dim', type=int, default=768, help='新闻初始向量维数')
    parser.add_argument('--word_embedding_dim', type=int, default=300, help='单词嵌入维数')
    parser.add_argument('--entity_embedding_dim', type=int, default=100, help='实体嵌入维数')
    parser.add_argument('--category_embedding_dim', type=int, default=100, help='主题总数')
    parser.add_argument('--subcategory_embedding_dim', type=int, default=100, help='自主题总数')
    parser.add_argument('--query_vector_dim', type=int, default=200, help='询问向量维数')
    # DKN参数
    parser.add_argument('--kcnn_num_filters', type=int, default=50, help='卷积核个数')
    parser.add_argument('--kcnn_window_sizes', type=list, default=[2, 3, 4], help='窗口大小')
    parser.add_argument('--use_context', type=bool, default=None, help='自主题总数')
    # NAML参数
    parser.add_argument('--cnn_num_filters', type=int, default=400, help='卷积核个数')
    parser.add_argument('--cnn_window_sizes', type=int, default=3, help='窗口大小')
    parser.add_argument('--drop_prob', type=bool, default=0.2, help='丢弃概率')
    # LSTUR参数
    parser.add_argument('--long_short_term_method', type=str, default='ini', help='ini or con')
    parser.add_argument('--lstur_num_filters', type=int, default=300, help='卷积核个数')
    parser.add_argument('--lstur_window_sizes', type=int, default=3, help='窗口大小')
    parser.add_argument('--masking_probability', type=int, default=0.3, help='遮掩概率')
    # LightGCN
    parser.add_argument("--lgn_layers", type=int, default=3, help="the number of layers in GCN")
    parser.add_argument("--keep_prob", type=float, default=0.6, help="the batch size for bpr loss training procedure")
    parser.add_argument("--dropout", type=float, default=0, help="using the dropout or not")
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 regularization weight')
    # MNN4Rec参数
    parser.add_argument('--topk_implicit', type=int, default=5, help='隐式邻居采样数')
    parser.add_argument('--topk_explicit', type=int, default=5, help='显式邻居采样数')
    parser.add_argument('--use_news_relation', type=bool, default=True, help='是否利用新闻关系')
    # MetaKG
    parser.add_argument('--meta_update_lr', type=float, default=0.001, help='meta update learning rate')
    parser.add_argument('--scheduler_lr', type=float, default=0.001, help='scheduler learning rate')
    parser.add_argument('--num_inner_update', type=int, default=2, help='number of inner update')
    parser.add_argument('--meta_batch_size', type=int, default=50, help='meta batch size')
    parser.add_argument('--n_hops', type=int, default=10, help='gcn hop')
    parser.add_argument("--mess_dropout", type=bool, default=True, help="consider message dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of node dropout")
    parser.add_argument('--select_idx', type=bool, default=False, help='是否选择样本')
    # KIM参数
    parser.add_argument('--feature_dim', type=int, default=300, help='新闻特征维度')
    # RCENR
    parser.add_argument('--depth', type=list, default=[5, 3, 2], help='K-跳深度')
    # RippleNet
    parser.add_argument('--ripplenet_n_hop', type=int, default=2, help='maximum hops')
    parser.add_argument('--ripplenet_n_memory', type=int, default=5, help='size of ripple set for each hop')
    parser.add_argument('--kge_weight', type=float, default=0.01, help='weight of the KGE term')
    parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of the l2 regularization term')
    parser.add_argument('--lr', type=float, default=0.02, help='learning rate')
    parser.add_argument('--n_epoch', type=int, default=10, help='the number of epochs')
    parser.add_argument('--update_mode', type=str, default='plus_transform', help='how to update item at the end of each hop')
    parser.add_argument('--using_all_hops', type=bool, default=True, help='whether using outputs of all hops or just the last hop when making prediction')
    # KPRN
    parser.add_argument('--kprn_path_long', type=int, default=6, help='路径长度')
    parser.add_argument('--kprn_max_path', type=int, default=5, help='每个用户项目对的最大个数')
    # ADAC
    parser.add_argument('--ADAC_path_long', type=int, default=5, help='路径长度')

    return parser.parse_args()

def parse_args_5w():
    parser = argparse.ArgumentParser()
    parser.add_argument('--user_data_mode', type=int, default=3)
    parser.add_argument('--news_data_mode', type=int, default=3)
    parser.add_argument('--mode', type=str, default='MNN4Rec')
    parser.add_argument('--epoch', type=int, default= 60)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--sample_size', type=int, default=5)
    parser.add_argument('--optimizer_type', type=str, default='adam')
    parser.add_argument('--loss_type', type=str, default='cross_entropy')
    parser.add_argument('--checkpoint_dir', type=str, default='./out/save_model/', help='模型保留位置')

    parser.add_argument('--user_num', type = int, default=50000)
    parser.add_argument('--user_clicked_num', type=int, default=50)
    parser.add_argument('--warm_user_num', type=int, default=25669, help='热用户数')
    parser.add_argument('--cold_user_num', type=int, default=24331, help='冷用户数')
    parser.add_argument('--news_num', type=int, default=51283, help='新闻总数')
    parser.add_argument('--warm_news_num', type=int, default=9423, help='冷新闻数')
    parser.add_argument('--cold_news_num', type=int, default=41860, help='冷新闻数')
    parser.add_argument('--category_num', type=int, default=18, help='类别向量总数')
    parser.add_argument('--subcategory_num', type=int, default=265, help='子类别向量总数')
    parser.add_argument('--word_num', type=int, default=47978, help='单词总数')
    parser.add_argument('--news_entity_num', type=int, default=27760, help='新闻实体特征个数')
    parser.add_argument('--total_entity_num', type=int, default=119574, help='总实体特征个数')
    parser.add_argument('--total_relation_num', type=int, default=422, help='总关系特征个数')
    parser.add_argument('--news_entity_size', type=int, default=20, help='单个新闻最大实体个数')
    parser.add_argument('--title_word_size', type=int, default=39, help='每个新闻标题中的单词数量')
    parser.add_argument('--entity_neigh_num', type=int, default=5, help='邻居节点个数')
    # MRNN
    parser.add_argument('--attention_heads', type=int, default=20, help='多头注意力的头数')
    parser.add_argument('--num_units', type=int, default=20, help='多头注意力输出维数')
    parser.add_argument('--attention_dim', type=int, default=20, help='注意力层的维数')
    parser.add_argument('--embedding_dim', type=int, default=100, help='新闻和用户向量')
    parser.add_argument('--title_embedding_dim', type=int, default=400, help='新闻初始向量维数')
    parser.add_argument('--word_embedding_dim', type=int, default=300, help='单词嵌入维数')
    parser.add_argument('--entity_embedding_dim', type=int, default=100, help='实体嵌入维数')
    parser.add_argument('--category_embedding_dim', type=int, default=100, help='主题总数')
    parser.add_argument('--subcategory_embedding_dim', type=int, default=100, help='自主题总数')
    parser.add_argument('--query_vector_dim', type=int, default=200, help='询问向量维数')
    # RippleNet
    parser.add_argument('--ripplenet_n_hop', type=int, default=2, help='maximum hops')
    parser.add_argument('--ripplenet_n_memory', type=int, default=5, help='size of ripple set for each hop')
    # DKN参数
    parser.add_argument('--kcnn_num_filters', type=int, default=50, help='卷积核个数')
    parser.add_argument('--kcnn_window_sizes', type=list, default=[2, 3, 4], help='窗口大小')
    parser.add_argument('--use_context', type=bool, default=None, help='自主题总数')
    # NAML参数
    parser.add_argument('--cnn_num_filters', type=int, default=400, help='卷积核个数')
    parser.add_argument('--cnn_window_sizes', type=int, default=3, help='窗口大小')
    parser.add_argument('--drop_prob', type=bool, default=0.2, help='丢弃概率')
    # LSTUR参数
    parser.add_argument('--long_short_term_method', type=str, default='ini', help='ini or con')
    parser.add_argument('--lstur_num_filters', type=int, default=300, help='卷积核个数')
    parser.add_argument('--lstur_window_sizes', type=int, default=3, help='窗口大小')
    parser.add_argument('--masking_probability', type=int, default=0.3, help='遮掩概率')
    # LightGCN
    parser.add_argument("--lgn_layers", type=int, default=3, help="the number of layers in GCN")
    parser.add_argument("--keep_prob", type=float, default=0.6, help="the batch size for bpr loss training procedure")
    parser.add_argument("--dropout", type=float, default=0, help="using the dropout or not")
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 regularization weight')
    # MNN4Rec参数
    parser.add_argument('--topk_implicit', type=int, default=5, help='隐式邻居采样数')
    parser.add_argument('--topk_explicit', type=int, default=5, help='显式邻居采样数')
    parser.add_argument('--use_news_relation', type=bool, default=True, help='是否利用新闻关系')
    # MetaKG
    parser.add_argument('--meta_update_lr', type=float, default=0.001, help='meta update learning rate')
    parser.add_argument('--scheduler_lr', type=float, default=0.001, help='scheduler learning rate')
    parser.add_argument('--num_inner_update', type=int, default=2, help='number of inner update')
    parser.add_argument('--meta_batch_size', type=int, default=50, help='meta batch size')
    parser.add_argument('--n_hops', type=int, default=10, help='gcn hop')
    parser.add_argument("--mess_dropout", type=bool, default=True, help="consider message dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of node dropout")
    parser.add_argument('--select_idx', type=bool, default=False, help='是否选择样本')
    # KIM参数
    parser.add_argument('--feature_dim', type=int, default=300, help='新闻特征维度')

    # RCENR
    parser.add_argument('--depth', type=list, default=[5, 3, 2], help='K-跳深度')
    parser.add_argument('--checkpoint_dir', type=str, default='../save_model/', help='模型保留位置')
    # RippleNet
    parser.add_argument('--ripplenet_n_hop', type=int, default=2, help='maximum hops')
    parser.add_argument('--ripplenet_n_memory', type=int, default=5, help='size of ripple set for each hop')
    parser.add_argument('--kge_weight', type=float, default=0.01, help='weight of the KGE term')
    parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of the l2 regularization term')
    parser.add_argument('--lr', type=float, default=0.02, help='learning rate')
    parser.add_argument('--n_epoch', type=int, default=10, help='the number of epochs')
    parser.add_argument('--update_mode', type=str, default='plus_transform', help='how to update item at the end of each hop')
    parser.add_argument('--using_all_hops', type=bool, default=True, help='whether using outputs of all hops or just the last hop when making prediction')
    # KPRN
    parser.add_argument('--kprn_path_long', type=int, default=6, help='路径长度')
    parser.add_argument('--kprn_max_path', type=int, default=5, help='每个用户项目对的最大个数')
    # ADAC
    parser.add_argument('--ADAC_path_long', type=int, default=5, help='路径长度')
    return parser.parse_args()

def parse_args_full():
    parser = argparse.ArgumentParser()
    parser.add_argument('--user_data_mode', type=int, default=3)
    parser.add_argument('--news_data_mode', type=int, default=3)
    parser.add_argument('--mode', type=str, default='NRMS')
    parser.add_argument('--epoch', type=int, default= 60)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--sample_size', type=int, default=5)
    parser.add_argument('--optimizer_type', type=str, default='adam')
    parser.add_argument('--loss_type', type=str, default='cross_entropy')
    parser.add_argument('--checkpoint_dir', type=str, default='../save_model/', help='模型保留位置')

    parser.add_argument('--user_num', type = int, default=711222)
    parser.add_argument('--user_clicked_num', type=int, default=50)
    parser.add_argument('--warm_user_num', type=int, default=539069, help='热用户数')
    parser.add_argument('--cold_user_num', type=int, default=172153, help='冷用户数')
    parser.add_argument('--news_num', type=int, default=101528, help='新闻总数')
    parser.add_argument('--warm_news_num', type=int, default=79547, help='热新闻数')
    parser.add_argument('--cold_news_num', type=int, default=21981, help='冷新闻数')
    parser.add_argument('--category_num', type=int, default=19, help='类别向量总数')
    parser.add_argument('--subcategory_num', type=int, default=286, help='子类别向量总数')
    parser.add_argument('--word_num', type=int, default=65829, help='单词总数')
    parser.add_argument('--news_entity_num', type=int, default=43432, help='新闻实体特征个数')
    parser.add_argument('--total_entity_num', type=int, default=141627, help='总实体特征个数')
    parser.add_argument('--total_relation_num', type=int, default=458, help='总关系特征个数')
    parser.add_argument('--news_entity_size', type=int, default=20, help='单个新闻最大实体个数')
    parser.add_argument('--title_word_size', type=int, default=65, help='每个新闻标题中的单词数量')
    parser.add_argument('--entity_neigh_num', type=int, default=5, help='邻居节点个数')
    # MRNN
    parser.add_argument('--attention_heads', type=int, default=20, help='多头注意力的头数')
    parser.add_argument('--num_units', type=int, default=20, help='多头注意力输出维数')
    parser.add_argument('--attention_dim', type=int, default=20, help='注意力层的维数')
    parser.add_argument('--embedding_dim', type=int, default=100, help='新闻和用户向量')
    parser.add_argument('--title_embedding_dim', type=int, default=768, help='新闻初始向量维数')
    parser.add_argument('--word_embedding_dim', type=int, default=300, help='单词嵌入维数')
    parser.add_argument('--entity_embedding_dim', type=int, default=100, help='实体嵌入维数')
    parser.add_argument('--category_embedding_dim', type=int, default=100, help='主题总数')
    parser.add_argument('--subcategory_embedding_dim', type=int, default=100, help='自主题总数')
    parser.add_argument('--query_vector_dim', type=int, default=200, help='询问向量维数')
    # DKN参数
    parser.add_argument('--kcnn_num_filters', type=int, default=50, help='卷积核个数')
    parser.add_argument('--kcnn_window_sizes', type=list, default=[2, 3, 4], help='窗口大小')
    parser.add_argument('--use_context', type=bool, default=None, help='自主题总数')
    # NAML参数
    parser.add_argument('--cnn_num_filters', type=int, default=400, help='卷积核个数')
    parser.add_argument('--cnn_window_sizes', type=int, default=3, help='窗口大小')
    parser.add_argument('--drop_prob', type=bool, default=0.2, help='丢弃概率')
    # LSTUR参数
    parser.add_argument('--long_short_term_method', type=str, default='ini', help='ini or con')
    parser.add_argument('--lstur_num_filters', type=int, default=300, help='卷积核个数')
    parser.add_argument('--lstur_window_sizes', type=int, default=3, help='窗口大小')
    parser.add_argument('--masking_probability', type=int, default=0.3, help='遮掩概率')
    # LightGCN
    parser.add_argument("--lgn_layers", type=int, default=3, help="the number of layers in GCN")
    parser.add_argument("--keep_prob", type=float, default=0.6, help="the batch size for bpr loss training procedure")
    parser.add_argument("--dropout", type=float, default=0, help="using the dropout or not")
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 regularization weight')
    # MNN4Rec参数
    parser.add_argument('--topk_implicit', type=int, default=5, help='隐式邻居采样数')
    parser.add_argument('--topk_explicit', type=int, default=5, help='显式邻居采样数')
    parser.add_argument('--use_news_relation', type=bool, default=True, help='是否利用新闻关系')
    # MetaKG
    parser.add_argument('--meta_update_lr', type=float, default=0.001, help='meta update learning rate')
    parser.add_argument('--scheduler_lr', type=float, default=0.001, help='scheduler learning rate')
    parser.add_argument('--num_inner_update', type=int, default=2, help='number of inner update')
    parser.add_argument('--meta_batch_size', type=int, default=50, help='meta batch size')
    parser.add_argument('--n_hops', type=int, default=10, help='gcn hop')
    parser.add_argument("--mess_dropout", type=bool, default=True, help="consider message dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of node dropout")
    parser.add_argument('--select_idx', type=bool, default=False, help='是否选择样本')
    # KIM参数
    parser.add_argument('--feature_dim', type=int, default=300, help='新闻特征维度')
    # RCENR
    parser.add_argument('--depth', type=list, default=[5, 3, 2], help='K-跳深度')
    # RippleNet
    parser.add_argument('--ripplenet_n_hop', type=int, default=2, help='maximum hops')
    parser.add_argument('--ripplenet_n_memory', type=int, default=5, help='size of ripple set for each hop')
    parser.add_argument('--kge_weight', type=float, default=0.01, help='weight of the KGE term')
    parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of the l2 regularization term')
    parser.add_argument('--lr', type=float, default=0.02, help='learning rate')
    parser.add_argument('--n_epoch', type=int, default=10, help='the number of epochs')
    parser.add_argument('--update_mode', type=str, default='plus_transform', help='how to update item at the end of each hop')
    parser.add_argument('--using_all_hops', type=bool, default=True, help='whether using outputs of all hops or just the last hop when making prediction')
    # KPRN
    parser.add_argument('--kprn_path_long', type=int, default=6, help='路径长度')
    parser.add_argument('--kprn_max_path', type=int, default=5, help='每个用户项目对的最大个数')
    # ADAC
    parser.add_argument('--ADAC_path_long', type=int, default=5, help='路径长度')
    return parser.parse_args()

def main(path, device):
    data_scale = "2wu"
    
    if data_scale == '2wu':
        args = parse_args_2w()
    elif data_scale == '5wu':
        args = parse_args_5w()
    elif data_scale == 'full':
        args = parse_args_full()

    data = load_data( args, path )
    if args.mode == "AnchorKG":
        model = AnchorKG_Train_Test(args, data, device)
        if args.train_flag:
            model.Train()
        if args.test_flag:
            model.Test()
    if args.mode == "PGPR":
        model = PGPR_Train_Test(args, data, device)
        if args.train_flag:
            model.Train()
        if args.test_flag:
            model.Test()
    if args.mode == "RippleNet":
        model = RippleNet_Train_Test(args, data, device)
        if args.train_flag:
            model.Train()
        if args.test_flag:
            model.Test()
    if args.mode == "ADAC":
        model = ADAC_Train_Test(args, data, device)
        if args.train_flag:
            model.Train()
        if args.test_flag:
            model.Test()
    if args.mode == "KPRN":
        model = KPRN_Train_Test(args, data, device)
        if args.train_flag:
            model.Train()
        if args.test_flag:
            model.Test()
    if args.mode == "MRNN":
        model = MRNN_Train_Test(args, data, device)
        if args.train_flag:
            model.Train()
        if args.test_flag:
            model.Test()
    if args.mode == "Baseline":
        model = Baseline_Train_Test(args, data, device)
        if args.train_flag:
            model.Train()
        if args.test_flag:
            model.Test()
    if args.mode == "KGIN":
        model = KGIN_Train_Test(args, data, device)
        if args.train_flag:
            model.Train()
        if args.test_flag:
            model.Test()
    if args.mode == "KGCN":
        model = KGCN_Train_Test(args, data, device)
        if args.train_flag:
            model.Train()
        if args.test_flag:
            model.Test()
    if args.mode == "KGAT":
        model = KGAT_Train_Test(args, data, device)
        if args.train_flag:
            model.Train()
        if args.test_flag:
            model.Test()
    if args.mode == "LightGCN":
        model = LightGCN_Train_Test(args, data, device)
        if args.train_flag:
            model.Train()
        if args.test_flag:
            model.Test()
    if args.mode == "RCENR":
        model = RCENR_Train_Test(args, data, device)
        if args.train_flag:
            model.Train()
        if args.test_flag:
            model.Test()
    if args.mode == "NRMS":
        model = NRMS_Train_Test(args, data, device)
        if args.train_flag:
            model.Train()
        if args.test_flag:
            model.Test()
    if args.mode == "NAML":
        model = NAML_Train_Test(args, data, device)
        if args.train_flag:
            model.Train()
        if args.test_flag:
            model.Test()
    if args.mode == "MNN4Rec":
        model = MNN4Rec_Train_Test(args, data, device)
        if args.train_flag:
            model.Train()
        if args.test_flag:
            model.Test()
    if args.mode == "LSTUR":
        model = LSTUR_Train_Test(args, data, device)
        if args.train_flag:
            model.Train()
        if args.test_flag:
            model.Test()
    if args.mode == "DKN":
        model = DKN_Train_Test(args, data, device)
        if args.train_flag:
            model.Train()
        if args.test_flag:
            model.Test()
    if args.mode == "LightGCN":
        model = LightGCN_Train_Test(args, data, device)
        if args.train_flag:
            model.Train()
        if args.test_flag:
            model.Test()
    if args.mode == "FM":
        model = FM_Train_Test(args, data, device)
        if args.train_flag:
            model.Train()
        if args.test_flag:
            model.Test()
    if args.mode == "MNN4Rec_update":
        model = MNN4Rec_update_Train_Test(args, data, device)
        if args.train_flag:
            model.Train()
        if args.test_flag:
            model.Test()
    if args.mode == "NPA":
        model = NPA_Train_Test(args, data, device)
        if args.train_flag:
            model.Train()
        if args.test_flag:
            model.Test()
    if args.mode == "KIM":
        model = KIM_Train_Test(args, data, device)
        if args.train_flag:
            model.Train()
        if args.test_flag:
            model.Test()
    if args.mode == "FIM":
        model = FIM_Train_Test(args, data, device)
        if args.train_flag:
            model.Train()
        if args.test_flag:
            model.Test()
    if args.mode == "SFT_NPA_ATT":
        model = SFT_NPA_Train_Test(args, data, device)
        if args.train_flag:
            model.Train()
        if args.test_flag:
            model.Test()
    if args.mode == "SFT_NRMS_ATT":
        model = SFT_NRMS_Train_Test(args, data, device)
        if args.train_flag:
            model.Train()
        if args.test_flag:
            model.Test()
    if args.mode == "SFT_NAML_ATT":
        model = SFT_NAML_Train_Test(args, data, device)
        if args.train_flag:
            model.Train()
        if args.test_flag:
            model.Test()
    if args.mode == "SFT_MRNN_ATT":
        model = SFT_MRNN_Train_Test(args, data, device)
        if args.train_flag:
            model.Train()
        if args.test_flag:
            model.Test()

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path = os.path.dirname(os.getcwd())
    main(path, device)
