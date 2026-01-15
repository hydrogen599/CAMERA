from math import sqrt
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from .ranking_utils import scores_mapping,ndcg

MAX_MIC = np.log10(2**13)

def cal_reg_metrics(pred,gt):
    """
    Return TopK MSE
    """
    # Sort by gt
    results =  [gt,pred]
    results = sorted(list(map(list, zip(*results))))
    results = list(map(list, zip(*results)))
    gt,pred = results[0],results[1]
    top10_mse = np.mean([(actual - predicted) ** 2 for actual, predicted in zip(gt[0:10], pred[0:10])])
    top30_mse = np.mean([(actual - predicted) ** 2 for actual, predicted in zip(gt[0:30], pred[0:30])])
    top50_mse = np.mean([(actual - predicted) ** 2 for actual, predicted in zip(gt[0:50], pred[0:50])])
    top100_mse = np.mean([(actual - predicted) ** 2 for actual, predicted in zip(gt[0:100], pred[0:100])])
    mse = np.mean([(actual - predicted) ** 2 for actual, predicted in zip(gt, pred)])
    pos_mse = np.mean([(actual - predicted) ** 2 
                                for actual, predicted in zip(gt, pred) 
                                if actual < MAX_MIC - 0.01])
    res_dict = {'top10_mse':top10_mse,'top30_mse':top30_mse,'top50_mse':top50_mse,'top100_mse':top100_mse, \
                'mse':mse,'pos_mse':pos_mse}
    return res_dict

def cal_ndcg_metrics_by_session(pred, gt, bacs):
    bacs = np.hstack(bacs)
    gt = np.hstack(gt)
    pred = np.hstack(pred)
    # print(len(bacs), len(gt), len(pred))
    result_df = pd.DataFrame({'bacs': bacs, 'gt': gt, 'pred': pred})

    session_ndcgs = pd.DataFrame(columns=['ndcg@5', 'ndcg@10', 'ndcg@30', 'ndcg@50', 'ndcg@100', 'ndcg@150', 'ndcg@all'])
    for bac in result_df.bacs.unique():
        result_bac = result_df[result_df.bacs == bac]
        pred_bac = result_bac['pred'].to_list()
        gt_bac = result_bac['gt'].to_list()

        ndcgs_bac = cal_ndcg_metrics(pred_bac, gt_bac)
        session_ndcgs = pd.concat([session_ndcgs, pd.DataFrame(ndcgs_bac, index=[0])], axis=0)
    
    return session_ndcgs.mean().to_dict()

def cal_ndcg_metrics(pred,gt):
    """
    Calculate ndcg metrics for all AMPs(exclude Non-AMPs)
    """
    # sorted by pred_mic
    results =  [pred,gt]
    results = sorted(list(map(list, zip(*results))))
    results = list(map(list, zip(*results)))
    pred, gt = np.array(results[0]),np.array(results[1])
    # exclude Non-AMPs
    amp_index = gt<MAX_MIC
    gt = gt[amp_index]
    pred = pred[amp_index]
    rel_scores = cal_rel_scores(gt)

    ndcg_at_5 = ndcg_at_k(rel_scores,5)
    ndcg_at_10 = ndcg_at_k(rel_scores,10)
    ndcg_at_30 = ndcg_at_k(rel_scores,30)
    ndcg_at_50 = ndcg_at_k(rel_scores,50)
    ndcg_at_100 = ndcg_at_k(rel_scores,100)
    ndcg_at_150 = ndcg_at_k(rel_scores,150)
    ndcg_at_all = ndcg_at_k(rel_scores,len(rel_scores))
    res_dict = {'ndcg@5':ndcg_at_5, 'ndcg@10':ndcg_at_10,'ndcg@30':ndcg_at_30,'ndcg@50':ndcg_at_50,'ndcg@100':ndcg_at_100, \
            'ndcg@150':ndcg_at_150,'ndcg@all':ndcg_at_all}
    return res_dict

def cal_rel_scores(mic_gt):
    """
    Calculate relevance_scores for each AMP, according to their groudtruth MIC label
    """
    mic_gt = np.array(mic_gt)
    scale = 2
    return np.exp((-scale)*(mic_gt - MAX_MIC))

def ndcg_at_k(relevance_scores,k):
    """
    Calculate ndcg@k.
    Args:
        relevance_scores: relevance socres of model output. Calculate by gt of MIC label.
        k: k of ndcg@k
    """
    relevance_scores = np.asarray(relevance_scores)
    k = min(k, len(relevance_scores))
    
    # DCG@k
    dcg = np.sum(relevance_scores[:k] / np.log2(np.arange(2, k + 2)))
    
    # IDCG@k
    sorted_scores = np.sort(relevance_scores)[::-1]
    idcg = np.sum(sorted_scores[:k] / np.log2(np.arange(2, k + 2)))
    
    # NDCG@k
    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    return ndcg
