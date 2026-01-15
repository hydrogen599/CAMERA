from typing import List,Dict
import imp
from unicodedata import name
import torch
import torch.nn as nn
import torch.nn.functional as F

class Ensemble_MSELoss(nn.Module):
    """
    Cross entropy loss for ensemble model
    """
    def __init__(self, loss_weight:Dict):
        super(Ensemble_MSELoss,self).__init__()
        self.loss_weight=loss_weight
    
    def forward(self, pred_results, labels):
        losses = {}
        for name,weight in self.loss_weight.items():
            weighted_loss = weight*F.mse_loss(pred_results[name], labels)
            losses[name] = weighted_loss
        sum_loss = sum([_value for _key,_value in losses.items()])
        return sum_loss

class Ensemble_ContrastiveLoss(torch.nn.Module):
    def __init__(self, loss_weight:Dict, sigma=1.0):
        super(Ensemble_ContrastiveLoss, self).__init__()
        self.loss_weight=loss_weight
        self.sigma = sigma

    def cal_loss(self, pred_result, Y):
        Y = Y.reshape(-1, 1)
        pred_result = pred_result.reshape(-1, 1)

        rel_diff = Y - Y.T
        rel_diff_tensor = torch.tensor(rel_diff).cuda()
        zero = torch.tensor(0, dtype=torch.float32).cuda()

        C = torch.where(
                rel_diff_tensor != 0, torch.logaddexp(self.sigma * pred_result, self.sigma * pred_result.t()), zero
            ) - torch.where(rel_diff_tensor > 0, self.sigma * pred_result, zero) \
              - torch.where(rel_diff_tensor < 0, self.sigma * pred_result.t(), zero)

        return C
    
    # TODO
    def forward(self, pred_results, labels):
        losses={}
        for name,weight in self.loss_weight.items():
            weighted_loss = weight*self.cal_loss(pred_results[name], labels)
            losses[name] = weighted_loss
        sum_loss = sum([_value for _key,_value in losses.items()])
        return sum_loss

class Ensemble_ListLoss(torch.nn.Module):
    def __init__(self, loss_weight:Dict, sigma=1.0):
        super(Ensemble_ListLoss, self).__init__()
        self.loss_weight=loss_weight
        self.sigma = sigma
    
    def cal_loss(self, pred_result, Y):
        Y = Y.reshape(-1, 1)
        pred_result = pred_result.reshape(-1, 1)

        P_y = F.softmax(Y, dim=0)
        P_pred = F.softmax(pred_result, dim=0)
        return torch.sum(P_y*torch.log(P_pred))
    
    # TODO
    def forward(self, pred_results, labels):
        losses={}
        for name,weight in self.loss_weight.items():
            weighted_loss = weight*self.cal_loss(pred_results[name], labels)
            losses[name] = weighted_loss
        sum_loss = sum([_value for _key,_value in losses.items()])
        return sum_loss

def build_loss_fn(loss_name, loss_cfg=None):
    if loss_name == 'Ensemble_MSELoss':
        return Ensemble_MSELoss(**loss_cfg.kwargs)
    elif loss_name == 'Ensemble_ContrastiveLoss':
        return Ensemble_ContrastiveLoss(**loss_cfg.kwargs)
    elif loss_name == 'Ensemble_ListLoss':
        return Ensemble_ListLoss(**loss_cfg.kwargs)
    
    else:
        loss_fn = getattr(torch.nn,loss_name)
        return loss_fn(**loss_cfg.kwargs)

class CombinedLossEvaluator(object):
    """
    Combined multiple loss evaluator
    """
    def __init__(self, loss_evaluators, loss_weights):

        self.loss_evaluators = loss_evaluators
        self.loss_weights = loss_weights
        
    def __call__(self, pred_results, gt, **kwargs):
        comb_loss_dict = {}
        for loss_name, loss_evaluator in self.loss_evaluators.items():
            loss = loss_evaluator(pred_results,gt)
            weight = self.loss_weights[loss_name]
            if isinstance(loss,dict):
                loss = {k:v*weight for k,v in loss.items()}
            else:
                comb_loss_dict[loss_name] = loss*weight
        return comb_loss_dict

def build_loss_evaluator(cfg):
    loss_evaluators = dict()
    loss_weights = dict()
    loss_dict = cfg.model.losses.copy()
    for loss_name,loss_cfg in loss_dict.items():
        loss_evaluator = build_loss_fn(loss_name,loss_cfg)
        loss_evaluators[loss_name] = loss_evaluator
        loss_weights[loss_name] = loss_cfg.weight
    return CombinedLossEvaluator(loss_evaluators,loss_weights)