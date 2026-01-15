from .base import Hook
from Ampmm_base.utils.metrics import *

class RankEvalHook(Hook):
    def __init__(self) -> None:
        self.samples_num = 0
        self.y_true = []
        self.y_pred = []
        self.bacs = []

    def reset(self):
        self.samples_num = 0
        self.y_true = []
        self.y_pred = []
        self.bacs = []
    
    def calculate_total_score(self, result_dict):
        score_dict = {
                "ndcg@5":0.2,
                "ndcg@10":0.5,
                "ndcg@30":0.0,
                "ndcg@50":0.0,
                "ndcg@100":0.0,
                "ndcg@150":0.0,
                "ndcg@all":0.3}
        score = 0
        for k,v in score_dict.items():
            score += v*result_dict[k]
        return score

    def compare_results(self, cur_results, best_results):
        if self.calculate_total_score(cur_results) > self.calculate_total_score(best_results):
            return True
        return False

    def before_run(self, runner):
        self.reset()
        runner.best_val_results = {"ndcg@5":0, "ndcg@10":0,"ndcg@30":0,"ndcg@50":0,"ndcg@100":0,"ndcg@150":0,"ndcg@all":0}

    def before_val_epoch(self, runner):
        self.reset()
    
    def after_val_iter(self, runner):
        preds = runner.outputs['model_outputs']
        labels = runner.outputs['labels']
        bacterium = runner.outputs['bacterium']
        self.samples_num += len(preds)
        self.y_true.extend(list(labels.cpu().numpy()))
        self.y_pred.extend(list(preds.cpu().numpy()))
        self.bacs.extend(bacterium)


    def after_val_epoch(self, runner):
        ndcg_results = cal_ndcg_metrics_by_session(self.y_pred,self.y_true, self.bacs)
        results = {**ndcg_results}
        runner.cur_val_results = results 
        if self.compare_results(results, runner.best_val_results):
            runner.best_val_results = results
            if runner.local_rank == 0 and runner.mode == 'val':
                runner.save_checkpoint()


class RegEvalHook(Hook):
    def __init__(self) -> None:
        self.samples_num = 0
        self.y_true = []
        self.y_pred = []

    def reset(self):
        self.samples_num = 0
        self.y_true = []
        self.y_pred = []
    
    def calculate_total_score(self, result_dict):
        score_dict = {"top10_mse":0.00,"top30_mse":0., \
                      "top50_mse":0.00,"top100_mse":0.0,'mse':0.3,"pos_mse":0.7}
        score = 0
        for k,v in score_dict.items():
            score += v*result_dict[k]
        return score

    def compare_results(self, cur_results, best_results):
        if self.calculate_total_score(cur_results) < self.calculate_total_score(best_results):
            return True
        return False

    def before_run(self, runner):
        self.reset()
        runner.best_val_results = {"top10_mse":1e7,"top30_mse":1e7, "top50_mse":1e7, \
                                   "top100_mse":1e7,'mse':1e7,"pos_mse":1e7}

    def before_val_epoch(self, runner):
        self.reset()
    
    def after_val_iter(self, runner):
        preds = runner.outputs['model_outputs']
        labels = runner.outputs['labels']
        self.samples_num += len(preds)
        self.y_true.extend(list(labels.cpu().numpy()))
        self.y_pred.extend(list(preds.cpu().numpy()))

    def after_val_epoch(self, runner):
        reg_results = cal_reg_metrics(self.y_pred,self.y_true)
        results = {**reg_results}
        runner.cur_val_results = results 
        if self.compare_results(results, runner.best_val_results):
            runner.best_val_results = results
            if runner.local_rank == 0 and runner.mode == 'val':
                runner.save_checkpoint()