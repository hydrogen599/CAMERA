import os
import pandas as pd

from .base import Hook
from Ampmm_base.utils.metrics import *

class AnalyseHook(Hook):
    def __init__(self, work_dir) -> None:
        self.work_dir = work_dir
        self.samples_num = 0
        self.sequence = []
        self.y_true = []
        self.y_pred = []
        self.bacs = []
        
    def before_val_epoch(self, runner):
        # Load checkpoint
        ckpt_path = runner.cfg.ckpt_path
        if ckpt_path is not None and runner.local_rank == 0:
            runner.load_checkpoint(ckpt_path)
            runner.logger.info("Start test, load checkpoint from {}".format(ckpt_path))

    def after_val_iter(self, runner):
        sequence = runner.outputs['seq']
        preds = runner.outputs['model_outputs']
        labels = runner.outputs['labels']
        bacterium = runner.outputs['bacterium']
        self.samples_num += len(preds)
        self.y_true.extend(list(labels.cpu().numpy()))
        self.y_pred.extend(list(preds.cpu().numpy()))
        self.sequence.extend(list(sequence))
        self.bacs.extend(bacterium)

    def save_results(self):
        result_df = pd.DataFrame({'Sequence':self.sequence,'MIC_gt':self.y_true,'MIC_pred':self.y_pred,'Bacterium':self.bacs})
        # result_df.sort_values("MIC_gt", inplace=True)
        result_df.to_csv(os.path.join(self.work_dir,'prediction.csv'),index=False)

    def after_val_epoch(self, runner):
        reg_results = cal_reg_metrics(self.y_pred,self.y_true)
        ndcg_results = cal_ndcg_metrics_by_session(self.y_pred,self.y_true, self.bacs)
        results = {**reg_results,**ndcg_results}
        # for logger output logs
        runner.cur_val_results = results        
        runner.best_val_results = results
        # output test results
        self.save_results()


        ## train_and_test
        if runner.cfg.test_save:
            import time
            with open(os.path.join(self.work_dir, 'all.txt'), 'a') as log:
                log.write('time: {}\n'.format(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())))
                log.write(f'model_name: {runner.cfg.model.name}\n')
                log.write(f'task: {runner.cfg.task}\n')
                log.write(f'esm_mode: {runner.cfg.esm_mode}\n')
                log.write(f'epochs: {runner.cfg.epochs} batch_size:{runner.cfg.batch_size}\n')
                log.write(f'mse: {runner.cfg.mse} pair:{runner.cfg.pair}\n')
                for key, value in results.items():
                    log.write(f'\t{key}: \t{value}\n')
                log.write(f'#################################################################################\n')