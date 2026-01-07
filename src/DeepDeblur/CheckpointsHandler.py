

class CheckpointsHandler:
    def __init__(self, save_every: int, increasing_metric, output_path: str):
        self.increasing_metric = increasing_metric
        self.save_every = save_every
        self.previous_best_value = -float('inf') if increasing_metric else float('inf')

    def check_save_every(self, current_epoch) -> bool:
        return current_epoch%self.save_every == 0
    
    def metric_has_improved(self, metric_val):
        if self.increasing_metric:
            if metric_val > self.previous_best_value:
                self.previous_best_value = metric_val
                return True
        else:
            if metric_val < self.previous_best_value:
                self.previous_best_value = metric_val
                return True
        return False
    
    def save_model(self, model_dict, optim_state, epoch, save_type):
        pass

