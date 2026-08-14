import copy


class EarlyStopping:
    def __init__(self, patience = 5, model=None,min_delta=0.0) -> None:
        self.patience = patience    # self.patience =10
        self.min_delta = min_delta

        self.pre_loss = float('inf')
        self.no_improvement_count = 0
        self.best_model_wts = copy.deepcopy(model.state_dict())
        self.bestepoch = 0

    def should_stop(self, epoch_loss, epoch, model):
        if epoch == 0 or (epoch_loss < self.pre_loss - self.min_delta):
            self.pre_loss = epoch_loss
            self.bestepoch = epoch
            self.no_improvement_count = 0
            self.best_model_wts = copy.deepcopy(model.state_dict())
            return False
        else:
            self.no_improvement_count += 1
            return self.no_improvement_count >= self.patience

