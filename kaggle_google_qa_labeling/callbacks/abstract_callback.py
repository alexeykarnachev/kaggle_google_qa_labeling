from abc import ABC


class AbstractCallback(ABC):
    def on_opt_step(self, learner):
        pass

    def on_train_end(self, learner):
        pass

    def on_epoch_end(self, learner):
        pass

    def on_train_start(self, learner):
        pass

    def on_eval_end(self, learner):
        pass
