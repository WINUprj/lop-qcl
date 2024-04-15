import pennylane
import torch


class Trainer:
    def __init__(
        self,
        task,
        model,
        loss,
        optimizer,
        n_steps,
        logger,
        device,
    ) -> None:
        self.task = task
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.n_steps = n_steps
        self.logger = logger
        self.device = device

    def train_single_run(self):
        for t, (x, y) in enumerate(self.task):
            # Make prediction
            x, y = x.to(self.device), y.to(self.device)
            pred = self.model(x)

            self.optimizer.zero_grad()
            loss = self.loss(pred)
            self.optimizer.step()
            
            # TODO: Measure things

            if t + 1 >= self.n_steps:
                break

    def run_train(self, n_runs: int):
        self.model.to(self.device)
        self.model.train()
        
        run = 1
        while run <= n_runs:
            self.train_single_run()
            run += 1
