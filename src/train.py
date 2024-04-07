import pennylane
import torch


class Trainer:
    def __init__(self, task) -> None:
        self.task = task