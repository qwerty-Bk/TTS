import config


class NoamOpt:  # follows the one
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.n = 0
        self.warmup = config.warmup
        self.d_model = config.encoder_input_size
        self.lr = 0

    def step(self):
        self.n += 1
        self.lr = self.d_model ** (-0.5) * min(self.n ** (-0.5), self.n * self.warmup ** (-1.5))
        for p in self.optimizer.param_groups:
            p['lr'] = self.lr
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()
