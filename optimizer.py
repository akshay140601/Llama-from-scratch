from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer
import numpy as np


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]
                beta1 = group['betas'][0]
                beta2 = group['betas'][1]
                eps = group['eps']
                weight_decay = group['weight_decay']
                correct_bias = group['correct_bias']
                #grad += weight_decay * p.data

                # Initialization for the first update
                if len(state) == 0:
                    state['t'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                    state['m_hat'] = torch.zeros_like(p.data)
                    state['v_hat'] = torch.zeros_like(p.data)
                    state['alpha_t'] = alpha

                state['t'] += 1
                # Update first and second moments of the gradients
                    
                state['m'] = beta1 * state['m'] + (1 - beta1) * grad
                state['v'] = beta2 * state['v'] + (1 - beta2) * (grad ** 2)

                # Bias correction

                state['m_hat'] = state['m'] / (1 - (beta1 ** state['t']))
                state['v_hat'] = state['v'] / (1 - (beta2 ** state['t']))

                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980

                state['alpha_t'] = (alpha * np.sqrt(1 - (beta2 ** state['t']))) / (1 - (beta1 ** state['t']))

                # Update parameters
                # Add weight decay after the main gradient-based updates.
                # Please note that the learning rate should be incorporated into this update.

                # Reference (https://arxiv.org/pdf/1711.05101.pdf)

                p.data = p.data - ((state['alpha_t'] * state['m']) / (torch.sqrt(state['v']) + eps)) - alpha * weight_decay * p.data

        return loss
