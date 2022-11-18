# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# LARS optimizer, implementation from MoCo v3:
# https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

import paddle
import plsc.optimizer


class LARS(plsc.optimizer.optimizer.Optimizer):
    """
    LARS optimizer, no rate scaling or weight decay for parameters <= 1D.
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, trust_coefficient=0.001):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, trust_coefficient=trust_coefficient)
        super().__init__(params, defaults)

    @paddle.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim > 1: # if not normalization gamma/beta or bias
                    dp = dp + g['weight_decay'] * p
                    param_norm = paddle.norm(p)
                    update_norm = paddle.norm(dp)
                    one = paddle.ones_like(param_norm)
                    q = paddle.where(param_norm > 0.,
                                    paddle.where(update_norm > 0,
                                    (g['trust_coefficient'] * param_norm / update_norm), one),
                                    one)
                    dp = dp.multiply(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = paddle.zeros_like(p)
                mu = param_state['mu']
                mu.copy_(mu * g['momentum'] + dp, False)
                p.copy_(p - mu * g['lr'], False)