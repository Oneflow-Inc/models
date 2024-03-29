import math
import oneflow as flow
from oneflow.nn.optimizer.lr_scheduler import LrScheduler


class PolynomialLR(LrScheduler):
    """This operator creates a polynomial decayed learning rate scheduler.
    The learning rate will be updated as follows:
    If cycle is `True`, the equation is:
    .. math::
        & decay\\_batch = decay\\_batch*ceil(\\frac{current\\_batch}{decay\\_batch})
        & learning\\_rate = (base\\_lr-end\\_lr)*(1-\\frac{current\\_batch}{decay\\_batch})^{pow}+end\\_lr
    If cycle is `False`, the equation is:
    .. math::
        & decay\\_batch = min(decay\\_batch, current\\_batch)
        & learning\\_rate = (base\\_lr-end\\_lr)*(1-\\frac{current\\_batch}{decay\\_batch})^{pow}+end\\_lr
    Args:
        steps (int): The decayed steps
        end_learning_rate (float, optional): The final learning rate. Defaults to 0.0001.
        power (float, optional): The power of polynomial. Defaults to 1.0.
        cycle (bool, optional): If cycle is true, the scheduler will decay the learning rate every decay steps. Defaults to False.
    For example:
        .. code-block:: python
            import oneflow as flow
           
            ... 
            polynomial_scheduler = flow.optimizer.lr_scheduler.PolynomialScheduler(optimizer,
                                                                           steps=5,
                                                                           end_learning_rate=0.00001,
                                                                           power=2)
            for epoch in range(num_epoch):
                train(...)
                polynomial_scheduler.step()
    """

    def __init__(
        self,
        optimizer,
        steps: int,
        end_learning_rate: float = 0.0001,
        power: float = 1.0,
        cycle: bool = False,
        last_step=-1,
        verbose=False,
    ):
        assert steps > 0, f"steps must greater than zero, but got {steps}"
        self.max_decay_steps = steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.cycle = cycle
        super().__init__(optimizer, last_step, verbose)

    def get_lr(self):
        decay_batch = self.max_decay_steps
        cur_batch = self.last_step
        if self.cycle:
            decay_batch = decay_batch * math.ceil(cur_batch / decay_batch)
        else:
            cur_batch = min(cur_batch, decay_batch)
        return [
            (base_lr - self.end_learning_rate)
            * ((1 - cur_batch / decay_batch) ** (self.power))
            + self.end_learning_rate
            for base_lr in self.base_lrs
        ]

    def _generate_conf_for_graph(self, opt_confs):
        # CosineDecayLR is the same as CosineDecayConf in nn.Graph
        for opt_conf in opt_confs:
            learning_rate_decay_conf = opt_conf.mutable_learning_rate_decay()
            learning_rate_decay_conf.mutable_polynomial_conf().set_decay_batches(
                self.max_decay_steps
            )
            learning_rate_decay_conf.mutable_polynomial_conf().set_end_learning_rate(
                self.end_learning_rate
            )
            learning_rate_decay_conf.mutable_polynomial_conf().set_power(self.power)
            learning_rate_decay_conf.mutable_polynomial_conf().set_cycle(self.cycle)
