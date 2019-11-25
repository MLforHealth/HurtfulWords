import torch.nn as nn
from torch.autograd import Function

class GradientReversalFunction(Function):
    """
    Adapted from https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/utils.py
    
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()  # Keep data the same for forward pass

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)  # Creates tensor with lambda as data but same dtype and device as grads
        dx = -lambda_ * grads  # Reverse the gradient and broadcast
        return dx, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

