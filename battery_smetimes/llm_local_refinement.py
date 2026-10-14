"""TRAIN-only local-head optimization after freezing visible LLM encodings.

This changes optimization, never initializes from answer-fit coefficients.
The closure must use the ordinary supervised forecasting objective.
"""
import torch


def local_refinement_optimizer(network):
    parameters=[p for name,p in network.named_parameters()
                if p.requires_grad and (name.startswith('local_head.') or name=='local_query')]
    if not parameters:raise ValueError('refinement requires a local LLM decoder')
    return torch.optim.LBFGS(parameters,lr=1.,max_iter=1,history_size=20,line_search_fn='strong_wolfe')


def local_refinement_step(network,optimizer,loss_function):
    parameters=[p for group in optimizer.param_groups for p in group['params']]
    evaluations=0
    def closure():
        nonlocal evaluations
        network.zero_grad(set_to_none=True)
        loss=loss_function()
        if not torch.isfinite(loss):raise ValueError('nonfinite local refinement objective')
        gradients=torch.autograd.grad(loss,parameters)
        for parameter,gradient in zip(parameters,gradients):
            if not torch.isfinite(gradient).all():raise ValueError('nonfinite local refinement gradient')
            parameter.grad=gradient
        evaluations+=1
        return loss
    optimizer.step(closure)
    with torch.no_grad():value=loss_function()
    return float(value),evaluations
