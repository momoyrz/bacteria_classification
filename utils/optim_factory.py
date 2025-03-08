import json

import torch


def create_optimizer(args, model, filter_bias_and_bn=True):
    opt_lower = args.opt.lower()

    if filter_bias_and_bn:
        parameters = get_parameter_groups(model, weight_decay=args.weight_decay)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    opt_args = dict(lr=args.lr, weight_decay=args.weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas

    opt_split = opt_lower.split('_')
    if opt_split[0] == 'sgd':
        optimizer = torch.optim.SGD(parameters, momentum=0.9, nesterov=(opt_lower == 'nesterov'), **opt_args)
    elif opt_split[0] == 'adam':
        optimizer = torch.optim.Adam(parameters, **opt_args)
    elif opt_split[0] == 'adamw':
        optimizer = torch.optim.AdamW(parameters, **opt_args)
    else:
        raise ValueError('Optimizer {} not supported'.format(args.opt))

    return optimizer



def get_parameter_groups(model, weight_decay=1e-5):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay

        if group_name not in parameter_group_names:

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())