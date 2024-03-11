# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import torch
import types

import dataclasses
from dataclasses import dataclass, field
import typing
import argparse


def patch_lr_scheduler(scheduler: torch.optim.lr_scheduler._LRScheduler) \
     -> torch.optim.lr_scheduler._LRScheduler:
    """ Monkey patches an lr scheduler to apply lr_scale to the learning rate. """

    old_step = scheduler.__class__.step

    # Hack to apply lr_scale to the learning rate
    # Because some optimziers use the current lr to update the learning rate, we need to scale it ourselves
    def scaled_step(self: torch.optim.lr_scheduler._LRScheduler, *args, **kwdargs):
        groups = self.optimizer.param_groups

        # Undo scaling frop previous step
        for group in groups:
            if "lr_unscaled" in group:
                group["lr"] = group["lr_unscaled"]
        
        old_step(self, *args, **kwdargs)

        # Apply scaling to the learning rate
        for group in groups:
            group["lr_unscaled"] = group["lr"]
            group["lr"] = group["lr"] * group.get("lr_scale", 1.0)


    # Monkey patch the step function
    scheduler.step = types.MethodType(scaled_step, scheduler)

    # Make sure the initial LR is scaled
    for group in scheduler.optimizer.param_groups:
        group["lr_unscaled"] = group["lr"]
        group["lr"] = group["lr"] * group.get("lr_scale", 1.0)

    return scheduler




def make_param_groups(model: torch.nn.Module, weight_decay: float, layer_decay: float):
    """
    Adds parameters groups to implement layer decay.
    !! Important !! You must use the `lr_scale` parameter in your lr scheduler to apply the layer decay.

    Model should implement:
        - no_weight_decay() -> list of parameter names that should not be weight decayed
        - num_layers() -> int, number of layers in the model
        - get_layer_id(str) -> int, get the layer id of a parameter

    Adapted from BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    no_weight_decay = model.no_weight_decay() if hasattr(model, 'no_weight_decay') else []
    num_layers = model.num_layers() if hasattr(model, 'num_layers') else 1
    layer_scales = [layer_decay ** (num_layers - i) for i in range(num_layers + 1)]

    if layer_decay < 1.0 and not hasattr(model, 'get_layer_id'):
        raise ValueError("Layer decay is not supported for this model. Set layer decay to 1.0 or implement get_layer_id()")

    param_groups = {}

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # Don't apply weight decay to certain parameters
        if p.ndim == 1 or n in no_weight_decay:
            group_name_decay = "no_decay"
            this_weight_decay = 0.
        else:
            group_name_decay = "decay"
            this_weight_decay = weight_decay
        
        layer_id = model.get_layer_id(n) if hasattr(model, 'get_layer_id') else 1
        group_name = f"{group_name_decay}_layer{layer_id}"

        if group_name not in param_groups:
            param_groups[group_name] = {
                "weight_decay": this_weight_decay,
                # Note: lr_scale isn't used by pytorch--we have to apply it ourselves!
                "lr_scale": layer_scales[layer_id],
                "params": [],
            }
        
        # print(f"{n} -> {group_name} (decay: {this_weight_decay}, lr_scale: {layer_scales[layer_id]})")

        param_groups[group_name]["params"].append(p)

    return list(param_groups.values())





def config(cls):
    """ Decorator that turns a dataclass object into something json serializable."""
    cls = dataclass(cls)

    def save(self):
        def _save(x):
            if dataclasses.is_dataclass(x):
                return x.save()
            elif isinstance(x, (list, tuple)):
                return [_save(y) for y in x]
            elif type(x) in (bool, int, float, str):
                return x
            else:
                raise ValueError(f"Config field {field.name} is not a primitive type or config object.")
            
        return {
            field.name: _save(getattr(self, field.name))
            for field in dataclasses.fields(self)
        }
    
    cls.save = save
    
    @classmethod
    def load(cls, dict):
        def _load(x, type):
            if dataclasses.is_dataclass(type):
                return type.load(x)
            elif typing.get_origin(type) in (list, tuple):
                assert len(typing.get_args(type)) == 1, "Only homogenous lists/tuples are supported."
                return typing.get_origin(type)(_load(y, typing.get_args(type)[0]) for y in x)
            elif type in (bool, int, float, str):
                return type(x)
            else:
                raise ValueError(f"Config field {field.name} is not a primitive type or config object.")
        
        return cls(**{
            field.name: _load(dict[field.name], field.type)
            for field in dataclasses.fields(cls)
        })
    
    cls.load = load
    
    def mutate(self, x:dict=None, **kwargs):
        """ Returns a new config with the given kwarg fields mutated. Will error out on unknown fields. """
        
        if x is not None and len(kwargs) > 0:
            raise ValueError("Cannot mutate with both a dict and kwargs.")
        

        # Allow keys like "a.b.c" to be set as "a: { b: { c: ... } }"
        if x is not None:
            x_new = {}
            for k, v in x.items():
                x_base = x_new
                ks = k.split(".")
                for kp in k.split(".")[:-1]:
                    if kp not in x_base:
                        x_base[kp] = {}
                    x_base = x_base[kp]
                x_base[ks[-1]] = v
            x = x_new


        def _mutate(obj, value):
            if dataclasses.is_dataclass(obj):
                # If the value is a dataclass, we'll overwrite the whole object with it
                if dataclasses.is_dataclass(value):
                    return value.save()  # Return copy of value
                # If the value is a dict, we'll mutate the object with it
                elif isinstance(value, dict):
                    out = obj.save()
                    field_names = {f.name for f in dataclasses.fields(obj)}
                    for k, v in value.items():
                        if k not in field_names:
                            raise ValueError(f"Couldn't mutate: unknown field {k} for config {obj.__class__.__name__}.")
                        out[k] = _mutate(getattr(obj, k), v)
                    return out
                else:
                    return ValueError(f"Cannot mutate config with {value} of type {type(value)}.")
            elif isinstance(obj, (list, tuple)):
                return obj.__class__(_mutate(x, y) for x, y in zip(obj, value))
            else:
                return value
            
        return cls.load(_mutate(self, x if x is not None else kwargs))

    cls.mutate = mutate


    @classmethod
    def parse(cls, parser: argparse.ArgumentParser, prefix=""):
        for field in dataclasses.fields(cls):
            if dataclasses.is_dataclass(field.type):
                field.type.parse(parser, prefix + field.name + ".")
            else:
                parser.add_argument(f"--{prefix}{field.name}", type=field.type)

    cls.parse = parse

    return cls

