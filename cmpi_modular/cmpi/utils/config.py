"""YAML configs with dotted command-line overrides, e.g. train.epochs=10."""

import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg, overrides):
    for item in overrides or []:
        key, value = item.split('=', 1)
        node = cfg
        *parents, leaf = key.split('.')
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    return cfg
