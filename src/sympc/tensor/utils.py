import torch

def modulo(x, session):
    config = session.config

    max_val = config.max_value
    min_val = config.min_value
    ring_size = config.ring_size

    mask_pos = x > max_val
    mask_neg = x < min_val
    mask = -mask_pos.long() + mask_neg.long()

    result = (x + mask * ring_size).long()

    return result
