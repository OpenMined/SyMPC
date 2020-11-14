import torch

def modular_to_real(x, session):
    config = session.conf

    max_val = config.max_value
    min_val = config.min_value
    ring_size = config.ring_size

    mask_pos = x > max_val
    mask_neg = x < min_val
    mask = -mask_pos.long() + mask_neg.long()

    real = x + mask * ring_size
    return real


def simulate_get():
    def get(self, *args, **kwargs):
        return self

    torch.Tensor.get = get
