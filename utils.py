
import torch

SIGNAL_CROP_LEN = 2560
SIGNAL_NON_ZERO_START = 571

def get_inputs(batch, apply = "nothing", device = "cuda"):
    # (B, C, L)
    if batch.shape[1] > batch.shape[2]:
        batch = batch.permute(0, 2, 1)

    B, n_leads, signal_len = batch.shape

    if apply == "non_zero":
        transformed_data = torch.zeros(B, n_leads, SIGNAL_CROP_LEN)
        for b in range(B):
            start = SIGNAL_NON_ZERO_START
            diff = signal_len - start
            if start > diff:
                correction = start - diff
                start -= correction
            end = start + SIGNAL_CROP_LEN
            for l in range(n_leads):
                transformed_data[b, l, :] = batch[b, l, start:end]
    else:
        transformed_data = batch.float()

    return transformed_data.to(device)