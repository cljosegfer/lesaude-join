
import torch.nn.functional as F

def join_l1(input, target, alpha = 5.56, beta = 1.23):
    loss_l = F.binary_cross_entropy_with_logits(input['logits'], target['label'])
    loss_t = F.l1_loss(input['signal_embedding'], target['text_embedding'])
    return alpha * loss_l + beta * loss_t
