from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import torch
import torch.nn as nn
from torch.autograd import Variable
import random

def if_use_att(caption_model):
    # Decide if load attention feature according to caption model
    if caption_model in ['show_tell', 'all_img', 'fc']:
        return False
    return True

def seq_tolist(ids):
    a = ids.tolist()
    while a and a[-1] == 0:
        a.pop()
    return a

def tolist(ids):
    return list(map(seq_tolist, ids.cpu().numpy()))

def decode_sequence(loader, ids):
    words = loader.ids_to_words(ids)
    pad = loader.ids_to_words(0)
    ret = []
    for i in range(len(words)):
        sent = list(words[i])
        while sent and sent[-1] == pad:
            sent.pop()
        ret.append(' '.join(sent))
    return ret

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

def to_onehot(a, n, dtype=torch.long, device=None):
    if device is None:
        device = a.device
    ret = torch.zeros(a.shape + (n,), dtype=dtype, device=device)
    a = a.unsqueeze(-1)
    ret.scatter_(-1, a, torch.ones(a.shape, dtype=dtype, device=device))
    return ret

def make_teach_mask(length, opt):
    prefix_length = opt.current_teach_mask_prefix_length
    gap = opt.teach_gap
    cont = opt.teach_cont
    n = gap + cont
    r = random.randrange(n)
    return [opt.teach_bos] + [i >= prefix_length or (i+r) % n >= gap for i in range(length-1)]

def mask_probs(probs, onehot, teach_mask):
    teach_mask = torch.tensor(teach_mask, device=onehot.device).unsqueeze(-1)
    return torch.where(teach_mask, onehot, probs)

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask, reduce=True):
        # truncate to the same size
        seq_len = input.size(1)
        target = target[:, :seq_len]
        mask = mask[:, :seq_len]
        target = target.unsqueeze(-1)
        output = - input.gather(-1, target).squeeze(-1) * mask
        if reduce:
            return output.sum() / mask.sum()
        else:
            return output.sum(1) / mask.sum(1)

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)
