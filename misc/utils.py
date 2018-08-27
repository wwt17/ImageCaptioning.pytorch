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

# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i,j]
            if ix > 0 :
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(int(ix.cpu()))]
            else:
                break
        out.append(txt)
    return out

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
    return [True] + [i >= prefix_length or (i+r) % n >= gap for i in range(length-1)]

def mask_probs(probs, onehot, teach_mask):
    teach_mask = torch.tensor(teach_mask, device=onehot.device).unsqueeze(-1)
    return torch.where(teach_mask, onehot, probs)

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]
        input = to_contiguous(input).view(-1, input.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        output = - input.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)
