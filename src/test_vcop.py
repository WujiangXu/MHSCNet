import itertools
import torch
import random
import numpy as np
import torch.nn.functional as F

def clip_label_generate():
    random_clip_order = [0,1,2,3,4,5]#[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
    clip_label_dict = {0:[1,2,3],1:[1,3,2],2:[2,1,3],3:[2,3,1],4:[3,1,2],5:[3,2,1]}
    random.shuffle(random_clip_order)
    label = list()
    for tmp in random_clip_order:
        label.append(clip_label_dict[tmp])
    return np.array(random_clip_order),label

def self_supervise_clip(x,order):
    x = F.adaptive_avg_pool1d(x.unsqueeze(0).permute(0,2,1),24)
    x = x.permute(0,2,1).squeeze().reshape(3,-1)
    x1,x2,x3 = x[0,:],x[1,:],x[2,:]
    x_l = [x1,x2,x3]
    x_r = list()
    for l_tmp in order:
        x_tmp = list()
        for num in l_tmp:
            x_tmp.append(x_l[num-1])
        x_tmp = torch.cat(x_tmp,dim=0)
        x_r.append(x_tmp.unsqueeze(0))
    x_r = torch.cat(x_r,dim=0)
    return x_r

label,random_clip_order = clip_label_generate()
seq = torch.zeros(768,1024)
seq_order = self_supervise_clip(seq,random_clip_order)
print(seq_order.shape)