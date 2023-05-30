import numpy as np
import torch
import clip
from pathlib import Path
import PIL.Image as Image
from torchvision import transforms, models
from torch.utils.data import Dataset,DataLoader
import os
import h5py
import torch.nn.functional as F

def detect_cap(str_tmp):
    start_idx = str_tmp.find("sentence")
    if start_idx != '-1':
        return str_tmp[start_idx+12:-1]
    else:
        return "e"
read_file_name_rect = "caption_youtube.txt"
name = list()
caption = list()
with open(read_file_name_rect, "r") as ins:
    i = 0 
    for line in ins:   
        if line != '\n':
            line = line.replace('\n','')
            if i%2==0: # save name
                line = Path(line).name
                line = line.replace('video name:','')
                name.append(line)
            else: # save caption
                line = line.split('}')
                tmp_caption = list()
                for tmp in line:
                    process_sentence = detect_cap(tmp)
                    if process_sentence != "":
                        tmp_caption.append(detect_cap(tmp))
                caption.append(tmp_caption)
            i += 1
caption_dict = dict()
for i in range(len(name)):
    caption_dict[name[i]] = caption[i]
with torch.no_grad():
    model, preprocess = clip.load("/home/jovyan/video_summary_dataset/CLIP-main/ViT-B-32.pt")
    model.cuda().eval()
for name,sentences in caption_dict.items():
    name = name.replace('.mp4','')
    print(name)
    # print(len(sentences))
    text_tokens = clip.tokenize([ desc for desc in sentences]).cuda()
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.view(1,-1).unsqueeze(0)
        #print(text_features.shape)
        text_features = F.adaptive_avg_pool1d(text_features,1024)
    text_features = text_features.squeeze().cpu().numpy()
    np.save("/home/jovyan/video_summary_dataset/video_summary_multi_modality/Ours_VS/caption_youtube/"+name+'.npy',text_features)