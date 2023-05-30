import random
from os import PathLike
from pathlib import Path
from typing import Any, List, Dict
from collections import defaultdict
import PIL.Image as Image
from torchvision import transforms, models
import os
import h5py
import numpy as np
import yaml
import torch
import pandas as pd
import cv2
from src.helpers import data_helper, vsumm_helper, bbox_helper

tvsumOrH5 = h5py.File('/home/jovyan/video_summary_dataset/video_summary_multi_modality/DSNet-main/datasets/eccv16_dataset_tvsum_google_pool5.h5')
origin_video = "/home/jovyan/video_summary_dataset/tvsum50/ydata-tvsum50-v1_1/video/"
tvsum_new_h5_vggish_path = '/home/jovyan/video_summary_dataset/video_summary_multi_modality/DSNet-main/datasets/eccv16_dataset_tvsum_google_pool5_vggish_caption.h5'
tsv_name = "/home/jovyan/video_summary_dataset/tvsum50/ydata-tvsum50-v1_1/data/ydata-tvsum50-info.tsv"
save_video_gt = "/home/jovyan/video_summary_dataset/tvsum50_gt/"
def load_tsv_hash_dict(tsv_name):
    tsvTmp = pd.read_csv(tsv_name, sep='\t')
    video_names = tsvTmp['video_id'].tolist()
    reHashDict = dict()
    i = 1
    for video_name in video_names:
        tmpName = "video_"+str(i)
        reHashDict[tmpName] = video_name
        i += 1
    return reHashDict
# summeOrH5 = h5py.File('/home/jovyan/video_summary_dataset/video_summary_multi_modality/DSNet-main/datasets/eccv16_dataset_summe_google_pool5.h5')
# vgg_path = "/home/jovyan/video_summary_dataset/summe_vggish/content/BMT/tmp/"
# caption_path = "/home/jovyan/video_summary_dataset/video_summary_multi_modality/Ours_VS/caption_npy/"
# summe_new_h5_vggish_path = '/home/jovyan/video_summary_dataset/video_summary_multi_modality/DSNet-main/datasets/eccv16_dataset_summe_google_pool5_vggish_caption.h5'
# shot_num = 0
# i = 0 
# for key in summeOrH5.keys():
#     orName = summeOrH5[key]['video_name'][...]
#     replaceOrName = str(orName.astype(str)).replace(' ','')
#     caption_name = os.path.join(caption_path,replaceOrName+'.npy')
#     vggish_name = os.path.join(vgg_path,replaceOrName+"_vggish.npy")
#     if not os.path.exists(vggish_name):
#         audio_vggish = np.zeros((100,128))
#     else:
#         audio_vggish = np.load(vggish_name)
#     if os.path.exists(caption_name):
#         caption_emb = np.load(caption_name)
#         #print(caption_emb.shape)
#     else:
#         print("not exists :{}".format(caption_name))
#     video_file = summeOrH5[key]
#     video_key = key
#     seq = video_file['features'][...].astype(np.float32)
#     gtscore = video_file['gtscore'][...].astype(np.float32)
#     cps = video_file['change_points'][...].astype(np.int32)
#     n_frames = video_file['n_frames'][...].astype(np.int32)
#     nfps = video_file['n_frame_per_seg'][...].astype(np.int32)
#     picks = video_file['picks'][...].astype(np.int32)
#     n_steps = video_file['n_steps'][...].astype(np.int32)
#     gtsummary = video_file['gtsummary'][...].astype(np.int32)
#     user_summary = video_file['user_summary'][...].astype(np.int32)
#     shot_num += cps.shape[0]
#     i += 1
#     # with h5py.File(summe_new_h5_vggish_path, 'a') as h5out:
#     #     h5out.create_dataset(f'{video_key}/vggish', data=audio_vggish)
#     #     h5out.create_dataset(f'{video_key}/caption', data=caption_emb)
#     #     h5out.create_dataset(f'{video_key}/features', data=seq)
#     #     h5out.create_dataset(f'{video_key}/gtscore', data=gtscore)
#     #     h5out.create_dataset(f'{video_key}/user_summary', data=user_summary)
#     #     h5out.create_dataset(f'{video_key}/change_points', data=cps)
#     #     h5out.create_dataset(f'{video_key}/n_frame_per_seg', data=nfps)
#     #     h5out.create_dataset(f'{video_key}/n_frames', data=n_frames)
#     #     h5out.create_dataset(f'{video_key}/picks', data=picks)
#     #     h5out.create_dataset(f'{video_key}/n_steps', data=n_steps)
#     #     h5out.create_dataset(f'{video_key}/gtsummary', data=gtsummary)
#     #     h5out.create_dataset(f'{video_key}/video_name', data=orName)
# # print(shot_num/i)
# checkH5File(summe_new_h5_vggish_path)
shot_num = 0
i = 0 
dict_video = load_tsv_hash_dict(tsv_name)
gt_time_dur = defaultdict(list)
for key in tvsumOrH5.keys():
    orName = dict_video[key]
    replaceOrName = orName#str(orName.astype(str)).replace(' ','')
    video_file = tvsumOrH5[key]
    video_key = key
    seq = video_file['features'][...].astype(np.float32)
    gtscore = video_file['gtscore'][...].astype(np.float32)
    cps = video_file['change_points'][...].astype(np.int32)
    n_frames = video_file['n_frames'][...].astype(np.int32)
    nfps = video_file['n_frame_per_seg'][...].astype(np.int32)
    picks = video_file['picks'][...].astype(np.int32)
    n_steps = video_file['n_steps'][...].astype(np.int32)
    gtsummary = video_file['gtsummary'][...].astype(np.int32)
    user_summary = video_file['user_summary'][...].astype(np.int32)
    keyshot_summ = vsumm_helper.get_keyshot_summ(
                gtscore, cps, n_frames, nfps, picks)
    video_path = os.path.join(origin_video,replaceOrName+".mp4")
    cap = cv2.VideoCapture(str(video_path))
    shot_num += cps.shape[0]
    i += 1
    fps_load = cap.get(5)
    # print(replaceOrName)
    # print(int(keyshot_summ.shape[0]/fps_load/60))
    # print(int(keyshot_summ.shape[0]/fps_load-int(keyshot_summ.shape[0]/fps_load/60)*60))
    idx = 0
    #idx // fps == all seconds -->all seconds / 60 = mins,all seconds-60 * mins = seconds
    while(idx<keyshot_summ.shape[0]):
        if keyshot_summ[idx] == True:
            idx_start_tmp = idx
            while(idx<keyshot_summ.shape[0] and keyshot_summ[idx] == True):
                idx+=1
            idx_end_tmp = idx-1
            s_tmp_min = int((idx_start_tmp/fps_load)/60)
            s_tmp_sec = int(idx_start_tmp/fps_load - s_tmp_min*60)

            e_tmp_min = int((idx_end_tmp/fps_load)/60)
            e_tmp_sec = int(idx_end_tmp/fps_load - e_tmp_min*60)
            #print(s_tmp_min,s_tmp_sec,e_tmp_min,e_tmp_sec,idx_start_tmp,idx_end_tmp)
            gt_time_dur[replaceOrName].append(str(int(idx_start_tmp/fps_load))+"-"+str(int(idx_end_tmp/fps_load)))
        else:
            idx += 1
    # assert cap is not None, f'Cannot open video: {video_path}'
    # print(cps.shape)
    # features = []
    # n_frames_read = 0
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break

    #     if n_frames_read<keyshot_summ.shape[0] and keyshot_summ[n_frames_read]:
    #         features.append(frame)

    #     n_frames_read += 1
    # H = features[0].shape[0]
    # W = features[0].shape[1]
    cap.release()
    # save_path = os.path.join(save_video_gt,replaceOrName+".mp4")
    # # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter(save_path,fourcc, 15.0, (W,H)) #W,H

    # for frame in features:
    #     out.write(frame)

    # # Release everything if job is finished
    # out.release()
    # print("process done write:{}".format(save_path))
#print(shot_num/i)
for k,v in gt_time_dur.items():
    print(k,v)