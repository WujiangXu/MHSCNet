import numpy as np
import torch
from pathlib import Path
import os
import h5py
import pandas as pd

def checkH5File(h5_name):
    fileTmp = h5py.File(h5_name)
    for key in fileTmp.keys():
        print(key)
        print(fileTmp[key])

summeOrH5 = h5py.File('/home/jovyan/video_summary_dataset/video_summary_multi_modality/DSNet-main/datasets/eccv16_dataset_summe_google_pool5.h5')
vgg_path = "/home/jovyan/video_summary_dataset/summe_vggish/content/BMT/tmp/"
caption_path = "/home/jovyan/video_summary_dataset/video_summary_multi_modality/Ours_VS/caption_npy/"
summe_new_h5_vggish_path = '/home/jovyan/video_summary_dataset/video_summary_multi_modality/DSNet-main/datasets/eccv16_dataset_summe_google_pool5_vggish_caption.h5'
for key in summeOrH5.keys():
    orName = summeOrH5[key]['video_name'][...]
    replaceOrName = str(orName.astype(str)).replace(' ','')
    print(key,replaceOrName)
    caption_name = os.path.join(caption_path,replaceOrName+'.npy')
    vggish_name = os.path.join(vgg_path,replaceOrName+"_vggish.npy")
    if not os.path.exists(vggish_name):
        audio_vggish = np.zeros((100,128))
    else:
        audio_vggish = np.load(vggish_name)
    if os.path.exists(caption_name):
        caption_emb = np.load(caption_name)
        #print(caption_emb.shape)
    else:
        print("not exists :{}".format(caption_name))
    video_file = summeOrH5[key]
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
    # with h5py.File(summe_new_h5_vggish_path, 'a') as h5out:
    #     h5out.create_dataset(f'{video_key}/vggish', data=audio_vggish)
    #     h5out.create_dataset(f'{video_key}/caption', data=caption_emb)
    #     h5out.create_dataset(f'{video_key}/features', data=seq)
    #     h5out.create_dataset(f'{video_key}/gtscore', data=gtscore)
    #     h5out.create_dataset(f'{video_key}/user_summary', data=user_summary)
    #     h5out.create_dataset(f'{video_key}/change_points', data=cps)
    #     h5out.create_dataset(f'{video_key}/n_frame_per_seg', data=nfps)
    #     h5out.create_dataset(f'{video_key}/n_frames', data=n_frames)
    #     h5out.create_dataset(f'{video_key}/picks', data=picks)
    #     h5out.create_dataset(f'{video_key}/n_steps', data=n_steps)
    #     h5out.create_dataset(f'{video_key}/gtsummary', data=gtsummary)
    #     h5out.create_dataset(f'{video_key}/video_name', data=orName)
# checkH5File(summe_new_h5_vggish_path)

# tvsumOrH5 = h5py.File('/home/jovyan/video_summary_dataset/video_summary_multi_modality/DSNet-main/datasets/eccv16_dataset_tvsum_google_pool5.h5')
# vgg_path = "/home/jovyan/video_summary_dataset/tvsum_vggish/content/BMT/tmp/"
# caption_path = "/home/jovyan/video_summary_dataset/video_summary_multi_modality/Ours_VS/caption_npy_new/"
# tvsum_new_h5_vggish_path = '/home/jovyan/video_summary_dataset/video_summary_multi_modality/DSNet-main/datasets/eccv16_dataset_tvsum_google_pool5_vggish_caption_new.h5'
# tsv_name = "/home/jovyan/video_summary_dataset/tvsum50/ydata-tvsum50-v1_1/data/ydata-tvsum50-info.tsv"
# def load_tsv_hash_dict(tsv_name):
#     tsvTmp = pd.read_csv(tsv_name, sep='\t')
#     video_names = tsvTmp['video_id'].tolist()
#     reHashDict = dict()
#     i = 1
#     for video_name in video_names:
#         tmpName = "video_"+str(i)
#         reHashDict[tmpName] = video_name
#         i += 1
#     return reHashDict
# dict_video = load_tsv_hash_dict(tsv_name)
# for key in tvsumOrH5.keys():
#     orName = dict_video[key]
#     replaceOrName = orName#str(orName.astype(str)).replace(' ','')
#     caption_name = os.path.join(caption_path,replaceOrName+'.npy')
#     vggish_name = os.path.join(vgg_path,replaceOrName+"_vggish.npy")
#     # print(caption_name)
#     # print(vggish_name)
#     if not os.path.exists(vggish_name):
#         print("not exists :{}".format(vggish_name))
#         audio_vggish = np.zeros((100,128))
#     else:
#         audio_vggish = np.load(vggish_name)
#     if os.path.exists(caption_name):
#         caption_emb = np.load(caption_name)
#         #print(caption_emb.shape)
#     else:
#         print("not exists :{}".format(caption_name))
#     video_file = tvsumOrH5[key]
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
#     with h5py.File(tvsum_new_h5_vggish_path, 'a') as h5out:
#         h5out.create_dataset(f'{video_key}/vggish', data=audio_vggish)
#         h5out.create_dataset(f'{video_key}/caption', data=caption_emb)
#         h5out.create_dataset(f'{video_key}/features', data=seq)
#         h5out.create_dataset(f'{video_key}/gtscore', data=gtscore)
#         h5out.create_dataset(f'{video_key}/user_summary', data=user_summary)
#         h5out.create_dataset(f'{video_key}/change_points', data=cps)
#         h5out.create_dataset(f'{video_key}/n_frame_per_seg', data=nfps)
#         h5out.create_dataset(f'{video_key}/n_frames', data=n_frames)
#         h5out.create_dataset(f'{video_key}/picks', data=picks)
#         h5out.create_dataset(f'{video_key}/n_steps', data=n_steps)
#         h5out.create_dataset(f'{video_key}/gtsummary', data=gtsummary)
#         h5out.create_dataset(f'{video_key}/video_name', data=orName)
# checkH5File(tvsum_new_h5_vggish_path)
# tvsumOrH5 = h5py.File('/home/jovyan/video_summary_dataset/video_summary_multi_modality/DSNet-main/datasets/eccv16_dataset_tvsum_google_pool5.h5')
# vgg_path = "/home/jovyan/video_summary_dataset/tvsum_vggish/content/BMT/tmp/"
# tvsum_new_h5_vggish_path = '/home/jovyan/video_summary_dataset/video_summary_multi_modality/DSNet-main/datasets/eccv16_dataset_tvsum_google_pool5_vggish.h5'
# tsv_name = "/home/jovyan/video_summary_dataset/tvsum50/ydata-tvsum50-v1_1/data/ydata-tvsum50-info.tsv"
# def load_tsv_hash_dict(tsv_name):
#     tsvTmp = pd.read_csv(tsv_name, sep='\t')
#     video_names = tsvTmp['video_id'].tolist()
#     reHashDict = dict()
#     i = 1
#     for video_name in video_names:
#         tmpName = "video_"+str(i)
#         reHashDict[tmpName] = video_name
#         i += 1
#     return reHashDict
# dict_video = load_tsv_hash_dict(tsv_name)
# for key in tvsumOrH5.keys():
#     orName = dict_video[key]
#     vggish_name = os.path.join(vgg_path,orName+"_vggish.npy")
#     if not os.path.exists(vggish_name):
#         audio_vggish = np.zeros((100,128))
#     else:
#         audio_vggish = np.load(vggish_name)
#     video_file = tvsumOrH5[key]
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
#     with h5py.File(tvsum_new_h5_vggish_path, 'a') as h5out:
#         h5out.create_dataset(f'{video_key}/vggish', data=audio_vggish)
#         h5out.create_dataset(f'{video_key}/features', data=seq)
#         h5out.create_dataset(f'{video_key}/gtscore', data=gtscore)
#         h5out.create_dataset(f'{video_key}/user_summary', data=user_summary)
#         h5out.create_dataset(f'{video_key}/change_points', data=cps)
#         h5out.create_dataset(f'{video_key}/n_frame_per_seg', data=nfps)
#         h5out.create_dataset(f'{video_key}/n_frames', data=n_frames)
#         h5out.create_dataset(f'{video_key}/picks', data=picks)
#         h5out.create_dataset(f'{video_key}/n_steps', data=n_steps)
#         h5out.create_dataset(f'{video_key}/gtsummary', data=gtsummary)
#         h5out.create_dataset(f'{video_key}/video_name', data=orName)
# checkH5File(tvsum_new_h5_vggish_path)

# youtubeOrH5 = h5py.File('/home/jovyan/video_summary_dataset/video_summary_multi_modality/DSNet-main/datasets/eccv16_dataset_youtube_google_pool5.h5')
# vgg_path = "/home/jovyan/video_summary_dataset/youtube_vggish/content/drive/MyDrive/youtube_vggish/"
# caption_path = "/home/jovyan/video_summary_dataset/video_summary_multi_modality/Ours_VS/caption_youtube/"
# youtube_new_h5_vggish_path = '/home/jovyan/video_summary_dataset/video_summary_multi_modality/Ours_VS/datasets/eccv16_dataset_youtube_google_pool5_vggish_caption.h5'
# youtube_dict = dict()
# for key in youtubeOrH5.keys():
#     num_tmp = int(key[-2:])
#     youtube_dict[key] = 'v'+str(num_tmp+60)
#     # else:
#     #     num_tmp += 
# print(youtube_dict)
# for key in youtubeOrH5.keys():
#     video_file = youtubeOrH5[key]
#     orName = youtube_dict[key]
#     replaceOrName = orName#str(orName.astype(str)).replace(' ','')
#     caption_name = os.path.join(caption_path,replaceOrName+'.npy')
#     vggish_name = os.path.join(vgg_path,replaceOrName+"_vggish.npy")
#     # print(caption_name)
#     # print(vggish_name)
#     if not os.path.exists(vggish_name):
#         print("not exists :{}".format(vggish_name))
#         audio_vggish = np.zeros((100,128))
#     else:
#         audio_vggish = np.load(vggish_name)
#     if os.path.exists(caption_name):
#         caption_emb = np.load(caption_name)
#         #print(caption_emb.shape)
#     else:
#         print("not exists :{}".format(caption_name))
#     video_key = key
#     seq = video_file['features'][...].astype(np.float32)
#     gtscore = video_file['gtscore'][...].astype(np.float32)
#     cps = video_file['change_points'][...].astype(np.int32)
#     n_frames = video_file['n_frames'][...].astype(np.int32)
#     nfps = video_file['n_frame_per_seg'][...].astype(np.int32)
#     picks = video_file['picks'][...].astype(np.int32)
#     gtsummary = video_file['gtsummary'][...].astype(np.int32)
#     with h5py.File(youtube_new_h5_vggish_path, 'a') as h5out:
#         h5out.create_dataset(f'{video_key}/vggish', data=audio_vggish)
#         h5out.create_dataset(f'{video_key}/caption', data=caption_emb)
#         h5out.create_dataset(f'{video_key}/features', data=seq)
#         h5out.create_dataset(f'{video_key}/gtscore', data=gtscore)
#         h5out.create_dataset(f'{video_key}/change_points', data=cps)
#         h5out.create_dataset(f'{video_key}/n_frame_per_seg', data=nfps)
#         h5out.create_dataset(f'{video_key}/n_frames', data=n_frames)
#         h5out.create_dataset(f'{video_key}/picks', data=picks)
#         h5out.create_dataset(f'{video_key}/gtsummary', data=gtsummary)
#         h5out.create_dataset(f'{video_key}/video_name', data=orName)
# checkH5File(youtube_new_h5_vggish_path)

# ovpOrH5 = h5py.File('/home/jovyan/video_summary_dataset/video_summary_multi_modality/DSNet-main/datasets/eccv16_dataset_ovp_google_pool5.h5')
# vgg_path = "/home/jovyan/video_summary_dataset/ovp_vggish/content/drive/MyDrive/ovp_vggish/"
# caption_path = "/home/jovyan/video_summary_dataset/video_summary_multi_modality/Ours_VS/caption_ovp/"
# ovp_new_h5_vggish_path = '/home/jovyan/video_summary_dataset/video_summary_multi_modality/Ours_VS/datasets/eccv16_dataset_ovp_google_pool5_vggish_caption_new.h5'
# ovp_dict = dict()
# for key in ovpOrH5.keys():
#     num_tmp = int(key.replace('video_',''))+20
#     ovp_dict[key] = 'v'+str(num_tmp)
# # print(ovp_dict)
# for key in ovpOrH5.keys():
#     orName = ovp_dict[key]
#     replaceOrName = orName#str(orName.astype(str)).replace(' ','')
#     caption_name = os.path.join(caption_path,replaceOrName+'.npy')
#     vggish_name = os.path.join(vgg_path,replaceOrName+"_vggish.npy")
#     # print(caption_name)
#     # print(vggish_name)
#     if not os.path.exists(vggish_name):
#         print("not exists :{}".format(vggish_name))
#         audio_vggish = np.zeros((100,128))
#     else:
#         audio_vggish = np.load(vggish_name)
#     if os.path.exists(caption_name):
#         caption_emb = np.load(caption_name)
#         #print(caption_emb.shape)
#     else:
#         print("not exists :{}".format(caption_name))
#     video_file = ovpOrH5[key]
#     video_key = key
#     seq = video_file['features'][...].astype(np.float32)
#     gtscore = video_file['gtscore'][...].astype(np.float32)
#     cps = video_file['change_points'][...].astype(np.int32)
#     n_frames = video_file['n_frames'][...].astype(np.int32)
#     nfps = video_file['n_frame_per_seg'][...].astype(np.int32)
#     picks = video_file['picks'][...].astype(np.int32)
#     gtsummary = video_file['gtsummary'][...].astype(np.int32)
#     with h5py.File(ovp_new_h5_vggish_path, 'a') as h5out:
#         h5out.create_dataset(f'{video_key}/vggish', data=audio_vggish)
#         h5out.create_dataset(f'{video_key}/caption', data=caption_emb)
#         h5out.create_dataset(f'{video_key}/features', data=seq)
#         h5out.create_dataset(f'{video_key}/gtscore', data=gtscore)
#         h5out.create_dataset(f'{video_key}/change_points', data=cps)
#         h5out.create_dataset(f'{video_key}/n_frame_per_seg', data=nfps)
#         h5out.create_dataset(f'{video_key}/n_frames', data=n_frames)
#         h5out.create_dataset(f'{video_key}/picks', data=picks)
#         h5out.create_dataset(f'{video_key}/gtsummary', data=gtsummary)
#         h5out.create_dataset(f'{video_key}/video_name', data=orName)
# checkH5File(ovp_new_h5_vggish_path)