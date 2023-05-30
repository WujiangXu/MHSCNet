import numpy as np
import torch
import clip
from pathlib import Path
import PIL.Image as Image
from torchvision import transforms, models
from torch.utils.data import Dataset,DataLoader
import os
import h5py

model, preprocess = clip.load("/home/jovyan/video_summary_dataset/CLIP-main/ViT-B-32.pt")
model.cuda().eval()

class CustomImageDataset(Dataset):
    def __init__(self, imgdir):
        self.imgdir = imgdir
        self.len_dataset = len(sorted(Path(imgdir).glob('*.JPG')))
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path = os.path.join(self.imgdir,str(idx)+".JPG")
        img = self.preprocess(Image.open(img_path))
        return img

# summe_path = "/home/jovyan/video_summary_dataset/video_summary_multi_modality/DSNet-main/datasets/summe/"
# summe_path_h5_old = {"/home/jovyan/video_summary_dataset/video_summary_multi_modality/DSNet-main/datasets/eccv16_dataset_summe_google_pool5.h5"}
# k1 = "/home/jovyan/video_summary_dataset/video_summary_multi_modality/DSNet-main/datasets/eccv16_dataset_summe_google_pool5.h5"
# summe_datasets = {path: h5py.File(path, 'r') for path in summe_path_h5_old}
# summe_h5_new_clip32_path = "/home/jovyan/video_summary_dataset/video_summary_multi_modality/DSNet-main/datasets/eccv16_dataset_summe_clip_vit32.h5"
# tvsum_path = "/home/jovyan/video_summary_dataset/video_summary_multi_modality/DSNet-main/datasets/tvsum/"
# tvsum_path_h5_old = {"/home/jovyan/video_summary_dataset/video_summary_multi_modality/DSNet-main/datasets/eccv16_dataset_tvsum_google_pool5.h5"}
# k2 = "/home/jovyan/video_summary_dataset/video_summary_multi_modality/DSNet-main/datasets/eccv16_dataset_tvsum_google_pool5.h5"
# tvsum_datasets = {path: h5py.File(path, 'r') for path in tvsum_path_h5_old}
# tvsum_h5_new_clip32_path = "/home/jovyan/video_summary_dataset/video_summary_multi_modality/DSNet-main/datasets/eccv16_dataset_tvsum_clip_vit32.h5"
# summe_dir = os.listdir(summe_path)
# tvsum_dir = os.listdir(tvsum_path)
# for dir_tmp in summe_dir:
#     entire_path_img = os.path.join(summe_path,dir_tmp)
#     bs = len(sorted(Path(entire_path_img).glob('*.JPG')))
#     tmp_dataset = CustomImageDataset(entire_path_img)
#     dataload = DataLoader(tmp_dataset, batch_size=bs, shuffle=False,num_workers=8)
#     video_file = summe_datasets[k1][dir_tmp]
#     gtscore = video_file['gtscore'][...].astype(np.float32)
#     user_summary = video_file['user_summary'][...].astype(np.float32)
#     cps = video_file['change_points'][...].astype(np.int32)
#     nfps = video_file['n_frame_per_seg'][...].astype(np.int32)
#     n_frames = video_file['n_frames'][...].astype(np.int32)
#     picks = video_file['picks'][...].astype(np.int32)
#     n_steps = video_file['n_steps'][...].astype(np.int32)
#     gtsummary = video_file['gtsummary'][...].astype(np.int32)
#     for img in dataload:
#         print("load all images in the dataloader")
#         img = torch.Tensor(img).cuda()
#         with torch.no_grad():
#             img_emb = model.encode_image(img).float()
#             img_emb = img_emb.cpu().numpy()
#             print(img_emb.shape)
#     video_key = dir_tmp
#     with h5py.File(summe_h5_new_clip32_path, 'w') as h5out:
#         h5out.create_dataset(f'{video_key}/features', data=img_emb)
#         h5out.create_dataset(f'{video_key}/gtscore', data=gtscore)
#         h5out.create_dataset(f'{video_key}/user_summary', data=user_summary)
#         h5out.create_dataset(f'{video_key}/change_points', data=cps)
#         h5out.create_dataset(f'{video_key}/n_frame_per_seg', data=nfps)
#         h5out.create_dataset(f'{video_key}/n_frames', data=n_frames)
#         h5out.create_dataset(f'{video_key}/picks', data=picks)
#         h5out.create_dataset(f'{video_key}/n_steps', data=n_steps)
#         h5out.create_dataset(f'{video_key}/gtsummary', data=gtsummary)
#     print("down writeing h5 file for video : {}\n".format(video_key))
# for dir_tmp in tvsum_dir:
#     entire_path_img = os.path.join(tvsum_path,dir_tmp)
#     bs = len(sorted(Path(entire_path_img).glob('*.JPG')))
#     tmp_dataset = CustomImageDataset(entire_path_img)
#     dataload = DataLoader(tmp_dataset, batch_size=bs, shuffle=False,num_workers=8)
#     video_file = tvsum_datasets[k2][dir_tmp]
#     gtscore = video_file['gtscore'][...].astype(np.float32)
#     user_summary = video_file['user_summary'][...].astype(np.float32)
#     cps = video_file['change_points'][...].astype(np.int32)
#     nfps = video_file['n_frame_per_seg'][...].astype(np.int32)
#     n_frames = video_file['n_frames'][...].astype(np.int32)
#     picks = video_file['picks'][...].astype(np.int32)
#     n_steps = video_file['n_steps'][...].astype(np.int32)
#     gtsummary = video_file['gtsummary'][...].astype(np.int32)
#     for img in dataload:
#         print("load all images in the dataloader")
#         img = torch.Tensor(img).cuda()
#         with torch.no_grad():
#             img_emb = model.encode_image(img).float()
#             img_emb = img_emb.cpu().numpy()
#             print(img_emb.shape)
#     video_key = dir_tmp
#     with h5py.File(tvsum_h5_new_clip32_path, 'w') as h5out:
#         h5out.create_dataset(f'{video_key}/features', data=img_emb)
#         h5out.create_dataset(f'{video_key}/gtscore', data=gtscore)
#         h5out.create_dataset(f'{video_key}/user_summary', data=user_summary)
#         h5out.create_dataset(f'{video_key}/change_points', data=cps)
#         h5out.create_dataset(f'{video_key}/n_frame_per_seg', data=nfps)
#         h5out.create_dataset(f'{video_key}/n_frames', data=n_frames)
#         h5out.create_dataset(f'{video_key}/picks', data=picks)
#         h5out.create_dataset(f'{video_key}/n_steps', data=n_steps)
#         h5out.create_dataset(f'{video_key}/gtsummary', data=gtsummary)
#     print("down writeing h5 file for video : {}\n".format(video_key))
# for 
# with torch.no_grad():
#     image_features = model.encode_image(image_input).float()
#     # text_features = model.encode_text(text_tokens).float()
#     image_features /= image_features.norm(dim=-1, keepdim=True)
#     # text_features /= text_features.norm(dim=-1, keepdim=True)
summe_path = "/home/jovyan/video_summary_dataset/video_summary_multi_modality/DSNet-main/datasets/summe/"
summe_path_h5_old = {"/home/jovyan/video_summary_dataset/video_summary_multi_modality/DSNet-main/datasets/eccv16_dataset_summe_google_pool5.h5"}
k1 = "/home/jovyan/video_summary_dataset/video_summary_multi_modality/DSNet-main/datasets/eccv16_dataset_summe_google_pool5.h5"
summe_datasets = {path: h5py.File(path, 'r') for path in summe_path_h5_old}
summe_h5_new_clip32_path = "/home/jovyan/video_summary_dataset/video_summary_multi_modality/DSNet-main/datasets/eccv16_dataset_summe_clip_vit32.h5"
summe_dir = os.listdir(summe_path)
for dir_tmp in tvsum_dir:
    entire_path_img = os.path.join(tvsum_path,dir_tmp)
    bs = len(sorted(Path(entire_path_img).glob('*.JPG')))
    tmp_dataset = CustomImageDataset(entire_path_img)
    dataload = DataLoader(tmp_dataset, batch_size=bs, shuffle=False,num_workers=8)
    video_file = tvsum_datasets[k2][dir_tmp]
    gtscore = video_file['gtscore'][...].astype(np.float32)
    user_summary = video_file['user_summary'][...].astype(np.float32)
    cps = video_file['change_points'][...].astype(np.int32)
    nfps = video_file['n_frame_per_seg'][...].astype(np.int32)
    n_frames = video_file['n_frames'][...].astype(np.int32)
    picks = video_file['picks'][...].astype(np.int32)
    n_steps = video_file['n_steps'][...].astype(np.int32)
    gtsummary = video_file['gtsummary'][...].astype(np.int32)
    for img in dataload:
        print("load all images in the dataloader")
        img = torch.Tensor(img).cuda()
        with torch.no_grad():
            img_emb = model.encode_image(img).float()
            img_emb = img_emb.cpu().numpy()
            print(img_emb.shape)
    video_key = dir_tmp
    with h5py.File(tvsum_h5_new_clip32_path, 'w') as h5out:
        h5out.create_dataset(f'{video_key}/features', data=img_emb)
        h5out.create_dataset(f'{video_key}/gtscore', data=gtscore)
        h5out.create_dataset(f'{video_key}/user_summary', data=user_summary)
        h5out.create_dataset(f'{video_key}/change_points', data=cps)
        h5out.create_dataset(f'{video_key}/n_frame_per_seg', data=nfps)
        h5out.create_dataset(f'{video_key}/n_frames', data=n_frames)
        h5out.create_dataset(f'{video_key}/picks', data=picks)
        h5out.create_dataset(f'{video_key}/n_steps', data=n_steps)
        h5out.create_dataset(f'{video_key}/gtsummary', data=gtsummary)
    print("down writeing h5 file for video : {}\n".format(video_key))