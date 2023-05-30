import cv2
import numpy as np
import torch

from helpers import init_helper, vsumm_helper, bbox_helper, video_helper, data_helper
from ms_level.model import MSAudioCaptionNet
# from modules.model_zoo import get_model
from pathlib import Path
import os

def main():
    args = init_helper.get_arguments()

    # load model
    print('Loading DSNet model ...')
    model = MSAudioCaptionNet(6,10,[8096,8096,8096,8096,8096,8096],1024,shot_first=True, \
                        long_head=8, long_num_encoder_layers=6, \
                        long_num_decoder_layers=6, long_dim_feedforward=2048, \
                        input_channel=1024,fusion_format="add",hidden_layer=128, \
                        need_recon=False,need_clip_predict=False,need_fusion_cap=True, \
                        need_sl=False,overlap_ratio=0.05,need_overlap=True).cuda()
    model = model.eval().to(args.device)
    tvsum_ckpt_path = "/home/jovyan/video_summary_dataset/video_summary_multi_modality/Ours_VS/more_abl/layer_2/checkpoint/tvsum_audio_cap.yml."#/home/jovyan/video_summary_dataset/video_summary_multi_modality/Ours_VS/more_abl/layer_6/checkpoint/tvsum_audio_cap.yml."#"/home/jovyan/video_summary_dataset/video_summary_multi_modality/Ours_VS/tvsum_new/shot4_8_overlap0.05/checkpoint/tvsum_audio_cap.yml."
    summe_ckpt_path = "/home/jovyan/video_summary_dataset/video_summary_multi_modality/Ours_VS/more_abl_summe/layer_6/checkpoint/summe_audio_cap.yml."#"/home/jovyan/video_summary_dataset/video_summary_multi_modality/Ours_VS/summe_new/best/checkpoint/summe_audio_cap.yml."#"/home/jovyan/video_summary_dataset/video_summary_multi_modality/Ours_VS/more_abl_summe/layer_1/checkpoint/summe_audio_cap.yml."
    # data_helper.dump_yaml(vars(args), model_dir / 'args.yml')
    tvsum_video_path = "/home/jovyan/video_summary_dataset/tvsum50/ydata-tvsum50-v1_1/video/"
    summe_video_path = "/home/jovyan/video_summary_dataset/SumMe/videos/"
    save_path = "/home/jovyan/video_summary_dataset/video_summary_multi_modality/Ours_VS/more_vis_img_tvsum/"
    save_img_path = "/home/jovyan/video_summary_dataset/video_summary_multi_modality/Ours_VS/save_images/Valparaiso_Downhill/"
    save_emb_path = "/home/jovyan/video_summary_dataset/video_summary_multi_modality/Ours_VS/emb_overlap/"
    for split_path in args.splits:
        split_path = Path(split_path)
        splits = data_helper.load_yaml(split_path)
        for split_idx, split in enumerate(splits):
            state_dict = torch.load(summe_ckpt_path+str(split_idx)+'.pt')
            model.load_state_dict(state_dict,strict=False)
            val_set = data_helper.InferDataset(split['test_keys'])
            val_loader = data_helper.DataLoader(val_set, shuffle=False)
            with torch.no_grad():
                for test_key, seq, gtscore, cps, n_frames, nfps, picks, user_summary, vggish, cap_emb, video_name in val_loader:
                    seq_len = len(seq)
                    seq_torch = torch.from_numpy(seq).unsqueeze(0).to(args.device)
                    vggish = torch.tensor(vggish, dtype=torch.float32).unsqueeze(0).to(args.device)
                    cap_emb = torch.tensor(cap_emb, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(args.device)
                    if args.need_recon:
                        if not args.need_sl:
                            pred_cls,seq_recon = model(seq_torch,vggish,cap_emb,isTrain=False)
                        else:
                            pred_cls,seq_recon,pred_score = model(seq_torch,vggish,cap_emb,isTrain=False)
                            pred_score = pred_score.squeeze()
                        seq_recon = seq_recon.squeeze()#.cpu()
                    else:
                        if not args.need_sl:
                            pred_cls = model(seq_torch,vggish,cap_emb,isTrain=False)
                        else:
                            pred_cls,pred_score = model(seq_torch,vggish,cap_emb,isTrain=False)
                            pred_score = pred_score.squeeze()
                    pred_cls = pred_cls.squeeze()
                    keyshot_summ = vsumm_helper.get_keyshot_summ(
                        gtscore, cps, n_frames, nfps, picks)
                    target = vsumm_helper.downsample_summ(keyshot_summ)
                    target = torch.Tensor(target).squeeze().cuda()
                    pred_cls = pred_cls.cpu()
                    pred_summ = vsumm_helper.get_keyshot_summ(pred_cls, cps, n_frames, nfps, picks)
                    
        # load video
                    eval_metric = 'avg' if 'tvsum' in test_key else 'max'
                    if 'tvsum' in test_key :
                        fscore = vsumm_helper.get_summ_f1score(
                            pred_summ, np.expand_dims(keyshot_summ,0), eval_metric)
                    else:
                        fscore = vsumm_helper.get_summ_f1score(
                            pred_summ, user_summary, eval_metric)

                    video_name = str(video_name).replace('b\'','').replace('\'','')
                    if not os.path.exists(os.path.join(save_path,video_name)):
                        os.makedirs(os.path.join(save_path,video_name))
                    source_video = summe_video_path + video_name+'.mp4'
                    cap = cv2.VideoCapture(source_video)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    #overlap_emb = overlap_emb.cpu().numpy()
                    #np.save(os.path.join(save_emb_path,video_name+'_nooverlap'+'.npy'),overlap_emb)
                    #np.save(os.path.join(save_emb_path,video_name+'.npy'),seq)
                    # print(source_video,width,height,fps)
                    frame_idx = 0
                    if video_name == 'Valparaiso_Downhill':
                        if not os.path.exists(os.path.join(save_img_path,video_name+'_'+str(fscore))):
                            os.mkdir(os.path.join(save_img_path,video_name+'_'+str(fscore)))
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            if frame_idx<pred_summ.shape[0]:
                                if pred_summ[frame_idx]:
                                    cv2.imwrite(os.path.join(save_img_path,video_name+'_'+str(fscore),str(frame_idx)+'.JPG'),frame)

                            frame_idx += 1
                    #print(gtscore.shape)
                    # np.save(os.path.join(save_path,video_name,'gtscore.npy'),gtscore)
                    # np.save(os.path.join(save_path,video_name,'score_layer4_'+str(fscore)+'.npy'), vsumm_helper.downsample_summ(pred_summ))
                    cap.release()


if __name__ == '__main__':
    main()
