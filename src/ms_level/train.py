import logging

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from ms_level.model import MSNet,MSAudioNet,MSAudioCaptionNet
from evaluate import evaluate,evaluateMS
from helpers import data_helper, vsumm_helper, bbox_helper
from ms_level.focalloss import FocalLoss
from ms_level.losses import calc_cls_loss
import random
import math

logger = logging.getLogger()
def copySeqLen(seq,seq_len):
    '''
        input : seq[x,1024]
                seq_len:320
        output : seq [x/320,320,1024]
    '''
    bs = math.ceil(seq.shape[0] / seq_len)
    seq_gen = torch.empty((bs,seq_len,seq.shape[1])).cuda()
    for i in range(bs):
        if i != bs-1:
            seq_gen[i,:,:] = seq[i*seq_len:(i+1)*seq_len,:]
        else: # pad 0
            len_tmp_r = (i+1)*seq_len - seq.shape[0]
            len_tmp_l = seq_len - len_tmp_r
            seq_gen[i,:len_tmp_l,:] = seq[i*seq_len:,:]
            tmp1 = torch.zeros((len_tmp_r,seq.shape[1]))
            seq_gen[i,len_tmp_l:,:] = tmp1
    return seq_gen

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def generate_weight(target):
    weight = torch.empty(len(target))
    true_num = 0
    for tmp in target:
        if tmp == True:
            true_num += 1
    all_num = len(target)
    weight_v = true_num/all_num
    weight_v2 = 1/all_num
    for i in range(len(target)):
        if target[i] == True:
            weight[i] = weight_v
        else:
            weight[i] = weight_v2
    return weight

def clip_label_generate():
    random_clip_order = [0,1,2,3,4,5]#[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
    clip_label_dict = {0:[1,2,3],1:[1,3,2],2:[2,1,3],3:[2,3,1],4:[3,1,2],5:[3,2,1]}
    random.shuffle(random_clip_order)
    label = list()
    for tmp in random_clip_order:
        label.append(clip_label_dict[tmp])
    return np.array(random_clip_order),label



def xavier_init(module):
    cls_name = module.__class__.__name__
    if 'Linear' in cls_name or 'Conv' in cls_name:
        nn.init.xavier_uniform_(module.weight, gain=np.sqrt(2.0))
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.1)


def train(args, split, save_path, split_idx=0):
    if args.dataset_type == 'v':
        model = MSNet(args.shot_layer_nums,args.shot_num,args.shot_channel,args.d_model,shot_first=True, \
                        long_head=args.long_head, long_num_encoder_layers=args.long_num_encoder_layers, \
                        long_num_decoder_layers=args.long_num_decoder_layers, long_dim_feedforward=args.long_dim_feedforward, \
                            input_channel=args.input_channel,fusion_format=args.fusion_format)
    elif args.dataset_type == 'va':
        model = MSAudioNet(args.shot_layer_nums,args.shot_num,args.shot_channel,args.d_model,shot_first=True, \
                        long_head=args.long_head, long_num_encoder_layers=args.long_num_encoder_layers, \
                        long_num_decoder_layers=args.long_num_decoder_layers, long_dim_feedforward=args.long_dim_feedforward, \
                            input_channel=args.input_channel,fusion_format=args.fusion_format)
    elif args.dataset_type == 'vac':
        model = MSAudioCaptionNet(args.shot_layer_nums,args.shot_num,args.shot_channel,args.d_model,shot_first=True, \
                        long_head=args.long_head, long_num_encoder_layers=args.long_num_encoder_layers, \
                        long_num_decoder_layers=args.long_num_decoder_layers, long_dim_feedforward=args.long_dim_feedforward, \
                        input_channel=args.input_channel,fusion_format=args.fusion_format,hidden_layer=args.hidden_layer, \
                        need_recon=args.need_recon,need_clip_predict=args.need_clip_predict,need_fusion_cap=args.need_fusion_cap, \
                        need_sl=args.need_sl,overlap_ratio=args.overlap_ratio,need_overlap=args.need_overlap)
    model = model.to(args.device)

    #model.apply(xavier_init) #test
    if args.loss_type == 'bce':
        loss_train = nn.BCELoss()
    elif args.loss_type == 'focal':
        loss_train = FocalLoss(gamma=args.gamma,alpha=args.alpha)
    elif args.loss_type == 'weight_bce':
        loss_train = nn.BCELoss(reduction='none')
    loss_recon = nn.SmoothL1Loss()
    #loss_f = FocalLoss(gamma=args.gamma,alpha=args.alpha)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=args.lr,
                                 weight_decay=args.weight_decay)
    ema = EMA(model, 0.999)
    ema.register()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=10)
    # if torch.cuda.is_available():
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False
    max_val_fscore = -1
    max_val_diversity = -1
    max_val_kendal = -1 
    max_val_spear = -1 
    if args.dataset_type == 'v':
        train_set = data_helper.VideoDataset(split['train_keys'])#data_helper.VideoImageDataset(split['train_keys'],tvsum_img_dir)#data_helper.VideoDataset(split['train_keys'])
        val_set = data_helper.VideoDataset(split['test_keys'])
    elif args.dataset_type == 'va':
        train_set = data_helper.VideoAudioDataset(split['train_keys'])#data_helper.VideoImageDataset(split['train_keys'],tvsum_img_dir)#data_helper.VideoDataset(split['train_keys'])
        val_set = data_helper.VideoAudioDataset(split['test_keys'])
    elif args.dataset_type == 'vac':
        train_set = data_helper.VideoAudioCaptionDataset(split['train_keys'])#data_helper.VideoImageDataset(split['train_keys'],tvsum_img_dir)#data_helper.VideoDataset(split['train_keys'])
        val_set = data_helper.VideoAudioCaptionDataset(split['test_keys'])
    # train_loader = DataLoader(train_set, batch_size=1, shuffle=True,
    #                                 num_workers=10, pin_memory=True)
    # val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
    #                                 num_workers=10, pin_memory=True)
    train_loader = data_helper.DataLoader(train_set, shuffle=True)
    val_loader = data_helper.DataLoader(val_set, shuffle=False)

    for epoch in range(args.max_epoch):
        model.train()
        stats = data_helper.AverageMeter('loss', 'cls_loss', 'ret_loss','fscore','gtf1')

        for _, seq, gtscore, cps, n_frames, nfps, picks, user_summary, vggish, cap_emb in train_loader:
            # seq = seq.squeeze().numpy()
            # gtscore = gtscore.squeeze().numpy()
            # cps = cps.squeeze().numpy()
            # n_frames = n_frames.squeeze().numpy()
            # picks = picks.squeeze().numpy()
            # vggish = vggish.squeeze().numpy()
            # cap_emb = cap_emb.squeeze().numpy()
            # print("seq shape:{}".format(seq.shape))
            # print("gtscore shape:{}".format(gtscore.shape))
            # print("cps shape:{}".format(cps.shape))
            # print("n_frames shape:{}".format(n_frames.shape))
            # print("picks shape:{}".format(picks.shape))
            # print("_ shape:{}".format(_.shape))
            # print("vggish shape:{}".format(vggish.shape))
            # print("cap_emb shape:{}".format(cap_emb.shape))
            # print("gtscore :{}".format(gtscore))
            # print(gtscore)
            if args.need_clip_predict:
                label_clip,clip_order_lists = clip_label_generate()
                label_clip = torch.Tensor(label_clip).cuda()
            keyshot_summ = vsumm_helper.get_keyshot_summ(
                gtscore, cps, n_frames, nfps, picks)
            #cps = vsumm_helper.downsample_cps(cps)
            #print(cps)
            target = vsumm_helper.downsample_summ(keyshot_summ)
            weight_frame = generate_weight(target) 
            #print("target shape:{}".format(target.shape))
            if not target.any():
                continue

            seq = torch.tensor(seq, dtype=torch.float32).to(args.device).unsqueeze(0)
            #seq = copySeqLen(seq,320) 
            if args.dataset_type == 'v':
                pred_cls = model(seq).squeeze()
            elif  args.dataset_type == 'va':
                vggish = torch.tensor(vggish, dtype=torch.float32).unsqueeze(0).to(args.device)
                pred_cls = model(seq,vggish).squeeze()
            elif  args.dataset_type == 'vac':
                vggish = torch.tensor(vggish, dtype=torch.float32).unsqueeze(0).to(args.device)
                cap_emb = torch.tensor(cap_emb, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(args.device)
                if args.need_recon:
                    if not args.need_clip_predict:
                        if not args.need_sl:
                            pred_cls,seq_recon = model(seq,vggish,cap_emb)
                        else:
                            pred_cls,seq_recon,pred_score = model(seq,vggish,cap_emb)
                            pred_score = pred_score.squeeze()
                        seq_recon = seq_recon.squeeze()#.cpu()
                    else:
                        if not args.need_sl:
                            pred_cls,seq_recon,clip_predict = model(seq,vggish,cap_emb,clip_order_lists)
                        else:
                            pred_cls,seq_recon,clip_predict,pred_score = model(seq,vggish,cap_emb,clip_order_lists)
                            pred_score = pred_score.squeeze()
                        seq_recon = seq_recon.squeeze()#.cpu()
                else:
                    if not args.need_clip_predict:
                        if not args.need_sl:
                            pred_cls = model(seq,vggish,cap_emb)
                        else:
                            pred_cls,pred_score = model(seq,vggish,cap_emb)
                            pred_score = pred_score.squeeze()
                    else:
                        if not args.need_sl:
                            pred_cls,clip_predict = model(seq,vggish,cap_emb,clip_order_lists)
                        else:
                            pred_cls,clip_predict,pred_score = model(seq,vggish,cap_emb,clip_order_lists)
                            pred_score = pred_score.squeeze()
                pred_cls = pred_cls.squeeze()#.cpu()
                # pred_cls2 = pred_cls2.squeeze()
                # pred_cls3 = pred_cls3.squeeze()
            eval_metric = 'avg'
            pred_summ = vsumm_helper.get_keyshot_summ(pred_cls.detach().cpu(), cps, n_frames, nfps, picks)
            #print(user_summary)
            # fscore = vsumm_helper.get_summ_f1score(
            #     pred_summ, user_summary, eval_metric)
            # gtf1 = vsumm_helper.get_summ_f1score(
            #     keyshot_summ, user_summary, eval_metric)
            target = torch.Tensor(target).squeeze().cuda()
            # print(target)
            # print(pred_cls)
            seq = seq.squeeze()#.cpu()
            # pred_cls = pred_cls.unsqueeze(0)
            # target = target.unsqueeze(0)
            if args.loss_type == 'weight_bce':
                weight_frame = weight_frame.cuda()
                loss_cls = (loss_train(pred_cls,target)*weight_frame).mean()
            elif args.loss_type == 'bce':    
                loss_cls = loss_train(pred_cls,target)#(loss_bce(pred_cls,target)*weight_frame).mean()
            elif args.loss_type == 'ds_ce':    
                loss_cls = calc_cls_loss(pred_cls,target,'cross-entropy')
            elif args.loss_type == 'ds_focal':    
                loss_cls = calc_cls_loss(pred_cls,target,'focal')#+calc_cls_loss(pred_cls2,target,'focal')+calc_cls_loss(pred_cls3,target,'focal')
            if args.need_recon:
                loss_re = loss_recon(seq,seq_recon)
                loss = loss_cls + loss_re * args.recon_weight
            else:
                loss = loss_cls
            if args.need_clip_predict:
                # print("clip_predict shape:{}".format(clip_predict.shape))
                # print("label_clip shape:{}".format(label_clip.shape))
                loss_clip = nn.CrossEntropyLoss()(clip_predict,label_clip.long())
                loss = loss + loss_clip * args.clip_coff
            if args.need_sl:
                gtscore = torch.Tensor(gtscore).cuda()
                loss = loss + loss_recon(gtscore,pred_score) * 10
            #print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #ema.update() #test

            if args.need_recon:
                stats.update(loss=loss.item(),cls_loss=loss_cls.item(),ret_loss=loss_re.item()) #,fscore=fscore,gtf1=gtf1
            else:
                stats.update(loss=loss.item(),cls_loss=loss_cls.item(),ret_loss=0)
            
            if args.eval_each_sample:
                if not args.need_other_metric:
                    val_fscore, val_diversity, val_loss = evaluateMS(model, val_loader, args.device, args.dataset_type, args)
                else:
                    #ema.apply_shadow() #test
                    val_fscore, val_diversity, val_loss, val_kendal, val_spear = evaluateMS(model, val_loader, args.device, args.dataset_type, args)
                    max_val_kendal = max(val_kendal,max_val_kendal)
                    max_val_spear = max(val_spear,max_val_spear)
                    #ema.restore() #test

                if max_val_fscore < val_fscore:
                    max_val_fscore = val_fscore
                    torch.save(model.state_dict(), str(save_path))
                if max_val_diversity < val_diversity:
                    max_val_diversity = val_diversity
        
        if not args.eval_each_sample:
            if not args.need_other_metric:
                val_fscore, val_diversity, val_loss = evaluateMS(model, val_loader, args.device, args.dataset_type, args)
            else:
                #ema.apply_shadow() #test
                val_fscore, val_diversity, val_loss, val_kendal, val_spear = evaluateMS(model, val_loader, args.device, args.dataset_type, args)
                max_val_kendal = max(val_kendal,max_val_kendal)
                max_val_spear = max(val_spear,max_val_spear)
                #ema.restore() #test

            if max_val_fscore < val_fscore:
                max_val_fscore = val_fscore
                torch.save(model.state_dict(), str(save_path))
            if max_val_diversity < val_diversity:
                max_val_diversity = val_diversity

        if not args.need_other_metric:
            logger.info(f'Epoch: {epoch}/{args.max_epoch} '
                        f'Loss: {stats.cls_loss:.4f}/{stats.ret_loss:.4f}/{stats.loss:.4f} '
                        f'F-score cur/max: {val_fscore:.4f}/{max_val_fscore:.4f}'
                        f'diversity cur/max: {val_diversity:.4f}/{max_val_diversity:.4f}'
                        f'Val loss: {val_loss:.4f}')
        else:
            logger.info(f'Split_idx: {split_idx} '
                        f'Epoch: {epoch}/{args.max_epoch} '
                        f'Loss: {stats.cls_loss:.4f}/{stats.ret_loss:.4f}/{stats.loss:.4f} '
                        #f'gt f-score: {stats.gtf1:.4f}'
                        #f'train f-score : {stats.fscore:.4f}'
                        f'F-score cur/max: {val_fscore:.4f}/{max_val_fscore:.4f}'
                        f'diversity cur/max: {val_diversity:.4f}/{max_val_diversity:.4f}'
                        f'kendal cur/max: {val_kendal:.4f}/{max_val_kendal:.4f}'
                        f'spear cur/max: {val_spear:.4f}/{max_val_spear:.4f}'
                        f'Val loss: {val_loss:.4f}')
        #scheduler.step()
    return max_val_fscore
