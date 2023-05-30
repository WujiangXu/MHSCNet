import logging
from pathlib import Path
from helpers.vsumm_helper import get_keyshot_summ
import numpy as np
import torch
from torch import nn
from helpers import init_helper, data_helper, vsumm_helper, bbox_helper
#from modules.model_zoo import get_model
from ms_level.focalloss import FocalLoss
from ms_level.losses import calc_cls_loss

logger = logging.getLogger()
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

def evaluateMS(model, val_loader, device, dataset_type, args):
    model.eval()
    if not args.need_other_metric:
        stats = data_helper.AverageMeter('fscore', 'diversity', 'loss')
    else:
        stats = data_helper.AverageMeter('fscore', 'diversity', 'loss', 'kendal', 'spear')
    #loss_f = FocalLoss(gamma=2,alpha=0.2)
    #loss_bce = nn.BCELoss(reduction='none')
    if args.loss_type == 'bce':
        loss_train = nn.BCELoss()
    elif args.loss_type == 'focal':
        loss_train = FocalLoss(gamma=args.gamma,alpha=args.alpha)
    elif args.loss_type == 'weight_bce':
        loss_train = nn.BCELoss(reduction='none')
    loss_recon = nn.SmoothL1Loss()

    with torch.no_grad():
        for test_key, seq, gtscore, cps, n_frames, nfps, picks, user_summary, vggish, cap_emb in val_loader:
            seq_len = len(seq)
            seq_torch = torch.from_numpy(seq).unsqueeze(0).to(device)
            if dataset_type=='v':
                pred_cls = model(seq_torch).cpu().squeeze()
            elif dataset_type=='va':
                vggish = torch.from_numpy(vggish).unsqueeze(0).to(device)
                pred_cls = model(seq_torch,vggish).cpu().squeeze()
            elif dataset_type == 'vac':
                vggish = torch.tensor(vggish, dtype=torch.float32).unsqueeze(0).to(device)
                cap_emb = torch.tensor(cap_emb, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                # pred_cls,seq_recon = model(seq_torch,vggish,cap_emb)
                # pred_cls = pred_cls.squeeze()#.cpu()
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
                pred_cls = pred_cls.squeeze()#.cpu()
                # pred_cls2 = pred_cls2.squeeze()#.cpu()
                # pred_cls3 = pred_cls3.squeeze()#.cpu()
                # pred_cls = pred_cls1 + pred_cls2 + pred_cls3
            keyshot_summ = vsumm_helper.get_keyshot_summ(
                gtscore, cps, n_frames, nfps, picks)
            target = vsumm_helper.downsample_summ(keyshot_summ)
            weight_frame = generate_weight(target).cuda()
            target = torch.Tensor(target).squeeze().cuda()
            if args.loss_type == 'weight_bce':
                weight_frame = weight_frame.cuda()
                loss_cls = (loss_train(pred_cls,target)*weight_frame).mean()
            elif args.loss_type == 'bce':    
                loss_cls = loss_train(pred_cls,target)#(loss_bce(pred_cls,target)*weight_frame).mean()
            elif args.loss_type == 'ds_ce':    
                loss_cls = calc_cls_loss(pred_cls,target,'cross-entropy')
            elif args.loss_type == 'ds_focal':    
                loss_cls = calc_cls_loss(pred_cls,target,'focal')
            if args.need_recon:
                loss_re = loss_recon(seq_torch,seq_recon)
                loss = loss_cls + loss_re * args.recon_weight
            else:
                loss = loss_cls
            #loss = (loss_bce(pred_cls,target)*weight_frame).mean()
            #pred_cls = pred_cls[:,1].numpy()
            pred_cls = pred_cls.cpu()
            if not args.need_sl:
                #print(pred_cls)
                #print(target)
                pred_summ = get_keyshot_summ(pred_cls, cps, n_frames, nfps, picks)
                #print(vsumm_helper.downsample_cps(cps))
            else:
                pred_score = pred_score.cpu()
                pred_summ = get_keyshot_summ(pred_score, cps, n_frames, nfps, picks)
            eval_metric = 'avg' if 'tvsum' in test_key else 'max'
            if 'tvsum' in test_key :
                fscore = vsumm_helper.get_summ_f1score(
                    pred_summ, np.expand_dims(keyshot_summ,0), eval_metric)
            else:
                fscore = vsumm_helper.get_summ_f1score(
                    pred_summ, user_summary, eval_metric)
            if args.need_other_metric:
                kendal_t,spear_t = vsumm_helper.get_statistics_metric(gtscore,pred_cls.numpy())
            pred_summ = vsumm_helper.downsample_summ(pred_summ)
            diversity = vsumm_helper.get_summ_diversity(pred_summ, seq)
            if not args.need_other_metric:
                stats.update(fscore=fscore, diversity=diversity,loss=loss.item())
            else:
                stats.update(fscore=fscore, diversity=diversity,loss=loss.item(),kendal=kendal_t,spear=spear_t)
        if not args.need_other_metric:
            return stats.fscore, stats.diversity, stats.loss
        else:
            return stats.fscore, stats.diversity, stats.loss, stats.kendal, stats.spear

def evaluate(model, val_loader, nms_thresh, device):
    model.eval()
    stats = data_helper.AverageMeter('fscore', 'diversity')

    with torch.no_grad():
        for test_key, seq, _, cps, n_frames, nfps, picks, user_summary in val_loader:
            seq_len = len(seq)
            seq_torch = torch.from_numpy(seq).unsqueeze(0).to(device)

            pred_cls, pred_bboxes = model.predict(seq_torch)

            pred_bboxes = np.clip(pred_bboxes, 0, seq_len).round().astype(np.int32)

            pred_cls, pred_bboxes = bbox_helper.nms(pred_cls, pred_bboxes, nms_thresh)
            pred_summ = vsumm_helper.bbox2summary(
                seq_len, pred_cls, pred_bboxes, cps, n_frames, nfps, picks)

            eval_metric = 'avg' if 'tvsum' in test_key else 'max'
            fscore = vsumm_helper.get_summ_f1score(
                pred_summ, user_summary, eval_metric)

            pred_summ = vsumm_helper.downsample_summ(pred_summ)
            diversity = vsumm_helper.get_summ_diversity(pred_summ, seq)
            stats.update(fscore=fscore, diversity=diversity)

    return stats.fscore, stats.diversity


def main():
    args = init_helper.get_arguments()

    init_helper.init_logger(args.model_dir, args.log_file)
    init_helper.set_random_seed(args.seed)

    logger.info(vars(args))
    model = get_model(args.model, **vars(args))
    model = model.eval().to(args.device)

    for split_path in args.splits:
        split_path = Path(split_path)
        splits = data_helper.load_yaml(split_path)

        stats = data_helper.AverageMeter('fscore', 'diversity')

        for split_idx, split in enumerate(splits):
            ckpt_path = data_helper.get_ckpt_path(args.model_dir, split_path, split_idx)
            state_dict = torch.load(str(ckpt_path),
                                    map_location=lambda storage, loc: storage)
            model.load_state_dict(state_dict)

            val_set = data_helper.VideoDataset(split['test_keys'])
            val_loader = data_helper.DataLoader(val_set, shuffle=False)

            fscore, diversity = evaluate(model, val_loader, args.nms_thresh, args.device)
            stats.update(fscore=fscore, diversity=diversity)

            logger.info(f'{split_path.stem} split {split_idx}: diversity: '
                        f'{diversity:.4f}, F-score: {fscore:.4f}')

        logger.info(f'{split_path.stem}: diversity: {stats.diversity:.4f}, '
                    f'F-score: {stats.fscore:.4f}')


if __name__ == '__main__':
    main()
