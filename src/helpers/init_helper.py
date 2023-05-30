import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_logger(log_dir: str, log_file: str) -> None:
    logger = logging.getLogger()
    format_str = r'[%(asctime)s] %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        datefmt=r'%Y/%m/%d %H:%M:%S',
        format=format_str
    )
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(str(log_dir / log_file))
    fh.setFormatter(logging.Formatter(format_str))
    logger.addHandler(fh)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # model type
    parser.add_argument('model', type=str,
                        choices=('ms'))

    # training & evaluation
    parser.add_argument('--device', type=str, default='cuda',
                        choices=('cuda', 'cpu'))
    parser.add_argument('--dataset_type', type=str, default='v',
                        choices=('v', 'va','vac'))
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--splits', type=str, nargs='+', default=[])
    parser.add_argument('--max-epoch', type=int, default=100)
    parser.add_argument('--model-dir', type=str, default='../models/model')
    parser.add_argument('--log-file', type=str, default='log.txt')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=5e-5)
    parser.add_argument('--split_num', type=int, default=1)
    parser.add_argument('--gamma', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--loss_type', type=str, default='bce',
                        choices=('bce', 'focal','weight_bce','ds_ce','ds_focal'))
    parser.add_argument('--need_recon', type=bool, default=False)
    parser.add_argument('--recon_weight', type=float, default=1.0)
    #clip = 3
    parser.add_argument('--need_clip_predict', type=bool, default=False)
    parser.add_argument('--clip_coff', type=float, default=0.5)
    # need Kendall’s/Spearman’s
    parser.add_argument('--need_other_metric', type=bool, default=True) 
    parser.add_argument('--need_fusion_cap', type=bool, default=True) 

    parser.add_argument('--eval_each_sample', type=bool, default=True) #eval f1 score on the 

    parser.add_argument('--need_sl', type=bool, default=False) #add smooth l1 loss

    parser.add_argument('--need_overlap', type=bool, default=False)
    parser.add_argument('--overlap_ratio', type=float, default=0.25)

    # inference
    parser.add_argument('--ckpt-path', type=str, default=None)
    parser.add_argument('--sample-rate', type=int, default=15)
    parser.add_argument('--source', type=str, default=None)
    parser.add_argument('--save-path', type=str, default=None)

    # shot-level encoder
    parser.add_argument('--input_channel', type=int, default=1024)
    parser.add_argument('--fusion_format', type=str, default="add")
    parser.add_argument('--shot_layer_nums', type=int, default=1)
    parser.add_argument('--shot_num', type=int, default=10)
    parser.add_argument('--shot_channel',nargs='+', type=int, default=8096)
    parser.add_argument('--shot_first', type=bool, default=True)

    # video-level encoder
    parser.add_argument('--d_model', type=int, default=1024,help="the number of expected features in the encoder/decoder inputs")
    parser.add_argument('--long_head', type=int, default=8,help="the number of heads in the multiheadattention models")
    parser.add_argument('--long_num_encoder_layers', type=int, default=6,help="the number of sub-encoder-layers in the encoder")
    parser.add_argument('--long_num_decoder_layers', type=int, default=6,help="the number of sub-decoder-layers in the decoder")
    parser.add_argument('--long_dim_feedforward', type=int, default=2048,help="the dimension of the feedforward network model")
    parser.add_argument('--hidden_layer', type=int, default=128,help="the channel number of hidden layer")

    return parser


def get_arguments() -> argparse.Namespace:
    parser = get_parser()
    args = parser.parse_args()
    return args
