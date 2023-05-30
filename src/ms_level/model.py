import torch
from torch import nn
from torchvision import transforms, models
import torch.nn.functional as F
import math

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

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class GCNExtractor(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.gcn = GCNConv(in_feature, out_feature)

    def forward(self, x):
        x = x.squeeze(1)
        edge_indices, edge_weights = self.create_graph(x, keep_ratio=0.3)
        print(edge_indices)
        print(edge_weights)
        out = self.gcn(x, edge_indices, edge_weights)
        out = out.unsqueeze(1)
        return out

    @staticmethod
    def create_graph(x, keep_ratio=0.3):
        seq_len, _ = x.shape
        keep_top_k = int(keep_ratio * seq_len * seq_len)

        edge_weights = torch.matmul(x, x.t())
        edge_weights = edge_weights - torch.eye(seq_len, seq_len).to(x.device)
        edge_weights = edge_weights.view(-1)
        edge_weights, edge_indices = torch.topk(
            edge_weights, keep_top_k, sorted=False)

        edge_indices = edge_indices.unsqueeze(0)
        edge_indices = torch.cat(
            [edge_indices / seq_len, edge_indices % seq_len])

        return edge_indices, edge_weights

class GCNLayers(nn.Module):
    def __init__(self, num_feature, num_layers=3):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.gcn = GCNConv(num_feature, num_feature)

    def forward(self, x):
        x = x.squeeze(0)
        edge_indices, edge_weights = self.create_graph(x, keep_ratio=0.3)
        out = self.gcn(x, edge_indices, edge_weights)
        out = out.unsqueeze(0)
        return out

    @staticmethod
    def create_graph(x, keep_ratio=0.3):
        seq_len, _ = x.shape
        keep_top_k = int(keep_ratio * seq_len * seq_len)

        edge_weights = torch.matmul(x, x.t())
        edge_weights = edge_weights - torch.eye(seq_len, seq_len).to(x.device)
        edge_weights = edge_weights.view(-1)
        edge_weights, edge_indices = torch.topk(
            edge_weights, keep_top_k, sorted=False)

        edge_indices = edge_indices.unsqueeze(0)
        edge_indices = torch.cat(
            [edge_indices / seq_len, edge_indices % seq_len])

        return edge_indices, edge_weights

def mergePooling(x,bf_size):
    '''
    input:
        x : [10,128]
        shot_num : 10
        bf_size : 768
    '''
    shot_num = x.shape[0]
    split_channel = bf_size//shot_num
    output = torch.empty((bf_size,x.shape[1])).cuda()
    last_channel = bf_size-split_channel*(shot_num-1)
    for i in range(shot_num-1):
        tmp = x[i,:]
        tmp = tmp.expand((split_channel,x.shape[1]))
        output[i*split_channel:(i+1)*split_channel,:] = tmp
    tmp = x[shot_num-1,:]
    tmp = tmp.expand((last_channel,x.shape[1]))
    output[(shot_num-1)*split_channel:,:] = tmp
    return output.cuda()

class shotLevelExtraction(nn.Module):
    def __init__(self, shot_num, shot_channel=8096,input_channel=1024):
        super().__init__()
        self.fc_up_shot = nn.Linear(input_channel,shot_channel)
        self.shot_num = shot_num
        self.frame_to_shot = nn.AdaptiveAvgPool1d(shot_num)
        '''
            1: 3*3 conv1d
            2: 5*5 conv1d
            3: 7*7 conv1d
            4: 1*1 conv1d
        '''
        self.shot_cross1 = nn.Conv1d(shot_num, shot_num, 3, stride=4)
        self.shot_cross2 = nn.Conv1d(shot_num, shot_num, 3, dilation=2, stride=4)
        self.shot_cross3 = nn.Conv1d(shot_num, shot_num, 3, dilation=3,stride=4)
        self.shot_cross4 = nn.Conv1d(shot_num, shot_num, 1, stride=4)
        self.pool = nn.AdaptiveAvgPool1d(shot_channel)
        #self.act = nn.RReLU()
        self.fc_down_shot = nn.Linear(shot_channel,input_channel)


    def forward(self, seq):
        '''
            seq: [1,seq_len,emb_size]
            
        '''
        seq_inital = seq
        seq = self.fc_up_shot(seq.squeeze()).unsqueeze(0)
        seq = seq.permute(0,2,1) # [1,emb_size,seq_len]
        bf_size = seq.shape[2]
        seq_shot = self.frame_to_shot(seq)
        seq_shot = seq_shot.permute(0,2,1) # [1,seq_len,emb_size]

        seq_shot_cross1 = self.shot_cross1(seq_shot)
        seq_shot_cross2 = self.shot_cross2(seq_shot)
        seq_shot_cross3 = self.shot_cross3(seq_shot)
        seq_shot_cross4 = self.shot_cross4(seq_shot)
        seq_shot_cross_sum = torch.cat((seq_shot_cross1,seq_shot_cross2,seq_shot_cross3,seq_shot_cross4),dim=2)
        seq_shot_cross_sum = self.pool(seq_shot_cross_sum).squeeze()
        seq_shot_cross_sum = self.fc_down_shot(seq_shot_cross_sum)
        seq_shot_cross_up =  mergePooling(seq_shot_cross_sum,bf_size).unsqueeze(0)
        return seq_inital+seq_shot_cross_up

# [100,1024] -> [shot_num,shot_channel],[shot_num*2,shot_channel*2],[shot_num*3,shot_channel*3]
# att concat有问题
# shot 里面高斯分布，中间的重要一些
# dsnet 得分
# multi label,
# multi score.

class shotLevelExtractionMS(nn.Module):
    def __init__(self, shot_num, shot_channel=8096,input_channel=1024):
        super().__init__()
        self.fc_up_shot = nn.Linear(input_channel,shot_channel)
        self.shot_num = shot_num
        self.frame_to_shot1 = nn.AdaptiveAvgPool1d(shot_num)
        self.frame_to_shot2 = nn.AdaptiveAvgPool1d(int(shot_num*1.5)) #1.5
        self.frame_to_shot3 = nn.AdaptiveAvgPool1d(int(shot_num*0.5))
        self.frame_to_shot4 = nn.AdaptiveAvgPool1d(int(shot_num*1.2))
        self.frame_to_shot5 = nn.AdaptiveAvgPool1d(int(shot_num*0.8))

        '''
            1: 3*3 conv1d
            2: 5*5 conv1d
            3: 7*7 conv1d
            4: 1*1 conv1d
        '''
        self.shot_cross11 = nn.Conv1d(shot_num, shot_num, 3, stride=4)
        self.shot_cross12 = nn.Conv1d(shot_num, shot_num, 3, dilation=2, stride=4)
        self.shot_cross13 = nn.Conv1d(shot_num, shot_num, 3, dilation=3,stride=4)
        self.shot_cross14 = nn.Conv1d(shot_num, shot_num, 1, stride=4)
        # self.shot_cross15 = nn.Conv1d(shot_num, shot_num, 3, dilation=4, stride=2)
        # self.shot_cross16 = nn.Conv1d(shot_num, shot_num, 3, dilation=5,stride=2)

        self.shot_cross21 = nn.Conv1d(int(shot_num*1.5), int(shot_num*1.5), 3, stride=4) #tvsum :1.5
        self.shot_cross22 = nn.Conv1d(int(shot_num*1.5), int(shot_num*1.5), 3, dilation=2, stride=4)
        self.shot_cross23 = nn.Conv1d(int(shot_num*1.5), int(shot_num*1.5), 3, dilation=3,stride=4)
        self.shot_cross24 = nn.Conv1d(int(shot_num*1.5), int(shot_num*1.5), 1, stride=4)
        # self.shot_cross25 = nn.Conv1d(int(shot_num*1.5), int(shot_num*1.5), 3, dilation=4, stride=2)
        # self.shot_cross26 = nn.Conv1d(int(shot_num*1.5), int(shot_num*1.5), 3, dilation=5,stride=2)

        self.shot_cross31 = nn.Conv1d(int(shot_num*0.5), int(shot_num*0.5), 3, stride=4)
        self.shot_cross32 = nn.Conv1d(int(shot_num*0.5), int(shot_num*0.5), 3, dilation=2, stride=4)
        self.shot_cross33 = nn.Conv1d(int(shot_num*0.5), int(shot_num*0.5), 3, dilation=3,stride=4)
        self.shot_cross34 = nn.Conv1d(int(shot_num*0.5), int(shot_num*0.5), 1, stride=4)
        # self.shot_cross35 = nn.Conv1d(int(shot_num*0.5), int(shot_num*0.5), 3, dilation=4, stride=2)
        # self.shot_cross36 = nn.Conv1d(int(shot_num*0.5), int(shot_num*0.5), 3, dilation=5,stride=2)

        # self.shot_cross41 = nn.Conv1d(int(shot_num*1.2), int(shot_num*1.2), 3, stride=4)
        # self.shot_cross42 = nn.Conv1d(int(shot_num*1.2), int(shot_num*1.2), 3, dilation=2, stride=4)
        # self.shot_cross43 = nn.Conv1d(int(shot_num*1.2), int(shot_num*1.2), 3, dilation=3,stride=4)
        # self.shot_cross44 = nn.Conv1d(int(shot_num*1.2), int(shot_num*1.2), 1, stride=4)

        # self.shot_cross51 = nn.Conv1d(int(shot_num*0.8), int(shot_num*0.8), 3, stride=4)
        # self.shot_cross52 = nn.Conv1d(int(shot_num*0.8), int(shot_num*0.8), 3, dilation=2, stride=4)
        # self.shot_cross53 = nn.Conv1d(int(shot_num*0.8), int(shot_num*0.8), 3, dilation=3,stride=4)
        # self.shot_cross54 = nn.Conv1d(int(shot_num*0.8), int(shot_num*0.8), 1, stride=4)

        self.pool = nn.AdaptiveAvgPool1d(shot_channel)
        self.act = nn.ReLU()
        self.fc_down_shot1 = nn.Sequential(
                                nn.Linear(shot_channel, input_channel),
                                nn.ReLU(),
                                nn.LayerNorm(input_channel)
                            )#nn.Linear(shot_channel,input_channel)
        self.fc_down_shot2 = nn.Sequential(
                                nn.Linear(shot_channel, input_channel),
                                nn.ReLU(),
                                nn.LayerNorm(input_channel)
                            )
        self.fc_down_shot3 = nn.Sequential(
                                nn.Linear(shot_channel, input_channel),
                                nn.ReLU(),
                                nn.LayerNorm(input_channel)
                            )
        # self.fc_down_shot4 = nn.Sequential(
        #                         nn.Linear(shot_channel, input_channel),
        #                         nn.ReLU(),
        #                         nn.LayerNorm(input_channel)
        #                     )
        # self.fc_down_shot5 = nn.Sequential(
        #                         nn.Linear(shot_channel, input_channel),
        #                         nn.ReLU(),
        #                         nn.LayerNorm(input_channel)
        #                     )

    def forward(self, seq):
        '''
            seq: [1,seq_len,emb_size]
            
        '''
        seq_inital = seq
        seq = self.fc_up_shot(seq.squeeze()).unsqueeze(0)
        seq = seq.permute(0,2,1) # [1,emb_size,seq_len]
        bf_size = seq.shape[2]
        seq_shot1 = self.frame_to_shot1(seq)
        seq_shot2 = self.frame_to_shot2(seq)
        seq_shot3 = self.frame_to_shot3(seq)
        # seq_shot4 = self.frame_to_shot4(seq)
        # seq_shot5 = self.frame_to_shot5(seq)

        seq_shot1 = seq_shot1.permute(0,2,1) # [1,seq_len,emb_size]
        seq_shot2 = seq_shot2.permute(0,2,1) # [1,seq_len,emb_size]
        seq_shot3 = seq_shot3.permute(0,2,1) # [1,seq_len,emb_size]
        # seq_shot4 = seq_shot4.permute(0,2,1) # [1,seq_len,emb_size]
        # seq_shot5 = seq_shot5.permute(0,2,1) # [1,seq_len,emb_size]


        seq_shot_cross11 = self.act(self.shot_cross11(seq_shot1))
        seq_shot_cross12 = self.act(self.shot_cross12(seq_shot1))
        seq_shot_cross13 = self.act(self.shot_cross13(seq_shot1))
        seq_shot_cross14 = self.act(self.shot_cross14(seq_shot1))
        # seq_shot_cross15 = self.act(self.shot_cross15(seq_shot1))
        # seq_shot_cross16 = self.act(self.shot_cross16(seq_shot1))
        seq_shot_cross_sum1 = torch.cat((seq_shot_cross11,seq_shot_cross12,seq_shot_cross13,seq_shot_cross14),dim=2)
        #print("seq_shot_cross_sum1 shape:{}".format(seq_shot_cross_sum1.shape)) #
        seq_shot_cross_sum1 = self.pool(seq_shot_cross_sum1).squeeze()
        seq_shot_cross_sum1 = self.fc_down_shot1(seq_shot_cross_sum1)

        seq_shot_cross21 = self.act(self.shot_cross21(seq_shot2))
        seq_shot_cross22 = self.act(self.shot_cross22(seq_shot2))
        seq_shot_cross23 = self.act(self.shot_cross23(seq_shot2))
        seq_shot_cross24 = self.act(self.shot_cross24(seq_shot2))
        # seq_shot_cross25 = self.act(self.shot_cross25(seq_shot2))
        # seq_shot_cross26 = self.act(self.shot_cross26(seq_shot2))
        seq_shot_cross_sum2 = torch.cat((seq_shot_cross21,seq_shot_cross22,seq_shot_cross23,seq_shot_cross24),dim=2)
        #print("seq_shot_cross_sum2 shape:{}".format(seq_shot_cross_sum2.shape)) #
        seq_shot_cross_sum2 = self.pool(seq_shot_cross_sum2).squeeze()
        seq_shot_cross_sum2 = self.fc_down_shot2(seq_shot_cross_sum2)

        seq_shot_cross31 = self.act(self.shot_cross31(seq_shot3))
        seq_shot_cross32 = self.act(self.shot_cross32(seq_shot3))
        seq_shot_cross33 = self.act(self.shot_cross33(seq_shot3))
        seq_shot_cross34 = self.act(self.shot_cross34(seq_shot3))
        # seq_shot_cross35 = self.act(self.shot_cross35(seq_shot3))
        # seq_shot_cross36 = self.act(self.shot_cross36(seq_shot3))
        seq_shot_cross_sum3 = torch.cat((seq_shot_cross31,seq_shot_cross32,seq_shot_cross33,seq_shot_cross34),dim=2)
        #print("seq_shot_cross_sum3 shape:{}".format(seq_shot_cross_sum3.shape)) #
        seq_shot_cross_sum3 = self.pool(seq_shot_cross_sum3).squeeze()
        seq_shot_cross_sum3 = self.fc_down_shot3(seq_shot_cross_sum3)

        # seq_shot_cross41 = self.act(self.shot_cross41(seq_shot4))
        # seq_shot_cross42 = self.act(self.shot_cross42(seq_shot4))
        # seq_shot_cross43 = self.act(self.shot_cross43(seq_shot4))
        # seq_shot_cross44 = self.act(self.shot_cross44(seq_shot4))
        # seq_shot_cross_sum4 = torch.cat((seq_shot_cross41,seq_shot_cross42,seq_shot_cross43,seq_shot_cross44),dim=2)
        # seq_shot_cross_sum4 = self.pool(seq_shot_cross_sum4).squeeze()
        # seq_shot_cross_sum4 = self.fc_down_shot4(seq_shot_cross_sum4)

        # seq_shot_cross51 = self.act(self.shot_cross51(seq_shot5))
        # seq_shot_cross52 = self.act(self.shot_cross52(seq_shot5))
        # seq_shot_cross53 = self.act(self.shot_cross53(seq_shot5))
        # seq_shot_cross54 = self.act(self.shot_cross54(seq_shot5))
        # seq_shot_cross_sum5 = torch.cat((seq_shot_cross51,seq_shot_cross52,seq_shot_cross53,seq_shot_cross54),dim=2)
        # seq_shot_cross_sum5 = self.pool(seq_shot_cross_sum5).squeeze()
        # seq_shot_cross_sum5 = self.fc_down_shot5(seq_shot_cross_sum5)
        # print("seq_shot_cross_sum1 shape:{}".format(seq_shot_cross_sum1.shape)) #[10, 1024]
        # print("seq_shot_cross_sum2 shape:{}".format(seq_shot_cross_sum2.shape)) #[15, 1024]
        # print("seq_shot_cross_sum3 shape:{}".format(seq_shot_cross_sum3.shape)) #[5, 1024]
        seq_shot_cross_sum = torch.cat((seq_shot_cross_sum1,seq_shot_cross_sum2,seq_shot_cross_sum3),0)
        seq_shot_cross_up = mergePooling(seq_shot_cross_sum,bf_size).unsqueeze(0)
        # seq_shot_cross_up1 =  mergePooling(seq_shot_cross_sum1,bf_size).unsqueeze(0)
        # seq_shot_cross_up2 =  mergePooling(seq_shot_cross_sum2,bf_size).unsqueeze(0)
        # seq_shot_cross_up3 =  mergePooling(seq_shot_cross_sum3,bf_size).unsqueeze(0)

        return seq_inital+seq_shot_cross_up

class shotLevelExtractionMS2(nn.Module):
    def __init__(self, shot_num, shot_channel=8096,input_channel=1024):
        super().__init__()
        '''
            1: 3*3 conv1d
            2: 5*5 conv1d
            3: 7*7 conv1d
            4: 1*1 conv1d
        '''
        self.shot_num = shot_num
        self.shot_cross11 = nn.Conv1d(shot_num, shot_num, 3, stride=4)
        self.shot_cross12 = nn.Conv1d(shot_num, shot_num, 3, dilation=2, stride=4)
        self.shot_cross13 = nn.Conv1d(shot_num, shot_num, 3, dilation=3,stride=4)
        self.shot_cross14 = nn.Conv1d(shot_num, shot_num, 1, stride=4)

        self.shot_cross21 = nn.Conv1d(shot_num*2, shot_num*2, 3, stride=4)
        self.shot_cross22 = nn.Conv1d(shot_num*2, shot_num*2, 3, dilation=2, stride=4)
        self.shot_cross23 = nn.Conv1d(shot_num*2, shot_num*2, 3, dilation=3,stride=4)
        self.shot_cross24 = nn.Conv1d(shot_num*2, shot_num*2, 1, stride=4)

        self.shot_cross31 = nn.Conv1d(int(shot_num*0.5), int(shot_num*0.5), 3, stride=4)
        self.shot_cross32 = nn.Conv1d(int(shot_num*0.5), int(shot_num*0.5), 3, dilation=2, stride=4)
        self.shot_cross33 = nn.Conv1d(int(shot_num*0.5), int(shot_num*0.5), 3, dilation=3,stride=4)
        self.shot_cross34 = nn.Conv1d(int(shot_num*0.5), int(shot_num*0.5), 1, stride=4)

        self.act = nn.ReLU()
        self.pool1 = nn.AdaptiveAvgPool1d(1024)
        self.fc_down = nn.Sequential(
                                nn.Linear(3072, 1024),
                                nn.ReLU(),
                                nn.LayerNorm(1024)
                            )

    def forward(self, seq):
        len_tmp = seq.shape[1]
        seq1 = copySeqLen(seq.squeeze(),self.shot_num)
        seq2 = copySeqLen(seq.squeeze(),self.shot_num*2)
        seq3 = copySeqLen(seq.squeeze(),int(self.shot_num*0.5))

        seq11 = self.act(self.shot_cross11(seq1))
        seq12 = self.act(self.shot_cross12(seq1))
        seq13 = self.act(self.shot_cross13(seq1))
        seq14 = self.act(self.shot_cross14(seq1))
        seq1 = self.pool1(torch.cat((seq11,seq12,seq13,seq14),2))
        seq1 = seq1.reshape(1,-1,1024)

        seq21 = self.act(self.shot_cross21(seq2))
        seq22 = self.act(self.shot_cross22(seq2))
        seq23 = self.act(self.shot_cross23(seq2))
        seq24 = self.act(self.shot_cross24(seq2))
        seq2 = self.pool1(torch.cat((seq21,seq22,seq23,seq24),2))
        seq2 = seq2.reshape(1,-1,1024)

        seq31 = self.act(self.shot_cross31(seq3))
        seq32 = self.act(self.shot_cross32(seq3))
        seq33 = self.act(self.shot_cross33(seq3))
        seq34 = self.act(self.shot_cross34(seq3))
        seq3 = self.pool1(torch.cat((seq31,seq32,seq33,seq34),2))
        seq3 = seq3.reshape(1,-1,1024)
        seq1 = seq1[:,:len_tmp,:]
        seq2 = seq2[:,:len_tmp,:]
        seq3 = seq3[:,:len_tmp,:]
        seq_merge = self.fc_down(torch.cat((seq1,seq2,seq3),2).squeeze())
        seq = seq_merge.unsqueeze(0)+seq
        return seq


class shotLevelExtractionML(nn.Module):
    def __init__(self, shot_num, shot_channel=8096,input_channel=1024):
        super().__init__()
        self.fc_up_shot1 = nn.Sequential(
                                nn.Linear(input_channel, shot_channel),
                                nn.ReLU(),
                                nn.LayerNorm(shot_channel)
                            )
        self.fc_up_shot2 = nn.Sequential(
                                nn.Linear(input_channel, shot_channel),
                                nn.ReLU(),
                                nn.LayerNorm(shot_channel)
                            )
        self.fc_up_shot3 = nn.Sequential(
                                nn.Linear(input_channel, shot_channel),
                                nn.ReLU(),
                                nn.LayerNorm(shot_channel)
                            )
        self.shot_num = shot_num
        self.frame_to_shot1 = nn.AdaptiveAvgPool1d(shot_num)
        self.frame_to_shot2 = nn.AdaptiveAvgPool1d(int(shot_num*1.5))
        self.frame_to_shot3 = nn.AdaptiveAvgPool1d(int(shot_num*0.5))

        '''
            1: 3*3 conv1d
            2: 5*5 conv1d
            3: 7*7 conv1d
            4: 1*1 conv1d
        '''
        self.shot_cross11 = nn.Conv1d(shot_num, shot_num, 3, stride=4)
        self.shot_cross12 = nn.Conv1d(shot_num, shot_num, 3, dilation=2, stride=4)
        self.shot_cross13 = nn.Conv1d(shot_num, shot_num, 3, dilation=3,stride=4)
        self.shot_cross14 = nn.Conv1d(shot_num, shot_num, 1, stride=4)

        self.shot_cross21 = nn.Conv1d(int(shot_num*1.5), int(shot_num*1.5), 3, stride=4)
        self.shot_cross22 = nn.Conv1d(int(shot_num*1.5), int(shot_num*1.5), 3, dilation=2, stride=4)
        self.shot_cross23 = nn.Conv1d(int(shot_num*1.5), int(shot_num*1.5), 3, dilation=3,stride=4)
        self.shot_cross24 = nn.Conv1d(int(shot_num*1.5), int(shot_num*1.5), 1, stride=4)

        self.shot_cross31 = nn.Conv1d(int(shot_num*0.5), int(shot_num*0.5), 3, stride=4)
        self.shot_cross32 = nn.Conv1d(int(shot_num*0.5), int(shot_num*0.5), 3, dilation=2, stride=4)
        self.shot_cross33 = nn.Conv1d(int(shot_num*0.5), int(shot_num*0.5), 3, dilation=3,stride=4)
        self.shot_cross34 = nn.Conv1d(int(shot_num*0.5), int(shot_num*0.5), 1, stride=4)


        self.pool = nn.AdaptiveAvgPool1d(shot_channel)
        self.act = nn.ReLU()
        self.fc_down_shot1 = nn.Sequential(
                                nn.Linear(shot_channel, input_channel),
                                nn.ReLU(),
                                nn.LayerNorm(input_channel)
                            )#nn.Linear(shot_channel,input_channel)
        self.fc_down_shot2 = nn.Sequential(
                                nn.Linear(shot_channel, input_channel),
                                nn.ReLU(),
                                nn.LayerNorm(input_channel)
                            )
        self.fc_down_shot3 = nn.Sequential(
                                nn.Linear(shot_channel, input_channel),
                                nn.ReLU(),
                                nn.LayerNorm(input_channel)
                            )

    def forward(self, seq1,seq2,seq3):
        '''
            seq: [1,seq_len,emb_size]
            
        '''
        seq_inital1 = seq1
        seq_inital2 = seq2
        seq_inital3 = seq3

        seq1 = self.fc_up_shot1(seq1.squeeze()).unsqueeze(0)
        seq1 = seq1.permute(0,2,1) # [1,emb_size,seq_len]
        seq2 = self.fc_up_shot2(seq2.squeeze()).unsqueeze(0)
        seq2 = seq2.permute(0,2,1)
        seq3 = self.fc_up_shot3(seq3.squeeze()).unsqueeze(0)
        seq3 = seq3.permute(0,2,1)

        bf_size = seq1.shape[2]
        seq_shot1 = self.frame_to_shot1(seq1)
        seq_shot2 = self.frame_to_shot2(seq2)
        seq_shot3 = self.frame_to_shot3(seq3)

        seq_shot1 = seq_shot1.permute(0,2,1) # [1,seq_len,emb_size]
        seq_shot2 = seq_shot2.permute(0,2,1) # [1,seq_len,emb_size]
        seq_shot3 = seq_shot3.permute(0,2,1) # [1,seq_len,emb_size]


        seq_shot_cross11 = self.act(self.shot_cross11(seq_shot1))
        seq_shot_cross12 = self.act(self.shot_cross12(seq_shot1))
        seq_shot_cross13 = self.act(self.shot_cross13(seq_shot1))
        seq_shot_cross14 = self.act(self.shot_cross14(seq_shot1))

        seq_shot_cross_sum1 = torch.cat((seq_shot_cross11,seq_shot_cross12,seq_shot_cross13,seq_shot_cross14),dim=2)
        seq_shot_cross_sum1 = self.pool(seq_shot_cross_sum1).squeeze()
        seq_shot_cross_sum1 = self.fc_down_shot1(seq_shot_cross_sum1)

        seq_shot_cross21 = self.act(self.shot_cross21(seq_shot2))
        seq_shot_cross22 = self.act(self.shot_cross22(seq_shot2))
        seq_shot_cross23 = self.act(self.shot_cross23(seq_shot2))
        seq_shot_cross24 = self.act(self.shot_cross24(seq_shot2))
        # seq_shot_cross25 = self.act(self.shot_cross25(seq_shot2))
        # seq_shot_cross26 = self.act(self.shot_cross26(seq_shot2))
        seq_shot_cross_sum2 = torch.cat((seq_shot_cross21,seq_shot_cross22,seq_shot_cross23,seq_shot_cross24),dim=2)
        seq_shot_cross_sum2 = self.pool(seq_shot_cross_sum2).squeeze()
        seq_shot_cross_sum2 = self.fc_down_shot2(seq_shot_cross_sum2)

        seq_shot_cross31 = self.act(self.shot_cross31(seq_shot3))
        seq_shot_cross32 = self.act(self.shot_cross32(seq_shot3))
        seq_shot_cross33 = self.act(self.shot_cross33(seq_shot3))
        seq_shot_cross34 = self.act(self.shot_cross34(seq_shot3))
        # seq_shot_cross35 = self.act(self.shot_cross35(seq_shot3))
        # seq_shot_cross36 = self.act(self.shot_cross36(seq_shot3))
        seq_shot_cross_sum3 = torch.cat((seq_shot_cross31,seq_shot_cross32,seq_shot_cross33,seq_shot_cross34),dim=2)
        seq_shot_cross_sum3 = self.pool(seq_shot_cross_sum3).squeeze()
        seq_shot_cross_sum3 = self.fc_down_shot3(seq_shot_cross_sum3)

        seq_shot_cross_up1 =  mergePooling(seq_shot_cross_sum1,bf_size).unsqueeze(0)
        seq_shot_cross_up2 =  mergePooling(seq_shot_cross_sum2,bf_size).unsqueeze(0)
        seq_shot_cross_up3 =  mergePooling(seq_shot_cross_sum3,bf_size).unsqueeze(0)

        return seq_inital1+seq_shot_cross_up1,seq_inital2+seq_shot_cross_up2,seq_inital3+seq_shot_cross_up3

class shotExtractionModule(nn.Module):
    def __init__(self, shot_num, shot_channel=8096,input_channel=1024,expand_ratio=1.0):
        super().__init__()
        self.shot_channel = int(shot_channel*expand_ratio)
        self.fc_up_shot = nn.Linear(input_channel,self.shot_channel)
        self.shot_num = int(shot_num * expand_ratio)
        self.frame_to_shot = nn.AdaptiveAvgPool1d(self.shot_num)

        '''
            1: 3*3 conv1d
            2: 5*5 conv1d
            3: 7*7 conv1d
            4: 1*1 conv1d
        '''
        self.shot_cross1 = nn.Conv1d(int(shot_num*expand_ratio), int(shot_num*expand_ratio), 3, stride=2)
        self.shot_cross2 = nn.Conv1d(int(shot_num*expand_ratio), int(shot_num*expand_ratio), 3, dilation=2, stride=2)
        self.shot_cross3 = nn.Conv1d(int(shot_num*expand_ratio), int(shot_num*expand_ratio), 3, dilation=3,stride=4)
        self.shot_cross4 = nn.Conv1d(int(shot_num*expand_ratio), int(shot_num*expand_ratio), 1, stride=4)
        self.LayerNorm1 =  nn.LayerNorm(self.shot_channel) 
        self.pool = nn.AdaptiveAvgPool1d(self.shot_channel)
        self.act = nn.ReLU()
        self.fc_down_shot = nn.Sequential(
                                nn.Linear(self.shot_channel, input_channel),
                                nn.ReLU(),
                                nn.LayerNorm(input_channel)
                            )#nn.Linear(shot_channel,input_channel)

    def forward(self, seq):
        '''
            seq: [1,seq_len,emb_size]
            
        '''
        seq_inital = seq
        seq = self.fc_up_shot(seq.squeeze()).unsqueeze(0)
        seq = seq.permute(0,2,1) # [1,emb_size,seq_len]
        bf_size = seq.shape[2]
        seq_shot1 = self.frame_to_shot(seq)
        seq_shot1 = seq_shot1.permute(0,2,1) # [1,seq_len,emb_size]

        #print(seq_shot1.shape)
        seq_shot_cross11 = self.shot_cross1(seq_shot1)
        seq_shot_cross12 = self.shot_cross2(seq_shot1)
        seq_shot_cross13 = self.shot_cross3(seq_shot1)
        seq_shot_cross14 = self.shot_cross4(seq_shot1)
        seq_shot_cross_sum1 = torch.cat((seq_shot_cross11,seq_shot_cross12,seq_shot_cross13,seq_shot_cross14),dim=2)
        seq_shot_cross_sum1 = self.LayerNorm1(self.act(self.pool(seq_shot_cross_sum1))).squeeze()
        seq_shot_cross_sum1 = self.fc_down_shot(seq_shot_cross_sum1)

        seq_shot_cross_up =  mergePooling(seq_shot_cross_sum1,bf_size).unsqueeze(0)
        #print(seq_shot_cross_up.shape)
        return seq_inital+seq_shot_cross_up

def shotAvgPool(seq,shot_num,shot_channel):
    '''
        input:
        seq: [1,seq_len,emb_size]
        shot_num
        output:
        [1,shot_num,shot_channel]
    '''
    shot_emb = torch.empty(1,shot_num,shot_channel).cuda()
    split_size = seq.shape[1] // shot_num
    for i in range(shot_num):
        tmp = seq[:,split_size*i,:]
        for j in range((i-1)*split_size+1,i*split_size):
            tmp = torch.cat((tmp,seq[:,j,:]),1)
        tmp = tmp.unsqueeze(1)
        tmp = F.adaptive_avg_pool1d(tmp,shot_channel)
        shot_emb[:,i,:] = tmp
    return shot_emb

class shotLevelExtractionV2(nn.Module):
    def __init__(self, shot_num, shot_channel=8096,input_channel=1024):
        super().__init__()
        '''
            1: 3*3 conv1d
            2: 5*5 conv1d
            3: 7*7 conv1d
            4: 1*1 conv1d
        '''
        self.shot_cross1 = nn.Conv1d(shot_num, shot_num, 3, stride=4)
        self.shot_cross2 = nn.Conv1d(shot_num, shot_num, 3, dilation=2, stride=4)
        self.shot_cross3 = nn.Conv1d(shot_num, shot_num, 3, dilation=3,stride=4)
        self.shot_cross4 = nn.Conv1d(shot_num, shot_num, 1, stride=4)
        self.pool = nn.AdaptiveAvgPool1d(shot_channel*shot_num)
        self.shot_num = shot_num
        self.shot_channel = shot_channel 
        self.pool2 = nn.AdaptiveAvgPool1d(shot_channel)
        self.act = nn.ReLU()
        self.pool3 = nn.AdaptiveAvgPool1d(input_channel)
        self.fc_down_shot = nn.Sequential(
                                nn.Linear(shot_channel, input_channel),
                                nn.Tanh(),
                                nn.LayerNorm(input_channel)
                            )#nn.Linear(shot_channel,input_channel)


    def forward(self, seq):
        '''
            seq: [1,seq_len,emb_size]
            
        '''
        seq_inital = seq
        seq_len = seq.shape[1]
        seq_shot = shotAvgPool(seq,self.shot_num,self.shot_channel)
        seq_shot_cross1 = self.shot_cross1(seq_shot)
        seq_shot_cross2 = self.shot_cross2(seq_shot)
        seq_shot_cross3 = self.shot_cross3(seq_shot)
        seq_shot_cross4 = self.shot_cross4(seq_shot)
        seq_shot_cross_sum = torch.cat((seq_shot_cross1,seq_shot_cross2,seq_shot_cross3,seq_shot_cross4),dim=2)
        seq_shot_cross_sum = self.pool2(seq_shot_cross_sum).squeeze()
        #print(seq_shot_cross_sum.shape) [shot_num,shot_channel]
        seq_shot_cross_sum = seq_shot_cross_sum.cuda()
        #print(seq_shot_cross_sum)
        seq_shot_cross_sum = self.fc_down_shot(seq_shot_cross_sum)
        # shot_num_overlap -> shot_num
        seq_shot_cross_up =  mergePooling(seq_shot_cross_sum,seq_len).unsqueeze(0)
        return seq_inital+seq_shot_cross_up

def largeOverlap(feature,shot_num):
    feature_overlap = torch.empty(1,shot_num,feature.shape[2]).cuda()
    for i in range(shot_num):
        if i % 2==0: # copy
            feature_overlap[:,i,:] = feature[:,i//2,:]
        else: #overlap part
            feature_overlap[:,i,:] = (feature[:,i//2,:] + feature[:,i//2+1,:])/2
    return feature_overlap.cuda()

def overlapSeqGenerate(seq,overlap_ratio,shot_num):
    '''
        input: seq [1,seq_len,emb_size]

        output: seq_overlap = [1,seq_len+overlap_frame*(shot_num-1),emb_size] 
    '''
    seq_len = seq.shape[1]
    split_channel = seq_len // shot_num
    overlap_frame = max(int(overlap_ratio * split_channel),1)
    seq_overlap = torch.empty(1,seq_len+overlap_frame*shot_num,seq.shape[2]).cuda()
    for i in range(shot_num):
        if i==0:
            seq_overlap[:,:overlap_frame,:] = seq[:,-overlap_frame:,:]
        else:
            seq_overlap[:,i*(split_channel+overlap_frame):i*(split_channel+overlap_frame)+overlap_frame,:] = seq[:,i*split_channel-overlap_frame:i*split_channel,:]
        seq_overlap[:,i*(split_channel+overlap_frame)+overlap_frame:(i+1)*(split_channel+overlap_frame),:] = seq[:,i*split_channel:(i+1)*split_channel,:]
    seq_overlap[:,(split_channel+overlap_frame)*shot_num:,:] = seq[:,split_channel*shot_num:,:]
    return seq_overlap

class shotLevelExtractionOverlap(nn.Module):
    def __init__(self, shot_num, shot_channel=8096,input_channel=1024,overlap_ratio=0.25):
        super().__init__()
        '''
            1: 3*3 conv1d
            2: 5*5 conv1d
            3: 7*7 conv1d
            4: 1*1 conv1d
        '''
        self.fc_up_shot = nn.Linear(input_channel,shot_channel)
        self.frame_to_shot1 = nn.AdaptiveAvgPool1d(shot_num)
        self.frame_to_shot2 = nn.AdaptiveAvgPool1d(int(shot_num*1.5))
        self.frame_to_shot3 = nn.AdaptiveAvgPool1d(int(shot_num*0.5))

        self.shot_cross11 = nn.Conv1d(shot_num, shot_num, 3, stride=4)
        self.shot_cross12 = nn.Conv1d(shot_num, shot_num, 3, dilation=2, stride=4)
        self.shot_cross13 = nn.Conv1d(shot_num, shot_num, 3, dilation=3,stride=4)
        self.shot_cross14 = nn.Conv1d(shot_num, shot_num, 1, stride=4)

        self.shot_cross21 = nn.Conv1d(int(shot_num*1.5), int(shot_num*1.5), 3, stride=4)
        self.shot_cross22 = nn.Conv1d(int(shot_num*1.5), int(shot_num*1.5), 3, dilation=2, stride=4)
        self.shot_cross23 = nn.Conv1d(int(shot_num*1.5), int(shot_num*1.5), 3, dilation=3,stride=4)
        self.shot_cross24 = nn.Conv1d(int(shot_num*1.5), int(shot_num*1.5), 1, stride=4)

        self.shot_cross31 = nn.Conv1d(int(shot_num*0.5), int(shot_num*0.5), 3, stride=4)
        self.shot_cross32 = nn.Conv1d(int(shot_num*0.5), int(shot_num*0.5), 3, dilation=2, stride=4)
        self.shot_cross33 = nn.Conv1d(int(shot_num*0.5), int(shot_num*0.5), 3, dilation=3,stride=4)
        self.shot_cross34 = nn.Conv1d(int(shot_num*0.5), int(shot_num*0.5), 1, stride=4)

        self.pool = nn.AdaptiveAvgPool1d(shot_channel*shot_num)
        self.shot_num = shot_num
        self.shot_channel = shot_channel 
        self.overlap_ratio = overlap_ratio
        self.pool2 = nn.AdaptiveAvgPool1d(shot_channel)
        self.act = nn.ReLU()
        self.pool3 = nn.AdaptiveAvgPool1d(input_channel)
        self.fc_down_shot1 = nn.Sequential(
                                nn.Linear(shot_channel, input_channel),
                                nn.ReLU(),
                                nn.LayerNorm(input_channel)
                            )#nn.Linear(shot_channel,input_channel)
        self.fc_down_shot2 = nn.Sequential(
                                nn.Linear(shot_channel, input_channel),
                                nn.ReLU(),
                                nn.LayerNorm(input_channel)
                            )
        self.fc_down_shot3 = nn.Sequential(
                                nn.Linear(shot_channel, input_channel),
                                nn.ReLU(),
                                nn.LayerNorm(input_channel)
                            )


    def forward(self, seq):
        '''
            seq: [1,seq_len,emb_size]
            
        '''
        seq_inital = seq
        seq_len = seq.shape[1]
        seq = overlapSeqGenerate(seq,self.overlap_ratio,self.shot_num)
        seq = self.fc_up_shot(seq.squeeze()).unsqueeze(0)
        seq = seq.permute(0,2,1) # [1,emb_size,seq_len]
        seq_shot1 = self.frame_to_shot1(seq)
        seq_shot2 = self.frame_to_shot2(seq)
        seq_shot3 = self.frame_to_shot3(seq)
        seq_shot1 = seq_shot1.permute(0,2,1) 
        seq_shot2 = seq_shot2.permute(0,2,1) 
        seq_shot3 = seq_shot3.permute(0,2,1) 

        # seq_shot1 = shotAvgPool(seq,self.shot_num,self.shot_channel)
        # seq_shot2 = shotAvgPool(seq,int(self.shot_num*1.5),self.shot_channel)
        # seq_shot3 = shotAvgPool(seq,int(self.shot_num*0.5),self.shot_channel)

        seq_shot_cross11 = self.shot_cross11(seq_shot1)
        seq_shot_cross12 = self.shot_cross12(seq_shot1)
        seq_shot_cross13 = self.shot_cross13(seq_shot1)
        seq_shot_cross14 = self.shot_cross14(seq_shot1)
        seq_shot_cross_sum1 = torch.cat((seq_shot_cross11,seq_shot_cross12,seq_shot_cross13,seq_shot_cross14),dim=2)
        seq_shot_cross_sum1 = self.pool2(seq_shot_cross_sum1).squeeze()
        seq_shot_cross_sum1 = seq_shot_cross_sum1.cuda()
        seq_shot_cross_sum1 = self.fc_down_shot1(seq_shot_cross_sum1)

        seq_shot_cross21 = self.shot_cross21(seq_shot2)
        seq_shot_cross22 = self.shot_cross22(seq_shot2)
        seq_shot_cross23 = self.shot_cross23(seq_shot2)
        seq_shot_cross24 = self.shot_cross24(seq_shot2)
        seq_shot_cross_sum2 = torch.cat((seq_shot_cross21,seq_shot_cross22,seq_shot_cross23,seq_shot_cross24),dim=2)
        seq_shot_cross_sum2 = self.pool2(seq_shot_cross_sum2).squeeze()
        seq_shot_cross_sum2 = seq_shot_cross_sum2.cuda()
        seq_shot_cross_sum2 = self.fc_down_shot2(seq_shot_cross_sum2)

        seq_shot_cross31 = self.shot_cross31(seq_shot3)
        seq_shot_cross32 = self.shot_cross32(seq_shot3)
        seq_shot_cross33 = self.shot_cross33(seq_shot3)
        seq_shot_cross34 = self.shot_cross34(seq_shot3)
        seq_shot_cross_sum3 = torch.cat((seq_shot_cross31,seq_shot_cross32,seq_shot_cross33,seq_shot_cross34),dim=2)
        seq_shot_cross_sum3 = self.pool2(seq_shot_cross_sum3).squeeze()
        seq_shot_cross_sum3 = seq_shot_cross_sum3.cuda()
        seq_shot_cross_sum3 = self.fc_down_shot3(seq_shot_cross_sum3)
    

        # shot_num_overlap -> shot_num
        # seq_shot_cross_sum = seq_shot_cross_sum3
        seq_shot_cross_sum = torch.cat((seq_shot_cross_sum1,seq_shot_cross_sum2,seq_shot_cross_sum3),0)
        seq_shot_cross_up = mergePooling(seq_shot_cross_sum,seq_len).unsqueeze(0)
        return seq_inital+seq_shot_cross_up

class shotLevelEncoderMS(nn.Module):
    def __init__(self, layer_nums=1, shot_num=10, shot_channel=[8096],input_channel=1024):
        super().__init__()
        assert layer_nums==len(shot_channel),"lose the shot_channel for other shot encoder layers"
        self.encoder_layers = nn.ModuleList([shotLevelExtractionMS(shot_num,shot_channel[i],input_channel) for i in range(layer_nums)])

    def forward(self,x):
        for layer in self.encoder_layers:
            x = layer(x) #res unit
        return x

class shotLevelEncoderOverlap(nn.Module):
    def __init__(self, layer_nums=1, shot_num=10, shot_channel=[8096],input_channel=1024,overlap_ratio=0.25):
        super().__init__()
        assert layer_nums==len(shot_channel),"lose the shot_channel for other shot encoder layers"
        self.encoder_layers = nn.ModuleList([shotLevelExtractionOverlap(shot_num,shot_channel[i],input_channel,overlap_ratio) for i in range(layer_nums)])

    def forward(self,x):
        for layer in self.encoder_layers:
            x = layer(x) #res unit
        return x

class shotLevelEncoderML(nn.Module):
    def __init__(self, layer_nums=1, shot_num=10, shot_channel=[8096],input_channel=1024,overlap_ratio=0.25):
        super().__init__()
        assert layer_nums==len(shot_channel),"lose the shot_channel for other shot encoder layers"
        self.encoder_layers = nn.ModuleList([shotLevelExtractionML(shot_num,shot_channel[i],input_channel) for i in range(layer_nums)])

    def forward(self,x):
        i = 0 
        for layer in self.encoder_layers:
            if i ==0:
                x1,x2,x3 = layer(x,x,x)
            else:
                x1,x2,x3 = layer(x1,x2,x3)
            i += 1
        return x1,x2,x3

class shotLevelConv(nn.Module):
    def __init__(self, layer_nums=1, shot_num=10, shot_channel=[8096],input_channel=1024,overlap_ratio=0.25):
        super().__init__()
        assert layer_nums==len(shot_channel),"lose the shot_channel for other shot encoder layers"
        self.shot_channel = shot_channel[0]
        self.conv1 = shotExtractionModule(shot_num, shot_channel=self.shot_channel,input_channel=input_channel,expand_ratio=2.0)
        self.conv2 = shotExtractionModule(shot_num, shot_channel=self.shot_channel,input_channel=input_channel,expand_ratio=1.5)
        self.conv3 = shotExtractionModule(shot_num, shot_channel=self.shot_channel,input_channel=input_channel,expand_ratio=1.0)
        self.conv4 = shotExtractionModule(shot_num, shot_channel=self.shot_channel,input_channel=input_channel,expand_ratio=0.7)
        self.conv5 = shotExtractionModule(shot_num, shot_channel=self.shot_channel,input_channel=input_channel,expand_ratio=0.5)

    def forward(self,x):
        x = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x)))))
        return x

class MSNet(nn.Module):
    def __init__(self,shot_layer_nums,shot_num,shot_channel,d_model,shot_first=True, \
                    long_head=8, long_num_encoder_layers=6, long_num_decoder_layers=6, long_dim_feedforward=2048,input_channel=1024):
        super().__init__()
        self.shot_fusion = shotLevelEncoder(layer_nums=shot_layer_nums,shot_num=shot_num,shot_channel=shot_channel,input_channel=input_channel)
        self.transformer_long = nn.Transformer(d_model=d_model, nhead=long_head, num_encoder_layers=long_num_encoder_layers, \
                                                    num_decoder_layers=long_num_decoder_layers, dim_feedforward=long_dim_feedforward)
        self.score_projection = nn.Linear(d_model,1)
        self.shot_first = shot_first

    def forward(self, seq):
        '''
        input:
            audio_emb:
            cap_emb:
            seq:
        output:
            label
        '''
        if self.shot_first:
            seq_fusion = self.shot_fusion(seq)
            seq_long = self.transformer_long(seq_fusion,seq_fusion).squeeze()
        else:
            seq_long = self.transformer_long(seq,seq).squeeze()
            seq_long = self.shot_fusion(seq_long).squeeze()
        pred_cls = self.score_projection(seq_long).sigmoid()
        return pred_cls

class visualAudioFuison(nn.Module):
    def __init__(self,input_channel=1024,fusion_format='add'):
        super().__init__()
        self.fusion_format = fusion_format
        if self.fusion_format == 'add':
            self.a2v_fc = nn.Linear(128,input_channel)
        elif self.fusion_format == 'attention_add':
            self.a2v_fc = nn.Linear(128,input_channel)
            self.att_multi = nn.MultiheadAttention(embed_dim=input_channel,num_heads=8,batch_first=True)


    def forward(self,x,y):
        y = y.permute(0,2,1)
        y = F.adaptive_avg_pool1d(y,x.shape[1])
        y = y.permute(0,2,1)
        if self.fusion_format == 'add':
            x += self.a2v_fc(y.squeeze()).unsqueeze(0)
        elif self.fusion_format == 'concat':
            x = torch.cat((x,y),dim=2)
        elif self.fusion_format == 'attention_add':
            y = self.a2v_fc(y.squeeze()).unsqueeze(0)
            x += self.att_multi(query=x,key=y,value=y)
        return x

class MSAudioNet(nn.Module):
    def __init__(self,shot_layer_nums,shot_num,shot_channel,d_model,shot_first=True, \
                    long_head=8, long_num_encoder_layers=6, long_num_decoder_layers=6, long_dim_feedforward=2048,input_channel=1024,fusion_format='add'):
        super().__init__()
        self.visual_audio_fuison = visualAudioFuison(input_channel=input_channel,fusion_format=fusion_format)
        if fusion_format=='concat':
            input_channel += 128
            d_model += 128
        self.shot_fusion = shotLevelEncoder(layer_nums=shot_layer_nums,shot_num=shot_num,shot_channel=shot_channel,input_channel=input_channel)
        self.transformer_long = nn.Transformer(d_model=d_model, nhead=long_head, num_encoder_layers=long_num_encoder_layers, \
                                                    num_decoder_layers=long_num_decoder_layers, dim_feedforward=long_dim_feedforward)
        self.score_projection = nn.Linear(d_model,1)
        self.shot_first = shot_first


    def forward(self, seq, vggish):
        '''
        input:
            audio_emb:
            cap_emb:
            seq:
        output:
            label
        '''
        seq = self.visual_audio_fuison(seq,vggish)
        if self.shot_first:
            seq_fusion = self.shot_fusion(seq)
            seq_long = self.transformer_long(seq_fusion,seq_fusion).squeeze()
        else:
            seq_long = self.transformer_long(seq,seq).squeeze()
            seq_long = self.shot_fusion(seq_long).squeeze()
        pred_cls = self.score_projection(seq_long).sigmoid()
        return pred_cls

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


class MSAudioCaptionNet(nn.Module):
    def __init__(self,shot_layer_nums,shot_num,shot_channel,d_model,shot_first=True, \
                    long_head=8, long_num_encoder_layers=6, long_num_decoder_layers=6, \
                    long_dim_feedforward=2048,input_channel=1024,fusion_format='add', \
                    hidden_layer=128,need_recon=False,need_clip_predict=False, \
                    need_fusion_cap=False,need_sl=False,overlap_ratio=0.25,need_overlap=False):
        super().__init__()
        self.visual_audio_fuison = visualAudioFuison(input_channel=input_channel,fusion_format=fusion_format)
        self.need_fusion_cap = need_fusion_cap
        if fusion_format=='concat':
            input_channel += 128
            d_model += 128
        if need_overlap == False:
            self.shot_fusion = shotLevelEncoderMS(layer_nums=shot_layer_nums,shot_num=shot_num,shot_channel=shot_channel,input_channel=input_channel)
        else:
            self.shot_fusion = shotLevelEncoderOverlap(layer_nums=shot_layer_nums,shot_num=shot_num,shot_channel=shot_channel,input_channel=input_channel,overlap_ratio=overlap_ratio)

        # self.transformer_long = nn.Transformer(d_model=d_model, nhead=long_head, num_encoder_layers=long_num_encoder_layers, \
        #                                             num_decoder_layers=long_num_decoder_layers, dim_feedforward=long_dim_feedforward)
        # self.lstm = nn.LSTM(input_size=1024,hidden_size=512,num_layers=5)
        # self.transformer_long2 = nn.Transformer(d_model=d_model, nhead=long_head, num_encoder_layers=long_num_encoder_layers, \
        #                                             num_decoder_layers=long_num_decoder_layers, dim_feedforward=long_dim_feedforward)
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8)
        # self.transformer_long = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.score_projection = nn.Sequential(
            nn.Linear(1024, hidden_layer),
            nn.Tanh(),#nn.ReLU(),#nn.Tanh(),
            nn.Dropout(0.5),
            nn.LayerNorm(hidden_layer),
            nn.Linear(hidden_layer,1)
        )
        # self.score_projection2 = nn.Sequential(
        #     nn.Linear(1024, hidden_layer),
        #     nn.Tanh(),#nn.ReLU(),#nn.Tanh(),
        #     nn.Dropout(0.5),
        #     nn.LayerNorm(hidden_layer),
        #     nn.Linear(hidden_layer,1)
        # )
        # self.score_projection3 = nn.Sequential(
        #     nn.Linear(1024, hidden_layer),
        #     nn.Tanh(),#nn.ReLU(),#nn.Tanh(),
        #     nn.Dropout(0.5),
        #     nn.LayerNorm(hidden_layer),
        #     nn.Linear(hidden_layer,1)
        # )
        # self.gcn1 = GCNExtractor(1024,2048)
        # self.gcn2 = GCNExtractor(2048,2048)
        # self.gcn3 = GCNExtractor(2048,1024)

        # self.score_projection = nn.Sequential(
        #     nn.Linear(1024, 256),
        #     nn.ReLU(),#nn.Tanh(),
        #     nn.Dropout(0.5),
        #     nn.BatchNorm1d(256),
        #     nn.Linear(256, 32),
        #     nn.ReLU(),#nn.Tanh(),
        #     nn.Dropout(0.5),
        #     nn.BatchNorm1d(32),
        #     nn.Linear(32,1)
        # )
        self.need_recon = need_recon
        # if self.need_recon:
        #     #self.reconstruct_fc = nn.Transformer(d_model=d_model, nhead=long_head, num_encoder_layers=1, \
        #                                 #num_decoder_layers=1, dim_feedforward=long_dim_feedforward)
        #     self.reconstruct_fc = nn.Sequential(
        #         nn.Linear(1, hidden_layer),
        #         nn.Tanh(),
        #         nn.Dropout(0.5),
        #         nn.LayerNorm(hidden_layer),
        #         nn.Linear(hidden_layer,d_model)
        #     )
        self.need_clip_predict = need_clip_predict
        # if self.need_clip_predict:
        #     self.fc_selfspv = nn.Sequential(
        #         nn.Linear(24576, 128),
        #         nn.Tanh(),
        #         nn.Dropout(0.5),
        #         nn.LayerNorm(128),
        #         nn.Linear(128,6)
        #     )
        #self.fusion_cap_frame = nn.Transformer(d_model=d_model, nhead=long_head, num_encoder_layers=1, \
                                        # num_decoder_layers=1, dim_feedforward=long_dim_feedforward)
        self.shot_first = shot_first
        self.need_sl = need_sl
        # if self.need_sl:
        #     self.score_projection2 = nn.Sequential(
        #         nn.Linear(d_model, hidden_layer),
        #         nn.Tanh(),
        #         nn.Dropout(0.5),
        #         nn.LayerNorm(hidden_layer),
        #         nn.Linear(hidden_layer,1)
        #     )
        self.fuse_cap_att = nn.MultiheadAttention(embed_dim=1024, num_heads=8)

    def forward(self, seq, vggish, cap_emb, order=None, isTrain=True):
        '''
        input:
            audio_emb:
            cap_emb:
            seq:
        output:
            label
        '''
        seq_inital = seq
        len_inital = seq.shape[1]
        seq = self.visual_audio_fuison(seq,vggish)
        if self.shot_first:
            seq_fusion = seq.permute(1,0,2)
            seq_fusion_cap,_ = self.fuse_cap_att(seq_fusion,cap_emb,cap_emb)
            seq_fusion = seq_fusion + seq_fusion_cap 
            seq = seq_fusion.permute(1,0,2)
            #seq = seq.squeeze()
            #seq = copySeqLen(seq,320) #[bs,320,1024]
            seq_fusion = self.shot_fusion(seq)
            seq_fusion = seq_fusion.reshape(1,-1,1024)
            seq_long= seq_fusion.squeeze()# + seq_inital
            seq_long = seq_long[:len_inital,:]
            # seq_fusion1,seq_fusion2,seq_fusion3 = self.shot_fusion(seq)
            #seq = seq_fusion.permute(1,0,2)
            #print(seq_fusion)seq = seq.permute(1,0,2)
            # seq_long1 = seq_fusion1.squeeze()
            # seq_long2 = seq_fusion2.squeeze()
            # seq_long3 = seq_fusion3.squeeze()
            #print(seq_fusion)
            #seq_fusion = seq.permute(1,0,2)
            #self.fusion_cap_frame(cap_emb,seq_fusion)
            # if not self.need_fusion_cap:
            #     #print(cap_emb.shape)
            #     #print(seq_fusion.shape)
            #     seq_long = self.transformer_long(cap_emb,seq_fusion).squeeze()
            # else:
            #     seq = seq.permute(1,0,2)
            #     seq_long = self.fusion_cap_frame(cap_emb,seq_fusion)
            #     #seq_long = self.gcn3(self.gcn2(self.gcn1(seq_long)))
            #     seq_long = self.transformer_long(seq_long,seq_long).squeeze()
                # seq_long,_1 = self.lstm(seq_long)
                # seq_long = seq_long.squeeze()
                #seq = self.transformer_long2(seq,seq).squeeze()
                #seq_long = torch.cat((seq.squeeze(),seq_long),1)
                #seq_long = seq_long + seq_fusion.squeeze() # residual format
            #seq_long = seq_long + seq_fusion.squeeze() # residual format
            #print(seq_long)
        else:
            seq_long = self.transformer_long(seq,cap_emb).squeeze()
            seq_long = self.shot_fusion(seq_long).squeeze()
        # if self.need_clip_predict and isTrain:
        #     clip_order = self.fc_selfspv(self_supervise_clip(seq_long,order))
        pred_cls = self.score_projection(seq_long)
        # pred_cls2 = self.score_projection2(seq_long2)
        # pred_cls3 = self.score_projection3(seq_long3)
        # if self.need_recon:
        #     seq_long = seq_long.unsqueeze(0)
        #     seq_recon = self.reconstruct_fc(pred_cls)
        
        pred_cls = pred_cls.sigmoid()
        # pred_cls2 = pred_cls2.sigmoid()
        # pred_cls3 = pred_cls3.sigmoid()
        return pred_cls#,seq_long
        #print(pred_cls)
        # if self.need_sl:
        #     pred_score = self.score_projection2(seq_long).sigmoid()
        # if self.need_recon:
        #     if not self.need_clip_predict or not isTrain:
        #         if not self.need_sl:
        #             return pred_cls,seq_recon
        #         else:
        #             return pred_cls,seq_recon,pred_score
        #     else :
        #         if not self.need_sl:
        #             return pred_cls,seq_recon,clip_order
        #         else:
        #             return pred_cls,seq_recon,clip_order,pred_score
        # else:
        #     if not self.need_clip_predict or not isTrain:
        #         if not self.need_sl:
        #             return pred_cls
        #         else:
        #             return pred_cls,pred_score
        #     else:
        #         if not self.need_sl:
        #             return pred_cls,clip_order
        #         else:
        #             return pred_cls,clip_order,pred_score

if __name__ == '__main__':
    m = nn.Transformer(d_model=128).cuda()
    x = torch.randn(1,768,128).cuda()
    print(m(x,x).shape)
    # m1 = nn.Conv1d(10, 10, 3, stride=4)
    # m2 = nn.Conv1d(10, 10, 3, dilation=2, stride=4)
    # m3 = nn.Conv1d(10, 10, 3, dilation=3,stride=4)
    # m4 = nn.Conv1d(10, 10, 1, stride=4)
    # # m4 = nn.AdaptiveAvgPool1d(10)
    # # in2 = torch.randn(1,128,768)
    # # in3 = torch.randn(1,128,512)
    # # print(m4(in2).shape)
    # # print(m4(in3).shape)
    # in1 = torch.randn(20, 10, 50)
    # o1 = m1(in1)
    # o2 = m2(in1)
    # o3 = m3(in1)
    # o4 = m4(in1)
    # print(o1.shape,o2.shape,o3.shape,o4.shape)
