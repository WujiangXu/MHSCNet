import matplotlib.pyplot as plt
import numpy as np
from IPython.core.pylabtools import figsize # import figsize
from sklearn.manifold import TSNE
import os
def revalue(data,label):
    idx = 0 
    value = list()
    for i in label:
        if i == True:
            value.append(data[idx])
        else:
            value.append(0)
        idx += 1
    return np.array(value)

# def tsne_draw(data,name):
#     tsne = TSNE(n_components=2, init='pca', random_state=0)
#     result = tsne.fit_transform(data)
#     frame = plt.gca()
#     frame.axes.get_yaxis().set_visible(False)
#     frame.axes.get_xaxis().set_visible(False)
#     plt.scatter(result[:, 0], result[:, 1], cmap=plt.cm.Spectral)
#     plt.savefig(name,dpi=300)
#     plt.close('all')

# load_path = "/home/jovyan/video_summary_dataset/video_summary_multi_modality/Ours_VS/emb_overlap/"
# save_path = "/home/jovyan/video_summary_dataset/video_summary_multi_modality/Ours_VS/tsne_imgs/"
# dir_load = os.listdir(load_path)
# new_dir_lists = list()
# for tmp in dir_load:
#     if tmp.find('overlap')==-1:
#         new_dir_lists.append(tmp)
# for tmp in new_dir_lists:
#     tmp = tmp[:-4]
#     print("process {}".format(tmp))
#     data = np.load("/home/jovyan/video_summary_dataset/video_summary_multi_modality/Ours_VS/emb_overlap/"+tmp+".npy")
#     data2 = np.load("/home/jovyan/video_summary_dataset/video_summary_multi_modality/Ours_VS/emb_overlap/"+tmp+'_overlap.npy')
#     data3 = np.load("/home/jovyan/video_summary_dataset/video_summary_multi_modality/Ours_VS/emb_overlap/"+tmp+'_nooverlap.npy')
#     tsne_draw(data,os.path.join(save_path,tmp+'.JPG'))
#     tsne_draw(data2,os.path.join(save_path,tmp+'_overlap.JPG'))
#     tsne_draw(data3,os.path.join(save_path,tmp+'_nooverlap.JPG'))

score = np.load("/home/jovyan/video_summary_dataset/video_summary_multi_modality/Ours_VS/more_vis_img_tvsum/AwmHb44_ouw/gtscore.npy")
pred = np.load("/home/jovyan/video_summary_dataset/video_summary_multi_modality/Ours_VS/more_vis_img_tvsum/AwmHb44_ouw/score_layer6_0.6740157480314961.npy")
print(pred)
score = score+0.5
rescore = revalue(score,pred)
plt.figure(figsize=(40,2))
plt.xticks([])
plt.yticks([])
plt.axis('off')
frame = plt.gca()
frame.axes.get_yaxis().set_visible(False)
frame.axes.get_xaxis().set_visible(False)
plt.bar(range(score.shape[0]), score, width=1,color='lightgrey')
plt.bar(range(rescore.shape[0]), rescore, width=1,color='firebrick')
#plt.hist(score,bins=223,color='peru',density=True,)
#plt.hist(y,bins=50,color='firebrick',density=True,label=None)

plt.savefig("/home/jovyan/video_summary_dataset/video_summary_multi_modality/Ours_VS/save_images/AwmHb44_ouw/layer6_vis_67.4.jpg",dpi=300)