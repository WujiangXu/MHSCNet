from helpers import init_helper, data_helper
from pathlib import Path
import numpy as np
import h5py


splits = data_helper.load_yaml("../splits/tvsum.yml")
for split_idx, split in enumerate(splits):        
    dataset_paths = {str(Path(key).parent) for key in split['train_keys']}
    print(dataset_paths)
    datasets = {path: h5py.File(path, 'r') for path in dataset_paths}

    key = split['train_keys'][0]
    video_path = Path(key)
    dataset_name = str(video_path.parent)
    video_name = video_path.name
    print(video_name,datasets,dataset_name)
    video_file = datasets[dataset_name][video_name]
    seq = video_file['features'][...].astype(np.float32)
    gtscore = video_file['gtscore'][...].astype(np.float32)
    cps = video_file['change_points'][...].astype(np.int32)
    n_frames = video_file['n_frames'][...].astype(np.int32)
    nfps = video_file['n_frame_per_seg'][...].astype(np.int32)
    picks = video_file['picks'][...].astype(np.int32)
    n_steps = video_file['n_steps'][...].astype(np.int32)
    gtsummary = video_file['gtsummary'][...].astype(np.int32)
    user_summary = video_file['user_summary'][...].astype(np.int32)
    print(seq.shape)
