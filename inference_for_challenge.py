import os
import torch
import json
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from slowfast.utils.parser import load_config
from slowfast.utils.misc import assert_and_infer_cfg
from slowfast.utils import distributed as du
from slowfast.utils import checkpoint as cu
from slowfast.models import build_model
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Load JSON data
with open('/home/datasets/holoassist/challenge/json_for_dataloader.json') as json_file:
    json_for_dataloader = json.load(json_file)

grouped_frames_holoassist, grouped_labels_holoassist = [], []
invalid_counter = 0

for e in json_for_dataloader:
    frame_list = json_for_dataloader[e]['paths']
    if len(frame_list) >= 8:
        gaze_x = json_for_dataloader[e]['gaze_x']
        gaze_y = json_for_dataloader[e]['gaze_y']
        sample_size = 8
        step = len(frame_list) / sample_size
        uniform_sampled_list_paths = ['/path_to_exported_frames/' + frame_list[int(i * step)] for i in range(sample_size)]
        uniform_sampled_list_gaze_x = [gaze_x[int(i * step)] for i in range(sample_size)]
        uniform_sampled_list_gaze_y = [gaze_y[int(i * step)] for i in range(sample_size)]

        tensor = np.zeros((sample_size, 4))
        tensor[:, 0] = uniform_sampled_list_gaze_x
        tensor[:, 1] = uniform_sampled_list_gaze_y
        tensor[:, 2] = 1
        tensor[:, 3] = 0
        tensor = torch.tensor(tensor)
        grouped_frames_holoassist.append(uniform_sampled_list_paths)
        grouped_labels_holoassist.append(tensor)
    else:
        invalid_counter += 1

# Namespace for arguments
class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Load configuration and initialize model
pretrain = 'egta'
if pretrain == 'egta':
    args = Namespace(
        cfg_file='/configs/Egtea/MVIT_B_16x4_CONV.yaml',
        init_method='tcp://localhost:9999',
        num_shards=1,
        opts=[
            'TRAIN.ENABLE', 'False',
            'TEST.BATCH_SIZE', '32',
            'NUM_GPUS', '1',
            'OUTPUT_DIR', 'checkpoints/GLC',
            'TEST.CHECKPOINT_FILE_PATH', '/0_HoloAssist/only_global_traj/checkpoints/checkpoint_epoch_00002.pyth',
        ],
        shard_id=0
    )

cfg = load_config(args)
cfg = assert_and_infer_cfg(cfg)
du.init_distributed_training(cfg)
torch.manual_seed(cfg.RNG_SEED)

model_both = build_model(cfg)
cu.load_test_checkpoint(cfg, model_both)

# Data grouping function
def group_data(frames, labels, errors, group_size):
    assert len(frames) == len(labels) == len(errors), "Input lists must have the same length"
    num_groups = len(frames) // group_size
    grouped_frames, grouped_labels, grouped_errors = [], [], []

    for i in range(num_groups):
        grouped_frames.append(frames[i * group_size: (i + 1) * group_size])
        grouped_labels.append(torch.stack(labels[i * group_size: (i + 1) * group_size], dim=0))
        grouped_errors.append(torch.stack(errors[i * group_size: (i + 1) * group_size], dim=0))
        
    return grouped_frames, grouped_labels, grouped_errors

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, frames, labels, errors, transform=None):
        self.frames = frames
        self.labels = labels
        self.errors = errors
        self.to_tensor = ToTensor()
        self.transform = transform

    def __len__(self):
        return len(self.frames)

    def _get_gaussian_map(self, heatmap, center, kernel_size, sigma):
        h, w = heatmap.shape
        mu_x, mu_y = round(center[0]), round(center[1])
        left = max(mu_x - (kernel_size - 1) // 2, 0)
        right = min(mu_x + (kernel_size - 1) // 2, w - 1)
        top = max(mu_y - (kernel_size - 1) // 2, 0)
        bottom = min(mu_y + (kernel_size - 1) // 2, h - 1)

        if left < right and top < bottom:
            kernel_1d = cv2.getGaussianKernel(ksize=kernel_size, sigma=sigma)
            kernel_2d = kernel_1d * kernel_1d.T
            heatmap[top:bottom + 1, left:right + 1] = kernel_2d[
                (kernel_size - 1) // 2 - mu_y + top: (kernel_size - 1) // 2 + bottom - mu_y + 1,
                (kernel_size - 1) // 2 - mu_x + left: (kernel_size - 1) // 2 + right - mu_x + 1
            ]

    def __getitem__(self, idx):
        frame_paths = self.frames[idx]
        label = self.labels[idx]
        errors = self.errors[idx]
        
        images = [Image.open(p).convert('RGB') for p in frame_paths]
        image_tensors = [self.to_tensor(img) for img in images]
        stacked_images = torch.stack(image_tensors, dim=0)

        if self.transform:
            stacked_images = torch.stack([self.transform(img) for img in stacked_images], dim=0)

        reshaped_tensor = stacked_images.permute(1, 0, 2, 3)
        label_hm = np.zeros((reshaped_tensor.size(1), reshaped_tensor.size(2) // 4, reshaped_tensor.size(3) // 4))

        for i in range(label_hm.shape[0]):
            if label[i, 2] == 0:
                label_hm[i, :, :] = 1 / (label_hm.shape[1] * label_hm.shape[2])
            else:
                self._get_gaussian_map(
                    label_hm[i, :, :],
                    center=((label[i, 0] * label_hm.shape[2]).item(), (label[i, 1] * label_hm.shape[1]).item()),
                    kernel_size=19,
                    sigma=-1
                )
            d_sum = label_hm[i, :, :].sum()
            if d_sum != 1:
                label_hm[i, :, :] /= d_sum

        label_hm = torch.as_tensor(label_hm).float()
        return reshaped_tensor, label, label_hm, errors, idx, {}




def create_holo_assist_loader(epoch, splitting, batch_size, error, fixation_only, sequence_len, error_len=0):
    np.random.seed(123456)
    json_file_path = '/home/datasets/Epic-Tent/2ite3tu1u53n42hjfh3886sa86/exported_frames_smi/Epic_Tent_annotations_exported.json'

    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    assert error in ["no", "only", "both"], f"Error mode '{error}' not supported"

    splits = {
        'train': [1, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 17, 19, 20, 22, 23, 26, 27, 28],
        'test': [3, 10, 15, 18, 21, 25, 29],
        'valid': [2, 16, 24]
    }
    selected_split = splits.get(splitting, [])

    all_path, all_gazes, all_errors = [], [], []
    for key, frame_data in data.items():
        subject_id = int(key.split('_')[0])
        if subject_id in selected_split and os.path.exists(f"/home/datasets/Epic-Tent/2ite3tu1u53n42hjfh3886sa86/exported_frames_smi/frames/{key}.jpg"):
            if 0 < frame_data['gaze_x'] < frame_data['frame_width'] and 0 < frame_data['gaze_y'] < frame_data['frame_height']:
                path = f"/home/datasets/Epic-Tent/2ite3tu1u53n42hjfh3886sa86/exported_frames_smi/frames/{key}.jpg"
                gaze_tensor = torch.tensor([frame_data['gaze_x'] / frame_data['frame_width'], frame_data['gaze_y'] / frame_data['frame_height'], 1, 0])

                add_sample = False
                if fixation_only and frame_data['gaze_type'] == 'Fixation':
                    add_sample = error == 'both' or (error == 'no' and frame_data['error'] == 99) or (error == 'only' and frame_data['error'] != 99)
                elif not fixation_only:
                    add_sample = error == 'both' or (error == 'no' and frame_data['error'] == 99) or (error == 'only' and frame_data['error'] != 99)

                if add_sample:
                    all_path.append(path)
                    all_gazes.append(gaze_tensor)
                    all_errors.append(torch.tensor(0 if frame_data['error'] == 99 else 1))

    frames, labels, errors = all_path[epoch:], all_gazes[epoch:], all_errors[epoch:]
    grouped_frames, grouped_labels, grouped_errors = group_data(frames, labels, errors, sequence_len)
    grouped_errors = [statistics.mode(tensor) for tensor in grouped_errors]

    print(f"Going to {splitting} on: {len(grouped_frames)}")

    transform = transforms.Compose([transforms.Resize(size=(256, 256), antialias=True)])
    custom_dataset = CustomDataset(grouped_frames, grouped_labels, grouped_errors, transform)
    
    return custom_dataset, grouped_frames, 123456


# Usage example
sequence_len = 8
batch_size = 1
inference_dataset, grouped_frames, seed = create_holo_assist_loader(0, "test", batch_size, error='both', fixation_only=True, sequence_len=sequence_len, error_len=10)
result_array = [[os.path.basename(path) for path in paths] for paths in grouped_frames]

# Specify file paths
result_file_path = "holoassist/dataloader/challenge/script/export/all_test_filename_global_both_both_fixationonly_true.json"
predictions_file_path = "inference_results.json"

def save_to_json(data, file_path):
    """Saves data to a JSON file."""
    with open(file_path, 'w') as jsonfile:
        json.dump(data, jsonfile, default=convert_to_builtin_type)
    print(f"The array has been saved to {file_path}")

def convert_to_builtin_type(obj):
    """Convert numpy types to built-in types for JSON serialization."""
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

# Save initial result_array to JSON
save_to_json(result_array, result_file_path)

# Initialize global result lists
global_results = []
global_gazes = []
global_errors = []
all_gazes_to_save = []

for i_counter, (frames, gazes, label_hm, errors, index, _) in enumerate(inference_dataset):
    frames_tensor = frames.unsqueeze(0).cuda()
    label_hm = label_hm.unsqueeze(0).cuda()
    model_both.eval()
    heatmaps_global_tensor = model_both([frames_tensor], label_hm)
    global_results.append(heatmaps_global_tensor.detach().cpu().numpy())
    global_gazes.append(gazes.numpy())
    global_errors.append(errors.numpy())
    frames_np = frames.numpy().transpose(1, 2, 3, 0)  # Transpose frames to shape [H, W, C, T]
    point_x_list = gazes[:, 0].numpy() * 256
    point_y_list = gazes[:, 1].numpy() * 256
    tmp_gt = [{"x": x, "y": y} for x, y in zip(point_x_list, point_y_list)]
    tmp_global = []
    for element in heatmaps_global_tensor[0][0]:
        element_np = element.detach().cpu().numpy()
        heatmap_error = (element_np - np.min(element_np)) / (np.max(element_np) - np.min(element_np) + 1e-8)
        heatmap_error = cv2.resize(heatmap_error, (256, 256))
        gaze_x_error, gaze_y_error = np.unravel_index(np.argmax(heatmap_error), heatmap_error.shape)
        tmp_global.append({"x": gaze_x_error, "y": gaze_y_error, "heatmap": element_np})
    batch_8 = [{"GT": tmp_gt, "global": tmp_global}]
    all_gazes_to_save.append(batch_8)

# Save all predicted gazes to JSON
save_to_json(all_gazes_to_save, predictions_file_path)

