'''
Copyright 2024 Image Processing Research Group of University Federico
II of Naples ('GRIP-UNINA'). All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import torch
import os
import pandas as pd
import numpy as np
import tqdm
import yaml
from PIL import Image
from torchvision.transforms import CenterCrop, Resize, Compose, InterpolationMode
from utils.processing import make_normalize
from utils.fusion import apply_fusion
from networks import create_architecture, load_weights
import cv2

def get_config(model_name, weights_dir='./weights'):
    with open(os.path.join(weights_dir, model_name, 'config.yaml')) as fid:
        data = yaml.load(fid, Loader=yaml.FullLoader)
    model_path = os.path.join(weights_dir, model_name, data['weights_file'])
    return data['model_name'], model_path, data['arch'], data['norm_type'], data['patch_size']

def extract_frames_from_video(video_path, output_dir):
    """
    Extract frames from an MP4 video and save them as images in the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frame_paths = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_paths.append(frame_filename)
        frame_count += 1
    
    cap.release()
    return frame_paths

def generate_csv_from_frames(frame_paths, csv_path):
    """
    Create a CSV file with the list of frame image paths.
    """
    data = {'filename': frame_paths}
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)

def running_tests(input_csv, weights_dir, models_list, device, batch_size=1):
    table = pd.read_csv(input_csv)[['filename',]]
    rootdataset = os.path.dirname(os.path.abspath(input_csv))

    models_dict = dict()
    transform_dict = dict()
    print("Models:")
    for model_name in models_list:
        print(model_name, flush=True)
        _, model_path, arch, norm_type, patch_size = get_config(model_name, weights_dir=weights_dir)

        model = load_weights(create_architecture(arch), model_path)
        model = model.to(device).eval()

        transform = list()
        if patch_size is None:
            print('input none', flush=True)
            transform_key = 'none_%s' % norm_type
        elif patch_size == 'Clip224':
            print('input resize:', 'Clip224', flush=True)
            transform.append(Resize(224, interpolation=InterpolationMode.BICUBIC))
            transform.append(CenterCrop((224, 224)))
            transform_key = 'Clip224_%s' % norm_type
        elif isinstance(patch_size, tuple) or isinstance(patch_size, list):
            print('input resize:', patch_size, flush=True)
            transform.append(Resize(*patch_size))
            transform.append(CenterCrop(patch_size[0]))
            transform_key = 'res%d_%s' % (patch_size[0], norm_type)
        elif patch_size > 0:
            print('input crop:', patch_size, flush=True)
            transform.append(CenterCrop(patch_size))
            transform_key = 'crop%d_%s' % (patch_size, norm_type)
        
        transform.append(make_normalize(norm_type))
        transform = Compose(transform)
        transform_dict[transform_key] = transform
        models_dict[model_name] = (transform_key, model)
        print(flush=True)

    ### test
    with torch.no_grad():
        do_models = list(models_dict.keys())
        do_transforms = set([models_dict[_][0] for _ in do_models])
        print(do_models)
        print(do_transforms)
        print(flush=True)
        
        print("Running the Tests")
        batch_img = {k: list() for k in transform_dict}
        batch_id = list()
        last_index = table.index[-1]
        for index in tqdm.tqdm(table.index, total=len(table)):
            filename = os.path.join(rootdataset, table.loc[index, 'filename'])
            for k in transform_dict:
                batch_img[k].append(transform_dict[k](Image.open(filename).convert('RGB')))
            batch_id.append(index)

            if (len(batch_id) >= batch_size) or (index == last_index):
                for k in do_transforms:
                    batch_img[k] = torch.stack(batch_img[k], 0)

                for model_name in do_models:
                    out_tens = models_dict[model_name][1](batch_img[models_dict[model_name][0]].clone().to(device)).cpu().numpy()

                    if out_tens.shape[1] == 1:
                        out_tens = out_tens[:, 0]
                    elif out_tens.shape[1] == 2:
                        out_tens = out_tens[:, 1] - out_tens[:, 0]
                    else:
                        assert False
                    
                    if len(out_tens.shape) > 1:
                        logit1 = np.mean(out_tens, (1, 2))
                    else:
                        logit1 = out_tens

                    for ii, logit in zip(batch_id, logit1):
                        table.loc[ii, model_name] = logit

                batch_img = {k: list() for k in transform_dict}
                batch_id = list()

            assert len(batch_id) == 0
        
    return table

def analyze_video(csv_path, models, fusion_column='fusion', threshold=0):
    """
    Analyze the output CSV to determine if a video is synthetic or not.
    
    Args:
        csv_path (str): Path to the output CSV file.
        models (list): List of model column names to evaluate (e.g., ['clipdet_latent10k_plus', 'Corvi2023']).
        fusion_column (str): Column name for the fusion score.
        threshold (float): Threshold above which a frame is classified as synthetic.
    
    Returns:
        str: 'Synthetic' if the video is classified as synthetic, 'Real' otherwise.
    """
    try:
        data = pd.read_csv(csv_path)

        required_columns = models + [fusion_column]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in the CSV: {missing_columns}")

        is_synthetic_by_models = any((data[model] > threshold).any() for model in models)

        is_synthetic_by_fusion = (data[fusion_column] > threshold).any()

        return "Synthetic" if is_synthetic_by_fusion or is_synthetic_by_models else "Real"
    
    except Exception as e:
        raise RuntimeError(f"Error analyzing video: {e}")

if __name__ == "__main__":
    parent_path = os.path.dirname(os.path.abspath(__file__))
        
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    video_paths_fake = [
        os.path.join(parent_path, "Luma/Purse_Fake.mp4"),
        os.path.join(parent_path, "Luma/Horse_Fake.mp4"),
        os.path.join(parent_path, "Luma/Cow_Fake.mp4"),
        os.path.join(parent_path, "Luma/Skier_Fake.mp4"),
        os.path.join(parent_path, "Luma/TV_Fake.mp4"),
        os.path.join(parent_path, "Luma/Sofa_Fake.mp4"),
        os.path.join(parent_path, "Luma/Soldier_Fake.mp4")
    ]
    video_paths_true = [
        os.path.join(parent_path, "Real_ones/Cow_Real.mp4"),
        os.path.join(parent_path, "Real_ones/Skier_Real.mp4"),
        os.path.join(parent_path, "Real_ones/Soldier_Real.mp4")
    ]
    
    video_path = video_paths_true[0]
    
    print("\n\n\nRunning tests on video: ", video_path, "\n\n\n")
    
    weights_dir = os.path.join(parent_path, "weights")
    temp_dir = os.path.join(parent_path, "temp_frames")
    models = ['clipdet_latent10k_plus', 'Corvi2023']
    fusion = 'soft_or_prob'

    csv_path = os.path.join(temp_dir, "input_images.csv")
    output_csv = os.path.join(parent_path, "results.csv")
    
    print("Extracting frames from video...")
    frame_paths = extract_frames_from_video(video_path, temp_dir)
    
    generate_csv_from_frames(frame_paths, csv_path)
    print(f"Frames extracted and CSV generated: {csv_path}")

    table = running_tests(csv_path, weights_dir, models, device)
    if fusion is not None:
        table['fusion'] = apply_fusion(table[models].values, fusion, axis=-1)
    
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    table.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

    # all frames are processed, now we can classify the video based on the results
    # question: how to classify the video? by the average of the frames? by the majority of the frames
    # now we classify the video by the majority of the frames
    
    # logic: LLR > 0 --> synthetic
    # consider the fusion score as well (?) --> if the fusion score is high, then the video is synthetic
    
    video_classification = analyze_video(output_csv, models)
    print(f"The video is classified as: {video_classification}")
