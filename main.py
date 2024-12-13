import os
import re
import cv2
import tqdm
import yaml
import torch
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from utils.fusion import apply_fusion
from utils.processing import make_normalize
from networks import create_architecture, load_weights
from torchvision.transforms import CenterCrop, Resize, Compose, InterpolationMode

parent_path = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(parent_path, "results")

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

import pandas as pd

def save_results(string_videos, video_name, csv_path, models, fusion_methods, threshold=0, just_soft_or_prob=False):
    try:
        data = pd.read_csv(csv_path)
        file_path = os.path.join(RESULTS_PATH, f'results_{string_videos}.csv')

        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            with open(file_path, 'w') as f:
                if just_soft_or_prob:
                    f.write("filename,prediction\n")
                    
                else:
                    f.write("filename, prediction_clipdet_latent10k, prediction_clipdet_latent10k_plus, prediction_Corvi2023, prediction_fusion[mean_logit], prediction_fusion[max_logit], prediction_fusion[median_logit], prediction_fusion[lse_logit], prediction_fusion[mean_prob], prediction_fusion[soft_or_prob]\n")
                    df = pd.DataFrame(columns=['filename', 'prediction_clipdet_latent10k', 'prediction_clipdet_latent10k_plus', 'prediction_Corvi2023', 'prediction_fusion[mean_logit]', 'prediction_fusion[max_logit]', 'prediction_fusion[median_logit]', 'prediction_fusion[lse_logit]', 'prediction_fusion[mean_prob]', 'prediction_fusion[soft_or_prob'])           # save the results in a csv file
       
        required_columns = models + [f'fusion[{method}]' for method in fusion_methods]
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns in the CSV: {missing_columns}")
        
        # results = []
        if just_soft_or_prob:
            synthetic_count = (data['fusion[soft_or_prob]'] > threshold).sum()
            total_frames = len(data)
            
            with open(file_path, 'a') as f:
                f.write(f"{video_name},{synthetic_count/total_frames}\n")
        else:
            for model in models + [f'fusion[{method}]' for method in fusion_methods]:
                synthetic_count = (data[model] > threshold).sum()
                total_frames = len(data)
                # results.append((model, synthetic_count/total_frames))
                df = df.append({'filename': video_name, f'prediction_{model}': synthetic_count/total_frames}, ignore_index=True)

            with open(file_path, 'a') as f:
                df.to_csv(f, header=False, index=False)
                f.write("\n")
            
    except Exception as e:
        raise RuntimeError(f"Error analyzing video: {e}")

if __name__ == "__main__":
    parent_path = os.path.dirname(os.path.abspath(__file__))
    
    luma_folder = os.path.join(parent_path, f"tests/luma_dream_machine")
    real_folder = os.path.join(parent_path, f"tests/real")
    cogvideo_folder = os.path.join(parent_path, f"tests/CogVideoX-5b")

    video_paths_luma = [os.path.join(luma_folder, f) for f in os.listdir(luma_folder) if f.endswith('.mp4')]
    video_paths_real = [os.path.join(real_folder, f) for f in os.listdir(real_folder) if f.endswith('.mp4')]
    video_paths_cogvideo = [os.path.join(cogvideo_folder, f) for f in os.listdir(cogvideo_folder) if f.endswith('.mp4')]
    
    video = input("Which video do you want to test? \n 1. Luma \n 2. Real \n 3. CogVideoX-5B \n")
    while video not in ['1', '2', '3']:
        video = input("Invalid input. Please enter 1, 2, or 3: ")
    
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print("\n\n\nRunning tests on device: ", device, "\n\n\n")
    
    if video == '1':
        video_paths = video_paths_luma
        string_videos = "luma"
    elif video == '2':
        video_paths = video_paths_real
        string_videos = "real"
    else:
        video_paths = video_paths_cogvideo
        string_videos = "cogvideo"
    
    for video_path in video_paths:
    
        print("\n\n\nRunning tests on video: ", video_path, "\n\n\n")
        
        weights_dir = os.path.join(parent_path, "weights")
        temp_dir = os.path.join(parent_path, "temp_frames")
        models = ['clipdet_latent10k', 'clipdet_latent10k_plus', 'Corvi2023']

        fusion = 'soft_or_prob'
        fusion_methods = ['mean_logit', 'max_logit', 'median_logit', 'lse_logit', 'mean_prob', 'soft_or_prob']
        
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        csv_path = os.path.join(temp_dir, "input_images.csv")
        output_csv = os.path.join(RESULTS_PATH, f"frames_results_{string_videos}_{video_name}.csv")
        
        print("Extracting frames from video...")
        frame_paths = extract_frames_from_video(video_path, temp_dir)
        
        generate_csv_from_frames(frame_paths, csv_path)
        print(f"Frames extracted and CSV generated: {csv_path}")

        table = running_tests(csv_path, weights_dir, models, device)

        for fusion_method in fusion_methods:
            table[f'fusion[{fusion_method}]'] = apply_fusion(table[models].values, fusion_method, axis=-1)

        # filename,clipdet_latent10k,clipdet_latent10k_plus,Corvi2023,fusion[max_logit],fusion[mean_logit],fusion[median_logit],fusion[lse_logit],fusion[mean_prob],fusion[soft_or_prob]

        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        table.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")
        
        save_results(string_videos, video_name, output_csv, models, fusion_methods, just_soft_or_prob=True)            
        
        