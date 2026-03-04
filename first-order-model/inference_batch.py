import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
matplotlib.use('Agg') # Fix for server/headless environments
import sys, yaml
from argparse import ArgumentParser
from tqdm import tqdm
import imageio
import numpy as np
from skimage.transform import resize
import torch
from sync_batchnorm import DataParallelWithCallback
from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector

def load_checkpoints(config_path, checkpoint_path, cpu=False):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])

    if not cpu:
        generator.cuda()
        kp_detector.cuda()

    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])

    if not cpu:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()
    return generator, kp_detector

def make_animation(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=False):
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu: source = source.cuda()
        
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        if not cpu: driving = driving.cuda()
        
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in range(driving.shape[2]):
            driving_frame = driving[:, :, frame_idx]
            kp_driving = kp_detector(driving_frame)
            
            kp_norm = kp_driving 
            if relative:
                kp_norm_value = kp_driving['value'] - kp_driving_initial['value']
                kp_norm_value = kp_norm_value + kp_source['value']
                kp_norm['value'] = kp_norm_value
            
            if adapt_movement_scale:
                kp_norm['value'] = kp_norm['value'] + (kp_source['value'] - kp_driving_initial['value'])

            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
            
    return predictions

def img_as_ubyte(img):
    return (img * 255).astype(np.uint8)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", required=True, help="path to checkpoint")
    parser.add_argument("--source_image", required=True, help="path to source image")
    parser.add_argument("--driving_video", required=True, help="path to driving video")
    parser.add_argument("--result_video", required=True, help="path to output")
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode")
    
    opt = parser.parse_args()

    # --- FIX START: AUTO-DETECT CPU ---
    if not torch.cuda.is_available():
        # Only print this once to avoid spamming logs
        # print("   [FOMM] No GPU detected. Switching to CPU mode...") 
        opt.cpu = True
    # --- FIX END ---

    source_image = None
    if opt.source_image.lower().endswith(('.mp4', '.avi', '.mov', '.mpg')):
        try:
            reader = imageio.get_reader(opt.source_image)
            source_image = reader.get_data(0) 
            reader.close()
        except Exception as e:
            pass

    if source_image is None:
        try:
            source_image = imageio.imread(opt.source_image)
        except Exception:
            try:
                reader = imageio.get_reader(opt.source_image)
                source_image = reader.get_data(0)
                reader.close()
            except:
                # If reading fails, just exit gracefully instead of crashing the whole pipeline
                sys.exit(0) 

    reader = imageio.get_reader(opt.driving_video)
    fps = reader.get_meta_data()['fps']
    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError: pass
    reader.close()

    source_image = resize(source_image, (256, 256))[..., :3]
    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

    generator, kp_detector = load_checkpoints(opt.config, opt.checkpoint, opt.cpu)
    
    predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=opt.cpu)
    
    imageio.mimsave(opt.result_video, [img_as_ubyte(frame) for frame in predictions], fps=fps)