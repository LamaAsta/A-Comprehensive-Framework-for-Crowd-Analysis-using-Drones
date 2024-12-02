from transformers import ViTImageProcessor
from PIL import Image
import torch
import cv2


def preprocess(video_path):     
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, frame_count // 16)

    for i in range(0, frame_count, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            video = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = processor(images=video, return_tensors="pt")
            preprocessed = inputs['pixel_values'].squeeze(0)
            frames.append(preprocessed)
        
    cap.release()

    if len(frames) < 16:
        for _ in range(16 - len(frames)):
            frames.append(torch.zeros_like(frames[0]))

    frames = torch.stack(frames)
    return frames

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')