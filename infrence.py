import torch
from super_gradients.training import models

best_model = models.get('yolo_nas_s',
                        num_classes=1,
                        checkpoint_path="checkpoints/parcel/ckpt_best.pth")

device = 0 if torch.cuda.is_available() else "cpu"

input_video_path = "../parcel-test.mp4"
output_video_path = "../parcel-detect-new2.mp4"
#device=0

best_model.to(device).predict(input_video_path,conf=0.6).save(output_video_path)
