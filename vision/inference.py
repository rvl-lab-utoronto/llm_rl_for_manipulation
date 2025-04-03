import sys
import time

PATH_TO_GROUNDED_SAM_2 = "./Grounded_SAM_2"

sys.path.append("./Grounded_SAM_2")
import argparse
import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from supervision.draw.color import ColorPalette
from utils.supervision_utils import CUSTOM_COLOR_MAP
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 

"""
Hyper parameters
"""
parser = argparse.ArgumentParser()
parser.add_argument('--grounding-model', default="IDEA-Research/grounding-dino-tiny")
parser.add_argument("--text-prompt", default="car. tire.")
parser.add_argument("--img-path", default=f"./test1.jpg")
parser.add_argument("--sam2-checkpoint", default=f"{PATH_TO_GROUNDED_SAM_2}/checkpoints/sam2.1_hiera_large.pt")
parser.add_argument("--sam2-model-config", default=f"configs/sam2.1/sam2.1_hiera_l.yaml")
parser.add_argument("--output-dir", default=f"outputs/test_sam2.1")
parser.add_argument("--no-dump-json", action="store_true")
parser.add_argument("--force-cpu", action="store_true")
args = parser.parse_args()

GROUNDING_MODEL = "IDEA-Research/grounding-dino-tiny"
TEXT_PROMPT = "red cube. yellow cube. green cube. blue cube." #args.text_prompt
IMG_PATH = "./test1.jpg"
SAM2_CHECKPOINT = f"{PATH_TO_GROUNDED_SAM_2}/checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("outputs/test_sam2.1")
DUMP_JSON_RESULTS = True

# create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class SegmentationModel():
    def __init__(self):
        torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # build SAM2 image predictor
        sam2_checkpoint = SAM2_CHECKPOINT
        model_cfg = SAM2_MODEL_CONFIG
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)
        
        # build grounding dino from huggingface
        model_id = GROUNDING_MODEL
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(DEVICE)

    def predict(self, image, text_prompt):
        self.sam2_predictor.set_image(np.array(image.convert("RGB")))

        inputs = self.processor(images=image, text=text, return_tensors="pt").to(DEVICE)
        print("here")
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]
        )
        
        # get the box prompt for SAM 2
        input_boxes = results[0]["boxes"].cpu().numpy()

        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        #self._post_process_results(results, input_boxes, masks, scores, logits)
    
    def _post_process_results(self, results, input_boxes, masks, scores, logits):
        """
        Post-process the output of the model to get the masks, scores, and logits for visualization
        """
        # convert the shape to (n, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)


        confidences = results[0]["scores"].cpu().numpy().tolist()
        class_names = results[0]["labels"]
        class_ids = np.array(list(range(len(class_names))))

        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence
            in zip(class_names, confidences)
        ]

        """
        Visualize image with supervision useful API
        """
        img = cv2.imread(img_path)
        detections = sv.Detections(
            xyxy=input_boxes,  # (n, 4)
            mask=masks.astype(bool),  # (n, h, w)
            class_id=class_ids
        )

        """
        Note that if you want to use default color map,
        you can set color=ColorPalette.DEFAULT
        """
        box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

        label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "groundingdino_annotated_image.jpg"), annotated_frame)

        mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "grounded_sam2_annotated_image_with_mask.jpg"), annotated_frame)



# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot
text = TEXT_PROMPT
img_path = IMG_PATH

image = Image.open(img_path)

seg_model = SegmentationModel()
t1 = time.time()
seg_model.predict(image, text)
t2 = time.time()
print(t2 - t1, "seconds to run the model")