# This code is adapted from the authors of Grounded SAM 2 and their file: grounded_sam2_hf_model_demo
import sys

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
parser.add_argument("--img-path", default=f"./real.jpg")
parser.add_argument("--sam2-checkpoint", default=f"{PATH_TO_GROUNDED_SAM_2}/checkpoints/sam2.1_hiera_large.pt")
parser.add_argument("--sam2-model-config", default=f"configs/sam2.1/sam2.1_hiera_l.yaml")
parser.add_argument("--output-dir", default=f"outputs/test_sam2.1")
parser.add_argument("--no-dump-json", action="store_true")
parser.add_argument("--force-cpu", action="store_true")
args = parser.parse_args()

GROUNDING_MODEL = args.grounding_model
TEXT_PROMPT = "red plate. blue plate. pink cup. blue cup." #args.text_prompt
IMG_PATH = args.img_path
SAM2_CHECKPOINT = args.sam2_checkpoint
SAM2_MODEL_CONFIG = args.sam2_model_config
DEVICE = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
OUTPUT_DIR = Path(args.output_dir)
DUMP_JSON_RESULTS = not args.no_dump_json

# create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# environment settings
# use bfloat16
torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# build SAM2 image predictor
sam2_checkpoint = SAM2_CHECKPOINT
model_cfg = SAM2_MODEL_CONFIG
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)
print(DEVICE)
# build grounding dino from huggingface
model_id = GROUNDING_MODEL
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(DEVICE)


# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot
text = TEXT_PROMPT
img_path = IMG_PATH

image = Image.open(img_path)

sam2_predictor.set_image(np.array(image.convert("RGB")))

inputs = processor(images=image, text=text, return_tensors="pt").to(DEVICE)
print("here")
with torch.no_grad():
    outputs = grounding_model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=0.4,
    text_threshold=0.3,
    target_sizes=[image.size[::-1]]
)

print(results)

"""
Results is a list of dict with the following structure:
[
    {
        'scores': tensor([0.7969, 0.6469, 0.6002, 0.4220], device='cuda:0'), 
        'labels': ['car', 'tire', 'tire', 'tire'], 
        'boxes': tensor([[  89.3244,  278.6940, 1710.3505,  851.5143],
                        [1392.4701,  554.4064, 1628.6133,  777.5872],
                        [ 436.1182,  621.8940,  676.5255,  851.6897],
                        [1236.0990,  688.3547, 1400.2427,  753.1256]], device='cuda:0')
    }
]
"""

# get the box prompt for SAM 2
input_boxes = results[0]["boxes"].cpu().numpy()
#input_boxes = input_boxes[[0,1,2,4]]
masks, scores, logits = sam2_predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_boxes,
    multimask_output=False,
)
print("here2")

"""
Post-process the output of the model to get the masks, scores, and logits for visualization
"""
# convert the shape to (n, H, W)
if masks.ndim == 4:
    masks = masks.squeeze(1)


confidences = results[0]["scores"].cpu().numpy() #[[0,1,2,4]].tolist()
class_names = np.array(results[0]["labels"]) #[[0,1,2,4]]
class_ids = np.array(list(range(len(class_names))))

labels = [
    f"{class_name} {confidence:.2f}"
    for class_name, confidence
    in zip(class_names, confidences)
    #if len(class_name.split(' ')) < 3
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
print(detections)

"""
Note that if you want to use default color map,
you can set color=ColorPalette.DEFAULT
"""
print(CUSTOM_COLOR_MAP)
#pallete = ['444']
pallete = ['#2a41a3', '#a83232', '#c75a5a', '#5f91c9']
#pallete = ['#e6194b', '#ffe119', '#0082c8', '#3cb44b', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#d2f53c', '#fabebe', '#008080', '#e6beff', '#aa6e28', '#fffac8', '#800000', '#aaffc3']
box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(pallete))
annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(pallete))
annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
cv2.imwrite(os.path.join(OUTPUT_DIR, "groundingdino_annotated_image.jpg"), annotated_frame)

mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(pallete))
annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
cv2.imwrite(os.path.join(OUTPUT_DIR, "grounded_sam2_annotated_image_with_mask.jpg"), annotated_frame)


"""
Dump the results in standard format and save as json files
"""

def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

if DUMP_JSON_RESULTS:
    # convert mask into rle format
    mask_rles = [single_mask_to_rle(mask) for mask in masks]

    input_boxes = input_boxes.tolist()
    scores = scores.tolist()
    # save the results in standard format
    results = {
        "image_path": img_path,
        "annotations" : [
            {
                "class_name": class_name,
                "bbox": box,
                "segmentation": mask_rle,
                "score": score,
            }
            for class_name, box, mask_rle, score in zip(class_names, input_boxes, mask_rles, scores)
        ],
        "box_format": "xyxy",
        "img_width": image.width,
        "img_height": image.height,
    }
    
    with open(os.path.join(OUTPUT_DIR, "grounded_sam2_hf_model_demo_results.json"), "w") as f:
        json.dump(results, f, indent=4)
