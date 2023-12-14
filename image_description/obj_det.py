## object detection (https://huggingface.co/facebook/detr-resnet-101)

from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests
from collections import Counter

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101")

def det(img):
    # download the image and convert to PIL object

    key = "AIzaSyC28vZubuuLq0i2spQ0JQL4Eo4OIrw6Fnw"
    
    image = Image.open(requests.get((img + key),stream = True).raw).convert('RGB') ####uncomment it for url based analysis

    # image = Image.open(img)  ####uncomment it for image based analysis
    
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    # print(results)

    items = []

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        items.append(model.config.id2label[label.item()])

    # Count the occurrences of each element in the list
    item_counts = Counter(items)

    return item_counts

