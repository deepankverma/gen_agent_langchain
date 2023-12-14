#### segmentation

from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from collections import Counter
from torch import nn
from huggingface_hub import hf_hub_download
import json
from PIL import Image
import requests

# extractor = SegformerFeatureExtractor.from_pretrained("nickmuchi/segformer-b4-finetuned-segments-sidewalk")
# model = SegformerForSemanticSegmentation.from_pretrained("nickmuchi/segformer-b4-finetuned-segments-sidewalk")

extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")

key = "AIzaSyC28vZubuuLq0i2spQ0JQL4Eo4OIrw6Fnw"

# img = "https://maps.googleapis.com/maps/api/streetview?size=600x400&location=6.86803,79.880366&heading=329.16&pitch=-0.76&key="



def segmentation(img):
    # download the image and convert to PIL object

    image = Image.open(requests.get(img + key, stream=True).raw)  ####uncomment it for url based analysis

    # image = Image.open(img) ####uncomment it for image based analysis
    

    inputs = extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)

    # First, rescale logits to original image size
    upsampled_logits = nn.functional.interpolate(
                        logits,
                        size=image.size[::-1], # (height, width)
                        mode='bilinear',
                        align_corners=False
                        )

    # Second, apply argmax on the class dimension
    pred_seg = upsampled_logits.argmax(dim=1)[0]

    predictions = pred_seg.cpu().detach().numpy()
    predictions1 = predictions.flatten()

    ##managing data specific  labels

    # filename = "id2label.json"
    # id2label = json.load(
    #             open(hf_hub_download("segments/sidewalk-semantic", filename, repo_type="dataset"), "r")
    #             )
    # id2label = json.load(open("image_description/id2label.json", "r"))

    id2label = json.load(open("image_description/id2label_nvidia.json", "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    num_labels = len(id2label)

    # num_labels, list(label2id.keys())

    ##counting all the predictions and converting to the key value pair and in percentages.


    pixel_cnt = Counter(list(predictions1))

    def convert_to_percentages(numbers):
        total_sum = sum(numbers)
        percentages = [(number / total_sum) * 100 for number in numbers]
        formatted_percentages = [f"{percentage:.2f}" for percentage in percentages]
        return formatted_percentages

    percentages = convert_to_percentages(pixel_cnt.values())
    # print(percentages)

    y = [list(label2id.keys())[x] for x in pixel_cnt.keys()]  
    result = {key: value for key, value in zip(y, percentages)}
    # print(result)


    thresholds = {
    "vegetation": {"high": 25, "average": 12.5},
    "sky": {"high": 25, "average": 12.5},
    "building": {"high": 25, "average": 12.5},
    "road": {"high": 25, "average": 12.5},
    "wall": {"high": 5, "average": 2.5},
    "fence": {"high": 5, "average": 2.5},
    "pole": {"high": 5, "average": 2.5}
    }

    # Define category labels
    category_labels = {
        "high": "High",
        "average": "Average",
        "low": "Low",
        "present": "Present"
    }

    # Categorize confidence scores and map to qualitative labels
    confidence_categories = {}
    for class_label, score in result.items():
        # print(class_label,score)
        if class_label in thresholds:
            high_threshold = thresholds[class_label]["high"]
            average_threshold = thresholds[class_label]["average"]
            
            if float(score) > float(high_threshold):
                confidence_categories[class_label] = category_labels["high"]
            elif float(score) >= float(average_threshold):
                confidence_categories[class_label] = category_labels["average"]
            else:
                confidence_categories[class_label] = category_labels["low"]
        else:
            confidence_categories[class_label] = category_labels["present"]

    # print("Confidence Categories:", confidence_categories)

    return(confidence_categories)