

## Captioning_model (https://huggingface.co/Salesforce/blip-image-captioning-base)

import requests
from transformers import BlipProcessor, BlipForConditionalGeneration
# from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")


def caption(img):
   
    key = "YOUR_GOOGLE_API_KEY"

    i_image = img ### for reading images from drive
    
    # i_image = Image.open(requests.get((img + key),stream = True).raw).convert('RGB') ####uncomment it for url based analysis

    # conditional image captioning
# text = "a photography of"
# inputs = processor(raw_image, text, return_tensors="pt").to("cuda")

# out = model.generate(**inputs)
# print(processor.decode(out[0], skip_special_tokens=True))
# >>> a photography of a woman and her dog

# unconditional image captioning
    inputs = processor(i_image, return_tensors="pt").to("cuda")

    out = model.generate(**inputs)
       
    
    return processor.decode(out[0], skip_special_tokens=True)




