from image_description.segment import segmentation
from image_description.places365.run_placesCNN_unified import places365
from image_description.obj_det import det

import glob
import os
import csv
from PIL import Image

def create_sandbox():
# Initialize an empty list to store the image details
    image_details = []

    for img in glob.glob("C:/Users/isu_v/Desktop/langchain/image_description/sandbox_img_selection/selected/*.png"):

        print(img)

        name = img.split("/")[-1]

        img = Image.open(img)
        seg = segmentation(img)
        scene_cat, scene_attr = places365(img)
        dets = det(img)

        image_detail = {
            'ImageName': name,
            'Segmentation': seg,
            'SceneCategory': scene_cat,
            'SceneAttributes': scene_attr,
            'Detections': dets
        }

        # Append the details to the list
        image_details.append(image_detail)

    # Define the CSV file path
    csv_file = 'C:/Users/isu_v/Desktop/langchain/image_description/sandbox_img_selection/selected_imgs_details.csv'

    # Check if the CSV file already exists or create a new one
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='') as file:
        fieldnames = ['ImageName', 'Segmentation', 'SceneCategory', 'SceneAttributes', 'Detections']
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write headers if the file doesn't exist
        if not file_exists:
            writer.writeheader()

        # Write the details for each image
        writer.writerows(image_details)

    # return("csv_created")