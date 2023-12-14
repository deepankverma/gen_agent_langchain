# PlacesCNN to predict the scene category, attribute, and class activation map in a single pass
# by Bolei Zhou, sep 2, 2017
# updated, making it compatible to pytorch 1.x in a hacky way

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
import numpy as np
import cv2
from PIL import Image
import requests
from image_description.places365 import wideresnet



def places365(img):

    key = "AIzaSyC28vZubuuLq0i2spQ0JQL4Eo4OIrw6Fnw"
    
    i_image = Image.open(requests.get((img + key),stream = True).raw).convert('RGB')  ####uncomment it for url based analysis

    # i_image = Image.open(img) ####uncomment it for image based analysis
    

    # hacky way to deal with the Pytorch 1.0 update
    def recursion_change_bn(module):
        if isinstance(module, torch.nn.BatchNorm2d):
            module.track_running_stats = 1
        else:
            for i, (name, module1) in enumerate(module._modules.items()):
                module1 = recursion_change_bn(module1)
        return module

    def load_labels():
        # prepare all the labels
        # scene category relevant
        file_name_category = 'image_description/places365/categories_places365.txt'
        if not os.access(file_name_category, os.W_OK):
            synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
            os.system('wget ' + synset_url)
        classes = list()
        with open(file_name_category) as class_file:
            for line in class_file:
                classes.append(line.strip().split(' ')[0][3:])
        classes = tuple(classes)

        # indoor and outdoor relevant
        file_name_IO = 'image_description/places365/IO_places365.txt'
        if not os.access(file_name_IO, os.W_OK):
            synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt'
            os.system('wget ' + synset_url)
        with open(file_name_IO) as f:
            lines = f.readlines()
            labels_IO = []
            for line in lines:
                items = line.rstrip().split()
                labels_IO.append(int(items[-1]) -1) # 0 is indoor, 1 is outdoor
        labels_IO = np.array(labels_IO)

        # scene attribute relevant
        file_name_attribute = 'image_description/places365/labels_sunattribute.txt'
        if not os.access(file_name_attribute, os.W_OK):
            synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/labels_sunattribute.txt'
            os.system('wget ' + synset_url)
        with open(file_name_attribute) as f:
            lines = f.readlines()
            labels_attribute = [item.rstrip() for item in lines]
        file_name_W = 'image_description/places365/W_sceneattribute_wideresnet18.npy'
        if not os.access(file_name_W, os.W_OK):
            synset_url = 'http://places2.csail.mit.edu/models_places365/W_sceneattribute_wideresnet18.npy'
            os.system('wget ' + synset_url)
        W_attribute = np.load(file_name_W)

        return classes, labels_IO, labels_attribute, W_attribute

    def hook_feature(module, input, output):
        features_blobs.append(np.squeeze(output.data.cpu().numpy()))

    def returnCAM(feature_conv, weight_softmax, class_idx):
        # generate the class activation maps upsample to 256x256
        size_upsample = (256, 256)
        nc, h, w = feature_conv.shape
        output_cam = []
        for idx in class_idx:
            cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
            cam = cam.reshape(h, w)
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255 * cam_img)
            output_cam.append(cv2.resize(cam_img, size_upsample))
        return output_cam

    def returnTF():
    # load the image transformer
        tf = trn.Compose([
            trn.Resize((224,224)),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return tf


    def load_model():
        # this model has a last conv feature map as 14x14

        model_file = 'image_description/places365/wideresnet18_places365.pth.tar'
        if not os.access(model_file, os.W_OK):
            os.system('wget http://places2.csail.mit.edu/models_places365/' + model_file)
            os.system('wget https://raw.githubusercontent.com/csailvision/places365/master/wideresnet.py')

        
        model = wideresnet.resnet18(num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
        
        # hacky way to deal with the upgraded batchnorm2D and avgpool layers...
        for i, (name, module) in enumerate(model._modules.items()):
            module = recursion_change_bn(model)
        model.avgpool = torch.nn.AvgPool2d(kernel_size=14, stride=1, padding=0)
        
        model.eval()

        # model.eval()
        # hook the feature extractor
        features_names = ['layer4','avgpool'] # this is the last conv layer of the resnet
        for name in features_names:
            model._modules.get(name).register_forward_hook(hook_feature)
        return model


    # load the labels
    classes, labels_IO, labels_attribute, W_attribute = load_labels()

    # load the model
    features_blobs = []
    model = load_model()

    # load the transformer
    tf = returnTF() # image transformer

    # get the softmax weight
    params = list(model.parameters())
    weight_softmax = params[-2].data.numpy()
    weight_softmax[weight_softmax<0] = 0

    # # load the test image
    # img_url = 'http://places.csail.mit.edu/demo/6.jpg'
    # os.system('wget %s -q -O test.jpg' % img_url)
    # img = Image.open(i_image)
    input_img = V(tf(i_image).unsqueeze(0))

    # forward pass
    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    # print('RESULT ON ' + img_url)

    # output the IO prediction
    io_image = np.mean(labels_IO[idx[:10]]) # vote for the indoor or outdoor
    # if io_image < 0.5:
    #     print('--TYPE OF ENVIRONMENT: indoor')
    # else:
    #     print('--TYPE OF ENVIRONMENT: outdoor')

    # output the prediction of scene category

    output_dict = {}

    # print('--SCENE CATEGORIES:')
    for i in range(0, 5):
        # print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
        probability = round(probs[i], 2)
        output_dict[classes[idx[i]]] = probability

    # print(output_dict)

    # output the scene attributes
    responses_attribute = W_attribute.dot(features_blobs[1])
    idx_a = np.argsort(responses_attribute)
    # print('--SCENE ATTRIBUTES:')
    # print(', '.join([labels_attribute[idx_a[i]] for i in range(-1,-10,-1)]))

    # print([labels_attribute[idx_a[i]] for i in range(-1,-10,-1)])


    return output_dict, [labels_attribute[idx_a[i]] for i in range(-1,-10,-1)]


# places365("https://maps.googleapis.com/maps/api/streetview?size=600x400&location=6.86574,79.885480&heading=348.43&pitch=-0.76&key=")