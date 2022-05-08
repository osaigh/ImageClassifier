import model_utility as mu
import json
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch import optim
from collections import OrderedDict
from PIL import Image

def predict(image_path, checkpoint_path,use_gpu = False ,topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Load model
    model = mu.LoadCheckpoint(checkpoint_path)

    #use gpu if available 
    device = torch.device('cpu')
    if use_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    #load PILImage
    pilimage = Image.open(image_path)
    
    #Convert from PILImage to Tensor 
    tensorimage = process_image(pilimage)
    
    #Convert data from torch DoubleTensor to Float Tensor
    if device.type == 'cuda':
        tensorimage = tensorimage.type(torch.cuda.FloatTensor)
    else:
        tensorimage = tensorimage.type(torch.FloatTensor)
        
    tensorimage = tensorimage.to(device)

    # Switch from training mode to evaluation mode
    model.eval()
    
    # use scope with grad turned off
    with torch.no_grad():
        output = model.forward(tensorimage.view(1,tensorimage.shape[0],tensorimage.shape[1],tensorimage.shape[2]))
        ps = torch.exp(output.cpu())
        top_p, top_class = ps.topk(topk,dim=1)
    
    #convert to numpy array
    top_p_arr = top_p.view((top_p.shape[1],)).numpy()
    top_class_arr_clone = top_class.view((top_class.shape[1],)).numpy() 
    
    #convert from index to class
    tempDict = dict()
    for c,i in model.class_to_idx.items():
        tempDict[str(i)]= str(c)
    
    top_class_str_list = list()
    for ind in top_class_arr_clone:
        top_class_str_list.append(tempDict[str(ind)])
    
    top_class_arr = np.array(top_class_str_list)
    
    return top_p_arr, top_class_arr

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    #Resize
    if image.height < image.width:
        image = image.resize((int((256/image.height)* image.width),256))
    else:
        image = image.resize((256,int((256/image.width)* image.height)))    
        
    #Crop
    left = (image.width / 2) - 112
    top = (image.height / 2) - 112
    image = image.crop((left,top, left+224, top+ 224))
    
    #color channel
    np_image = np.array(image)/255   
    
    #normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (np_image - mean)/std
    
    #transpose
    img = img.transpose((2,0,1))
    return torch.from_numpy(img)  

def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor. This was ripped from the helper module provided during class"""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax

def view_classify(image_path, top_p, top_class,cat_to_name_file = 'NULL'):
    """view_classify. Displays the image along with its top predictions"""

    #load PILImage
    pilimage = Image.open(image_path)
    img = process_image(pilimage)
    
    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=1, nrows=2)
    imshow(img, ax=ax1, normalize=False)
    ax1.axis('off')
    
    # get the real class name using the mapping from index to class and then class to category
    top_class_str_arr = list()
    if cat_to_name_file != 'NULL':
        with open('cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f)
        for e in top_class:
            top_class_str_arr.append(cat_to_name[str(e)])
    else:
        for e in top_class:
            top_class_str_arr.append(str(e))
    
    ax2.barh(np.arange(top_class.shape[0]), top_p)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(top_class.shape[0]))
    ax2.set_yticklabels(top_class_str_arr)
    
    ax1.set_title(top_class_str_arr[0])

    plt.tight_layout()


