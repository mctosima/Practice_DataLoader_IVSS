# %%
# Import the modules here!
from __future__ import print_function, division
import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.transforms import Normalize
from PIL import Image
import datetime
from labels import Labels
plt.ion() # interactive mode
"""
Inherit the proper class from PyTorch module.
"""
class CustomDataset(Dataset):                                   #this class function's purpose is to prepare the dataset path and showing the results
    def __init__(self, root="dataset", transform=None):
        self.imagespath  = glob.glob('dataset/images/*.png')    #get the image's directory, I'm familiar with glob
        self.labelspath = glob.glob('dataset/labels/*.png')     #get the label's directory, I'm familiar with glob
        self.root = root                                        #determining that root is /dataset/ directory
        
        # try to print the path for checking purpose
        print("Img Path: ", self.imagespath)
        print("Label Path: ", self.labelspath)
        self.transform = transform
    
    def __len__(self) -> int:        
        return len(self.imagespath)
    
    def __getitem__(self,index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        #load image Dataset
        img_data = self.imagespath[index]
        image = Image.open(img_data)
        
        #load label Dataset
        label_data = self.labelspath[index]
        label = Image.open(label_data)
        
        #dictionary between image and label
        sample_data = {"image":image, "label":label}
        
        if self.transform:
            sample_data = self.transform(sample_data)
        return sample_data
            
    
    def shows(self, image, label):                              #This function is preparing the processed image to be shown again:
        for datasetIndex in range(image.shape[0]):
            image_ = ImageDenormalize()(image[datasetIndex])    #denormalize image (still in tensor format)
            image_ = transforms.ToPILImage()(image_)            #convert from tensor to PILImage
            image_ = np.asarray(image_)                         #convert into numpy array
            label_ = np.asarray(label[datasetIndex])            #convert label into numpy array
            _, axs = plt.subplots(2, figsize=(15, 15))          #create 2 figure in plotting using matplotlib
            axs[0].imshow(image_)                               #showing the image output                 
            axs[0].set_title("Image Output")                    #create the figure title
            label_ = Labels.colorize(label_)                    #colorize the label based on labels.py file
            axs[1].imshow(label_)                               #showing the label output
            axs[1].set_title("Colorized Label Output")          #create the figure title
            
            fname = datetime.datetime.now()
            print(fname)
            plt.imsave(arr=image_, fname=os.path.join("output", f"{fname}.png"))
            plt.imsave(arr=label_, fname=os.path.join("output", f"{fname}_label.png"))
            
class HorizontalFlip():
    def __init__(self)-> None:
        pass
    
    def __call__(self, sample) -> dict:
        image = sample["image"]
        label = sample["label"]
        transformedImage = transforms.RandomHorizontalFlip(p=1)(image) #do a random horizontal flip (probability 1, means all image will be flipped)
        transformedLabel = transforms.RandomHorizontalFlip(p=1)(label) #do a random horizontal flip (probability 1, means all label will be flipped)
        
        return {"image":transformedImage, "label":transformedLabel}
        
class RandomCrop():
    def __init__(self) -> None:
            self.output_size = (512, 512)        # define the output size (size after cropped)
    
    def __call__(self,sample) -> dict:
        image = sample["image"]
        label = sample["label"]

        w, h = image.size                       # get image size
        new_h, new_w = self.output_size         # determine new image size
        
        left = np.random.randint(0, w - new_w)  # create horizontal random location for cropping
        top = np.random.randint(0, h - new_h)   # create vertical random location for cropping
        print("image height :",h,"image weight :",w)    #printing for checking purpose
        print("height crop location :",top,"weight crop location :",left) #printing for checking purpose
        
        imagecropped = transforms.functional.crop(image, top,left,new_h,new_w) #crop image using the defined parameter before
        labelcropped = transforms.functional.crop(label, top,left,new_h,new_w) #crop label using the same defined parameter before

        return {"image": imagecropped, "label": labelcropped} #return the results
    
class ImageNormalized():                                                #class for normalizing the image dataset
    def __init__(self) -> None:
        self.mean = (0.485, 0.456, 0.406)                               #define the mean value
        self.std = (0.229, 0.224, 0.225)                                #define the std value
        
    def __call__(self,sample) -> dict:
        image = sample["image"]                                         #load image dataset based on given index
        label = sample["label"]                                         #load label dataset based on given index
        trans = transforms.Compose([                                    #transforms.Compose is used to create a sequential process consist of ToTensor and Normalize
            transforms.ToTensor(),
            Normalize(self.mean, self.std)
        ])
        imageNormalized = trans(image)
        labelTensored = torch.tensor(np.asarray(label))                 #converting the label into tensor
        
        return {"image":imageNormalized, "label":labelTensored}         #returning the normalized image and tensored label

class ImageDenormalize():    
    def __init__(self) -> None:
        self.mean = (-0.485/0.229, -0.456/0.224, -0.406/0.255)          #to denormalize simply by dividing the minus of previous mean with previous std
        self.std = (1/0.229, 1/0.224, 1/0.255)                          #to denormalize simply by dividing 1 with the previous std

    def __call__(self, sample) -> dict:                                 #since I use ImageDenormalize() in def shows() in CustomDataset() class, I don't have to load the image and label here
        # image = sample["image"]
        # label = sample["label"]
        trans = Normalize(self.mean, self.std)                          #load the Normalize function (with denormalize value into imagedenorm)
        imagedenorm = trans(sample)                                     #process the denomrlaization
        return imagedenorm                                              #return the denormalized image
    

if __name__ == "__main__":                                              # main function for calling the transformation and batch iteration
    alltransformation = transforms.Compose([                            # composing a sequence of transformation
        RandomCrop(),HorizontalFlip(),ImageNormalized()])
    load = CustomDataset(transform = alltransformation)                 # put the transformation instruction into new variable

    dataloader = DataLoader(load, batch_size =1, shuffle = True)        # using DataLoader function to load multiple files and define the transformation into the Loader
    

    # plt.imshow(transforms.ToPILImage()(trial["image"]))
    # plt.figure()
    # plt.imshow(transforms.ToPILImage()(trial["label"]))
    
    for i_batch, sample_batched in enumerate(dataloader):               # do the batch processing using an iteration
        print(i_batch, sample_batched["image"].size(), sample_batched["label"].size())
        load.shows(sample_batched["image"], sample_batched["label"])
# %%
