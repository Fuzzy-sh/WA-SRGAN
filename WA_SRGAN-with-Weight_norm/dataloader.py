#################################
# UTILS
###############################
# import liberaries for Data utils 
import os
from os import listdir
from os.path import join
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms
from torchvision.transforms import Compose, RandomCrop, CenterCrop, Resize, ToTensor, ToPILImage, Normalize

####################################
# check if the file is image
#################################
def is_image_file(filename):
  return any(filename.endswith(extension) for extension in ['.png','.jpg','.jpeg','.PNG','.JPEG', '.JPG'])




#################################
# Proper augmentation
#################################
def calculate_valid_crop_size(crop_size, upscale_factor):
  return crop_size-(crop_size % upscale_factor)

def normalize():
  return(Compose([Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]))

# transformation for high and low resolution
def train_hr_transform(crop_size):
  return(Compose([RandomCrop(crop_size), ToTensor(),]))


def train_lr_transform(crop_size,upscale_factor):
  return (Compose([ToPILImage(),
                   Resize(crop_size//upscale_factor, interpolation=Image.BICUBIC),
                   ToTensor(),]))

def display_transform(crop_size):
  return (Compose([ToPILImage(),
                   Resize(crop_size),
                   CenterCrop(crop_size), 
                   ToTensor(),]))



# classes to read train, validation and test dataset. Also, transform the imgages and return the filenames
class TrainDatasetFromFolder(Dataset):
  def __init__(self, dataset_dir, crop_size, upscale_factor):
    super(TrainDatasetFromFolder, self).__init__()
    self.image_filenames=[join(dataset_dir,x) for x in listdir(dataset_dir) if is_image_file(x)]
    crop_size=calculate_valid_crop_size(crop_size, upscale_factor)
    self.hr_transform=train_hr_transform(crop_size)
    self.lr_transform=train_lr_transform(crop_size, upscale_factor)
    
  def __getitem__(self, index):
    hr_image=self.hr_transform(Image.open(self.image_filenames[index]))
    lr_image=self.lr_transform(hr_image)
    
    return hr_image, lr_image
  def __len__(self):
    return len(self.image_filenames)
  
class ValDatasetFromFolder(Dataset):
  def __init__(self, dataset_dir, upscale_factor,crop_size):
    super(ValDatasetFromFolder, self).__init__()
    self.upscale_factor=upscale_factor
    self.image_filenames=[join(dataset_dir,x) for x in listdir(dataset_dir) if is_image_file(x)]
    self.crop_size=crop_size

  def __getitem__(self, index):
    hr_image = Image.open(self.image_filenames[index])
    crop_size = calculate_valid_crop_size(self.crop_size, self.upscale_factor)
    lr_scale = Resize(self.crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
    hr_scale = Resize(self.crop_size, interpolation=Image.BICUBIC)
    hr_image = CenterCrop(self.crop_size)(hr_image)
    lr_image = lr_scale(hr_image)
    hr_restore_img = hr_scale(lr_image)
    lr_image= ToTensor()(lr_image)
    hr_restore_img=ToTensor()(hr_restore_img)
    hr_image= ToTensor()(hr_image)
    return lr_image, hr_restore_img, hr_image

  def __len__(self):
    return len(self.image_filenames)

class TesDatasetFromFolder(Dataset):
  def __init__(self, dataset_dir, upscale_factor ):
    super(TesDatasetFromFolder, self).__init__()
    self.lr_path=datasete_dir+'/data/'
    self.hr_path=dataset_dir+'/target/'
    self.lr_filenames=[join(lr_path,x) for x in listdir(self.lr_path) if is_image_file(x)]
    self.hr_filenames=[join(hr_path,x) for x in listdir(self.hr_path) if is_image_file(x)]
    self.upscale_factor=upscale_factor
  def __getitem__(self,index):
    image_name=self.lr_filenames[index].split('/')[-1]
    hr_image=Image.open(self.hr_filenames[index])
    lr_image=Image.open(self.lr_filenames[index])
    w,h=lr_image.size()
    hr_scale=Resize(h*self.upscale_factor, w*self.upscale_factor, interpolation=Image.BICUBIC)
    hr_restore_image=hr_scale(lr_image)
    return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_image), ToTensor()(hr_image)
  def __len__(self):
    return len(self.lr_filenames)


