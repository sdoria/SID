import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
   
import glob
import numpy as np 
import rawpy

from dirs import *


def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out


def get_paths_fns(train_id):
    #input filename
    in_files = glob.glob(input_dir + '%05d_00*.ARW' % train_id)


    in_path_list = in_files   #[np.random.randint(0, len(in_files))]
    in_fn_list = [os.path.basename(in_path) for in_path in in_path_list]

    #ground truth filename 
    gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % train_id)
    gt_path = gt_files[0]
    gt_fn = os.path.basename(gt_path)
    
    return in_path_list, gt_path, in_fn_list, gt_fn
    

def get_imgs(train_id):
    # get dataset's input and ground truth processed images for a given train_id
    
    in_path_list, gt_path, in_fn_list, gt_fn = get_paths_fns(train_id)

    # exposure ratio between input and ground truth
    in_exposure_list = [float(in_fn[9:-5]) for in_fn in in_fn_list]
    gt_exposure = float(gt_fn[9:-5])
    ratio_list = [min(gt_exposure / in_exposure, 300) for in_exposure in in_exposure_list]

    # reading input
    input_img_list = [rawpy.imread(in_path) for in_path in in_path_list]
    input_img_list = [pack_raw(input_img) * ratio   for input_img, ratio in zip( input_img_list,ratio_list)]    # (H,W,C)  C = 4 

    # reading ground truth
    gt_img = rawpy.imread(gt_path)
    gt_img = gt_img.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
    gt_img = np.float32(gt_img / 65535.0)
    
    return input_img_list, gt_img



def get_imgs_processed(train_id):
    # get dataset's input and ground truth processed images for a given train_id
    # here input is processed same as ground truth
    
    in_path_list, gt_path, in_fn_list, gt_fn = get_paths_fns(train_id)

    # exposure ratio between input and ground truth
    in_exposure_list = [float(in_fn[9:-5]) for in_fn in in_fn_list]
    gt_exposure = float(gt_fn[9:-5])
    ratio_list = [min(gt_exposure / in_exposure, 300) for in_exposure in in_exposure_list]

    # reading input
    input_img_list = [rawpy.imread(in_path) for in_path in in_path_list]
    #input_img_list = [np.float32(input_img.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)* ratio/65535.0)
    input_img_list = [np.float32(input_img.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16,
                                   demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR,no_auto_scale=True,
                                  output_color = rawpy.ColorSpace.raw,
                                  user_black = 512)* ratio/65535.0)
                      for input_img, ratio in zip( input_img_list,ratio_list)]
    
    # reading ground truth
    gt_img = rawpy.imread(gt_path)
    gt_img = gt_img.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
    gt_img = np.float32(gt_img / 65535.0)
    
    return input_img_list, gt_img




def transform(input_img, gt_img, ps = 512, raw_input=True):#, switch_greens = False):
    
    
   
    
    # crop starts here
    H = input_img.shape[0]
    W = input_img.shape[1]

    xx = np.random.randint(0, W - ps)   # ps is patch size (e.g. 512)
    yy = np.random.randint(0, H - ps)   # xx, yy are corners of patch
    input_patch = input_img[yy:yy + ps, xx:xx + ps, :]
    
    if raw_input:
        gt_patch = gt_img[ yy * 2:yy * 2 + ps * 2, xx * 2:xx * 2 + ps * 2, :]
    else:
        gt_patch = gt_img[yy:yy + ps, xx:xx + ps, :]
    # crop ends here

    # transformations  
    # (shape is (H,W,C))
    # use .copy() to avoid negative strides which Pytorch does not like
    if np.random.randint(2, size=1)[0] == 1:  # random flip  50% chance, axis=H
        input_patch = np.flip(input_patch, axis=0).copy()
        gt_patch = np.flip(gt_patch, axis=0).copy()
    if np.random.randint(2, size=1)[0] == 1:  # random flip  50% chance, , axis=W
        input_patch = np.flip(input_patch, axis=1).copy()
        gt_patch = np.flip(gt_patch, axis=1).copy()
    if np.random.randint(2, size=1)[0] == 1:  # random transpose 50% chance
        input_patch = np.transpose(input_patch, ( 1, 0, 2)).copy()
        gt_patch = np.transpose(gt_patch, ( 1, 0, 2)).copy()
    #if True: #switch_greens:
        #if np.random.randint(2, size=1)[0] == 1:    # switch the two green channels (input only)
           # green1 = input_patch[:,:,0].copy()
           # green2 = input_patch[:,:,3].copy()
           # input_patch[:,:,0] = green2
           # input_patch[:,:,3] = green1
            
            


    input_patch = np.minimum(input_patch, 1.0)   
    
    return input_patch, gt_patch



class ImageDataset (Dataset):
    def __init__(self, train_ids, transform= True, test= False, raw_input=True):
        
        '''
        raw_input = True implies a 4 channel input as in the original SID paper
        raw_input = False implies a minimally processed 3 channel input
        '''
        
        self.ps = 512   #patch size
        
        self.transform =transform
        self.test=test
        self.raw_input = raw_input
        
        self.train_ids = train_ids
       
        
        self.input_img_lists = []
        self.gt_imgs = []
        
        
            
        
        
        for idx in range(len(self.train_ids)):

            train_id = train_ids[idx]

            if raw_input:
                input_img_list, gt_img = get_imgs(train_id)
            else:
                input_img_list, gt_img = get_imgs_processed(train_id)

            if test:          # return multiple images per train_id
                self.input_img_lists += input_img_list
                self.gt_imgs += [gt_img]*len(input_img_list)
                
            else:            # only one image per train_id
                self.input_img_lists += [input_img_list]
                self.gt_imgs += [gt_img]

        
            
    def __len__(self):
        
        
        return len(self.input_img_lists)
        
        #return len(self.train_ids)
        
        
                 
    def __getitem__(self, idx):
        
        # get the path from image id
        # multiple exposure time for a given id, we pick one at random
        '''
        For example, in "10019_00_0.033s.RAF",
        the first digit "1" means it is from the test set 
        ("0" for training set and "2" for validation set); "0019" is the image ID; 
        the following "00" is the number in the sequence/burst; 
        "0.033s" is the exposure time 1/30 seconds.
        '''
        
        
        
        
          
            
       

        self.input_img_list = self.input_img_lists[idx]
        self.gt_img = self.gt_imgs[idx]

        if self.test:
            input_img = self.input_img_list     #not a list in this case
        else :   #pick randomly from the list
            input_img = self.input_img_list[np.random.randint(0, len(self.input_img_list))]
        gt_img = self.gt_img
        
        if not self.transform:
            input_img = np.minimum(input_img, 1.0)      #fixes issue with streetlights
            return torch.from_numpy(input_img).permute(2,0,1) ,torch.from_numpy(gt_img).permute(2,0,1)  #(C,H,W) 
        
        
        
        
        
        input_img_tfm, gt_img_tfm = transform(input_img, gt_img, self.ps, self.raw_input)
        input_img_tfm = torch.from_numpy(input_img_tfm).permute(2,0,1)
        gt_img_tfm = torch.from_numpy(gt_img_tfm).permute(2,0,1)

        

        
        return input_img_tfm, gt_img_tfm   #(C,H,W) 