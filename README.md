# Learning to See in the Dark implemented in Pytorch

Pytorch implementation of "Learning to See in the Dark" [1], a model using a U-net architecture[2] 

We obtain results comparable to the original Tensorflow model (PNSR 28.39, SSIM 0.784) on the Sony dataset.

Saved model parameters can be downloaded [here](https://drive.google.com/file/d/11Xvyk0pvrrOdukC9erHSykprkcXeXnmF/view?usp=sharing) 


Modifications that were tried and did not improve our results (not published in this repo):

- Using VGG16 feature loss (a.k.a. perceptual loss) instead of L1 Loss (inspired by [3] and [4])
- Replace first layer by three 3*3 convolutional layers that progressively increase number of channels - PNSR = 28.45(inspired by [5])
- Building another U-net model that uses a pretrained Resnet34 model as a backbone (PNSR = 21.7) [6] 
- Adding short skip connections (no loss improvement, but potentially faster training) [5]
- Combining our fully trained baseline U-net with a "dynamic Unet" [6] to form a W shaped model (very poor results)
- Replacing transpose convolutions with other upsampling methods
- Using minimally processed 3-channel inputs (bilinear interpolation) instead of the original 4-channel raw input



# References:

[1] Learning to See in the Dark in CVPR 2018, by Chen Chen, Qifeng Chen, Jia Xu, and Vladlen Koltun.
[arXiv](https://arxiv.org/abs/1805.01934), [github](https://github.com/cchen156/Learning-to-See-in-the-Dark),
    [Project Website](http://cchen156.web.engr.illinois.edu/SID.html)
    
[2] U-Net: Convolutional Networks for BiomedicalImage Segmentation, by Olaf Ronneberger, Philipp Fischer, and Thomas Brox

[3] Perceptual Losses for Real-Time Style Transfer and Super-Resolution, by Justin Johnson, Alexandre Alahi, Li Fei-Fei

[4] [fastai implementation of feature loss](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson7-superres.ipynb)
    
[5]  Bag of tricks for image classification with convolutional neural networks, by T. He, Z. Zhang, H. Zhang, Z. Zhang, J. Xie, and M. Li.

[6] [Dynamic Unet, fastai library](https://docs.fast.ai/vision.models.unet.html)





