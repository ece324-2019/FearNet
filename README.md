# FearNet

Goal:

Notes:

Hyper Params: 128x128img, LR = 0.001, Epoch = 20, Batch Size = 64, No BN (unless model inherent, i.e. VGG)   
-------------------------------------------------------
Baseline: 13.67%   
3-DCNN Ensemble: 14.48%   
Resnet152 (1 unfrozen fc layer): ~69%  
Resnet152 (1 unfrozen fc layer + 1 ext. fc layer): ~75%  
Resnet152 (1 unfrozen fc layer + 2 ext. fc layers): ~75%  
VGG19_BN (1 unfrozen fc layer): ~73%  
VGG19_BN (1 unfrozen fc layer + 1 ext. fc layer): ~73-74%  
Densenet161 (1 unfrozen fc layer): 75%  
Densenet161 (1 unfrozen fc layer + 1 ext. fc layer): ~75-76%  

Using Updated (Cleaned) Dataset (~5% overall val. acc. improvement)  
-----------------------------------------------------
Densenet161 (1 unfrozen fc layer): ~79%    
Densenet161 (1 unfrozen fc layer + 1 ext. fc layer): ~80-81%  
Densenet161 (1 unfrozen fc layer + 2 ext. fc layer): ~80%  
Resnext101 (1 unfrozen fc layer): ~78%   
Resnext101 (1 unfrozen fc layer + 1 ext. fc layer): ~78-79%    
Resnext101 (1 unfrozen fc layer + 2 ext. fc layer): ~78-79%   
Wres101 (1 unfrozen fc layer): ~75%  
Wres101 (1 unfrozen fc layer + 1 ext. fc layer): ~76%     
Alexnet (1 unfrozen fc layer): ~70%    
Googlenet (1 unfrozen fc layer): ~72%   
Shufflenet (1 unfrozen fc layer): ~73%    

W/ Batch Norm on modified linear layers (~2% overall val. acc. improvement)  
--------------------------------------
Densenet161 (1 unfrozen fc layer + 1 ext. fc layer): ~80.5%   
Resnext101 (1 unfrozen fc layer + 1 ext. fc layer): ~80-81%   
Resnext101 (1 unfrozen fc layer + 2 ext. fc layer): ~80-81%   
Wres101 (1 unfrozen fc layer + 1 ext. fc layer): ~77%     
Alexnet (1 unfrozen fc layer + 1 ext. fc layer): ~73-74%    
Googlenet (1 unfrozen fc layer + 1 ext. fc layer): ~73%     
Shufflenet (1 unfrozen fc layer + 1 ext. fc layer): ~75% (was 77 up to 50 epochs)      

Ensembling Results   
-------------------
8-Transfer-Learned-Model Ensemble w/ Averaged Output: ~86-87%   
8-TL Ensemble w/ 1 layer output: ~84%   
