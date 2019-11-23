# FearNet

Goal:

Notes:

Hyper Params: LR = 0.001, Epoch = 20, Batch Size = 64
-----------------------------------------------------
Baseline: 13.67%   
3-DCNN Ensemble: 14.48%   
Resnet152 (1 unfrozen fc layer): ~69%  
Resnet152 (1 unfrozen fc layer + 1 external fc layer): ~75%  
Resnet152 (1 unfrozen fc layer + 2 external fc layers): ~75%  
VGG19_BN (1 unfrozen fc layer): ~73%  
VGG19_BN (1 unfrozen fc layer + 1 external fc layer): ~73-74%  
Densenet161 (1 unfrozen fc layer): 75%
Densenet161 (1 unfrozen fc layer + 1 external fc layer): ~75-76%
