## DewarpNet: An end-to-end approach for document dewarping using single image (In Progress).

### This codes are heavily structured on pytorch-semseg (https://github.com/meetshah1995/pytorch-semseg)

### Requirements

* pytorch >=0.4.0
* torchvision ==0.2.0
* visdom >=1.0.1 (for loss and results visualization)
* scipy
* tqdm

#### One-line installation
    
`pip install -r requirements.txt`

#### Training codes:
/train[...].py
- trainS3dbmnoimg.py/trainS3dbmnoimgv2.py : code to train backward mapping regression from world coordinates (model:dnetccnl)
- trainS3dWchtanHglassCRGB.py : code to train backward mapping regression from world coordinates (model:hourglass_cat)
- trainS3dWchtanHglassRGB.py : code to train backward mapping regression from world coordinates (model:hourglass)
- trainS3dWchtanUnetNCRGB.py/trainS3dWchtanUnetNCRGBv2.py : code to train world coord regression from RGB Image (model: unetnc)

#### Testing codes:
/test[...].py
- testBatch.py : test wc regression in a batch
- testBenchmE2E.py : test dewarping on benchmark data
- valWc.py : validate wc regression models

#### Loader:
/src/loader/
- swat3dbmnoimg_loader.py : loader for backward mapping (alb,wc,bm)
- swat3dbmnoimgd_loader.py : loader for backward mapping (alb,depth,wc,bm)
- swat3de2e_loader.py : loader for end-to-end model(alb,RGB img,wc,bm)
- swat3dlapsrn_loader.py : loader for lapsrn model(RGB img,wc)
- swat3dwc_loader.py : loader for wc regression model(RGB img,wc)
- swat3dwcg_loader.py : loader for wc regression model(Gray img,wc)

#### Model:
/src/models/
- densenet_.py : Densenet enc-dec with single fc layer
- densenet.py : Densenet enc-dec with two intermediate fc layers and dropout
- densenetcc.py : Densenet enc-dec with two intermediate fc layers and dropout + initial layer coordconv
- densenetccnl.py : Densenet enc-dec + initial layer coordconv
- densenetns.py : Densenet enc-dec with single fc layer no dropout and no final activation
- hourglass_cat.py : Hourglass module with concatenation (2 unets)
- hourglass.py : Hourglass module with residual connection (2 unets)
- lapsrn.py : LAPSRN for upscaling
- unet.py : Unet with center cropped skip connections
- unetnc.py : Unet with skip connections (no crop)




