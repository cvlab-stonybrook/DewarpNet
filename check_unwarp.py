import numpy as np
import torch
from hdf5storage import loadmat
import cv2
import torch.nn.functional as F


t = loadmat('/media/hilab/HiLabData/Sagnik/FoldedDocumentDataset/data/DewarpNet/bm/1/DCX100001.mat')
t = t['bm'].astype(np.float32)
t = t / np.array([156, 187])
t = (t - 0.5) * 2
print 'min: %f, max%f'%(np.min(t), np.max(t))
t = np.reshape(t, (1, 448, 448, 2))
bm = torch.from_numpy(t).float()

im = cv2.imread('/media/hilab/HiLabData/Sagnik/FoldedDocumentDataset/data/DewarpNet/images/1/10Xpp_Page_248X69X0001.png').astype(np.float32) / 255.0
im = im.transpose((2, 0, 1))
im = np.expand_dims(im, 0)
im = torch.from_numpy(im)

res = F.grid_sample(im, bm)
res = res[0].numpy().transpose((1, 2, 0))
print np.max(res)

# cv2.imshow('a', res)
# cv2.waitKey(0)