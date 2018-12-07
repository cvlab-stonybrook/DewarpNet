import argparse
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class WcToNormals(nn.Module):
    def __init__(self,cuda=False):
        super(WcToNormals, self).__init__()
        # self.opt = opt
        if cuda:
            self.fa9 = torch.cuda.FloatTensor(1,1,3,3).fill_(0).cuda()
            self.fb9 = torch.cuda.FloatTensor(1,1,3,3).fill_(0).cuda()
            self.fa8 = torch.cuda.FloatTensor(1,1,3,3).fill_(0).cuda()
            self.fb8 = torch.cuda.FloatTensor(1,1,3,3).fill_(0).cuda()
            self.fa7 = torch.cuda.FloatTensor(1,1,3,3).fill_(0).cuda()
            self.fb7 = torch.cuda.FloatTensor(1,1,3,3).fill_(0).cuda()
            self.fa6 = torch.cuda.FloatTensor(1,1,3,3).fill_(0).cuda()
            self.fb6 = torch.cuda.FloatTensor(1,1,3,3).fill_(0).cuda()
        else:
            self.fa9 = torch.FloatTensor(1,1,3,3).fill_(0)
            self.fb9 = torch.FloatTensor(1,1,3,3).fill_(0)
            self.fa8 = torch.FloatTensor(1,1,3,3).fill_(0)
            self.fb8 = torch.FloatTensor(1,1,3,3).fill_(0)
            self.fa7 = torch.FloatTensor(1,1,3,3).fill_(0)
            self.fb7 = torch.FloatTensor(1,1,3,3).fill_(0)
            self.fa6 = torch.FloatTensor(1,1,3,3).fill_(0)
            self.fb6 = torch.FloatTensor(1,1,3,3).fill_(0)
        # initialize horizontal and vertical filters
        self.fa9[:,:,2,0] = -1
        self.fa9[:,:,2,2] = 1
        self.fa9 = Variable(self.fa9, requires_grad=False)
        self.fb9[:,:,2,2] = -1
        self.fb9[:,:,0,2] = 1
        self.fb9 = Variable(self.fb9, requires_grad=False)
        
        self.fa8[:,:,2,2] = -1
        self.fa8[:,:,0,2] = 1
        self.fa8 = Variable(self.fa8, requires_grad=False)
        self.fb8[:,:,0,2] = -1
        self.fb8[:,:,0,0] = 1
        self.fb8 = Variable(self.fb8, requires_grad=False)
        
        self.fa7[:,:,0,2] = -1
        self.fa7[:,:,0,0] = 1
        self.fa7 = Variable(self.fa7, requires_grad=False)
        self.fb7[:,:,0,0] = -1
        self.fb7[:,:,2,0] = 1
        self.fb7 = Variable(self.fb7, requires_grad=False)
        
        self.fa6[:,:,0,0] = -1
        self.fa6[:,:,2,0] = 1
        self.fa6 = Variable(self.fa6, requires_grad=False)
        self.fb6[:,:,2,0] = -1
        self.fb6[:,:,2,2] = 1
        self.fb6 = Variable(self.fb6, requires_grad=False)

        self.padder = nn.ReplicationPad2d(1)

        self.getCrossProductNormals9 = waspCrossNormals()
        self.getCrossProductNormals8 = waspCrossNormals()
        self.getCrossProductNormals7 = waspCrossNormals()
        self.getCrossProductNormals6 = waspCrossNormals()
        
        
    def getCrossProductNormals(self,u1,u2,u3,v1,v2,v3):
        # normals1, the cross product of u and v
        n1 = torch.mul(u2,v3)-torch.mul(u3,v2)
        n2 = torch.mul(u3,v1)-torch.mul(u1,v3)
        n3 = torch.mul(u1,v2)-torch.mul(u2,v1)
        # normalize the vector
        namp = torch.sqrt(n1.pow(2) + n2.pow(2) + n3.pow(2))
        nx = torch.div(n1,namp+1e-12)
        ny = torch.div(n2,namp+1e-12)
        nz = torch.div(n3,namp+1e-12)
        # the normal map
        normals = torch.cat((nx,ny,nz),dim=1)
        return normals
        
    def forward(self, cx,cy,cz):
        # input is cx,cy,cz , where cz is depth map
        self.batchSize = cx.size(0)
        cx_pad = self.padder(cx)
        cy_pad = self.padder(cy)
        cz_pad = self.padder(cz)
        # horizontal vector h1
        a9x = F.conv2d(cx_pad, self.fa9, stride=1, padding=0)
        a9y = F.conv2d(cy_pad, self.fa9, stride=1, padding=0)
        a9z = F.conv2d(cz_pad, self.fa9, stride=1, padding=0)
        b9x = F.conv2d(cx_pad, self.fb9, stride=1, padding=0)
        b9y = F.conv2d(cy_pad, self.fb9, stride=1, padding=0)
        b9z = F.conv2d(cz_pad, self.fb9, stride=1, padding=0)
        
        a8x = F.conv2d(cx_pad, self.fa8, stride=1, padding=0)
        a8y = F.conv2d(cy_pad, self.fa8, stride=1, padding=0)
        a8z = F.conv2d(cz_pad, self.fa8, stride=1, padding=0)
        b8x = F.conv2d(cx_pad, self.fb8, stride=1, padding=0)
        b8y = F.conv2d(cy_pad, self.fb8, stride=1, padding=0)
        b8z = F.conv2d(cz_pad, self.fb8, stride=1, padding=0)
        
        a7x = F.conv2d(cx_pad, self.fa7, stride=1, padding=0)
        a7y = F.conv2d(cy_pad, self.fa7, stride=1, padding=0)
        a7z = F.conv2d(cz_pad, self.fa7, stride=1, padding=0)
        b7x = F.conv2d(cx_pad, self.fb7, stride=1, padding=0)
        b7y = F.conv2d(cy_pad, self.fb7, stride=1, padding=0)
        b7z = F.conv2d(cz_pad, self.fb7, stride=1, padding=0)
        
        a6x = F.conv2d(cx_pad, self.fa6, stride=1, padding=0)
        a6y = F.conv2d(cy_pad, self.fa6, stride=1, padding=0)
        a6z = F.conv2d(cz_pad, self.fa6, stride=1, padding=0)
        b6x = F.conv2d(cx_pad, self.fb6, stride=1, padding=0)
        b6y = F.conv2d(cy_pad, self.fb6, stride=1, padding=0)
        b6z = F.conv2d(cz_pad, self.fb6, stride=1, padding=0)
        # normals1, the cross product of h1 and c3
        normals9 = self.getCrossProductNormals9(a9x, a9y, a9z, b9x, b9y, b9z)
        normals8 = self.getCrossProductNormals8(a8x, a8y, a8z, b8x, b8y, b8z)
        normals7 = self.getCrossProductNormals7(a7x, a7y, a7z, b7x, b7y, b7z)
        normals6 = self.getCrossProductNormals6(a6x, a6y, a6z, b6x, b6y, b6z)
        # average the normals
        normals0 = normals9 + normals8 + normals7 + normals6
        # normalize the averaged normals
        n1 = normals0[:,0,:,:].unsqueeze(1)
        n2 = normals0[:,1,:,:].unsqueeze(1)
        n3 = normals0[:,2,:,:].unsqueeze(1)
        namp = torch.sqrt(n1.pow(2) + n2.pow(2) + n3.pow(2))
        nx = torch.div(n1,namp+1e-12)
        ny = torch.div(n2,namp+1e-12)
        nz = torch.div(n3,namp+1e-12)
        # the normal map
        normals = torch.cat((nx,ny,nz),dim=1)       
        return normals

class waspCrossNormals(nn.Module):
    def __init__(self, ngpu=1):
        super(waspCrossNormals, self).__init__()
        self.ngpu = ngpu
    def forward(self, u1,u2,u3,v1,v2,v3):

        n1 = torch.mul(u2,v3)-torch.mul(u3,v2)
        n2 = torch.mul(u3,v1)-torch.mul(u1,v3)
        n3 = torch.mul(u1,v2)-torch.mul(u2,v1)
        # normalize the vector
        namp = torch.sqrt(n1.pow(2) + n2.pow(2) + n3.pow(2))
        nx = torch.div(n1,namp+1e-12)
        ny = torch.div(n2,namp+1e-12)
        nz = torch.div(n3,namp+1e-12)
        # the normal map
        normals = torch.cat((nx,ny,nz),dim=1)
        return normals

def getBaseGridForCoord( N=(512,512), normalize = True, getbatch = False, batchSize = 1):
    a = torch.arange(-(N[1]-1), N[1], 2)
    if normalize:
        a = a/(N[1]-1)
    x = a.repeat(N[0],1)
    b = torch.arange(-(N[0]-1), N[0], 2)
    if normalize:
        b = b/(N[0]-1)
    y = -b.repeat(N[1],1).t()
    #print(x.shape)
    #print (y.shape)
    grid = torch.cat((x.unsqueeze(0), y.unsqueeze(0)),0)
    if getbatch:
        grid = grid.unsqueeze(0).repeat(batchSize,1,1,1)
    return grid 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--cuda', nargs='?', type=bool, default=True, 
                        help='Use GPU')
    args = parser.parse_args()
    img1_path="/media/hilab/HiLabData/Sagnik/FoldedDocumentDataset/data/DepthTrain/worldCoord/ImageWC_100010001.png"
    img2_path="/media/hilab/HiLabData/Sagnik/FoldedDocumentDataset/data/DepthTrain/worldCoord/ImageWC_100020001.png"
    img1=cv2.imread(img1_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    img2=cv2.imread(img2_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    # imgsize=img1.shape
    img1=np.array(img1,dtype=np.float)/255.0
    img2=np.array(img2,dtype=np.float)/255.0
    # print img1

    cx=torch.from_numpy(img1[:,:,0]).float().unsqueeze(0).unsqueeze(0).cuda()
    cy=torch.from_numpy(img1[:,:,1]).float().unsqueeze(0).unsqueeze(0).cuda()
    cz=torch.from_numpy(img1[:,:,2]).float().unsqueeze(0).unsqueeze(0).cuda()

    # img1b=torch.from_numpy(img1).float().unsqueeze(0).unsqueeze(0).cuda()

    # img1=torch.from_numpy(img1).float().unsqueeze(0).unsqueeze(0).cuda()
    # img2=torch.from_numpy(img2).float().unsqueeze(0).unsqueeze(0).cuda()
    # basegxy = getBaseGridForCoord(N=imgsize, getbatch = True, batchSize = 1)

    # if args.cuda:
    #     cx = ((basegxy[:,0,:,:].unsqueeze(1)+1)/2).cuda()
    #     cy = ((basegxy[:,1,:,:].unsqueeze(1)+1)/2).cuda()    
    # else:
    #     cx = (basegxy[:,0,:,:].unsqueeze(1)+1)/2
    #     cy = (basegxy[:,1,:,:].unsqueeze(1)+1)/2

    wc2n= WcToNormals(args)
    norm1=wc2n(cx,cy,cz)
    disp_surf_norm=norm1.squeeze(0).transpose(0,1).transpose(1,2)
    print(disp_surf_norm.shape)
    cv2.imshow('norm',disp_surf_norm.cpu().numpy())
    # cv2.imshow('img',img1)
    cv2.waitKey(0)
    