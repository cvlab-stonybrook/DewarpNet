#augmentations code for RGB images

import cv2
import os
import matplotlib.pyplot as plt 
import numpy as np
import tqdm
import random

image_path='images-newmesh/'
mask_path='mask-newmesh/'
# root='/media/hilab/HiLabData/Sagnik/FoldedDocumentDataset/data/DewarpNet/'

# filenames=['2/34275Xpr_Page_319X224X0001','2/30548Xcp_Page_0085X273X0001','2/30111Xtc_Page_081X259X0001',
#                '2/31362Xns_Page_639X309X0001','2/33711Xcp_Page_0685X265X0001','2/31864Xns_Page_458X58X0001',
#                '2/32548Xns_Page_002X13X0001','2/34952Xny_Page_060X291X0001','2/32718Xny_Page_261X18X0001']


def findbuff(msk):
    xindex,yindex=np.where(msk==255)
    xmin=np.min(xindex)
    xmax=np.max(xindex)
    ymin=np.min(yindex)
    ymax=np.max(yindex)
    # print(xmax,xmin,ymax,ymin)
    size=msk.shape
    # print (msk.shape)
    bot_buff=size[0]-xmax
    top_buff=xmin
    left_buff=ymin
    right_buff=size[1]-ymax
    # to debug
    # msk=cv2.line(msk,(1,0),(1,left_buff), (255,0,0), 1)
    # msk=cv2.line(msk,(10,0),(10,right_buff), (255,0,0), 1)
    # msk=cv2.line(msk,(20,0),(20,bot_buff), (255,0,0), 1)
    # msk=cv2.line(msk,(30,0),(30,top_buff), (255,0,0), 1)
    # plt.imshow(msk)
    # plt.show()
    return left_buff,right_buff,top_buff,bot_buff


def move(img,lbl,msk,l,r,t,b):
    rows,cols,ch=img.shape
    chance=random.uniform(0,1)
    if chance>0.5:
        if r-2 >0:#SANITY CHECK
            xmv=random.randint(0,r-2)
        else:
            xmv=random.randint(0,r)

        if b-2 >0:#SANITY CHECK
            ymv=random.randint(0,b-2)
        else:
            ymv=random.randint(0,b)

    else:
        if l-2 >0:#SANITY CHECK
            xmv=-random.randint(0,l-2)
        else:
            xmv=-random.randint(0,l)
        if t-2 >0:#SANITY CHECK
            ymv=-random.randint(0,t-2)
        else:
            ymv=-random.randint(0,t)

    #move +x, +y
    M = np.float32([[1,0,xmv],[0,1,ymv]])
    nimg = cv2.warpAffine(img,M,(cols,rows))
    nlbl = cv2.warpAffine(lbl,M,(cols,rows))
    nmsk = cv2.warpAffine(msk,M,(cols,rows))
    # to debug
    # f,axarr=plt.subplots(1,6)
    # axarr[0].imshow(nlbl)
    # axarr[1].imshow(nimg)
    # axarr[2].imshow(nmsk)
    # axarr[3].imshow(lbl)
    # axarr[4].imshow(img)
    # axarr[5].imshow(msk)
    # plt.show()

    return nimg,nlbl,nmsk

def cropimg(img,lbl,msk):
    chance=random.uniform(0,1)
    if chance>0.7:
        size=msk.shape
        l,r,t,b=findbuff(msk)
        if t-2>0 and l-2>0: #SANITY CHECK
            nimg=img[t-2:size[0]-b+2,l-2:size[1]-r+2]
            nlbl=lbl[t-2:size[0]-b+2,l-2:size[1]-r+2]
            nmsk=msk[t-2:size[0]-b+2,l-2:size[1]-r+2]
        elif size[0]-b+2 < size[0] and size[1]-r+2 < size[1]: #SANITY CHECK
            nimg=img[t:size[0]-b+2,l:size[1]-r+2]
            nlbl=lbl[t:size[0]-b+2,l:size[1]-r+2]
            nmsk=msk[t:size[0]-b+2,l:size[1]-r+2]
        else: #SANITY CHECK
            nimg=img[t:size[0],l:size[1]]
            nlbl=lbl[t:size[0],l:size[1]]
            nmsk=msk[t:size[0],l:size[1]]
        # f,axarr=plt.subplots(1,4)
        # axarr[0].imshow(img)
        # axarr[1].imshow(nimg)
        # axarr[2].imshow(nlbl)
        # axarr[3].imshow(nmsk)
        # plt.show()
        nimg=cv2.resize(nimg,(img.shape[1],img.shape[0]),interpolation=cv2.INTER_LANCZOS4)
        nlbl=cv2.resize(nlbl,(img.shape[1],img.shape[0]),interpolation=cv2.INTER_LANCZOS4)
        nmsk=cv2.resize(nmsk,(img.shape[1],img.shape[0]),interpolation=cv2.INTER_LANCZOS4)
        # print(nimg.shape)
        return nimg,nlbl,nmsk
    else:
        return img,lbl,msk


def augmentbg(root,im_name,img,lbl):
    chance=random.uniform(0,1)

    #read an image
    # im_path = os.path.join(image_path + filenames[0] +'.png')
    # img=cv2.imread(im_path).astype(np.uint8)
    # img_arr=np.array(img)

    # plt.imshow(img)
    # plt.show()

    # im_name_split = im_name.strip().split('X')
    # foldr,msk_id=im_name_split[0].split('/')
    # msk_name = 'MSX'+msk_id+im_name_split[3]+'.png'         #WCX20001.exr

    #read the mask
    # print(os.path.join(root,mask_path,foldr,msk_name))
    # msk=cv2.imread(os.path.join(root,mask_path,foldr,msk_name),0)
    msk=((lbl[:,:,0]!=0)&(lbl[:,:,1]!=0)&(lbl[:,:,2]!=0)).astype(np.uint8)*255

    # plt.imshow(msk)
    # plt.show()

    #extract fg
    msk=msk.astype(np.uint8)
    if chance>0.3:
        fg=cv2.bitwise_and(img,img,mask=msk)

        ########################################################################### To shift the image
        # #shift the image/ shift the mask/ shift the label
        # shift_chance=random.uniform(0,1)
        # #find buffer space to shift
        # if shift_chance > 1:
        #     l,r,t,b=findbuff(msk)
        #     fg,nlbl,nmsk=move(fg,lbl,msk,l,r,t,b)
        # else:
        #     nlbl=lbl
        #     nmsk=msk
        ###########################################################################
        # plt.imshow(fg)
        # plt.show()
        #flip the mask
        rmsk=255-msk


        #read a random background    5641
        tex_id=random.randint(1,5640)
        with open(os.path.join(root[:-7],'augtexnames.txt'),'r') as f:
            for i in range(tex_id):
                txpth=f.readline().strip()

        # print(tex_root+txpth)
        if chance <= 0.5 :
            tex=cv2.imread(os.path.join(root[:-7],txpth)).astype(np.uint8)
            tex=cv2.resize(tex,(img.shape[1],img.shape[0]),interpolation=cv2.INTER_LANCZOS4)
        elif chance > 0.5 and chance <= 0.8:
            #color bg
            tex=np.ones((img.shape[0],img.shape[1],3)).astype(float)
            tex[:,:,0]=random.uniform(0,1)*tex[:,:,0]*255
            tex[:,:,1]=random.uniform(0,1)*tex[:,:,1]*255
            tex[:,:,2]=random.uniform(0,1)*tex[:,:,2]*255
        else:
            #noise bg
            tex=np.random.rand(img.shape[0],img.shape[1],3)*255
        
        # print(tex.__class__)
        # plt.imshow(tex)
        # plt.show()
        tex=tex.astype(np.uint8)
        #multiply with a random background
        bg=cv2.bitwise_and(tex,tex,mask=rmsk)
        # plt.imshow(bg)
        # plt.show()

        #get the final image 
        fimg= cv2.bitwise_xor(bg,fg)
        # plt.imshow(fimg)
        # plt.show()
        # plt.imshow(lbl)
        # plt.show()

        return fimg,lbl,msk
    else:
        return img,lbl,msk
    

def change_intensity(img):
    chance=random.uniform(0,1)
    # print(chance)
    nimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if chance>0.3:
        inc=random.randint(20,60)
        # print(inc)
        #increase
        v = nimg[:, :, 2]
        v = np.where(v <= 255 - inc, v + inc, 255)
        nimg[:, :, 2] = v

    nimg = cv2.cvtColor(nimg, cv2.COLOR_HSV2BGR)
    # f,axarr=plt.subplots(1,2)
    # axarr[0].imshow(img)
    # axarr[1].imshow(nimg)
    # plt.show()
    return nimg


def change_hue_sat(img):
    chance=random.uniform(0,1)
    # print(chance)
    nimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if chance>0.3:
        inc=random.randint(20,40)
        # print(inc)
        #increase
        v = nimg[:, :, 0]
        v = np.where(v <= 255 - inc, v + inc, 255)
        nimg[:, :, 0] = v

    if chance>0.3:
        inc=random.randint(20,40)
        # print(inc)
        #increase
        v = nimg[:, :, 1]
        v = np.where(v <= 255 - inc, v + inc, 255)
        nimg[:, :, 1] = v
    
    nimg = cv2.cvtColor(nimg, cv2.COLOR_HSV2BGR)
    # f,axarr=plt.subplots(1,2)
    # axarr[0].imshow(img)
    # axarr[1].imshow(nimg)
    # plt.show()
    return nimg

def add_noise(img):
    chance=random.uniform(0,1)
    nimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    if chance>0.3:
        rows,cols,ch = nimg.shape 
        noise = np.random.randint(0,30,(rows, cols))
        zitter = np.zeros_like(nimg)
        zitter[:,:,1] = noise  
        nimg = cv2.add(nimg, zitter)
        
    nimg = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
    # f,axarr=plt.subplots(1,2)
    # axarr[0].imshow(img)
    # axarr[1].imshow(nimg)
    # plt.show()
    return nimg


def call_augmentations(root,im_name,img,lbl):
    nimg=change_intensity(img)
    nimg=change_hue_sat(nimg)
    nimg,nlbl,nmsk=augmentbg(root,im_name,nimg,lbl)
    # nimg=add_noise(nimg)
    # nimg,nlbl,nmsk=cropimg(nimg,nlbl,nmsk)
    # f,axarr=plt.subplots(1,2)
    # axarr[0].imshow(img)
    # axarr[1].imshow(nimg)
    # plt.show()
    return nimg,nlbl


# def main():
#     for im_name in filenames:
#         im_path = os.path.join(root,image_path,im_name +'.png')
#         img=cv2.imread(im_path).astype(np.uint8)

#         lbl_name = im_name.strip().split('X')
#         foldr,lbl_id=lbl_name[0].split('/')
#         lbl_name = 'WCX'+lbl_id+lbl_name[3]+'.exr'         #WCX20001.exr
#         lbl_path = os.path.join(root, 'wc-corrmesh',foldr,lbl_name)
#         lbl = cv2.imread(lbl_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

#         img,lbl=call_augmentations(root,im_name,img,lbl)

# if __name__ == '__main__':
#     main()
    
