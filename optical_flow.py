# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 21:05:11 2019

@author: Yang Xu
"""
import sys
import cv2
import numpy as np
import seaborn as sns
from scipy import signal

frame1_path=str(sys.argv[1])
frame2_path=str(sys.argv[2])
wd=int(sys.argv[3])
k_harris =float(sys.argv[4])

##convolution function for spatial derivative
def conv_de(img):
    
    kernal=np.array((-1,0,1))
    kernal=np.vstack((kernal,kernal,kernal))
    ##convolve2d is the only function used from scipy library
    dx = signal.convolve2d(img, kernal, boundary='symm', mode='same')
    dy = signal.convolve2d(img, kernal.T, boundary='symm', mode='same')
    
    return (dx,dy)

##lukas kanade algorithm to estimate optical flow
def lk_optcflow(frame1,frame2,window_size=5,k=0.05):
    
    ##use difference of gaussian to preprocess images and then use it for
    ##harris corner detection
    blur1 = cv2.GaussianBlur(frame1,(7,7),3)
    blur2 = cv2.GaussianBlur(frame1,(7,7),9)
    blurs = blur1-blur2
    blurs = np.float32(blurs)
    ##find corner regions in frame1 by harris corner detection
    ##use the same window size
    dst = cv2.cornerHarris(blurs,window_size,3,k)
    ##dilation for marking the corners
    dst = cv2.dilate(dst,None)
    ##threshhold for optimum result
    ##estimate optical flow of corner regions
    dst[dst>0.01*dst.max()]=255
    
    ##blur image
    f1 = cv2.GaussianBlur(frame1,(7,7),3)
    f1 = f1/255
    f2 = cv2.GaussianBlur(frame2,(7,7),3)
    f2 = f2/255
    It=f2-f1
    (dx,dy) = conv_de(f1)
    mag = np.zeros(frame1.shape)
    ang = np.zeros(frame1.shape)
    u = np.zeros(frame1.shape)
    v = np.zeros(frame2.shape)
    w = window_size//2
    for i in range(w,dst.shape[0]-w):
        for j in range(w,dst.shape[1]-w):
            if dst[i,j]==255:
                I=-(It[i-w:i+w+1,j-w:j+w+1].flatten())
                Ixy=np.column_stack((dx[i-w:i+w+1,j-w:j+w+1].flatten(),
                                     dy[i-w:i+w+1,j-w:j+w+1].flatten()))
                ata = np.dot(Ixy.T,Ixy)
                atb = np.dot(Ixy.T,I)
                uij,vij = np.linalg.lstsq(ata, atb, rcond=None)[0]
                u[i,j]=uij
                v[i,j]=vij
                mag[i,j]=uij**2+vij**2
                ang[i,j]=np.degrees(np.arctan2(vij,uij))
            ##I somehow thought I should do iterative lk, but it seems
            ##unnecessary in this case
            #if dst[i,j]==255:
                ##initialize u and v
            #    (ux,vy)=(0,0)
            #    iterative=True
            #    while iterative:
            #        I=f1[i-w+int(ux):i+w+1+int(ux),j-w+int(vy):j+w+1+int(vy)]\
            #        -f2[i-w:i+w+1,j-w:j+w+1]
            #        I=I.flatten()
            #        Ixy=np.column_stack((dx[i-w+int(ux):i+w+1+int(ux),\
            #                                j-w+int(vy):j+w+1+int(vy)].flatten(),\
            #                             dy[i-w+int(ux):i+w+1+int(ux),\
            #                                j-w+int(vy):j+w+1+int(vy)].flatten()))
            #        ata = np.dot(Ixy.T,Ixy)
            #        atb = np.dot(Ixy.T,I)
            #        uij,vij = np.linalg.lstsq(ata, atb, rcond=None)[0]
            #        ux=ux+uij
            #        vy=vy+vij
            #        if abs(uij) < 1 and abs(vij) < 1:
            #            iterative = False
            #    u[i,j]=ux
            #    v[i,j]=vy
            #    mag[i,j]=u**2+v**2
            #    ang[i,j]=np.degrees(np.arctan2(v,u))
                    
            else:
                continue
    return mag,ang,u,v
    
    
    
def main():
    
    f1 = cv2.imread(frame1_path,0)
    f2 = cv2.imread(frame2_path,0)
    
    mag,angle,ux,vy=lk_optcflow(f1,f2,\
                                window_size=wd,k=k_harris)
    
    move_ang=sns.heatmap(angle,cmap="RdBu")
    move_ang.get_figure().savefig("angle_optical_flow.jpeg",dpi=1200)
    
    move_mag=sns.heatmap(np.log(mag+1),cmap="RdBu")
    move_mag.get_figure().savefig("mag_optical_flow.jpeg",dpi=1200)

if __name__ == '__main__':
    main()
