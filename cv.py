#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 17:37:43 2021

@author: s174162
"""
import numpy as np

def box3d(n):
   N = 15*n;
   Q = np.zeros((N,3))
   val = np.linspace(-0.5,0.5,n);
   span = val;
   for j in range(0,5-1):
       span = np.append(span,val)

    # x-coords
   Q[0:4*n,0] = -0.5
   Q[4*n:8*n,0] = 0.5
   Q[8*n:13*n,0] = span
   Q[13*n:,0] = 0 # not necessary (already zeros)
    
    # y-coords
   span = np.append(val,val)
   Q[0:2*n,1]=span # x=-0.5, y = val, z=0.5 and z=-0.5. 2*n points
   Q[2*n:3*n,1]=0.5 # x=-0.5, y = 0.5, z=val. 1*n points
   Q[3*n:4*n,1]=-0.5 # x=-0.5, y = -0.5, z=val. 1*n points
   Q[4*n:6*n,1]=span # x=0.5, y = val, z=0.5 and z=-0.5. 2*n points
   Q[6*n:7*n,1]=0.5 # x=0.5, y=0.5, z=val, 1*n points
   Q[7*n:8*n,1]=-0.5 # x=0.5, y=-0.5, z=val, 1*n points
   Q[8*n:9*n,1]=0 # x=span, y=0, z=0, 1*n points
   Q[9*n:11*n,1]=0.5 # x=span, y=0.5, z=0.5 and z=-0.5. 2*n points
   Q[11*n:13*n,1]=-0.5 # x=span, y=-0.5, z=0.5 and z=-0.5. 2*n points
   Q[13*n:14*n,1]=val # x=0, y=span, z=0

    # z-coords
   Q[0:n,2]=0.5
   Q[n:2*n,2]=-0.5
   Q[2*n:4*n,2]=span
   Q[4*n:5*n,2]=0.5
   Q[5*n:6*n,2]=-0.5
   Q[6*n:8*n,2]=span
   Q[8*n:9*n,2]=0
   Q[9*n:10*n,2]=0.5
   Q[10*n:11*n,2]=-0.5
   Q[11*n:12*n,2]=0.5
   Q[12*n:13*n,2]=-0.5
   Q[13*n:14*n,2]=0 # not necessary (already zeros)
   Q[14*n:,2]=val
   
       
   return Q.T


def projectpoints(K, R, t, Q) :
    # K, R, t, and Q are, respectively, the camera matrix, rotation, 
    # translation, and a list of 𝑁 3D points to be projected into the camera.
    P = np.matmul(K,np.c_[R,t]) # projection matrix
    nQ = Q.shape[1] # number of points
    Qhomo = np.vstack((Q,np.ones(nQ))) # 3D points in homogeneous form (4 coordinates)
    qhomo = np.matmul(P,Qhomo) # projected points
    q = (qhomo[:2,:nQ]/qhomo[2,:]).T # 2D points
    
    return q


def projectpointsdist(K, R, t, dist, Q) :
    # K, R, t, dist and Q are, respectively, the camera matrix, rotation, 
    # translation, list of radial distortion coefficients and a list of 𝑁 3D points to be projected into the camera.
    cx = K[0,2]
    cy = K[1,2]
    alpha = 1
    beta = 0
    f = K[0,0]
    Ap = np.array([[f,0,0],[0,f,0],[0,0,1]])
    Aq = np.array([[1,beta,cx],[0,alpha,cy],[0,0,1]])
    Pp = np.matmul(Ap,np.c_[R,t])
    nQ = Q.shape[1] # number of points
    Qhomo = np.vstack((Q,np.ones(nQ))) # 3D points in homogeneous form (4 coordinates)
    pdhomo = np.matmul(Pp,Qhomo) # distorted projection coordinate
    pd = pdhomo[:2,:nQ]/pdhomo[2,:]
    r = np.sqrt(pd[0,:]**2+pd[1,:]**2) # radius
    scale = 0
    for i in range(len(dist)):
        scale += dist[i]*(r**((i+1)*2)) # radial distortion as a function of radius 
    pc = pd*(1+scale) # corrected projection coordinate
    pchomo = np.vstack((pc,np.ones(nQ)))
    q = np.matmul(Aq,pchomo).T
    q = q[:,:2]
    
    return q


def RotationMatrix(theta):
    R1 = np.array([[np.cos(theta[2]),-np.sin(theta[2]),0],[np.sin(theta[2]),np.cos(theta[2]),0],[0,0,1]])
    R2 = np.array([[np.cos(theta[1]),0,np.sin(theta[1])],[0,1,0],[-np.sin(theta[2]),0,np.cos(theta[1])]])
    R3 = np.array([[1,0,0],[0,np.cos(theta[0]),-np.sin(theta[0])],[0,np.sin(theta[0]),np.cos(theta[0])]])
    R = R1@R2@R3
    
    return R

def triangulate(q1,q2,P1,P2):
    x1 = q1[0];
    y1 = q1[1];
    x2 = q2[0];
    y2 = q2[1];
    B = np.array([P1[2,:]*x1-P1[0,:], P1[2,:]*y1-P1[1,:], P2[2,:]*x2-P2[0,:], P2[2,:]*y2-P2[1,:]])
    return B


def skewsymmetricmatrix(v):

    #Preallocate the matrix
    mat = np.zeros((3,3))

    #Use the equation.
    mat[0,1] = -v[2]
    mat[0,2] =  v[1]
    mat[1,0] =  v[2]
    mat[1,2] = -v[0]
    mat[2,0] = -v[1]
    mat[2,1] =  v[0]

    return mat


def NormalizationMatrix(points):
    #points = q[0:2,:] # without 1 in the end
    s = 1/np.std(points.reshape((-1),))
    dx = -s*np.mean(points[0,:])
    dy = -s*np.mean(points[1,:])
    T = np.array([[s,0,dx],
                   [0,s,dy],
                   [0,0,1]])
    return T


def estimateHomography(q1,q2):   
    B = []
    for i in range(3*3):
        B.append(np.kron(q1[:,i].T,skewsymmetricmatrix(q2[:,i])))
    
    B = np.array(B).reshape((q1.T.shape[1]*q2.shape[1],9)) # see week 5, slide 20

    [U,S,VT] = np.linalg.svd(B)
    V = VT.T
    Hnorm = V[:,-1].reshape((3,3)).T
    
    T1 = NormalizationMatrix(q1);
    T2 = NormalizationMatrix(q2);
    
    H = np.linalg.inv(T2)@Hnorm@T1 
    return H