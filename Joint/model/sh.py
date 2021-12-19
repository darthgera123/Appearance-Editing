import math, cv2
import numpy as np
import sys
import torch

def y_0_0_(colors, theta, phi):
    K = np.sqrt( 1.0/(np.pi) ) * 1.0/2.0
    return K * colors

#-----------------------------------------------------------------------------------------------

def y_1_n1_(colors, theta, phi):
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)

    K = np.sqrt( 3.0/(4*np.pi) )
    return K * y * colors

def y_1_0_(colors, theta, phi):
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)

    K = np.sqrt( 3.0/(4*np.pi) )
    return K * z * colors

def y_1_p1_(colors, theta, phi):
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)

    K = np.sqrt( 3.0/(4*np.pi) )
    return K * x * colors

#-----------------------------------------------------------------------------------------------

def y_2_n2_(colors, theta, phi):
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)

    K = np.sqrt( 15.0/(np.pi) ) * 1.0/2.0
    return K * x * y * colors

def y_2_n1_(colors, theta, phi):
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)

    K = np.sqrt( 15.0/(np.pi) ) * 1.0/2.0
    return K * y * z * colors

def y_2_0_(colors, theta, phi):
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)

    K = np.sqrt( 5.0/(16*np.pi) )
    return K * ( -torch.pow(x, 2)-torch.pow(y, 2)+2*torch.pow(z, 2) ) * colors

def y_2_p1_(colors, theta, phi):
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)

    K = np.sqrt( 15.0/(4*np.pi) )
    return K * z * x * colors

def y_2_p2_(colors, theta, phi):
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)

    K = np.sqrt( 15.0/(16*np.pi) )
    return K * (torch.pow(x, 2)-torch.pow(y, 2)) * colors
