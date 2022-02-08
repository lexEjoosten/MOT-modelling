from torch import Tensor
import numpy as np
from numpy import cos,sin,pi,sqrt
import torch







def rot(a,b,c):
    a=Tensor([[cos(a),-sin(a),0],[sin(a),cos(a),0],[0,0,1]]).cuda()
    b=Tensor([[cos(b),0,sin(b)],[0,1,0],[-sin(b),0,cos(b)]]).cuda()
    c=Tensor([[1,0,0],[0,cos(c),-sin(c)],[0,sin(c),cos(c)]]).cuda()
    return torch.mm(torch.mm(a,b),c)

def Rot(c,b,a):
    A=torch.stack([
        torch.stack([torch.cos(a),-torch.sin(a),torch.zeros(len(a)).cuda()]),
        torch.stack([torch.sin(a),torch.cos(a),torch.zeros(len(a)).cuda()]),
        torch.stack([torch.zeros(len(a)).cuda(),torch.zeros(len(a)).cuda(),torch.ones(len(a)).cuda()])
    ]).transpose(0,2).transpose(1,2)
    
    B=torch.stack([
        torch.stack([torch.cos(b),torch.zeros(len(b)).cuda(),torch.sin(b)]),
        torch.stack([torch.zeros(len(b)).cuda(),torch.ones(len(b)).cuda(),torch.zeros(len(b)).cuda()]),
        torch.stack([-torch.sin(b),torch.zeros(len(b)).cuda(), torch.cos(b)])
    ]).transpose(0,2).transpose(1,2)
    
    C=torch.stack([
        torch.stack([torch.ones(len(b)).cuda(),torch.zeros(len(b)).cuda(),torch.zeros(len(b)).cuda()]),
        torch.stack([torch.zeros(len(b)).cuda(),torch.cos(c),-torch.sin(c)]),
        torch.stack([torch.zeros(len(b)).cuda(),torch.sin(c),torch.cos(c)])
    ]).transpose(0,2).transpose(1,2)
    
    return torch.matmul(A,torch.matmul(B,C))




def WignerD(B):
    U = torch.stack([
                torch.stack([torch.cos(B/2)**2, -torch.sin(B)/sqrt(2), torch.sin(B/2)**2]),
                torch.stack([torch.sin(B)/sqrt(2), torch.cos(B), -torch.sin(B)/sqrt(2)]),
                torch.stack([torch.sin(B/2)**2, torch.sin(B)/sqrt(2), torch.cos(B/2)**2])]).transpose(0,2)
    return U


def poltocart(ep):
    epr=torch.matmul(Tensor([[-1/sqrt(2),0,1/sqrt(2)],[0,0,0],[0,1,0]]).cuda().unsqueeze(0).repeat(ep.shape[0],1,1),ep)
    epc=torch.matmul(Tensor([[0,0,0],[1/sqrt(2),0,1/sqrt(2)],[0,0,0]]).cuda().unsqueeze(0).repeat(ep.shape[0],1,1),ep)
    
    
    
    return epr,epc



def carttopol(ep):
    epr=ep[0]
    epc=ep[1]
    epr1=torch.matmul(Tensor([[-1/sqrt(2),0,0],[0,0,1],[1/sqrt(2),0,0]]).cuda().unsqueeze(0).unsqueeze(1).repeat(epr.shape[0],epr.shape[1],1,1),epr)
    epr2=torch.matmul(Tensor([[0,1/sqrt(2),0],[0,0,0],[0,1/sqrt(2),0]]).cuda().unsqueeze(0).unsqueeze(1).repeat(epr.shape[0],epr.shape[1],1,1),epc)
    
    epc1=torch.matmul(Tensor([[-1/sqrt(2),0,0],[0,0,1],[1/sqrt(2),0,0]]).cuda().unsqueeze(0).unsqueeze(1).repeat(epr.shape[0],epr.shape[1],1,1),epc)
    epc2=torch.matmul(Tensor([[0,-1/sqrt(2),0],[0,0,0],[0,-1/sqrt(2),0]]).cuda().unsqueeze(0).unsqueeze(1).repeat(epr.shape[0],epr.shape[1],1,1),epr)
    
    epr=epr1+epr2
    epc=epc1+epc2
    
    ep=torch.norm(torch.cat((epr.unsqueeze(0),epc.unsqueeze(0)),0),dim=0)
    return(ep)

    
    
    
    
def Mvect_outer(vector, Matrix):
    vector*Matrix
    
    

    
    
E=Tensor([pi/2,pi,2*pi,pi/2]).cuda()
x=Tensor([[1,0,0],[-1,0,0],[0,1,0],[0,0,1]]).cuda().unsqueeze(2)
x=poltocart(x)


def RatesbyBeam(u,l,Pa,En,Rb):
    s=En.Intensities(Pa.x)
    den=(1+4*(En.fulldtun(l,u)/Rb.Gamma)**2+s)
    Rate=s*Rb.Gamma/2*En.eploc(Pa.x)[:,:,(1+(l-u)),0]/den
    Rate=Rate*torch.sqrt(Rb.BranRat[l+2,u+3])
    return Rate
    







    


    