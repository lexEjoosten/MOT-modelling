import torch
from fun_ctions import Rot,poltocart,carttopol
from torch import Tensor
import numpy as np
from numpy import pi
import torch.functional as F

class particles():
    
    
    def create(N, R, T, m, F_l, F_u):
        k_B=1.3806503e-23
        lamb=np.sqrt((2*k_B*T)/m)
        
        #picking a point on a quadrant on the surface of the sphere
        z=torch.zeros((N)).cuda().uniform_()
        theta=torch.arcsin(z).cuda()
        phi=torch.zeros((N)).cuda().uniform_(0,np.pi/2)
        x=torch.zeros(N,3).cuda()
        x[:,0]=torch.sqrt(1-z**2)*torch.cos(phi)
        x[:,1]=torch.sqrt(1-z**2)*torch.sin(phi)
        x[:,2]=z
        
        v=torch.zeros((N,3)).to('cuda').uniform_(0,1)
        v=-(lamb/1.2)*torch.arctanh(v) #mapping from uniform to positive Max.boltz. distribution (see notes)
        
        e=1-2*torch.zeros((N,3)).cuda().random_(0,2) #creates an N*3 matrix of ones and minus ones
        particles.x=R*e*x
        particles.v=e*v
        particles.l=(torch.ones((N,2*F_l+1))/(2*(F_l+1))).cuda()
        particles.u=(torch.zeros((N,2*F_u+1))).cuda()
        
    def createbyinp(pos,vel,F_l,F_u):
        N=len(pos[:,0])
        particles.x=Tensor(pos).cuda()
        particles.v=Tensor(vel).cuda()
        particles.l=(torch.ones((N,2*F_l+1))/(2*(F_l+1))).cuda()
        particles.u=(torch.zeros((N,2*F_u+1))).cuda()

class Rubidium:
    mass=1.455181063e-25 #mass given in kilograms
    l=2  #This variable defines the angular momentum of the lower state.
    u=3  #This variable defines the angular momentum of the upper state.
    gl=1/2  #This variable defines the g-factor for the lower states.
    gu=2/3  #This variable defines the g-factor for the upper states.
    Gamma=38.11e6  #this is the natural decay rate in s^-1
    
    
    BranRat=Tensor([
        [15/21,5/21,1/21,0,0,0,0],
        [0,10/21,8/21,3/21,0,0,0],
        [0,0,6/21,9/21,6/21,0,0],
        [0,0,0,3/21,8/21,10/21,0],
        [0,0,0,0,1/21,5/21,15/21]
    ]).cuda() #This tensor encodes the branching ratio/CB coeff, which are used for calculating the transition rates.
    
    
    

class Environment():            
    #define lasers by intensity in their k-vector. (i.e. give directionality and orientation.)
    Lk=torch.zeros(6,3).cuda()
    for i in range(0,6):
        #define 6 orth. laser unit k-vectors.
        Lk[i,int(i/2)]=2*((i+1)%2)-1
        
    Lpol=torch.zeros(6,3).cuda()
    for i in range(0,4):
        #define polarization using (ep+,pi,ep-)
        Lpol[i,0]=1
    for i in range(4,6):
        #z has opposite polarization to k-vector, +z --> ep-
        Lpol[i,0]=1
        
    
    rinit=Rot(Tensor([0,0,pi/2,-pi/2,0,pi]).cuda(),Tensor([-pi/2,pi/2,0,0,0,0]).cuda(),Tensor([0,0,0,0,0,0]).cuda()) #this initializes the rotation tensor
    Lpolcart=poltocart(Lpol.unsqueeze(2))
    LpolcartR=torch.matmul(rinit,Lpolcart[0])
    LpolcartI=torch.matmul(rinit,Lpolcart[1])
    

    
    
    dtun=2*np.pi*12e6 #laser power detunement in rad/s
    A=0.15 #Magnetic field gradient in T/m
    B=Tensor([A,A,-2*A]).cuda() #
    LAnFr=2.4141913346e15  #Laser angular frequency (neednt be exact)
    rad=20000 #gaussian radius of the beams
    cutoff=10 #cutoff radius of the beams
    hbar=1.055e-34 #reduced planck constant in J/s
    c=3e8 #speed of light in m/s
    Is=hbar*4*pi**3*Rubidium.Gamma*LAnFr**3/(27*c**2) #saturation intensity in W m^-2
    #Is=2.0
    
    Imax=50.0 #maximum beam intensity in Watt m^-2
    Kmag=LAnFr/c  #photon momentum vector magnitude
    Kmag2=Kmag**2  #momentum vector squared important later.
    gravity=Tensor([0,0,-9.81]).cuda().unsqueeze(0)  #gravity accelaration tensor in m s^-2
    
    def Intensities(x):
        pos=torch.cat(((x[:,1]**2+x[:,2]**2).unsqueeze(1),(x[:,0]**2+x[:,2]**2).unsqueeze(1)),dim=1)
        pos=torch.cat((pos,(x[:,0]**2+x[:,1]**2).unsqueeze(1)),dim=1)
        Int=torch.pow(np.e,-(torch.pow(pos,2)/Environment.rad)).repeat(1,2)
        Int=torch.index_select(Int,1,torch.LongTensor([0,3,1,4,2,5]).cuda())
        Int=Int*Environment.Imax/Environment.Is
        return Int
    
    
    
    def Veldtun(v):
        kv = torch.inner(Environment.Lk,v)
        return -kv*Environment.Kmag
    
    
    def Bdtun(Ml,Mu,x):
        return 8.7941e10*torch.sqrt(torch.sum(torch.square(Environment.B*x),1))*(Rubidium.gl*Ml-Rubidium.gu*Mu)
    
    def eploc(x):
        R1=Environment.Brot(Environment.B,x).unsqueeze(1).repeat(1,6,1,1)
        epr=Environment.LpolcartR.unsqueeze(0).repeat(x.shape[0],1,1,1)
        epc=Environment.LpolcartI.unsqueeze(0).repeat(x.shape[0],1,1,1)
        epr=torch.matmul(R1,epr)
        epc=torch.matmul(R1,epc)
        ep=carttopol([epr,epc])
        ep=ep
        return ep
        
    
        
    def Brot(B,x):
        u=F.normalize(torch.matmul(Tensor([[0,1,0],[-1,0,0],[0,0,0]]).cuda().unsqueeze(0).repeat(x.shape[0],1,1),(B*x).unsqueeze(2)).squeeze())
        W=torch.inner(Tensor([[[0,0,0],[0,0,-1],[0,1,0]],[[0,0,1],[0,0,0],[-1,0,0]],[[0,-1,0],[1,0,0],[0,0,0]]]).cuda(),u).transpose(1,2).transpose(0,1)

        phi=-torch.arccos(F.normalize(x)[:,2])
        
        Id=Tensor([[1,0,0],[0,1,0],[0,0,1]]).cuda().unsqueeze(0).repeat(x.shape[0],1,1)
        
        R=Id+(W.transpose(0,2)*torch.sin(phi)).transpose(0,2)+((torch.matmul(W,W)).transpose(0,2)*(2*torch.sin(phi/2)**2)).transpose(0,2)
        
        
        
        return R
    
    
    
    
    
    def fulldtun(Ml,Mu):
        return (Environment.dtun-Environment.Veldtun(particles.v)-Environment.Bdtun(Ml,Mu,particles.x)).transpose(0,1)
    

    
    
        
    
    

def RatesbyBeam(u,l,Pa,En,Rb):
    s=En.Intensities(Pa.x)
    den=(1+4*(En.fulldtun(l,u)/Rb.Gamma)**2+s)
    Rate=s*Rb.Gamma/2*En.eploc(Pa.x)[:,:,(1+(l-u)),0]/den
    Rate=Rate*torch.sqrt(Rb.BranRat[l+2,u+3])
    return Rate
    




