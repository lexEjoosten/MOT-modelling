import torch
from fun_ctions import rot,Rot,poltocart,carttopol
from torch import Tensor
import numpy as np
from numpy import pi
import torch.nn.functional as F
import MOT_vars as vr

class particles():
    
    
    def create(N=5000 , R=0.01 , T=180 , m=1.455181063e-25 , F_l=2 , F_u=3 ): #setup to handle rubidium D2
        k_B=1.3806503e-23 #K_B in J/K
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
        particles.l=(torch.ones((N,2*F_l+1))/((2*F_l)+1)).cuda()
        particles.u=(torch.zeros((N,2*F_u+1))).cuda()

    def createuniform(N=5000 , R=0.01 , T=180 , m=1.455181063e-25 , F_l=2 , F_u=3 ): #setup to handle rubidium D2
        k_B=1.3806503e-23 #K_B in J/K
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
        
        v=(np.sqrt(2*k_B*T/m)*(v.transpose(0,1)/torch.norm(v,dim=1))).transpose(0,1)






        e=1-2*torch.zeros((N,3)).cuda().random_(0,2) #creates an N*3 matrix of ones and minus ones
        particles.x=R*e*x
        particles.v=e*v
        particles.l=(torch.ones((N,2*F_l+1))/((2*F_l)+1)).cuda()
        particles.u=(torch.zeros((N,2*F_u+1))).cuda()

    def keep(i):
        particles.x=particles.x[i,:]
        particles.v=particles.v[i,:]
        particles.l=particles.l[i,:]
        particles.u=particles.u[i,:]
        
        
        
    def createbyinp(pos,vel,F_l,F_u):
        N=len(pos[:,0])
        particles.x=Tensor(pos).cuda()
        particles.v=Tensor(vel).cuda()
        particles.l=(torch.ones((N,2*F_l+1))/((2*F_l)+1)).cuda()
        particles.u=(torch.zeros((N,2*F_u+1))).cuda()
    
    def init_track(Simlen=1000,n=0):
        x,v=torch.zeros((Simlen+1,3)).cuda(),torch.zeros((Simlen+1,3)).cuda()
        x[0],v[0]=particles.x[n],particles.v[0]
        return x,v

    def track(x,v,i,n=0):
        x[i+1]=particles.x[n]
        v[i+1]=particles.v[n]
        return x,v









class Rubidium:
    mass=1.455181063e-25 #mass given in kilograms
    l=2  #This variable defines the angular momentum of the lower state.
    u=3  #This variable defines the angular momentum of the upper state.
    gl=1/2  #This variable defines the g-factor for the lower states.
    gu=2/3  #This variable defines the g-factor for the upper states.
    Gamma=int(38.11e6)  #this is the natural decay rate in s^-1
    
    
    
    BranRat=Tensor([
        [15,5,1,0,0,0,0],
        [0,10,8,3,0,0,0],
        [0,0,6,9,6,0,0],
        [0,0,0,3,8,10,0],
        [0,0,0,0,1,5,15]
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
    

    
    
    hbar=1.055e-34 #reduced planck constant in J/s
    c=3e8 #speed of light in m/s
    dtun=vr.dtun
    A=vr.A
    BaHH=Tensor([A,A,-2*A]).cuda() #Anti-Helmholtz field
    aHHassym=rot()
    



    


    Wl=vr.wavelength
    
    LAnFr=2*pi*c/Wl

    rad=vr.rad
    
    cutoff=vr.cutoff #cutoff radius of the beams
    Is=vr.Saturation_isotropic#saturation intensity in W m^-2
    #Is=2.0
    
    Imax=50.0 #maximum beam intensity in Watt m^-2
    Kmag=LAnFr/c  #photon momentum vector magnitude (in meters per rad)
    Kmag2=Kmag**2  #momentum vector squared important later.
    gravity=Tensor([0,0,-9.81]).cuda().unsqueeze(0)  #gravity accelaration tensor in m s^-2
    
    def Intensities(x):
        #creating a vector of the distance from the center of each axis, for each particle.
        pos=torch.cat(((x[:,1]**2+x[:,2]**2).unsqueeze(1),(x[:,0]**2+x[:,2]**2).unsqueeze(1)),dim=1)
        pos=torch.cat((pos,(x[:,0]**2+x[:,1]**2).unsqueeze(1)),dim=1)
        #intensity from each beam from each direction (all of this needs to be fixed for better generality  )
        Int=torch.pow(np.e,-(torch.pow(pos,2)/Environment.rad)).repeat(1,2)

        Int=torch.index_select(Int,1,torch.LongTensor([0,3,1,4,2,5]).cuda())
        Int=Int*Environment.Imax/Environment.Is
        return Int
    
    
    
    def Veldtun(v):
        kv = torch.inner(Environment.Lk,v)
        return -kv*Environment.Kmag

    
    def Bdtun(Ml,Mu,x):
        return 8.7941e10*torch.sqrt(torch.sum(torch.square(Environment.BaHH*x),1))*(Rubidium.gl*Ml-Rubidium.gu*Mu)
    
    def eploc(x):
        R1=Environment.Brot(Environment.BaHH,x).unsqueeze(1).repeat(1,6,1,1)
        epr=Environment.LpolcartR.unsqueeze(0).repeat(x.shape[0],1,1,1)
        epc=Environment.LpolcartI.unsqueeze(0).repeat(x.shape[0],1,1,1)
        epr=torch.matmul(R1,epr)
        epc=torch.matmul(R1,epc)
        ep=carttopol([epr,epc])
        ep=ep
        return ep
        
    
        
    def Brot(B,x):
        u=F.normalize(torch.matmul(Tensor([[0,1,0],[-1,0,0],[0,0,0]]).cuda().unsqueeze(0).repeat(x.shape[0],1,1),(torch.matmul(Environment.aHHassym,(B*x).transpose(0,1)).transpose(0,1)).unsqueeze(2)).squeeze())
        W=torch.inner(Tensor([[[0,0,0],[0,0,-1],[0,1,0]],[[0,0,1],[0,0,0],[-1,0,0]],[[0,-1,0],[1,0,0],[0,0,0]]]).cuda(),u).transpose(1,2).transpose(0,1)

        phi=-torch.arccos(F.normalize(x)[:,2])
        
        Id=Tensor([[1,0,0],[0,1,0],[0,0,1]]).cuda().unsqueeze(0).repeat(x.shape[0],1,1)
        
        R=Id+(W.transpose(0,2)*torch.sin(phi)).transpose(0,2)+((torch.matmul(W,W)).transpose(0,2)*(2*torch.sin(phi/2)**2)).transpose(0,2)
        
        
        
        return R
    
    
    
    
    
    def fulldtun(Ml,Mu,dop,zee):
        return (Environment.dtun-dop*Environment.Veldtun(particles.v)-zee*Environment.Bdtun(Ml,Mu,particles.x)).transpose(0,1)

    

    
    
        
    
    

def RatesbyBeam(u,l,Pa,En,Rb,dop,zee):
    s=En.Intensities(Pa.x)
    #sprim=torch.sum(s,dim=2)
    den=(1+4*(En.fulldtun(l,u,dop,zee)/Rb.Gamma)**2+s)
    Rate=s*Rb.Gamma/2*En.eploc(Pa.x)[:,:,(1+(l-u)),0]/den
    Rate=Rate*torch.sqrt(Rb.BranRat[l+2,u+3])
    return Rate
    
    




def forward(Pa=particles,En=Environment,Ru=Rubidium,timestep=0.001,ratemults=1, dop=True, zee=True, den_sim=False, acceladj=False, grav=False):
    
    Rates=torch.zeros((Pa.x.shape[0],6,5,7)).cuda()
    #calculate rate eqns:
    for l in range(-2,3):
        for u in range(-3,4):
            if abs(l-u)<2:
                Rates[:,:,l+2,u+3]=RatesbyBeam(u,l,Pa,En,Ru,dop,zee)
        
    
    
    Rats=torch.sum(Rates,1)
    Pnr=Pa.x.shape[0]
    
    
    Br=Ru.BranRat.unsqueeze(0).repeat(Pa.x.shape[0],1,1)
    dt=timestep
    mul=ratemults
    Accel=torch.zeros(Pa.x.shape,device='cuda')
    for i in range(mul):
        RNl=torch.mul(Rates.transpose(2,3).transpose(0,2),Pa.l).transpose(0,2).transpose(2,3)
        RNu=torch.mul(Rates.transpose(0,2),Pa.u).transpose(0,2)
        Rachang=torch.sum(RNl-RNu,(2,3))


        Accel+=(Environment.hbar*Environment.Kmag/Rubidium.mass* torch.matmul(Rachang.unsqueeze(1).unsqueeze(1),Environment.Lk.unsqueeze(0).unsqueeze(3).transpose(1,2)).squeeze())/mul



        if den_sim:
            Pa.l+=(Ru.Gamma*7/5*torch.matmul(Br/21,Pa.u.unsqueeze(2)).squeeze()+torch.sum(RNu-RNl,(1,3)))*dt/mul
            Pa.u+=(-Ru.Gamma*Pa.u+torch.sum(RNl-RNu,(1,2)))*dt/mul
    
    if grav:
        Accel+=Environment.gravity.repeat(Accel.shape[0],1)
        

    if acceladj:
        Pa.v=Pa.v+Accel*dt
        Pa.x=Pa.x+Pa.v*dt
    return Accel,Rates