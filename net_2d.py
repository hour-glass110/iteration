from re import X
import torch
from torch import nn
from torch.nn.modules import linear
from torch.nn.modules.activation import Mish,  Mish

import torch.nn.functional as F

D_in,H,D_out=2,2,1

class net_2d(nn.Module):
    def __init__(self):
        super(net_2d,self).__init__()
        self.Din = nn.Linear(D_in, H)
        self.Mish1 = Mish()
        self.Linear2 = nn.Linear(H,H)
        self.Mish2 = Mish()
        self.Linear3 = nn.Linear(H,H)
        self.Mish3 = Mish()
        self.Linear4 = nn.Linear(H,H)
        self.Mish4 = Mish()
        self.Linear5 = nn.Linear(H,H)    
        self.Mish5 = Mish() 
        
    
        self.Linear6 = nn.Linear(H,H)
        self.Mish6 = Mish()
        self.Linear7 = nn.Linear(H,H)
        self.Mish7 = Mish()
        self.Linear8 = nn.Linear(H,H)
        self.Mish8 = Mish()
        self.Linear9 = nn.Linear(H,H)    
        self.Mish9 = Mish() 
        
        self.Linear10 = nn.Linear(H,H)
        self.Mish10 = Mish()
        self.Linear11 = nn.Linear(H,H)
        self.Mish11 = Mish()
        self.Linear12 = nn.Linear(H,H)
        self.Mish12 = Mish()
        self.Linear13 = nn.Linear(H,H)    
        self.Mish13 = Mish() 

        self.Linear14 = nn.Linear(H,H)
        self.Mish14 = Mish()
        self.Linear15 = nn.Linear(H,H)    
        self.Mish15 = Mish() 

        self.Dout = nn.Linear(H, D_out)  

         
        
        
    def forward(self,x):
        # x = self.models(x)
        # print(self.models)

        x = self.Din(x)
        x = self.Mish1(x)
        x = self.Linear2(x)
        x = self.Mish2(x)
        x = self.Linear3(x)
        x = self.Mish3(x) 
        x = self.Linear4(x)
        x = self.Mish4(x) 
        x = self.Linear5(x)
        x = self.Mish5(x) 

        x = self.Linear6(x)
        x = self.Mish6(x)
        x = self.Linear7(x)
        x = self.Mish7(x) 
        x = self.Linear8(x)
        x = self.Mish8(x) 
        x = self.Linear9(x)
        x = self.Mish9(x) 

        x = self.Linear10(x)
        x = self.Mish10(x)
        x = self.Linear11(x)
        x = self.Mish11(x) 
        x = self.Linear12(x)
        x = self.Mish12(x) 
        x = self.Linear13(x)
        x = self.Mish13(x) 

        x = self.Linear14(x)
        x = self.Mish14(x) 
        x = self.Linear15(x)
        x = self.Mish15(x) 
           

        x = self.Dout(x)

        return x



