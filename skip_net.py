from re import X, match
from matplotlib.pyplot import switch_backend
import torch
from torch import nn
from torch.nn.modules import linear
from torch.nn.modules.activation import Mish,  Mish
import numpy as np

import torch.nn.functional as F

D_in,H,D_out=1,1,1
list = ['Din','Linear2','Linear3','Linear4','Linear5','Linear6','Linear7','Linear8','Linear9','Linear10','Linear11','Linear12','Linear13','Dout'] 
class skip_net(nn.Module):
    def __init__(self):
        super(skip_net,self).__init__()
        
        self.k = 1
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
        

        # self.Linear14 = nn.Linear(H,H)
        # self.Mish14 = Mish()
        # self.Linear15 = nn.Linear(H,H)    
        # self.Mish15 = Mish() 

        self.Dout = nn.Linear(H, D_out)        
    
    # def choose_net(self):
    #     for name,para in skip_net.named_parameters():
    #         for i in range(self.k):
    #             if list[i] in name:
    #                 para.requires_grad = False
                    
        
        
    def forward(self,x):
        # x = self.models(x)
        # print(self.models)
        # self.choose_net()
        # for name,para in skip_net.named_parameters(self):
        #     para.requires_grad = True

        # for name,para in skip_net.named_parameters(self):
        #     for i in range(self.k):
        #         if list[i] in name:
        #             para.requires_grad = False
        # flag1 = np.ones([14])#控制原来输入
        # flag1[self.k-1] = 0
        # flag2 = np.zeros([14])#控制新的输入
        # flag2[self.k-1] = 1
        # temp = x

        # x = self.Din(x * flag1[0]  + temp * flag2[0])
        # x = self.Mish1(x)

        # x = self.Linear2(x * flag1[1]  + temp * flag2[1])
        # x = self.Mish2(x)
        # x = self.Linear3(x * flag1[2]  + temp * flag2[2])
        # x = self.Mish3(x) 
        # x = self.Linear4(x * flag1[3]  + temp * flag2[3])
        # x = self.Mish4(x) 
        # x = self.Linear5(x * flag1[4]  + temp * flag2[4])
        # x = self.Mish5(x) 

        # x = self.Linear6(x * flag1[5]  + temp * flag2[5])
        # x = self.Mish6(x)
        # x = self.Linear7(x * flag1[6]  + temp * flag2[6])
        # x = self.Mish7(x) 
        # x = self.Linear8(x * flag1[7]  + temp * flag2[7])
        # x = self.Mish8(x) 
        # x = self.Linear9(x * flag1[8]  + temp * flag2[8])
        # x = self.Mish9(x) 

        # x = self.Linear10(x * flag1[9]  + temp * flag2[9])
        # x = self.Mish10(x)
        # x = self.Linear11(x * flag1[10]  + temp * flag2[10])
        # x = self.Mish11(x) 
        # x = self.Linear12(x * flag1[11]  + temp * flag2[11])
        # x = self.Mish12(x) 
        # x = self.Linear13(x * flag1[12]  + temp * flag2[12])
        # x = self.Mish13(x) 

        # # x = self.Linear14(x)
        # # x = self.Mish14(x) 
        # # x = self.Linear15(x)
        # # x = self.Mish15(x) 
           

        # x = self.Dout(x * flag1[13]  + temp * flag2[13])

        

        # return x


        if self.k == 1 :
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
        
        # x = self.Linear14(x)
        # x = self.Mish14(x) 
        # x = self.Linear15(x)
        # x = self.Mish15(x) 
           

            x = self.Dout(x)

            return x
        elif self.k==2:
            
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
        
        # x = self.Linear14(x)
        # x = self.Mish14(x) 
        # x = self.Linear15(x)
        # x = self.Mish15(x) 
           

            x = self.Dout(x)

            return x
        elif self.k==3:
            
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
        
        # x = self.Linear14(x)
        # x = self.Mish14(x) 
        # x = self.Linear15(x)
        # x = self.Mish15(x) 
           

            x = self.Dout(x)

            return x
        elif self.k==4:
          
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
        
        # x = self.Linear14(x)
        # x = self.Mish14(x) 
        # x = self.Linear15(x)
        # x = self.Mish15(x) 
           

            x = self.Dout(x)

            return x

        elif self.k==5:
            
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
        
        # x = self.Linear14(x)
        # x = self.Mish14(x) 
        # x = self.Linear15(x)
        # x = self.Mish15(x) 
           

            x = self.Dout(x)

            return x
        elif self.k==6:
            
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
        
        # x = self.Linear14(x)
        # x = self.Mish14(x) 
        # x = self.Linear15(x)
        # x = self.Mish15(x) 
           

            x = self.Dout(x)

            return x
        elif self.k==7:
           
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
        
        # x = self.Linear14(x)
        # x = self.Mish14(x) 
        # x = self.Linear15(x)
        # x = self.Mish15(x) 
           

            x = self.Dout(x)

            return x
        elif self.k==8:
            
            
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
        
        # x = self.Linear14(x)
        # x = self.Mish14(x) 
        # x = self.Linear15(x)
        # x = self.Mish15(x) 
           

            x = self.Dout(x)

            return x
        elif self.k==9:
            
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
        
        # x = self.Linear14(x)
        # x = self.Mish14(x) 
        # x = self.Linear15(x)
        # x = self.Mish15(x) 
           

            x = self.Dout(x)

            return x
        elif self.k == 10:
            x = self.Linear10(x)
            x = self.Mish10(x)
            x = self.Linear11(x)
            x = self.Mish11(x) 
            x = self.Linear12(x)
            x = self.Mish12(x) 
            x = self.Linear13(x)
            x = self.Mish13(x) 
        
        # x = self.Linear14(x)
        # x = self.Mish14(x) 
        # x = self.Linear15(x)
        # x = self.Mish15(x) 
           

            x = self.Dout(x)
            return x
        elif self.k==11:
            x = self.Linear11(x)
            x = self.Mish11(x) 
            x = self.Linear12(x)
            x = self.Mish12(x) 
            x = self.Linear13(x)
            x = self.Mish13(x) 
        
        # x = self.Linear14(x)
        # x = self.Mish14(x) 
        # x = self.Linear15(x)
        # x = self.Mish15(x) 
           

            x = self.Dout(x)
            return x
        elif self.k==12:
            
            x = self.Linear12(x)
            x = self.Mish12(x) 
            x = self.Linear13(x)
            x = self.Mish13(x) 
        
        # x = self.Linear14(x)
        # x = self.Mish14(x) 
        # x = self.Linear15(x)
        # x = self.Mish15(x) 
           

            x = self.Dout(x)
            return x
        elif self.k==13:
            
            x = self.Linear13(x)
            x = self.Mish13(x) 
        
        # x = self.Linear14(x)
        # x = self.Mish14(x) 
        # x = self.Linear15(x)
        # x = self.Mish15(x) 
           

            x = self.Dout(x)
            return x
        else:
            
        # x = self.Linear14(x)
        # x = self.Mish14(x) 
        # x = self.Linear15(x)
        # x = self.Mish15(x) 
           

            x = self.Dout(x)
            return x
    def GetK(self,k):
        self.k = k

    # def loadthemodel(self):
        
    #     net.load_state_dict(torch.load("data2d_1.pth"))
