#import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing datset
dataset=pd.read_csv('Market_Basket_Optimisation.csv',header=None)
transactions=[]
for i in range(0,7501):
    items=[]
    j=0
    while(str(dataset.values[i,j])!='nan'):
        items.append(str(dataset.values[i,j]))
        j=j+1
        if(j==20):
            break     
    transactions.append(items)
    
#fitting data
from eclatpy import eclat
outi=[]
sets=eclat(transactions,supp=5,zmin=2,out=outi)
#if out parameter is a list that only it return list
#otherwise it will give integer with number of pairs
#In List return tuple 
# 1> Tuple with combination of elemnts brought together
# 2> Number of times they are brough together
