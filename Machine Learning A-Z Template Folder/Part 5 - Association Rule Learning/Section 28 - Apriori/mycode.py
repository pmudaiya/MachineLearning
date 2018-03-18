#importing librarieimport numpy as np
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
    
"""
#Instructor code
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])
    """
#fitting data
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# seeing rules
results = list(rules)

results_list = []
for i in range(0, len(results)):
    results_list.append('RULE:\t' + str(results[i][0]) + '\nSUPPORT:\t' + str(results[i][1])+'\nLHS:\t' + str(results[i][2][0][0]) + '\nRHS:\t' + str(results[i][2][0][1])+'\nConfidence:\t' + str(results[i][2][0][2]) + '\nLift:\t' + str(results[i][2][0][3]))
    
    
    
