"""
Problem
We are given d add and we have to find out which add is best to show and which gives highest click on add

the dataset we have teels which user will click on which add.

In real case scenario this will be dynamic hence we are using the dataset just to check whether prediction
is right or not
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Ads_CTR_Optimisation.csv')

#implementing UCB
import math
d=10
N=10000
ad_selected=[]
number_of_times_selected=[0]*d
sum_of_rewards=[0]*d
total_rewards=0
for n in range(0,N):
    maximum_confidence=0
    ad_index=0
    #finding upper bound and which bound to choose
    for i in range(0,d):
        if(number_of_times_selected[i]>0):
            average_reward=sum_of_rewards[i]/number_of_times_selected[i]
            confidence_interval=math.sqrt(((1.5)*math.log(n+1))/number_of_times_selected[i])
            upper_bound=average_reward+confidence_interval
        else:
            upper_bound=1000000000
        if(upper_bound>maximum_confidence):
            maximum_confidence=upper_bound
            ad_index=i
    ad_selected.append(ad_index)
    number_of_times_selected[ad_index]=number_of_times_selected[ad_index]+1
    sum_of_rewards[ad_index]=sum_of_rewards[ad_index]+dataset.values[n,ad_index]
    total_rewards=total_rewards+dataset.values[n,ad_index]
    
plt.hist(ad_selected)
plt.title("Ad Selceted")
plt.xlabel("Ad")
plt.ylabel("Number of time selected")
plt.show()