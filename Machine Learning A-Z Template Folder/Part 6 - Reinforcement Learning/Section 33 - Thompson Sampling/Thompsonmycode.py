# Upper Confidence Bound
"""
Problem
We are given d add and we have to find out which add is best to show and which gives highest click on add

the dataset we have teels which user will click on which add.

In real case scenario this will be dynamic hence we are using the dataset just to check whether prediction
is right or not
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing Thompson Sampling
import random
N = 10000
d = 10
ads_selected = []
number_of_reward_1=[0]*d
number_of_reward_0=[0]*d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_beta = 0
    for i in range(0,d):
        #finding value randomly from beat distribution
        beta=random.betavariate(number_of_reward_1[i]+1,number_of_reward_0[i]+1)
        if beta > max_beta:
            max_beta = beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    #updating reward
    if reward==1:
        number_of_reward_1[ad]=number_of_reward_1[ad]+1
    else:
        number_of_reward_0[ad]=number_of_reward_0[ad]+1
    total_reward = total_reward + reward

# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()