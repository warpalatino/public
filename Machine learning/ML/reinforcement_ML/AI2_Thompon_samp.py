import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math # for squared root calculation



# -------------------------
# REINFORCEMENT LEARNING
# Thompson sampling

# example: advertising team sends many different ads to promote an SUV; we need to register if a specific user will clikc on the ad (via 0/1)
# we assume 1 is a reward when a user clicks, while 0 is no reward
# we have to monitor 10.000 users to see which ad has the best click-through


# load data
# ------------
dataset = pd.read_csv('../data/Ads_CTR_Optimisation.csv')




# run the model
# ------------
N = 10000   # N is total users we monitor (or rounds in general)
d = 10  # d is the total variables (=ads) to monitor
ads_selected = []   #
numbers_of_selections = [0] * d     # initialize a list with counters => how many times each ad has been selected
sums_of_rewards = [0] * d       # sum the rewards for each click at each ad (one click makes the ad get one reward)
total_reward = 0

# iterate through all N customers/rounds to sum up the rewards
import random
N = 10000
d = 10
ads_selected = []
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
    total_reward = total_reward + reward



# visualizing results
# ------------
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
# plt.show()



