import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math # for squared root calculation



# -------------------------
# REINFORCEMENT LEARNING
# UCB - upper confidence bound

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
for n in range(0, N):
    ad = 0  # starting point: monitor from ad 0 (to reach the tenth ad => see second for loop to go through the 10 ads)
    max_upper_bound = 0 # confidence level: initialized at zero, but can grow
    # loop through the 10 different ad types
    for i in range(0, d):
        # check if an ad has been selected by a user
        if (numbers_of_selections[i] > 0):
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]  # avg reward is nothing else than the mean reward in the distribution
            # get the confidence interval according to UCB formula (see notes)
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i  # define the upper tail range in the reward distribution, as we will have to select the ad with the max UCB
        else:
            upper_bound = 1e400 # upper bound for the ad not selected it yet => we make it an incredibly high value = 1 with 400 zeros
        # let's check if the upper_bound is larger than max, make the max equal to the largest value
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i # i is the index selected in the loop, so we make an ad the selected ad
    ads_selected.append(ad) # add the selected ad to the list of se;ected ads
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1   # increment the selected ad to go through one by one
    reward = dataset.values[n, ad]  # access a specific dataset value (either 0 or 1) to see if the ad was cliocked, and thus which reward is granted
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward  # incrementing the reward per each ad
    total_reward = total_reward + reward    # incrementing the total reward



# visualizing results
# ------------
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
# plt.show()



