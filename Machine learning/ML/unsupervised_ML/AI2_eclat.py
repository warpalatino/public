import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# APRIORI package missing - need to find one to be added via Conda Forge

# -------------------------
# UNSUPERVISED LEARNING
# Apriori


# load and pre-process data
# ------------
dataset = pd.read_csv('../data/Market_Basket_Optimisation.csv', header = None)
# print(dataset.head())
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])
print(transactions)




# run the model
# ------------
# for details on the model, see here https://scikit-learn.org/stable/auto_examples/cluster/plot_ward_structured_vs_unstructured.html
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)
results = list(rules)
print(results)



# organise results in a pandas df
# ------------
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    return list(zip(lhs, rhs, supports))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Product 1', 'Product 2', 'Support'])

## Displaying the results sorted by descending supports
resultsinDataFrame.nlargest(n = 10, columns = 'Support')



