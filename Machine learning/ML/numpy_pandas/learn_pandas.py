import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

#----------------
#---on series

#create a list
labels = ['a', 'b', 'c']

#create another list
mylist = [10,20,30]

#create a pandas series
my_series = pd.Series(data=mylist)
print(my_series)

#create a pandas series with named labels
my_series = pd.Series(data=mylist, index=labels)
print(my_series)

#data can be mixed
my_new_list = [1, 'hey', 4.4]
my_series_two = pd.Series(my_new_list)
print(my_series_two)

#grab via the index
series1 = pd.Series([1,2,3,4], index=['USA', 'Germany', 'Italy', 'Japan'])
print(series1)
print(series1['USA'])

#perform operations via named indexes
series2 = pd.Series([1,2,3,5], index=['USA', 'Germany', 'Italy', 'Australia'])
my_sum = series1 + series2
print(my_sum)


#----------------
#---on dataframes

#build a dataframe
#first, create a matrix with data as content
seed = np.random.seed(101)
rand_matrix = np.random.rand(5,4)
# create named indexes
my_index = ['Madrid', 'London', 'Milan', 'San Juan', 'New York']
my_column = ['Sunny', 'Rainy', 'Foggy', 'Cloudy']
my_dataframe = pd.DataFrame(data=rand_matrix, index=my_index, columns=my_column)
print(my_dataframe)
print('---------')

#grab a column
grab_one = my_dataframe['Sunny']
print(grab_one)
print('---------')

#show first few lines of dataframe (head)
show = my_dataframe.head()

#create a new column
my_dataframe['Windy'] = [0.1, 0.2, 0.3, 0.4, 0.5]
print(my_dataframe)
print('---------')

#remove the new column
my_dataframe.drop('Windy', axis=1, inplace=True)
print(my_dataframe)
print('---------')

#grab a row
grab_two = my_dataframe.loc['Madrid']
print(grab_two)
print('---------')
grab_two = my_dataframe.iloc[0]
print(grab_two)
print('---------')

#grab rows and columns
grab_three = my_dataframe.loc[['Madrid', 'Milan'],['Sunny', 'Rainy']]
print(grab_three)
print('---------')
print('last')


#conditional selection
my_dataframe2 = pd.DataFrame(data=rand_matrix, index=my_index, columns=my_column)
update = my_dataframe2[my_dataframe2 > 0]
print(update)
print('---------')

#conditional selection on a single column
update = my_dataframe2[my_dataframe2['Sunny'] > 0.5]
print(update)
print('---------')

#organizing multiple conditions
cond1 = my_dataframe2['Sunny'] > 0.5
cond2 = my_dataframe2['Rainy'] < 0.5
update = my_dataframe2[(cond1) & (cond2)]
print(update)
print('---------')

#gathering information
update1 = my_dataframe2.info()
print(update1)
print('***')
update2 = my_dataframe2.dtypes
print(update2)
print('***')
update3 = my_dataframe2.describe()
print(update3)
print('***')
print('---------')


#sorting values of a certain column
sorted_df = my_dataframe.sort_values('Sunny', ascending=False)
print(sorted_df)
print('---------')

#sorting values of a certain column and grouping by a factor
#does not make sense on our df as there is nothing to group by, but just to show the ops...
grouped_df = my_dataframe.groupby('Sunny').sum()
print(grouped_df)
print('---------')

#math operations for dataframes
#creating a new column out of ops
my_dataframe['SUm_sun_rain'] = my_dataframe['Sunny'] + my_dataframe['Rainy']
print(my_dataframe)
print('---------')



#----------------
# on missing info on dataframes
df = pd.DataFrame({'A':[1,2, np.nan], 'B':[5, np.nan, np.nan], 'C':[1,2,3]})
print(df)
print('---------')

# we can drop lines with Nan by doing...
df.dropna()

#we can drop columns with NaN by doing 
df.dropna(axis=1)

#we can drop columns conditionally, only if NaN exceed a certain number per row...
# (we would drop a line below only if NaN would be at least 2 per row)
df.dropna(thresh=2)

#we can replace values with anything
df.fillna(value='fill value')

#we can replace values with a smart fill
df.fillna(df.mean())

#we can replace values with a smart fill for one single column
df['A'].fillna(df.mean())



# ------------------
#datetime object in pandas

#add timeseries as dataframe index
data = np.random.randn(3,2)
cols = ['A', 'B']
ind = pd.date_range('2020-01-01', periods=3, freq='D')

dataframe = pd.DataFrame(data, index=ind, columns=cols)
print(dataframe)

#to show a plot here we need to import matplotlib as plt, then run the show command
plot = dataframe.plot()
plt.show()

