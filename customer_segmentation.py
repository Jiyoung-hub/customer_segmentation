#!/usr/bin/env python
# coding: utf-8

# ## Read data

# In[1]:


import pandas as pd


# In[2]:


data = pd.read_csv('transactions_n100000.csv')


# In[3]:


data.head()


# ## Data preprocessing

# In[4]:


# deleting duplicates
df = data.drop_duplicates(subset="ticket_id")
df.head()


# In[5]:


# create a pivot table for items
items = pd.pivot_table(data, index='ticket_id', columns='item_name', values='item_count').fillna(0)
df = df.merge(items, on=['ticket_id','ticket_id'])


# In[6]:


df.head()


# In[7]:


# converting order_timestamp to 'datetime' type
from datetime import datetime
df['date'] = df['order_timestamp'].apply(lambda row: datetime.strptime(str(row), '%Y-%m-%d %H:%M:%S'))


# In[8]:


df = df.drop(columns = ['order_timestamp','item_name','item_count'])
df.head()


# In[9]:


import calendar
df['weekday'] =  [calendar.day_name[x.weekday()] for x in df['date']]
df['hour'] = df['date'].dt.hour


# In[10]:


def time_of_day(row):
    if 7 < row['hour'] < 12:
        return 'Morning'
    elif 12 <= row['hour'] < 18:
        return  'Afternoon'
    elif 18 <= row['hour'] <= 20:
        return 'Dinner'
    elif 20 < row['hour'] <= 23:
        return 'Late-night'
    elif 0 <= row['hour'] <= 6:
        return 'Late-night'

df['time_of_day'] =  df.apply(lambda row: time_of_day(row), axis=1)


# In[11]:


df['time_of_day'].value_counts()


# In[12]:


df = df.drop(columns = ['date','hour','lat','long'])
df.head()


# In[13]:


df['location'] = df['location'].astype('category')
cat = ['location', 'weekday', 'time_of_day']
num = ['burger','fries','salad','shake']
X_cat = df[cat] 

# Creating dummy variable dataframe from categorical variables.
data_X = pd.get_dummies(X_cat)
data_X = df[num].join(data_X)
data_X = data_X.set_index(df['ticket_id'])


# In[14]:


data_X.head()


# ## Split the data (train-test) & Scaling

# In[16]:


# split the data
from sklearn import model_selection
train,test = model_selection.train_test_split(data_X, test_size=0.3, random_state = 0)


# In[17]:


# scaling
from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()
train_scaled = minmax.fit_transform(train)
test_scaled = minmax.fit_transform(test)


# In[18]:


train_scaled = pd.DataFrame(train_scaled, columns = train.columns,index= train.index)
test_scaled = pd.DataFrame(test_scaled,columns = test.columns,index= test.index)


# ## Kproto Clustering

# In[19]:


import numpy as np
from kmodes.kprototypes import KPrototypes
kproto = KPrototypes(n_clusters=3, init='Cao')
clusters = kproto.fit_predict(train_scaled, categorical=[4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])


# In[20]:


kproto.cost_


# In[21]:


clusters_t = kproto.predict(test_scaled, categorical=[4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])


# ## Validation

# In[22]:


train_with_id = train.copy()
train_with_id['cluster_id'] = clusters
train_with_id.cluster_id.value_counts() / len(train)


# In[23]:


test_with_id = test.copy()
test_with_id['cluster_id'] = clusters_t
test_with_id.cluster_id.value_counts() / len(test)


# ## Interpretation of the clusters 

# In[24]:


train_with_id.groupby('cluster_id').mean()[['weekday_Friday', 'weekday_Monday',
       'weekday_Saturday', 'weekday_Sunday', 'weekday_Thursday',
       'weekday_Tuesday', 'weekday_Wednesday']]


# In[25]:


train_with_id.groupby('cluster_id').mean()[['time_of_day_Afternoon',
       'time_of_day_Dinner', 'time_of_day_Late-night', 'time_of_day_Morning']]


# In[26]:


test_with_id.groupby('cluster_id').mean()[['burger','fries','salad','shake','location_1','location_2','location_3',
                                           'location_4','location_5','location_6','location_7','location_8','location_9']]


# ## Choosing optimal K

# In[27]:


data_X_re = data_X.drop(columns=['weekday_Friday', 'weekday_Monday',
       'weekday_Saturday', 'weekday_Sunday', 'weekday_Thursday',
       'weekday_Tuesday', 'weekday_Wednesday'])
train,test = model_selection.train_test_split(data_X_re, test_size=0.3, random_state = 0)
train_scaled = minmax.fit_transform(train)
test_scaled = minmax.fit_transform(test)
train_scaled = pd.DataFrame(train_scaled, columns = train.columns,index= train.index)
test_scaled = pd.DataFrame(test_scaled,columns = test.columns,index= test.index)


# In[28]:


# elbow method
import matplotlib.pyplot as plt
cost = []
for num_clusters in list(range(1,6)):
    kproto = KPrototypes(n_clusters=num_clusters, init='Cao')
    kproto.fit_predict(train_scaled, categorical=[4,5,6,7,8,9,10,11,12,13,14,15,16])
    cost.append(kproto.cost_)
    print(kproto.cost_)


# In[30]:


import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.axes()

x = list(range(1,6))
y = cost
xticks=list(range(1,6))
ticklabels = ['1','2','3','4','5']
plt.xticks(xticks, ticklabels)

ax.plot(x, y,color='blue', linestyle='dashed', marker='o',
     markerfacecolor='black', markersize=7);


# # (Final) DataFrame with cluster_id

# In[31]:


ids_train = pd.DataFrame(clusters, index = train.index, columns=['cluster_id'])


# In[32]:


ids_test = pd.DataFrame(clusters_t, index = test.index, columns=['cluster_id'])


# In[34]:


ids = pd.concat([ids_train, ids_test],axis=0,sort=False)
ids.reset_index(level=0, inplace=True)
ids.head()


# In[35]:


df_t = df.merge(ids,left_on = 'ticket_id', right_on = 'ticket_id')
df_t.head()


# In[36]:


df_t.cluster_id.value_counts() / len(df)


# ## Looking at each segment

# ### segment 1 (mostly location 4,7,9)

# In[37]:


condition = df_t['cluster_id'] == 0
segment1 = df_t.loc[condition,:]


# In[38]:


segment1.ticket_id.nunique()


# In[39]:


segment1.head()


# In[43]:


segment1[['burger','fries','salad','shake']].mean()


# In[44]:


segment1.time_of_day.value_counts()


# ### segment 2 (mostly location 2,6)

# In[45]:


condition = df_t['cluster_id'] == 1
segment2 = df_t.loc[condition,:]


# In[46]:


segment2.ticket_id.nunique()


# In[47]:


segment2[['burger','fries','salad','shake']].mean()


# In[48]:


segment2.time_of_day.value_counts()


# ### segment 3 (mostly location 1,3,5,8)

# In[49]:


condition = df_t['cluster_id'] == 2
segment3 = df_t.loc[condition,:]


# In[50]:


segment3.ticket_id.nunique()


# In[51]:


segment3.time_of_day.value_counts()


# ### 3 clusters

# In[54]:


all_encoded = pd.concat([train_with_id, train_with_id], axis=0)


# In[56]:


all_encoded.groupby('cluster_id').mean()[['burger','fries','salad','shake','location_1','location_2','location_3',
                                           'location_4','location_5','location_6','location_7','location_8','location_9']]

