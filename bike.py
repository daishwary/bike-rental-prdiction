
# coding: utf-8

# ## Importing libraries

# In[2]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
import os


# In[3]:


os.chdir("C:/Users/asus/Desktop/project4")


# ## Importing data
# 

# In[4]:


bike_df = pd.read_csv("day.csv")
bike_df.info()


# ## Preprocessing

# In[5]:


# Renaming columns names to more readable names
bike_df.rename(columns={'instant':'rec_id',
                        'dteday':'datetime',
                        'holiday':'is_holiday',
                        'workingday':'is_workingday',
                        'weathersit':'weather_condition',
                        'hum':'humidity',
                        'mnth':'month',
                        'cnt':'total_count',
                        'yr':'year'},inplace=True)


# In[6]:




###########################
# Setting proper data types
###########################


# categorical variables
bike_df['season'] = bike_df.season.astype('category')
bike_df['is_holiday'] = bike_df.is_holiday.astype('category')
bike_df['weekday'] = bike_df.weekday.astype('category')
bike_df['weather_condition'] = bike_df.weather_condition.astype('category')
bike_df['is_workingday'] = bike_df.is_workingday.astype('category')
bike_df['month'] = bike_df.month.astype('category')
bike_df['year'] = bike_df.year.astype('category')


# ## Plotting

# In[7]:


# Configuring plotting visual and sizes
sns.set_style('whitegrid')
sns.set_context('talk')
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (30, 10),
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large'}

plt.rcParams.update(params)


# In[9]:


fig,ax = plt.subplots()
sns.pointplot(data=bike_df[['atemp',
                           'total_count',
                           'season']],
              x='atemp',
              y='total_count',
              hue='season',
              ax=ax)
ax.set(title="Season wise tem distribution of counts")


# In[10]:


fig,ax = plt.subplots()
sns.pointplot(data=bike_df[['humidity',
                           'total_count',
                           'weekday']],
              x='humidity',
              y='total_count',
              hue='weekday',
              ax=ax)
ax.set(title="Weekday wise hourly distribution of counts")


# In[11]:


fig,ax = plt.subplots()
sns.barplot(data=bike_df[['month',
                           'total_count']],
              x='month',
              y='total_count',
              ax=ax)
ax.set(title="Monthly distribution of counts")


# In[12]:


fig,ax = plt.subplots()
sns.barplot(data=bike_df[['season',
                           'total_count']],
              x='season',
              y='total_count',
              ax=ax)
ax.set(title="Seasonal distribution of counts")


# In[13]:


fig,ax = plt.subplots()
sns.violinplot(data=bike_df[['year',
                           'total_count']],
              x='year',
              y='total_count',
              ax=ax)
ax.set(title="Year distribution of counts")


# #### Checking for outliners:

# In[14]:


fig,(ax1,ax2) = plt.subplots(ncols=2)
sns.boxplot(data=bike_df[['total_count',
                          'casual',
                          'registered']],ax=ax1)
sns.boxplot(data=bike_df[['temp',
                          'windspeed']],ax=ax2)


# In[15]:


fig,ax = plt.subplots()
sns.boxplot(data=bike_df[['total_count',
                          'humidity']],x='humidity',y='total_count',ax=ax)
ax.set(title="Checking for outliners in humidity")


# #### Correlations

# In[16]:


corrMatt = bike_df[['temp',
                    'atemp', 
                    'humidity', 
                    'windspeed', 
                    'casual', 
                    'registered', 
                    'total_count']].corr()

mask = np.array(corrMatt)
# Turning the lower-triangle of the array to false
mask[np.tril_indices_from(mask)] = False
fig,ax = plt.subplots()
sns.heatmap(corrMatt, 
            mask=mask,
            vmax=.8, 
            square=True,
            annot=True,
            ax=ax)


# ## Feature Engineering
# Since the dataset contains multiple categorical variables, it is imperative that we encode the nominal ones before we use them in our modeling process.

# In[17]:


# Defining categorical variables encoder method
def fit_transform_ohe(df,col_name):
    """This function performs one hot encoding for the specified
column.
    Args:
        df(pandas.DataFrame): the data frame containing the mentioned column name
        col_name: the column to be one hot encoded
    Returns:
        tuple: label_encoder, one_hot_encoder, transformed column as pandas Series
    """
    # label encode the column
    le = preprocessing.LabelEncoder()
    le_labels = le.fit_transform(df[col_name])
    df[col_name+'_label'] = le_labels
    # one hot encoding
    ohe = preprocessing.OneHotEncoder()
    feature_arr = ohe.fit_transform(df[[col_name+'_label']]).toarray()
    feature_labels = [col_name+'_'+str(cls_label) for cls_label in le.classes_]
    features_df = pd.DataFrame(feature_arr, columns=feature_labels)
    return le,ohe,features_df

# given label encoder and one hot encoder objects, 
# encode attribute to ohe
def transform_ohe(df,le,ohe,col_name):
    """This function performs one hot encoding for the specified
        column using the specified encoder objects.

    Args:
        df(pandas.DataFrame): the data frame containing the mentioned column name
        le(Label Encoder): the label encoder object used to fit label encoding
        ohe(One Hot Encoder): the onen hot encoder object used to fit one hot encoding
        col_name: the column to be one hot encoded

    Returns:
        tuple: transformed column as pandas Series

    """
    # label encode
    col_labels = le.transform(df[col_name])
    df[col_name+'_label'] = col_labels
    
    # ohe 
    feature_arr = ohe.fit_transform(df[[col_name+'_label']]).toarray()
    feature_labels = [col_name+'_'+str(cls_label) for cls_label in le.classes_]
    features_df = pd.DataFrame(feature_arr, columns=feature_labels)
    
    return features_df


# In[18]:


# Divide the dataset into training and testing sets
X, X_test, y, y_test = train_test_split(bike_df.iloc[:,0:-3],
                                        bike_df.iloc[:,-1],
                                        test_size=0.33,
                                        random_state=42)
X.reset_index(inplace=True)
y = y.reset_index()

X_test.reset_index(inplace=True)
y_test = y_test.reset_index()


# In[19]:


# Encoding all the categorical features
cat_attr_list = ['season','is_holiday',
                 'weather_condition','is_workingday',
                 'weekday','month','year']
# though we have transformed all categoricals into their one-hot encodings, note that ordinal
# attributes such as hour, weekday, and so on do not require such encoding.
numeric_feature_cols = ['temp','humidity','windspeed',
                        'weekday','month','year']
subset_cat_features =  ['season','is_holiday','weather_condition','is_workingday']

###############
# Train dataset
###############
encoded_attr_list = []
for col in cat_attr_list:
    return_obj = fit_transform_ohe(X,col)
    encoded_attr_list.append({'label_enc':return_obj[0],
                              'ohe_enc':return_obj[1],
                              'feature_df':return_obj[2],
                              'col_name':col})


feature_df_list  = [X[numeric_feature_cols]]
feature_df_list.extend([enc['feature_df']                         for enc in encoded_attr_list                         if enc['col_name'] in subset_cat_features])

train_df_new = pd.concat(feature_df_list, axis=1)
print("Train dataset shape::{}".format(train_df_new.shape))
print(train_df_new.head())

##############
# Test dataset
##############
test_encoded_attr_list = []
for enc in encoded_attr_list:
    col_name = enc['col_name']
    le = enc['label_enc']
    ohe = enc['ohe_enc']
    test_encoded_attr_list.append({'feature_df':transform_ohe(X_test,
                                                              le,ohe,
                                                              col_name),
                                   'col_name':col_name})
    
    
test_feature_df_list = [X_test[numeric_feature_cols]]
test_feature_df_list.extend([enc['feature_df']                              for enc in test_encoded_attr_list                              if enc['col_name'] in subset_cat_features])

test_df_new = pd.concat(test_feature_df_list, axis=1) 
print("Test dataset shape::{}".format(test_df_new.shape))
print(test_df_new.head())


# ## Modeling

# In[20]:


X = train_df_new
y = y.total_count.values.reshape(-1,1)

lin_reg = linear_model.LinearRegression()

# using the k-fold cross validation (specifically 10-fold) to reduce overfitting affects
# cross_val_predict function returns cross validated prediction values as fitted by the model object.
predicted = cross_val_predict(lin_reg, X, y, cv=10)


# In[21]:


# Analysing residuals in our predictinos
fig,ax = plt.subplots(figsize=(15,15))
ax.scatter(y, y-predicted)
ax.axhline(lw=2,color='black')
ax.set_xlabel('Observed')
ax.set_ylabel('Residual')
ax.set_title('Residual Plot')
plt.show()


# In[22]:


# Evaluating model in cross-validation iteration

r2_scores = cross_val_score(lin_reg, X, y, cv=10)
mse = cross_val_score(lin_reg, X, y, cv=10,scoring='neg_mean_squared_error')

fig,ax = plt.subplots()
ax.plot(range(0,10),
        r2_scores)
ax.set_xlabel('Iteration')
ax.set_ylabel('R.Squared')
ax.set_title('Cross-Validation scores')
plt.show()


print("R-squared::{}".format(r2_scores))
print("MSE::{}".format(mse))


# ## Testing dataset evaluation

# In[23]:


# Predict model based on training dataset
lin_reg.fit(X,y)

# Constructing test dataset
X_test = test_df_new
y_test = y_test.total_count.values.reshape(-1,1)


y_pred = lin_reg.predict(X_test)
residuals = y_test-y_pred

fig, ax = plt.subplots(figsize=(15,15))
ax.scatter(y_test, residuals)
ax.axhline(lw=2,color='black')
ax.set_xlabel('Observed')
ax.set_ylabel('Residuals')
ax.title.set_text("Residual Plot with R-Squared={}".format(np.average(lin_reg.score(X_test,y_test))))
plt.show()

print("MSE: {}".format(metrics.mean_squared_error(y_test, y_pred)))


# In[27]:


from sklearn.tree import DecisionTreeRegressor

#Decision tree for regression
fit_DT = DecisionTreeRegressor(max_depth=2).fit(X, y)

#Apply model on test data
predictions_DT = fit_DT.predict(X_test)


# In[28]:


#Calculate MAPE
def MAPE(y_true, y_pred): 
    mape = np.mean(np.abs((y_true - y_pred) / y_true))*100
    return mape


# In[29]:


MAPE(y_test, predictions_DT)


# In[30]:


#Accuracy = 88.6%


# In[34]:


#Random Forest
from sklearn.ensemble import RandomForestRegressor

RF_model = RandomForestRegressor(n_estimators = 20).fit(X, y)


# In[35]:


RF_Predictions = RF_model.predict(X_test)


# In[37]:


#Calculate MAPE
MAPE(y_test,RF_Predictions)


# In[ ]:


#Accuracy = 94%

