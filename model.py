import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# import the dataframe
montreal_listing =pd.read_csv('montreal_airbnb.csv')

# clean Data
montreal_listing = montreal_listing.drop(['name','id','neighbourhood_group','host_name','last_review'], axis=1)
montreal_listing.isnull().sum()
montreal_listing.dropna(how='any',inplace=True)

#creating a sub-dataframe with no extreme values / less than 400 
sub_montreal_listing=montreal_listing[montreal_listing.price < 400]

# Features Engineering
feature_sub_montreal_listing = sub_montreal_listing.copy()
feature_sub_montreal_listing.drop(['latitude','longitude'],axis=1,inplace=True)

# Encoding categorical features (proposed 1)
categorical_features=['room_type', 'neighbourhood']

for feature in categorical_features:
    labels_ordered=feature_sub_montreal_listing.groupby([feature])['price'].mean().sort_values().index
    labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}
    feature_sub_montreal_listing[feature]=feature_sub_montreal_listing[feature].map(labels_ordered)

# Normalise Dataframe (proposed 2)
#num_features=['host_id','reviews_per_month','number_of_reviews','calculated_host_listings_count', 'minimum_nights', 'availability_365', 'price']

feature_scale=[feature for feature in feature_sub_montreal_listing.columns if feature not in ['host_id','price']]
data = pd.DataFrame()
for feature in feature_scale:
    data[feature] = (feature_sub_montreal_listing[feature] - feature_sub_montreal_listing[feature].mean())/ (feature_sub_montreal_listing[feature].std())

data.insert(loc=0, column='host_id', value=feature_sub_montreal_listing['host_id'])
data.insert(loc=1, column='price', value=feature_sub_montreal_listing['price'])
feature_sub_montreal_listing = data.copy()

# Feature selection
# Data filtering
# Filter the dataset for prices between 0 and $120
feature_sub_montreal_listing = feature_sub_montreal_listing.loc[(feature_sub_montreal_listing['price'] < 120)]

## Split data and feature slection data (proposed 1)
from sklearn.model_selection import train_test_split

x_train = feature_sub_montreal_listing.iloc[0:10000]
y_train = feature_sub_montreal_listing.iloc[0:10000]['price'].values
#y_train = np.log10(y_train)
x_test = feature_sub_montreal_listing.iloc[10000:]
y_test = feature_sub_montreal_listing.iloc[10000:]['price'].values
#y_test = np.log10(y_test)

selected_feat = ['neighbourhood', 'room_type', 'availability_365']
x_train=x_train[selected_feat]
x_test =x_test[selected_feat] 

# LR Prediction Model
#from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
#from sklearn.neighbors import KNeighborsRegressor
#from sklearn.metrics import classification_report
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import r2_score
#from sklearn.metrics import * # importer tout les metrics d'erreurs

#Prepare a Linear Regression (LR) Model
reg=LinearRegression()
reg.fit(x_train,y_train)

# Saving model to disk
pickle.dump(reg, open('model.pkl','wb')) 