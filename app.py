
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

def normalise_predect_feat(features):
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

    feature_sub_montreal_listing

    # Encoding categorical features (proposed 1)
    categorical_features=['room_type', 'neighbourhood']

    for feature in categorical_features:
        labels_ordered=feature_sub_montreal_listing.groupby([feature])['price'].mean().sort_values().index
        labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}
        feature_sub_montreal_listing[feature]=feature_sub_montreal_listing[feature].map(labels_ordered)

    # Normalise Dataframe (proposed 2)
    # add the predict features at the end of data
    # Pass the row elements as key value pairs to append() function 
    feature_sub_montreal_listing = feature_sub_montreal_listing.append({'neighbourhood' : features[0], 'room_type' : features[1], 'availability_365': features[2]} , ignore_index=True)

    feature_scale=[feature for feature in feature_sub_montreal_listing.columns if feature not in ['host_id','price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count']]
    data = pd.DataFrame()
    for feature in feature_scale:
        data[feature] = (feature_sub_montreal_listing[feature] - feature_sub_montreal_listing[feature].mean())/ (feature_sub_montreal_listing[feature].std())
    
    features_data = pd.DataFrame()
    row = len(data)-1
    features_data = data.loc[row,:]
    return(features_data.to_numpy()) # to_numpy(): convert df to array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    #final_features = [np.array(int_features)]
    
    final_features_1 = np.array(int_features)
    final_features_normalise = normalise_predect_feat(final_features_1)
    final_features_data = pd.DataFrame()
    final_features_data = [np.array(final_features_normalise)]
    
    prediction = model.predict(final_features_data)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Le prix d une chambre/Appartement devrait être $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)