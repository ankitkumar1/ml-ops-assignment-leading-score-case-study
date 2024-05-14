'''
filename: utils.py
functions: encode_features, load_model
creator: shashank.gupta
version: 1
'''

###############################################################################
# Import necessary modules
# ##############################################################################

import mlflow
import mlflow.sklearn
import pandas as pd

import sqlite3

import os
import logging

from datetime import datetime

from Lead_scoring_inference_pipeline.constants import *

###############################################################################
# Define the function to train the model
# ##############################################################################


def encode_features():
    '''
    This function one hot encodes the categorical features present in our  
    training dataset. This encoding is needed for feeding categorical data 
    to many scikit-learn models.

    INPUTS
        db_file_name : Name of the database file 
        db_path : path where the db file should be
        ONE_HOT_ENCODED_FEATURES : list of the features that needs to be there in the final encoded dataframe
        FEATURES_TO_ENCODE: list of features  from cleaned data that need to be one-hot encoded
        **NOTE : You can modify the encode_featues function used in heart disease's inference
        pipeline for this.

    OUTPUT
        1. Save the encoded features in a table - features

    SAMPLE USAGE
        encode_features()
    '''
    conn = sqlite3.connect(DB_PATH+DB_FILE_NAME)
    if conn:
        model_input_data = pd.read_sql_query("select * from model_input",conn)
        df_encoded = pd.DataFrame(columns=ONE_HOT_ENCODED_FEATURES)
        df_placeholder= pd.DataFrame()
        for feature in FEATURES_TO_ENCODE:
            if feature in model_input_data.columns:
                encode = pd.get_dummies(model_input_data[feature])
                encode = encode.add_prefix(feature+'_')
                df_placeholder = pd.concat([df_placeholder,encode],axis=1)
            else:
                print("feature not found")
        
        for feature in ONE_HOT_ENCODED_FEATURES:
            if feature in df_placeholder.columns:
                df_encoded[feature]= df_placeholder[feature]
            if feature in model_input_data.columns:
                df_encoded[feature]=model_input_data[feature]
                
        df_encoded=df_encoded.fillna(0)
        df_encoded.to_sql('features_inference',con=conn,index=False,if_exists='replace')

    conn.close()

###############################################################################
# Define the function to load the model from mlflow model registry
# ##############################################################################

def get_models_prediction():
    '''
    This function loads the model which is in production from mlflow registry and 
    uses it to do prediction on the input dataset. Please note this function will the load
    the latest version of the model present in the production stage. 

    INPUTS
        db_file_name : Name of the database file
        db_path : path where the db file should be
        model from mlflow model registry
        model name: name of the model to be loaded
        stage: stage from which the model needs to be loaded i.e. production


    OUTPUT
        Store the predicted values along with input data into a table

    SAMPLE USAGE
        load_model()
    '''
    conn = sqlite3.connect(DB_PATH+DB_FILE_NAME)
    df_new_data = pd.read_sql_query("select * from features_inference",conn)
    load_model = mlflow.pyfunc.load_model(MODEL_PATH)
    y_pred = load_model.predict(df_new_data)
    df_new_data['app_complete_flag']=y_pred
    df_new_data.to_sql("predicted_data",conn,if_exists='replace',index=False)
    conn.close()

###############################################################################
# Define the function to check the distribution of output column
# ##############################################################################

def prediction_ratio_check():
    '''
    This function calculates the % of 1 and 0 predicted by the model and  
    and writes it to a file named 'prediction_distribution.txt'.This file 
    should be created in the ~/airflow/dags/Lead_scoring_inference_pipeline 
    folder. 
    This helps us to monitor if there is any drift observed in the predictions 
    from our model at an overall level. This would determine our decision on 
    when to retrain our model.
    

    INPUTS
        db_file_name : Name of the database file
        db_path : path where the db file should be

    OUTPUT
        Write the output of the monitoring check in prediction_distribution.txt with 
        timestamp.

    SAMPLE USAGE
        prediction_col_check()
    '''
    conn = sqlite3.connect(DB_PATH+DB_FILE_NAME)
    df_predicted = pd.read_sql_query("select * from predicted_data",conn)
    per_value=df_predicted['app_complete_flag'].value_counts(normalize=True)
    ct = datetime.now()
    text = str(ct) +" % of 1="+str(per_value[0])+ " % of 0 =" + str(per_value[1])
    with open(FILE_PATH+"prediction_distribution.txt",'a') as f:
        f.write(text +"\n")
###############################################################################
# Define the function to check the columns of input features
# ##############################################################################


def input_features_check():
    '''
    This function checks whether all the input columns are present in our new
    data. This ensures the prediction pipeline doesn't break because of change in
    columns in input data.

    INPUTS
        db_file_name : Name of the database file
        db_path : path where the db file should be
        ONE_HOT_ENCODED_FEATURES: List of all the features which need to be present
        in our input data.

    OUTPUT
        It writes the output in a log file based on whether all the columns are present
        or not.
        1. If all the input columns are present then it logs - 'All the models input are present'
        2. Else it logs 'Some of the models inputs are missing'

    SAMPLE USAGE
        input_col_check()
    '''
    conn = sqlite3.connect(DB_PATH+DB_FILE_NAME)
    df_new_data = pd.read_sql_query("select * from features_inference",conn)
    if(set(df_new_data)==set(ONE_HOT_ENCODED_FEATURES)):
        print("All the models input are present")
    else:
        print("Some of the models inputs are missing")
       
