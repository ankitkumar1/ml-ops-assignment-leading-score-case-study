'''
filename: utils.py
functions: encode_features, get_train_model
creator: shashank.gupta
version: 1
'''

###############################################################################
# Import necessary modules
# ##############################################################################

import pandas as pd
import numpy as np

import sqlite3
from sqlite3 import Error

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from Lead_scoring_training_pipeline.constants import *
import logging
###############################################################################
# Define the function to encode features
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
       

    OUTPUT
        1. Save the encoded features in a table - features
        2. Save the target variable in a separate table - target


    SAMPLE USAGE
        encode_features()
        
    **NOTE : You can modify the encode_featues function used in heart disease's inference
        pipeline from the pre-requisite module for this.
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
        df_features = df_encoded.drop('app_complete_flag',axis=1)
        df_target = df_encoded['app_complete_flag']
        df_features.to_sql('features',con=conn,index=False,if_exists='replace')
        df_target.to_sql('target',con=conn,index=False,if_exists='replace')

    conn.close()
###############################################################################
# Define the function to train the model
# ##############################################################################

def get_trained_model():
    '''
    This function setups mlflow experiment to track the run of the training pipeline. It 
    also trains the model based on the features created in the previous function and 
    logs the train model into mlflow model registry for prediction. The input dataset is split
    into train and test data and the auc score calculated on the test data and
    recorded as a metric in mlflow run.   

    INPUTS
        db_file_name : Name of the database file
        db_path : path where the db file should be


    OUTPUT
        Tracks the run in experiment named 'Lead_Scoring_Training_Pipeline'
        Logs the trained model into mlflow model registry with name 'LightGBM'
        Logs the metrics and parameters into mlflow run
        Calculate auc from the test data and log into mlflow run  

    SAMPLE USAGE
        get_trained_model()
    '''
    conn = sqlite3.connect(DB_PATH+DB_FILE_NAME)
    if conn:
        X = pd.read_sql("select * from features",conn)
        y = pd.read_sql("select * from target",conn)
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
        mlflow.set_tracking_uri(TRACKING_URI)
        try:
            logging.info("creating mlflow experiment")
            mlflow.create_experiment(EXPERIMENT_NAME)
        except:
            pass
        logging.info("setting mlflow experiment")
        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(run_name=EXPERIMENT_NAME) as run:
            clf = lgb.LGBMClassifier()
            clf.set_params(**model_config)
            clf.fit(X_train,y_train)
            mlflow.sklearn.log_model(sk_model=clf,artifact_path="models",registered_model_name='LightGBM')
            mlflow.log_params(model_config)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_pred,y_test)
            auc = roc_auc_score(y_pred,y_test)
            mlflow.log_metric('test_accouracy',acc)
            mlflow.log_metric('test_auc',auc)
            runID = run.info.run_uuid
            print("Inside MLflow Run with id {}".format(runID))
   
