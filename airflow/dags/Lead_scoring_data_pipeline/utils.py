##############################################################################
# Import necessary modules and files
# #############################################################################


import pandas as pd
import os
import sqlite3
from sqlite3 import Error
from Lead_scoring_data_pipeline.constants import DB_FILE_NAME, DB_PATH, DATA_DIRECTORY, INTERACTION_MAPPING,NOT_FEATURES, INDEX_COLUMNS_TRAINING, INDEX_COLUMNS_INFERENCE
from Lead_scoring_data_pipeline.mapping.city_tier_mapping import city_tier_mapping
from Lead_scoring_data_pipeline.mapping.significant_categorical_level import *
###############################################################################
# Define the function to build database
# ##############################################################################

def build_dbs():
    '''
    This function checks if the db file with specified name is present 
    in the /Assignment/01_data_pipeline/scripts folder. If it is not present it creates 
    the db file with the given name at the given path. 


    INPUTS
        DB_FILE_NAME : Name of the database file 'utils_output.db'
        DB_PATH : path where the db file should exist  


    OUTPUT
    The function returns the following under the given conditions:
        1. If the file exists at the specified path
                prints 'DB Already Exists' and returns 'DB Exists'

        2. If the db file is not present at the specified loction
                prints 'Creating Database' and creates the sqlite db 
                file at the specified path with the specified name and 
                once the db file is created prints 'New DB Created' and 
                returns 'DB created'


    SAMPLE USAGE
        build_dbs()
    '''
    db_file_path = os.path.join(DB_PATH, DB_FILE_NAME)
    os.makedirs(DB_PATH, exist_ok=True)
    if os.path.exists(db_file_path):
        print('DB Already Exists')
        return 'DB Exists'
    else:
        # Create the database file
        print('Creating Database')
        conn = sqlite3.connect(db_file_path)
        conn.close()
        print('New DB Created')
        return 'DB created'

    
def check_if_table_has_value(cnx,table_name):
    check_table = pd.read_sql(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';", cnx).shape[0]
    if check_table == 1:
        return True
    else:
        return False
###############################################################################
# Define function to load the csv file to the database
# ##############################################################################

def load_data_into_db():
    '''
    Thie function loads the data present in data directory into the db
    which was created previously.
    It also replaces any null values present in 'toal_leads_dropped' and
    'referred_lead' columns with 0.


    INPUTS
        DB_FILE_NAME : Name of the database file
        DB_PATH : path where the db file should be
        DATA_DIRECTORY : path of the directory where 'leadscoring.csv' 
                        file is present
        

    OUTPUT
        Saves the processed dataframe in the db in a table named 'loaded_data'.
        If the table with the same name already exsists then the function 
        replaces it.


    SAMPLE USAGE
        load_data_into_db()
    '''
    # Build connection string
    conn_string = os.path.join(DB_PATH, DB_FILE_NAME)
    conn = sqlite3.connect(conn_string)
    data = pd.read_csv(os.path.join(DATA_DIRECTORY, 'leadscoring_inference.csv'), index_col=[0])
    data['total_leads_droppped'] = data['total_leads_droppped'].fillna(0)
    data['referred_lead']= data['referred_lead'].fillna(0)
    if not check_if_table_has_value(conn,'loaded_data'):
        print('test')
        data.to_sql(name="loaded_data",con=conn,if_exists='replace',index=False)
    conn.close()

###############################################################################
# Define function to map cities to their respective tiers
# ##############################################################################

    
def map_city_tier():
    '''
    This function maps all the cities to their respective tier as per the
    mappings provided in the city_tier_mapping.py file. If a
    particular city's tier isn't mapped(present) in the city_tier_mapping.py 
    file then the function maps that particular city to 3.0 which represents
    tier-3.


    INPUTS
        DB_FILE_NAME : Name of the database file
        DB_PATH : path where the db file should be
        city_tier_mapping : a dictionary that maps the cities to their tier

    
    OUTPUT
        Saves the processed dataframe in the db in a table named
        'city_tier_mapped'. If the table with the same name already 
        exsists then the function replaces it.

    
    SAMPLE USAGE
        map_city_tier()

    '''
    # Build connection string
    db_file_path = f"{DB_PATH}/{DB_FILE_NAME}"
    conn = sqlite3.connect(db_file_path)
    if not check_if_table_has_value(conn,'city_tier_mapped'):
        df = pd.read_sql_query("select * from loaded_data" ,conn)
        df["city_tier"] = df["city_mapped"].map(city_tier_mapping)
        df['city_tier']=df['city_tier'].fillna(3.0)
        df=df.drop('city_mapped',axis=1)
        df.to_sql(name="city_tier_mapped",con=conn,if_exists='replace',index=False)
    conn.close()

###############################################################################
# Define function to map insignificant categorial variables to "others"
# ##############################################################################


def map_categorical_vars():
    '''
    This function maps all the insignificant variables present in 'first_platform_c'
    'first_utm_medium_c' and 'first_utm_source_c'. The list of significant variables
    should be stored in a python file in the 'significant_categorical_level.py' 
    so that it can be imported as a variable in utils file.
    

    INPUTS
        DB_FILE_NAME : Name of the database file
        DB_PATH : path where the db file should be present
        list_platform : list of all the significant platform.
        list_medium : list of all the significat medium
        list_source : list of all rhe significant source

        **NOTE : list_platform, list_medium & list_source are all constants and
                 must be stored in 'significant_categorical_level.py'
                 file. The significant levels are calculated by taking top 90
                 percentils of all the levels. For more information refer
                 'data_cleaning.ipynb' notebook.
  

    OUTPUT
        Saves the processed dataframe in the db in a table named
        'categorical_variables_mapped'. If the table with the same name already 
        exsists then the function replaces it.

    
    SAMPLE USAGE
        map_categorical_vars()
    '''
    db_file_path = f"{DB_PATH}/{DB_FILE_NAME}"
    conn = sqlite3.connect(db_file_path)
    if not check_if_table_has_value(conn,"categorical_variables_mapped"):
            df = pd.read_sql("select * from city_tier_mapped",conn)
            new_df=df[~df['first_platform_c'].isin(list_platform)]
            new_df['first_platform_c']="others"
            old_df = df[df['first_platform_c'].isin(list_platform)]
            df = pd.concat([new_df,old_df])

            new_df=df[~df['first_utm_medium_c'].isin(list_medium)]
            new_df['first_utm_medium_c']="others"
            old_df = df[df['first_utm_medium_c'].isin(list_medium)]
            df = pd.concat([new_df,old_df])

            new_df=df[~df['first_utm_source_c'].isin(list_source)]
            new_df['first_utm_source_c']="others"
            old_df = df[df['first_utm_source_c'].isin(list_source)]
            df = pd.concat([new_df,old_df])
            
            df = df.drop_duplicates()
            df.to_sql("categorical_variables_mapped",con=conn,if_exists='replace',index=False)
    conn.close()

##############################################################################
# Define function that maps interaction columns into 4 types of interactions
# #############################################################################
def interactions_mapping():
    '''
    This function maps the interaction columns into 4 unique interaction columns
    These mappings are present in 'interaction_mapping.csv' file. 


    INPUTS
        DB_FILE_NAME: Name of the database file
        DB_PATH : path where the db file should be present
        INTERACTION_MAPPING : path to the csv file containing interaction's
                                   mappings
        INDEX_COLUMNS_TRAINING : list of columns to be used as index while pivoting and
                                 unpivoting during training
        INDEX_COLUMNS_INFERENCE: list of columns to be used as index while pivoting and
                                 unpivoting during inference
        NOT_FEATURES: Features which have less significance and needs to be dropped
                                 
        NOTE : Since while inference we will not have 'app_complete_flag' which is
        our label, we will have to exculde it from our features list. It is recommended 
        that you use an if loop and check if 'app_complete_flag' is present in 
        'categorical_variables_mapped' table and if it is present pass a list with 
        'app_complete_flag' column, or else pass a list without 'app_complete_flag'
        column.

    
    OUTPUT
        Saves the processed dataframe in the db in a table named 
        'interactions_mapped'. If the table with the same name already exsists then 
        the function replaces it.
        
        It also drops all the features that are not requried for training model and 
        writes it in a table named 'model_input'

    
    SAMPLE USAGE
        interactions_mapping()
    '''
    db_file_path = f"{DB_PATH}/{DB_FILE_NAME}"
    conn = sqlite3.connect(db_file_path)
    print("saving into model_input")
    if not check_if_table_has_value(conn,'interactions_mapped'):
            df = pd.read_sql("select * from categorical_variables_mapped",conn)
            df = df.drop_duplicates()
            df_event_mapping = pd.read_csv(INTERACTION_MAPPING,index_col=[0])
            # check if app_complete_flag is in df.columns
            df_unpivot=pd.melt(df,id_vars=INDEX_COLUMNS_TRAINING,var_name='interaction_type',value_name='interaction_value')
            df_unpivot['interaction_value'] = df_unpivot['interaction_value'].fillna(0)
            df=pd.merge(df_unpivot,df_event_mapping,how='left',on='interaction_type')
            df = df.drop('interaction_type',axis=1)
            df_pivot = df.pivot_table(values='interaction_value', index=INDEX_COLUMNS_INFERENCE, columns='interaction_mapping', aggfunc='sum')
            df_pivot = df_pivot.reset_index()
            df_pivot.to_sql('interactions_mapped',con=conn,if_exists='replace',index=False)     
            df_model_input = df_pivot.drop(NOT_FEATURES,axis=1)
            print("saving into model_input")
            df_model_input.to_sql('model_input',con=conn,if_exists='replace',index=False)
            
            
    conn.close()                              

    
