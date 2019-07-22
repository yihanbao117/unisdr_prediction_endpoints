#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
 
"""     
This script aims create the endpoints of prediction UNISDR's theme labels.
""" 

__author__ = "Yihan Bao"
__copyright__ =  "Copyright 2019, United Nations OICT PSGD ETT"
__credits__ = ["Kevin Bradley", "Yihan Bao", "Praneeth Nooli"]
__date__ = "08 July 2019"
__version__ = "0.1"
__maintainer__ = "Yihan Bao"
__email__ = "yihan.bao@un.org"
__status__ = "Development"
 
import json  # Used for json file operation
import ast  # Used for evaluate a Unicode or Latin-1 encoded string containing a Python literal
import pandas as pd # Used for data operation
import pickle  # Used for read and write pickle file
import sklearn  # Used for applying machine learning packages
import os  # Used for reading, writing and updating files
import logging  # Used for saving loogings info
import sys  # Used for integrating with system
import mysql.connector  # Used for connecting mysql database
sys.path.append('../ett/')  # Natigate to ett folder path
from helper import Helper as ett_h  # ETT helper package
from transformer import Transformer as ett_t  # ett transfer package 
from constants import JobType  # ETT constants for 
from constants import RegexFilter  # ETT constants for regexfilter
from constants import OrientPara  # ETT constants for orient parameters
from constants import ColumnName  # ETT constants for column names
from constants import EncodingType  # ETT constants for encoding type
from constants import MySQLDB  # ETT constants for mysql database
from constants import JobId  # ETT constants for job id
from constants import LabelType  # ETT constants for label types
sys.path.append('../classification/')  # Navigate to classification folder path
from unisdr import TextClassification as tc  # ETT text classification methods
from unisdr import TextModel as tm  # ETT model saving methods
from flask_restful import Resource  # Flask restful for create endpoints
from flask_jwt_extended import jwt_required  # Flask extension to provide the jwt 
from flask_restful import reqparse  # Flask restful to parse user's input parameters
from io import StringIO  # Used to convert bytes to string
from flask import request  # Flask methods for requesting binary input file
from run import db  # From run.py import application's database
from run import app  # From run.py import application
from flask import jsonify  # Flask methods used for jsonify object
from flask import session  # Flask methods used for reading from aws

# Initiate Parameter
base_folder_location = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
models = []
labels =  []
job_type = ['BATCH', 'SINGLE']
colnames = ["title", "textData"]
data = pd.DataFrame()

# Folder name
models_folder = "models"
dim_reductor_models_folder = "dim_reductor_models"
normalizar_models_folder = "normalizar_models"
vector_models_folder = "vector_models"

# Model file name
model_name = '_model.pickle'
vector_model_name = '_vectorizer.pickle'
dim_reductor_model_name = '_dim_reductor.pickle'
normalizar_model_name = '_normalizar.pickle'

# File Name
label_file_name = 'labels.txt'
data_file_name = 'data.json'

# List of loaded models
vector_models_list = []
dim_reductor_models_list = []
normalizar_models_list = []
models_list= []

# File directory list
folder_name = [vector_models_folder, dim_reductor_models_folder, normalizar_models_folder, models_folder]
file_name = [vector_model_name, dim_reductor_model_name, normalizar_model_name, model_name]
emp_list = [vector_models_list, dim_reductor_models_list, normalizar_models_list, models_list]

# New Text Models 
models_object = []

# MySQL connection
theme_db = mysql.connector.connect(host='localhost', user='root', password='password', database='unisdr_schema_db')
theme_cursor = theme_db.cursor()

# Load labels - list
@app.before_first_request
def load_theme_labels():
    abs_filename = ett_h.generate_dynamic_path([base_folder_location, LabelType.THEME.value, label_file_name])
    global labels
    labels = (ett_h.load_data_common_separated(abs_filename, RegexFilter.SINGLE_COMMA.value))

# load models
@app.before_first_request
def load_theme_models():
    for label in labels:
        print("Theme label: ", label)
        abs_filename_p = ett_h.generate_dynamic_path([base_folder_location, LabelType.THEME.value, models_folder, label+model_name]) 
        abs_filename_v = ett_h.generate_dynamic_path([base_folder_location, LabelType.THEME.value, vector_models_folder, label+vector_model_name]) 
        abs_filename_r = ett_h.generate_dynamic_path([base_folder_location, LabelType.THEME.value, dim_reductor_models_folder, label+dim_reductor_model_name]) 
        abs_filename_n = ett_h.generate_dynamic_path([base_folder_location, LabelType.THEME.value, normalizar_models_folder, label+normalizar_model_name]) 
        prediction_model = ett_h.load_model(abs_filename_p)
        vector_model = ett_h.load_model(abs_filename_v)
        dim_reductor_model = ett_h.load_model(abs_filename_r)
        normalizar_model = ett_h.load_model(abs_filename_n)
        new_model = tm(label, vector_model, prediction_model, dim_reductor_model, normalizar_model)
        global models_object
        models_object.append(new_model)

# Used for generting job id when user uploading files
@app.before_first_request
def counting():
        counting.counter += 1
counting.counter = -1  # The first job id starts with 1 

@app.route('/upload-theme', methods=['POST'])
def upload_theme():

    # Reead byte data from postman (eg:csv)
    bytes_data = request.stream.read()
    s = ett_t.bytes_to_str(bytes_data)
    bytes_data = StringIO(s)
    global input_data
    input_data = pd.read_csv(bytes_data)

    # Used for generating Job ID
    counting()
    num = counting.counter

    # Gnenerate Job ID
    input_data[ColumnName.JOBID.value] = num

    # Insert uploaded files into MySQL database 
    for index, row in input_data.iterrows():
        sql = "INSERT INTO Theme_Predictions (job_id, title, body) VALUES (%s, %s, %s)"
        val = (row[ColumnName.JOBID.value], row[ColumnName.TITLE.value], row[ColumnName.TEXTDATA.value])
        theme_cursor.execute(sql, val)
    theme_db.commit()
    return  "saving successfully"

class FetchDataframeTheme(Resource):


    # Needing a job ID to be the acceptance and extract the data and making the predictions
    def post(self):

        # Receive parameters from users input
        parser = reqparse.RequestParser()
        parser.add_argument(ColumnName.JOBID.value, help='This field cannot be blank', required=True)
        data = parser.parse_args()  
        
        # Based on job_id selected prediction_id,title and body
        job_id = data[ColumnName.JOBID.value]
        sql = "SELECT prediction_id, title, body FROM Theme_Predictions WHERE Theme_Predictions.job_id = %s"
        val = (job_id,)
        theme_cursor.execute(sql, val)
        
        # Make prediction on data corresponding to users requested job_id
        fet_df = theme_cursor.fetchall()
        fet_df = ett_h.provision_named_data_frame(fet_df, [ColumnName.PREDID.value, ColumnName.TITLE.value, ColumnName.TEXTDATA.value])
        global prediction_id
        prediction_id = fet_df[ColumnName.PREDID.value]
        tit_id_df = fet_df[ColumnName.TITLE.value]
        bod_id_df = fet_df[ColumnName.TEXTDATA.value]
        global data_df
        data_df = ett_t.transform_data_to_dataframe(job_type, fet_df, colnames)
        return "successfully fetching the data"

class UserPredictTheme(Resource):


    #@jwt_required
    def post(self):

        #Text classification happens here
        classification = tc(models_object, data_df, labels)
        results_df = classification.process_data()
        results_df = ett_h.combined_df(prediction_id, results_df, 1)

        # Updating the MySQL database using pred_prob and pred_lab
        theme_cursor.execute("SET SQL_SAFE_UPDATES=0")
        for index, row in results_df.iterrows():
            sql = "UPDATE Theme_Predictions SET labels = %s, probabilities = %s WHERE prediction_id = %s"
            val = (row['labels'], row['probabilities'], row[ColumnName.PREDID.value])
            theme_cursor.execute(sql, val)
        theme_db.commit()
        df_json = ett_t.df_to_json(results_df, OrientPara.INDEX.value)
        return df_json

class ThemeDataUpdateDB(Resource):

    
    def post(self):
        
        #Receive parameters from users input
        parser = reqparse.RequestParser()
        parser.add_argument(ColumnName.UPDATEDLABEL.value, help='Please verifying the predicted labels', required=True)
        parser.add_argument(ColumnName.PREDID.value, help='Please provide the prediction_id', required=True)
        data = parser.parse_args()
        
        #Receive the bytesdata and transfer into dataframe
        dict_updated = data[ColumnName.UPDATEDLABEL.value]
        dict_updated = ast.literal_eval(dict_updated)
        updated_labels = pd.DataFrame(dict_updated.values())
        updated_labels.columns = [ColumnName.UPDATEDLABEL.value]
        
        #Update the actuals in mysql database
        updated_id = data[ColumnName.PREDID.value]
        for index, row in updated_labels.iterrows():
            sql = "UPDATE Theme_Predictions SET actuals = %s WHERE prediction_id = %s"
            val = (row[ColumnName.UPDATEDLABEL.value], updated_id)
            theme_cursor.execute(sql, val)
        theme_db.commit()
        return {"message": "The actuals has been updated in the Theme Prediction Database"}

"""
@app.before_first_request
def load_theme_models():
    # AWS credentails

    session = boto3.Session(
        aws_access_key_id='',
        aws_secret_access_key='')
    s3 = session.client("s3")
    # Save the preload models in list
    for n in range(len(labels)):
        print(n)
        for i in range(len(folder_name)):
            print(i)
            # 1. Load the Pickle Object  - Unpickle
            key_file = ett_h.generate_dynamic_path([LabelType.THEME.value,folder_name[i],labels[n]+file_name[i]])
            print("before accessing s3")
            print(key_file)
            obj = s3.get_object(Bucket="oict-psdg-unisdr-prediction-models-v1", Key=key_file)
            print("after accessing s3")
            serializedObject = obj['Body'].read()
            global emp_list
            single_model = pickle.loads(serializedObject)
            emp_list[i].append(single_model)
            #print('Loaded Avalanche Model')
"""
