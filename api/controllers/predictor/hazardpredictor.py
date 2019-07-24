#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 

"""     
This script aims create the endpoints of prediction UNISDR's hazard labels.
""" 

__author__ = "Yihan Bao"
__copyright__ =  "Copyright 2019, United Nations OICT PSGD ETT"
__credits__ = ["Kevin Bradley", "Yihan Bao", "Praneeth Nooli"]
__date__ = "08 July 2019"
__version__ = "0.1"
__maintainer__ = "Yihan Bao"
__email__ = "yihan.bao@un.org"
__status__ = "Development"

#import boto3 
import json  # Used for json file operation
import ast  # Used for evaluate a Unicode or Latin-1 encoded string containing a Python literal
import pandas as pd # Used for data operation
import pickle  # Used for read and write pickle file
import sklearn  # Used for applying machine learning packages
import os  # Used for reading, writing and updating files
import logging  # Used for saving loogings info
import sys  # Used for integrating with system
sys.path.append('../ett/')  # Natigate to ett folder path
from helper import Helper as ett_h  # ETT helper package
from transformer import Transformer as ett_t  # ett transfer package 
from constants import JobType  # ETT constants for 
from constants import RegexFilter  # ETT constants for regexfilter
from constants import OrientPara  # ETT constants for orient parameters
from constants import ColumnName  # ETT constants for column names
from constants import EncodingType  # ETT constants for encoding type
from constants import JobId  # ETT constants for job id
from constants import LabelType  # ETT constants for label types
sys.path.append('../classification/')  # Navigate to classification folder path
from unisdr import TextClassification as tc  # ETT text classification methods
from unisdr import TextModel as tm  # ETT model saving methods
from flask_restful import Resource  # Flask restful for create endpoints
from flask_restful import reqparse  # Flask restful to parse user's input parameters
from flask import request  # Flask methods for requesting binary input file
from run import app  # From run.py import application
from flask import jsonify  # Flask methods used for jsonify object
from flask import session  # Flask methods used for reading from aws
import json

# Initiate Parameter
base_folder_location = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
models = []
labels =  []
job_type = ['BATCH', 'SINGLE']
colnames = ['title', 'textData']
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

# File name
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

# Load labels - list
@app.before_first_request
def load_hazard_labels():

    abs_filename = ett_h.generate_dynamic_path([base_folder_location, LabelType.HAZARD.value, label_file_name])
    global labels
    labels = (ett_h.load_data_common_separated(abs_filename, RegexFilter.SINGLE_COMMA.value))

# load models
@app.before_first_request
def load_hazard_models():

    for label in labels:
        print("Hazard label: ", label)
        abs_filename_p = ett_h.generate_dynamic_path([base_folder_location, LabelType.HAZARD.value, models_folder, label+model_name]) 
        abs_filename_v = ett_h.generate_dynamic_path([base_folder_location, LabelType.HAZARD.value, vector_models_folder, label+vector_model_name]) 
        abs_filename_r = ett_h.generate_dynamic_path([base_folder_location, LabelType.HAZARD.value, dim_reductor_models_folder, label+dim_reductor_model_name]) 
        abs_filename_n = ett_h.generate_dynamic_path([base_folder_location, LabelType.HAZARD.value, normalizar_models_folder, label+normalizar_model_name]) 
        prediction_model = ett_h.load_model(abs_filename_p)
        vector_model = ett_h.load_model(abs_filename_v)
        dim_reductor_model = ett_h.load_model(abs_filename_r)
        normalizar_model = ett_h.load_model(abs_filename_n)
        new_model = tm(label, vector_model, prediction_model, dim_reductor_model, normalizar_model)
        global models_object
        models_object.append(new_model)

@app.route('/upload-hazard', methods=['POST'])
def upload_hazard():
    
    bytes_data = request.stream.read()
    bytes_data = ett_t.bytes_to_str(bytes_data)
    print(bytes_data)
    bytes_data = json.loads(bytes_data) 
    global input_data
    input_data = pd.DataFrame(bytes_data)
    global data_df
    data_df = ett_t.transform_data_to_dataframe(job_type, input_data, colnames)
    print(data_df)
    return "Successfully uploading hazard data"

class PredictHazard(Resource):


    def post(self):
        
        # Text classification happens here
        classification = tc(models_object, data_df, labels)
        results_df = classification.process_data()
        results_json = results_df.to_json(orient='columns')
        return results_json  # Return labels and probabilities


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
            key_file = ett_h.generate_dynamic_path([label_type[1],folder_name[i],labels[n]+file_name[i]])
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
