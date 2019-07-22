#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
 
"""     
This script is the entry point of running the UNISDR web application, which also configuring the database and JWT here.
""" 

__author__ = "Yihan Bao"
__copyright__ =  "Copyright 2019, United Nations OICT PSGD ETT"
__credits__ = ["Kevin Bradley", "Yihan Bao", "Praneeth Nooli"]
__date__ = "08 July 2019"
__version__ = "0.1"
__maintainer__ = "Yihan Bao"
__email__ = "yihan.bao@un.org"
__status__ = "Development"

import pandas as pd
import os
import logging
import sys
from flask import Flask
from flask_restful import Api
from flask import request
from flask import jsonify
from io import StringIO
db_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),'ett')
sys.path.append(db_folder)
from transformer import Transformer as ett_t

# Save logging into local files
logFormatter = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(filename='myLogs.log', filemode='w',format=logFormatter, level=logging.DEBUG)

# The main entry point for the application
app = Flask(__name__)
app.secret_key = 'abcdefghi'
api = Api(app)

@app.route('/', methods=['GET'])
def index():
    return jsonify({'message' : 'UNISDR FLASK API started'})


# After configuring all the api and app, import resources to link to endpointss
from controllers.predictor import hazardpredictor
from controllers.predictor import themepredictor
db_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),'db')
sys.path.append(db_folder)

# Add a resource to the api
#app.add_resource(hazardpredictor.UploadHazard, '/upload/hazard')
#app.add_resource(hazardpredictor.UploadTheme, '/upload/theme')

api.add_resource(hazardpredictor.PredictHazard, '/predict/hazard')
api.add_resource(themepredictor.PredictTheme, '/predict/theme')

#api.add_resource(hazardpredictor.HazardUpdate, '/h-update-labels')
#api.add_resource(themepredictor.ThemeUpdate, '/t-update-labels')
