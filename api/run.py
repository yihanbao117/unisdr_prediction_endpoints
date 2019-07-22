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
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager
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

# Configure the database
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:password@localhost/unisdr_schema_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'some-secret-string'
db = SQLAlchemy(app)

@app.before_first_request
def create_tables():
    db.create_all()

# Configure the Json Web Tockenregistration
app.config['JWT_SECRET_KEY'] = 'jwt-secret-string'

# Initialize JWT by passing app instance to JWTManager class
jwt = JWTManager(app)
app.config['JWT_BLACKLIST_ENABLED'] = True
app.config['JWT_BLACKLIST_TOKEN_CHECKS'] = ['access', 'refresh']

@jwt.token_in_blacklist_loader
def check_if_token_in_blacklist(decrypted_token):
    jti = decrypted_token['jti']
    return database.RevokedTokenModel.is_jti_blacklisted(jti)

# After configuring all the api and app, import resources to link to endpointss
from controllers.resources import resources
from controllers.predictor import hazardpredictor
from controllers.predictor import themepredictor
db_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),'db')
sys.path.append(db_folder)
import database

# Add a resource to the api
api.add_resource(resources.UserRegistration, '/registration')
api.add_resource(resources.UserLogin, '/login')
api.add_resource(resources.AllUsers, '/user-table')
api.add_resource(hazardpredictor.UserPredictHazard, '/predict/hazard')
api.add_resource(themepredictor.UserPredictTheme, '/predict/theme')
api.add_resource(resources.UserLogoutAccess, '/logout/access')
api.add_resource(resources.UserLogoutRefresh, '/logout/refresh') 
api.add_resource(resources.TokenRefresh, '/token/refresh')
# api.add_resource(hazardpredictor.HazardSavedToDB, '/h-result-db')
# api.add_resource(themepredictor.ThemeSavedToDB, '/t-result-db')
api.add_resource(hazardpredictor.HazardDataUpdateDB, '/h-update-labels')
api.add_resource(themepredictor.ThemeDataUpdateDB, '/t-update-labels')
api.add_resource(hazardpredictor.FetchDataframeHazard, '/fetch-hazard-data')
api.add_resource(themepredictor.FetchDataframeTheme, '/fetch-theme-data')