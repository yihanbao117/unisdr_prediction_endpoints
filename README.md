# UNISDR Text Classification Flask API

## Description
This reporsitory contains the Flask Api to wrap up the text classification project for unisdr (UN Office of Disaster Risk Reduction). This project attempts to improve the efficiency of classifying articles on UNISDR website for Office for Disaster Risk Reduction. This project's main objective is to build a Web Prevention Tool for UNISDR.  

Currently the main focus is to allow:  

* The endpoint to register users for the system ;
* The endpoint for user login for the system; 
* The endpoint to uploading binary file for hazard and theme labels;
* The endpoint to fetch the data based on users' input job ID;
* The endpoint to make the prediction for the hazard and theme labels;
* The endpoint to receive users' updated labels(actural labels) and save into MySQL database;
* The endpoint to access all the users registration information.
 
The following are the main features of the prediction endpoint:

* Loading data, labels and models;
* Cleaning and transforming text data into numerical data;
* Geting the prediction output and probabilities for each article.

## Dependencies

There are a number of project dependencies required to develop and operate the UNISDR Web Prevention Tool.

The following list details project dependencies:

* (IDE) Visual Studio Code Version 1.31;
* (PACKAGE) pyenchant 2.0.0;
* (PACKAGE) nltk 3.4;
* (CORPUS) nltk stopwords corpus;
* (PACKAGE) pandas 0.24.1;
* (PACKAGE) pathos 0.2.3;
* (PACKAGE) multiprocess 0.70.7;
* (PACKAGE) numpy 1.16.1;
* (PACKAGE) scikit-learn 0.20.1;
* (PACKAGE) textblob 0.15.2;
* (CORPUS) textblob corpus;
* (PACKAGE) re build-in;
* (PACKAGE) logging build-in;
* (PACKAGE) pickle build-in;
* (PACKAGE) spark-parser 1.8.7;
* (PACKAGE) SQLAlchemy 1.3.5;
* (PACKAGE) PyJWT 1.7.1;
* (PACKAGE) mysql-connector-python 8.0.15;
* (PACKAGE) Flask 1.0.3;
* (PACKAGE) Flask-Cors 3.0.8;
* (PACKAGE) Flask-RESTful 0.3.7;

## Getting Started

* Clone the repository into your machine;
* Install all the above dependencies with correct Python version;
* Create "models" and "vector_models", "normalizar_models"  and "dim_reductor_models" folder in both ./unisdr/hazard and /unisdr/theme path and save all four types of models in right place;
* Run run.py to initiate the server.

## Documentation

Please refer to the following documents for this project:
* Project Initiation Document (PID) located at <Team_Directory>/projects/<Project_Name>/documents/;
* Design Document located at <Team_Directory>/projects/<Project_Name>/documents/;
* Analysis Document located at <Team_Directory>/projects/<Project_Name>/documents/.

## Author
* OICT/PSGD/ETT | ett@un.org
# unisdr_prediction_endpoints
