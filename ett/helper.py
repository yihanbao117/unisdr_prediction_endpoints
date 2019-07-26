#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Emerging Technologies Team Core Library for General Helper Functions
    
    Contains general helper based functions classes such as Enums and Exceptions
"""

__author__ = "Yihan Bao"
__copyright__ = "Copyright 2019, United Nations OICT PSGD ETT"
__credits__ = ["Kevin Bradley", "Yihan Bao", "Praneeth Nooli"]
__date__ = "27 February 2019"
__version__ = "0.1"
__maintainer__ = "Yihan Bao"
__email__ = "yihan.bao@un.org"
__status__ = "Development"

import pandas as pd  # DataFrame management
import logging  # Error Handling
import numpy as np  # Mathematical calcs
import pickle  # Used for loading models
import sklearn  # Used for model generation via pickle
from enum import Enum  # Used for custom Enums
from constants import Encoding  # Used to identify character encoding
from flask_restful import reqparse  # Used to receive users' input from UI

# Base Error class
class Error(Exception):


    """ETT Base class for other custom exceptions"""
    pass

# Custom Error Class
class InvalidLabelsToModelsError(Error):


   """The number of labels does not match the number of expected corresponding models"""
   pass
   
class Helper:

    
    ##
    # This function is used to load a CSV file based on the 
    # filename and return this output.
    # @param filename The source file, as an string.
    # @return A DataFrame of the CSV file data.
    # @see OSError
    # @see Exception
    @staticmethod
    def load_csv(filename):

        try:
            return pd.read_csv(filename, encoding=Encoding.LATIN_1.value)
        except OSError:
            logging.error("OSError occurred", exc_info=True)
        except Exception:
            logging.error("Exception occurred", exc_info=True)
    
    ##
    # This function is used to load a json file based on the 
    # filename and return this output.
    # @param filename The source file, as an string.
    # @return A DataFrame of the json file data.
    # @see OSError
    # @see Exception
    @staticmethod
    def load_json(filename):

        try:
            return pd.read_json(filename)
        except OSError:
            logging.error("OSError occurred", exc_info=True)
        except Exception:
            logging.error("Exception occurred", exc_info=True)
    
    ##
    # This function is used to load a Model based on the 
    # filename and return this object.
    # @param filename The source file, as an string.
    # @return An Object which is the model.
    @staticmethod
    def load_model(filename):

        try:
            with open(filename, 'rb') as model_file:
                return pickle.load(model_file)
        except OSError:
            logging.error("OSError occurred", exc_info=True)
        except Exception:
            logging.error("Exception occurred", exc_info=True)
     
    ## 
    # This function is used to load data based on the delimiter
    # @param filename String file path
    # @param delChar Character delimiter
    # @returns Tuple of values
    @staticmethod # Change the function  #  The outpurt is one list
    def load_data_common_separated(filename, delChar):

        try: 
            text_file = open(filename,'r')
            return text_file.read().split(delChar)
        except OSError:
            logging.error("OSError occurred", exc_info=True)
        except Exception:
            logging.error("Exception occurred", exc_info=True)
    
    ##
    # This function is used to create an empty DataFrame matching
    # the dimensions of the data expected to populate it
    # @param dataFrame DataFrame initial dataFrame
    # @param num Number of columns
    # @returns DataFrame with the correct dimensions 
    @staticmethod
    def provision_data_frame(dataFrame, num):

        try:
            provisioned_data = pd.DataFrame(np.zeros((len(dataFrame), num)))
            return provisioned_data
        except:
            if (dataFrame == None) | (num == None):
                logging.error("NameError occurred:", exc_info=True)
            else:
                logging.error("Exception occurred:", exc_info=True)
    
    ##
    # This function is used to create an empty DataFrame without any data in the dataframe
    # @param A list that contains the column names
    # @returns an empty DataFrame with correct column name
    @staticmethod
    def create_empty_df(colname):

        try:
            empty_df = pd.DataFrame(columns=colname)
            return empty_df
        except:
            logging.error("Exception occurred", exc_info=True)
    
    ##
    # Method used to create a DataFrame with the column names specified
    # @param dataFrame DataFrame of strings for each part of the path
    # @param colnames Tuple of strings with the column names
    # @returns DataFrame with the columns provisioned
    @staticmethod
    def provision_named_data_frame(dataFrame, colnames):

        try:
            provisioned_named_data = pd.DataFrame(dataFrame, columns=colnames)
            return provisioned_named_data
        except:
            if (dataFrame==None) | (colnames==None):
                logging.error("NameError occurred:", exc_info=True)
            else:
                logging.error("Exception occurred:", exc_info=True)
    
    ##
    # Method used to build a dynamic filepath
    # @param parts List of strings for each part of the path  
    # @returns String which is the combined file path
    @staticmethod
    def generate_dynamic_path(parts):

        try:
            if len(parts) > 1:
                return '/'.join(parts)
        except Exception:
                logging.error("Exception occurred", exc_info=True)
    
    ## 
    # Method used to string to dataframe
    # @param input_data A string
    # @param list_colname A list contains all the column names
    # @returns A dataframe with column name
    @staticmethod
    def string_to_dataframe(input_data, list_colname):

        try:
            input_data = pd.DataFrame(input_data, columns=list_colname)
            return input_data
        except:
            if (input_data == pd.DataFrame) | (list_colname == []):
                logging.error("NameError occurred:", exc_info=True)
            else:
                logging.error("Exception occurred:", exc_info=True)
    
    ##
    # This function simply concatinates columns 
    # @param colnames Tuple of column names as Strings
    # @param data DataFrame of the original data to subset and concat
    # @returns concat_cols DataFrame of concatinated columns imto a single columne
    @staticmethod
    def concatinate_data_columns(colnames, data):

        try:
            concat_cols = data[colnames].apply(lambda x: ''.join(x), axis=1)
            return concat_cols
        except:
            if (colnames == None) | (data == None):
                logging.error("NameError occurred:", exc_info=True)
            else:
                logging.error("Exception occurred:", exc_info=True)

    ##
    # This function used for saving dataframe to MySQL database
    # @para dataframe A dataframe that you want to save into the MySQL database
    # @para engine_para The engine that the connecting to your database
    # @para command_para The command of how to deal with the new record in the table
    # @para index_para The parameter to control the True or False of index
    # @returns Saving dataframe to MySQL
    @staticmethod
    def df_to_db(dataframe, database_name, engine_para, command_para, index_para):  

        dataframe.to_sql(database_name, con=engine_para, if_exists=command_para, index=index_para)  
        return "save to database"
    
    ##
    # This function used for combininig two dataframe together into one dataframe
    # @para dataframe1 The engine that the connecting to your database
    # @para dataframe2 The command of how to deal with the new record in the table
    # @para axis_para The parameter to control the True or False of index
    # @returns One dataframe combined both dataframe1 and dataframe2
    @staticmethod
    def combined_df(dataframe1,dataframe2, axis_para):

        combined_df = pd.concat([dataframe1,dataframe2], axis=axis_para)
        return combined_df