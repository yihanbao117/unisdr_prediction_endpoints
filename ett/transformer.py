#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Emerging Technologies Team Core Library for Transformation Functions

    Contains methods that are used to transform data in some way
"""

__author__ = "Yihan Bao"
__copyright__ = "Copyright 2019, United Nations OICT PSGD ETT"
__credits__ = ["Kevin Bradley", "Yihan Bao", "Praneeth Nooli"]
__date__ = "27 February 2019"
__version__ = "0.1"
__maintainer__ = "Yihan Bao"
__email__ = "yihan.bao@un.org"
__status__ = "Development"

import nltk  # Package used to do text analysis 
import logging  # Package used to handling errors
import pandas as pd  # Operation dataframe
from constants import JobType  # Enum for job type
from helper import Helper as ett_h  # Core package of ETT 
from textblob import Word  # Package used to do lemmatization
#nltk.data.path.append("/Users/kevin/Desktop/unisdr/nltk_data")
#from pathos.multiprocessing import ProcessingPool as Pool 
from multiprocessing import Pool  # Package used for multiprocessing
from nltk import WordNetLemmatizer  # Text data normalization-lemmatization
from nltk.stem import PorterStemmer  # Text data normalization-stemming
from sklearn import preprocessing  # Use for normalize the datasets to [0,1]

class Transformer:


    ##
    # This function simply transforms text to lowercase           
    # @param text String input text
    # @returns String lowercased text
    # Here the text actually is a dataframe with text data
    @staticmethod
    def lowercase(text):
        
        if text == pd.DataFrame:
            print("Your text data is empty, please check your input data")
        else:
            print(type(text))
            text = text.apply(lambda x: " ".join(x.lower() for x in x.split()))  
            return text

    ##
    # This function simply transforms text to lowercase           
    # @param text String input text
    # @returns String stemmed text
    # Here the text actually is a dataframe with text data
    @staticmethod
    def stemming(text):

        try:
            porter_stemmer = PorterStemmer()
            text = text.apply(lambda x: " ".join([porter_stemmer.stem(word) for word in x.split()]))
            return text
        except Exception:
            logging.error("Here occurred",exc_info=True)

    ##      
    # This function simply transforms text to lowercase           
    # @param text String input text
    # @returns String stemmed text
    # Here the text actually is a dataframe with text data
    @staticmethod
    def stemming_mp(text,cores=4):   

        if text == pd.DataFrame:
            print("Your text data is empty, please check your input data")
        else:
            with Pool(processes=cores) as pool:
                porter_stemmer = PorterStemmer()
                text = text.apply(lambda x: " ".join([porter_stemmer.stem(word) for word in x.split()]))
            return text

    ##                
    # This function simply transforms text morphologically by removing inflectional endings
    # for instance cats > cat                                          
    # @param text String input text                                    
    # @returns String rooted text
    @staticmethod                                                   
    def lemmatization(text):

        text = text.apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
        return text

    ##
    # This function is to transfer the input string into lowercase
    # @param text Text data that you want to lowercase it
    # @param core The number of cores that the computer used 
    # @return text data with all lowered case character
    # Here the text actually is a dataframe with text data
    @staticmethod
    def lemmatization_mp(text,cores=2):  

        if text == pd.DataFrame:
            print("Your text data is empty, please check your input data")
        else:
            with Pool(processes=cores) as pool:
                wlemm = WordNetLemmatizer()
                result = pool.map(wlemm.lemmatize, text)
            return result

    ##
    # This function simply calls the transform method of the model object
    # Method using these calculated parameters apply the transform 8ij ation to a particular dataset
    # @param model Object representing a model
    # @param dataFrame DataFrame to be used to transform
    # @returns concat_cols DataFrame of concatinated columns imto a single column   
    @staticmethod
    def perform_model_transformation(model, dataFrame): 

        try:
            return model.transform(dataFrame)
        except AttributeError:
            logging.error("AttributeError occurred: ", exc_info=True)
        except Exception:
            logging.error("Exception occurred", exc_info=True)
    
    ## 
    # Method used to transform the input data to a dataframe
    # @param input_data DataFrame the initial data loaded into the app 
    # @returns DataFrame
    @staticmethod
    def transform_data_to_dataframe_basic(input_data, list_column): 

        combined_data = ett_h.concatinate_data_columns(list_column, input_data)
        return combined_data

    ##
    # Method used to transform the input data to a dataframe
    # @param input_data DataFrame the initial data loaded into the app 
    # @returns DataFrame
    @staticmethod
    def transform_data_to_dataframe(job_type, input_data, list_column):   

        # SINGLE                                                        
        if job_type == JobType.BATCH.value:                             
            dataframe_single = ett_h.string_to_dataframe(input_data, list_column)
            combined_data = ett_h.concatinate_data_columns(list_column, dataframe_single)
        # BATCH
        else:  
            combined_data = ett_h.concatinate_data_columns(list_column, input_data)
        return combined_data
    
    ##
    # Method used to combine all dataframe in a list to one big dataframe based on x or y
    # @param list_dataframe A list contains all dataframe that you want to combine together
    # @param axis_num 1 or 0,axis_num = 1 for x; axis_num = 1 for y
    # @returns Combined dataframe
    @staticmethod
    def combine_dataframe(list_dataframe, axis_num): 

        for i in list_dataframe:
             result_model = pd.concat([i for i in list_dataframe], axis = axis_num)
        return result_model
    
    ##
    # Method used to combine the dataframe to json format
    # @param orient_para A parameter which regular the way to transfer dataframe
    # @param dataframe A dataframe
    # @returens Transfered json file
    @staticmethod
    def df_to_json(dataframe, orient_para):

        json = dataframe.to_json(orient=orient_para)
        return json
    
    ##
    # Method used to transfer other data type to string
    # @para data Any type of data
    # @returns String type of data
    @staticmethod
    def to_str(data):

        data = str(data)
        return data

    ##
    # Method used to transfer bytes data type to string
    # @para data Any type of data
    # @returns String type of data
    @staticmethod
    def bytes_to_str(data):

        data = str(data, "utf-8")
        return data