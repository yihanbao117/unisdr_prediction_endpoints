#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    UNISDR Prevention Web Text Classification Solution

    This module is the entry point for the multi label text classification
    problem for UNISDR
"""

__author__ = "Kevin Thomas Bradley"
__copyright__ = "Copyright 2019, United Nations OICT PSGD ETT"
__credits__ = ["Kevin Bradley", "Yihan Bao", "Praneeth Nooli"]
__date__ = "11 February 2019"
__version__ = "0.1"
__maintainer__ = "Kevin Thomas Bradley"
__email__ = "bradleyk@un.org"
__status__ = "Development"

import time  # Calculcate time differences
import re  # Package used to as replace function
import numpy as np  # Mathematic caculations
import pandas as pd  # Dataframe operations
from helper import Helper as ett_h  # ETT Helper methods
from helper import InvalidLabelsToModelsError  # Custom ETT Exception
from transformer import Transformer as ett_t  # ETT Transformer methods
from cleanser import Cleanser as ett_c  # ETT Cleanser methods
from constants import RegexFilter  # ETT constants methods for RegexFilter
from constants import Language  # ETT constants methods for language type
from enum import Enum  # Custom enum for classification type
from predictor import Predictor as ett_p  # ETT Predictor methods

# Custom UNISDR Classification Type (only two currently)
class ClassificationType(Enum):


    HAZARD = "HAZARD",
    THEME = "THEME"

# Custom Model Container Class
class TextModel:


    label = None
    vector_model = None
    prediction_model = None
    dim_reductor_model = None

    def __init__(self, label, vector_model, prediction_model, dim_reductor_model, normalizar_model):
        self.label = label
        self.vector_model = vector_model
        self.prediction_model = prediction_model
        self.dim_reductor_model = dim_reductor_model
        self.normalizar_model = normalizar_model

# Text Classification Class
class TextClassification:


    # Global variable declaration TODO array of models, populate these types-DONE
    models = []  # list .pickle file
    data = pd.DataFrame()  # dataframe
    labels = []  # list

    # Processing variables
    num_of_labels = 0  # number
    output_dataFrame = pd.DataFrame()  # dataframe

    ##
    # UNISDR application constructor
    # @param models Tuple of TextModel objects
    # @param data DataFrame of the data to be processed
    # @param labels Tuple of strings of the expected labels
    # @raise InvalidLabelsToModelsError custom ETT exception for incorrect label / model sizes
    def __init__(self, models, data, labels):
        self.models = models
        self.data = data
        self.labels = labels
        self.num_of_labels = len(labels)
        if self.num_of_labels != len(models):
            raise InvalidLabelsToModelsError

    # Entry point method to actually start the
    # classification operations
    def process(self):
        self.process_data()

    # Method which acts as the builder
    def process_data(self):

        # Clean the data
        cleansed_data = self.pre_process_text_cleanse(self.data)

        # Transform the data
        transformed_data = self.pre_process_text_transform(cleansed_data)

        # Construct the output data frame
        result_label = []
        self.output_dataFrame = ett_h.provision_data_frame(transformed_data, self.num_of_labels)
        
        # Get the prediction and probabilities
        for i in range(0,self.num_of_labels):
            cores=4

            # Vectorize the data
            vectorized_data = ett_t.perform_model_transformation(self.models[i].vector_model, transformed_data)
            #vectorized_data = pd.DataFrame(vectorized_data.toarray())

            # Dimension Reduction
            dim_data = ett_t.perform_model_transformation(self.models[i].dim_reductor_model, vectorized_data) 

            # Normalize data to [0,1]
            normalized_data = ett_t.perform_model_transformation(self.models[i].normalizar_model, dim_data)
            normalized_data = pd.DataFrame(normalized_data)
            
            # Chunking Data
            chunk_size = len(normalized_data) // cores + cores
            chunks = [df_chunk for g, df_chunk in normalized_data.groupby(np.arange(len(normalized_data)) // chunk_size)]
            
            # Get the prediciton labels
            labelled_data = ett_p.perform_model_predictions_mp(self.models[i].prediction_model, chunks, cores)
            labelled_data = np.concatenate(labelled_data)

            # Get the probabilities result
            probabilities_data = ett_p.perform_model_prob_predictions_mp(self.models[i].prediction_model, chunks, cores)
            probabilities_data  = np.concatenate(probabilities_data )
            result_label.append(labelled_data)
            
            # Get the probabilties dataframe
            self.output_dataFrame[i] = pd.DataFrame(probabilities_data)[1]

        # Get the prediction dataframe
        result_label = pd.DataFrame(result_label).T
        # Format the probabilities
        result_prob = self.output_dataFrame.applymap(lambda x : "%.2f" % (100*x))
        # Get the corresponding probability
        result_prob = result_label * result_prob.astype(float)

        # Map the prediction results to actual labels (e.g: Avalanche)
        for i in range(0,self.num_of_labels):
            result_label.iloc[:,i] = result_label.iloc[:,i].map({1: self.labels[i], 0:str(0)})

        # Combined output dataframe to one columns and remove characters
        result_label = self.combined_columns_format_outputs(result_label,str(0),str,RegexFilter.SQUARE_BRACKET_AND_SINGLE_QUOTATION.value,"","labels")
        result_prob = self.combined_columns_format_outputs(result_prob,float(0),str,RegexFilter.SQUARE_BRACKET_AND_SINGLE_QUOTATION.value,"","probabilities")

        # A list of dataframe that containes all columns you want to show in UI
        result_list = [result_label,result_prob]
        results = ett_t.combine_dataframe(result_list,1)
        return results

    ##
    # Method used to change array type to dataframe and assign a column name to it
    # @param dataframe Dataframe
    # @param char_remove Characters that you want to remove
    # @param datatype Datatype, such as str,int,float
    # @param old_char Characters that you want to change
    # @param new_char New characters
    # @param colname_name New name of column
    # @returns One column of data, which including all data in original dataframe without unnecessary characters
    def combined_columns_format_outputs(self, dataframe, char_remove,
                                        datatype, old_char, new_char, column_name):
        # Change dataframe to list
        dataframe = dataframe.values.tolist()
        # Remove char(zero) in array
        dataframe = self.remove_char_array(dataframe, char_remove)
        # Change list to array
        dataframe = self.list_to_array(dataframe)
        # Change arrary to dataframe and all data type to string
        dataframe = self.change_datatype_dataframe(dataframe, datatype)
        # Remove character in dataframe
        dataframe = self.replace_column_char(dataframe[0], str(old_char), str(new_char))
        # Assign new column name
        dataframe = self.assign_name_array_dataframe(dataframe, column_name)
        return dataframe

    ##
    # Method used to remove certain characters in array
    # @param array Array format: [[a,A],[b,B]]
    # @param char_remove Characters that you want to remove
    # @returns Array of original list
    def remove_char_array(self, array, char_remove):
        for i in array:
            while char_remove in i:
                i.remove(char_remove)
        return array

    ##
    # Method used to change list to numpy array
    # @param listname List
    # @returns Array of original list
    def list_to_array(self, listname):
        listname  = np.asarray(listname)
        return listname

    ##
    # Method used to change the data type in dataframe
    # @param dataframe Dataframe
    # @param datatype Datatype, such as str,int,float
    # @returns Dataframe of certain type of data
    def change_datatype_dataframe(self, dataframe, datatype):
        dataframe = pd.DataFrame(dataframe).astype(datatype)
        return dataframe

    ##
    # Method used to repalce old character to new character
    # @param column Column(dataframe)
    # @param old_char Characters that you want to change
    # @param new_char New characters
    # @returns Column with new characters and without old characters
    def replace_column_char(self, column, old_char, new_char):
        column = column.apply(lambda x: re.sub(old_char, new_char, x))
        return column

    ##
    # Method used to change array type to dataframe and assign a column name to it
    # @param arrayname Array
    # @param colname The name of new column
    # @returns Named dataFrame of original array
    def assign_name_array_dataframe(self, arrayname, colname):
        arrayname = arrayname.to_frame(colname)
        return arrayname

    ##
    # Method used to contain the various cleansing procedures performed on the data
    # @param initial_data DataFrame of the initial data
    # @returns DataFrame of the cleaned data according to these rules
    def pre_process_text_cleanse(self, initial_data):
        # Removed all non alphanumeric characters
        nan_cleaned_data = ett_c.clean_dataframe_by_regex(initial_data, RegexFilter.NON_ALPHA_NUMERIC.value) # TODO use ENUM-DONE
        # Removed all digits
        d_cleaned_data = ett_c.clean_dataframe_by_regex(nan_cleaned_data, RegexFilter.DIGITS_ONLY.value) # TODO use ENUM-DONE
        # Remove non-English text
        l_cleaned_data = ett_c.remove_non_iso_words(d_cleaned_data, Language.ENGLISH.value)  # TODO use ENUM-DONE
        # Remove English stop words
        rew_cleaned_data = ett_c.remove_language_stopwords(l_cleaned_data, Language.ENGLISH.name) #  TODO use ENUM-DONE
        # Return the newly cleaned data
        return rew_cleaned_data # TODO use rew_cleaned_data-DONE(change nothing?)

    ##
    # Method used to contain the various transformation procedures performed on the data
    # @param cleaned_data DataFrame of the cleansed data
    # @returns DataFrame of the transformed data according to these rules
    def pre_process_text_transform(self, cleaned_data):
        # Transform text to lowercase
        l_transformed_data = ett_t.lowercase(cleaned_data)
        # Transform text to core words i.e. playing > play
        le_transformed_data = ett_t.stemming_mp(l_transformed_data)
        # Return the newly transformed data
        return le_transformed_data
