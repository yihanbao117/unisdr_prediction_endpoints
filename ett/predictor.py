#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Emerging Technologies Team Core Library for Predictions

    Contains methods that are called to perform certain predictive functions on models
"""

__author__ = "Yihan Bao"
__copyright__ = "Copyright 2019, United Nations OICT PSGD ETT"
__credits__ = ["Kevin Bradley", "Yihan Bao", "Praneeth Nooli"]
__date__ = "11 February 2019"
__version__ = "0.1"
__maintainer__ = "Yihan Bao"
__email__ = "yihan.bao@un.org"
__status__ = "Development"

import logging  # Package used to handling errors
import pandas as pd  # Package used to operate dataframe
import numpy as np  # Pakcage used for scientific computing 
from pathos.multiprocessing import ProcessingPool as Pool  # Package used to multiprocessing the function

class Predictor:


    ##
    # This function calls the predict function on the model
    # @param model Object of type model to perform prediction on
    # @param dataFrame DataFrame of the cleansed and transformed data
    # @returns DataFrame housing 0 OR 1
    @staticmethod
    def perform_model_predictions(model, dataFrame):
        
        try:
            return model.predict
        except Exception:
            logging.error("Error occurred", exc_info=True)
    
    ##
    # This function calls the predict function on the model
    # @param model Object of type model to perform prediction on
    # @param dataFrame DataFrame of the cleansed and transformed data
    # @param cores The number of cores used for multiprocessing 
    # @returns DataFrame housing 0 OR 1
    @staticmethod
    def perform_model_predictions_mp(model, dataFrame, cores):

        try:
            with Pool(processes=cores) as pool:
                result = pool.map(model.predict, dataFrame)
                return result
        except Exception:
            logging.error("Error occurred", exc_info=True)
    
    ##
    # This function calls the predict probabilities function on the model
    # @param model Object of type model to perform proba prediction on
    # @param dataFrame DataFrame of the cleansed and transformed data
    # @returns DataFrame housing 0.0 > 1.0
    @staticmethod
    def perform_model_prob_predictions(model, dataFrame):
        try:
            return model.predict_proba
        except Exception:
            logging.error("Error occurred", exc_info=True)
    
    ##
    # This function calls the predict probabilities function on the model
    # @param model Object of type model to perform proba prediction on
    # @param dataFrame DataFrame of the cleansed and transformed data
    # @param cores The number of cores used for multiprocessing 
    # @returns DataFrame housing 0.0 > 1.0
    @staticmethod
    def perform_model_prob_predictions_mp(model, dataFrame, cores):

        try:
            with Pool(processes=cores) as pool:
                result = pool.map(model.predict_proba, dataFrame)
                return result
        except Exception:
            logging.error("Error occurred", exc_info=True)