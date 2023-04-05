# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 13:11:28 2023

@author: Blaise Ezeokeke
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import stats
"""
Importing the libraries: Pandas for structuring, 
Pyplot module of the Matplotlib Library for Visualisation,
Stats module of the Scipy Library for statistical functions
skew to identify measure of assymmetry 
kurtosis to identify "peakedness" of the distribution 
"""


#Define a function to read data into Pandas dataframe and Transpose the data

def ClimateData(filename):
    
    
    """
    
    Load and transform world bank climate data from a CSV file.

    Args:
        filename (str): The name of the CSV file containing the world bank climate data.

    Returns:
        tuple: A tuple of two dataframes. The first dataframe contains the raw world bank climate data,
            while the second dataframe contains a transposed and cleaned version of the data.

    Raises:
        FileNotFoundError: If the input filename does not exist in the file system.
        ValueError: If the input filename does not have the .csv file extension.

    try:
        # Load the world bank climate data into Pandas dataframe, skip 4 empty rows
        world_bank = pd.read_csv(filename, skiprows=4)

        # Transpose the Climate Dataframe
        world_bank_T = pd.DataFrame.transpose(world_bank)

        # Drop undesirable columns
        world_bank_T = world_bank_T.drop(['Country Code', 'Indicator Code', 'Unnamed: 66'], axis=0)
        world_bank_T.columns = world_bank_T.iloc[0]

        # Determining the second dataframe
        world_bank2_T = world_bank_T[1:]

        # Return the two dataframes
        return world_bank, world_bank2_T

    except FileNotFoundError:
        raise FileNotFoundError(f"{filename} not found in file system.")
    except ValueError:
        raise ValueError("Input file must have .csv file extension.")

    """
    
    
    #Load the world bank climate data into Pandas dataframe, skip 4 empty rows
    world_bank = pd.read_csv('world_bank_data.csv', skiprows=4) 
    
    #Transpose the Climate Dataframe
    world_bank_T = pd.DataFrame.transpose(world_bank)
    
    #Drop undesirable columns
    world_bank_T = world_bank_T.drop(['Country Code', 'Indicator Code', 'Unnamed: 66'], axis=0)
    world_bank_T.columns = world_bank_T.iloc[0]
    
    #Determining the second dataframe
    world_bank2_T = world_bank_T[1:]
    
    #Return the two dataframes
    return world_bank, world_bank2_T

#Print the returned dataframes
world_bank, world_bank2_T = ClimateData('world_bank_data.csv')

print(world_bank.head())
print(world_bank2_T.head())
