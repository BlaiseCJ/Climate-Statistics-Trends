# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 13:11:28 2023

@author: Blaise Ezeokeke
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

"""
Importing the libraries: Pandas for structuring, 
Pyplot module of the Matplotlib Library for Visualisation,
Seaborn for clearer visualisation
Stats module of the Scipy Library for statistical functions
skew to identify measure of assymmetry 
kurtosis to identify "peakedness" of the distribution 
"""

#Use set_option function to give no maximum number of column to be displayed
pd.set_option('display.max_columns', None)

#Read the csv file using pandas
world_bank = pd.read_csv('world_bank_data.csv', skiprows=4)

#Read the metadata using pandas
world_bank_col = pd.read_csv('Metadata_Country_API_19_DS2_en_csv_v2_5361599.csv')

#Remove the non_numbers (NaN)
world_bank_col1 = world_bank_col[~pd.isna(world_bank_col['IncomeGroup'])]

#Select the required columns
world_bank = world_bank[['Country Name', 'Country Code', 'Indicator Name', 
                         'Indicator Code','2001', '2002',
         '2003', '2004','2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012',
         '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']]

#Find the mean of the distribution 
world_bank['mean'] = world_bank[['2001', '2002', '2003', '2004', '2005', '2006', '2007',
       '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016',
       '2017', '2018', '2019', '2020', '2021']].mean(axis=1)

#Choose the indicators to work with
selected_indicators = ['Urban population (% of total population)', 'Urban population growth (annual %)', 
                       'Population, total', 'Population growth (annual %)',
 'Mortality rate, under-5 (per 1,000 live births)', 'CO2 emissions (kt)', 'Energy use (kg of oil equivalent per capita)',
 'Electric power consumption (kWh per capita)','Energy use (kg of oil equivalent) per $1,000 GDP (constant 2017 PPP)',
 'Access to electricity (% of population)','Forest area (% of land area)','Arable land (% of land area)', 'Agricultural land (% of land area)']

#Select the countries to visualise, particular interest is Nigeria
countries = ['Nigeria','Australia','Jamaica','China','United Arab Emirates','Germany','France',
             'Malaysia','Japan','Romania','New Zealand','Morocco','United States','Pakistan','Thailand']


#Drop the Country Code and the Indicator Code, as part of data cleaning
world_bank = world_bank.drop(['Country Code','Indicator Code'], axis=1)

#Define a function to extract and transform the world bank climate data

def worldbankdata(world_bank, indicator):
    
    
    """
    Extract and transform World Bank data for a given indicator.

    Args:
        world_bank (pandas.DataFrame): A pandas dataframe containing World Bank data.
        indicator (str): The name of the World Bank indicator to extract.

    Returns:
        tuple: A tuple of two dataframes. The first dataframe contains the extracted data, indexed by country name.
            The second dataframe contains a transposed version of the first dataframe, indexed by year.
    """
    
    
    # Extract the World Bank data for the given indicator
    world_bank_1 = world_bank[world_bank['Indicator Name'] == indicator]
    
    # Drop the 'Indicator Name' column
    world_bank_1 = world_bank_1.drop(['Indicator Name'], axis=1)
    
    # Index the dataframe by 'Country Name'
    world_bank_1.index = world_bank_1.loc[:, 'Country Name']
    world_bank_1 = world_bank_1.drop(['Country Name'], axis=1)
    
    # Transpose the dataframe
    world_bank_2 = world_bank_1.transpose()
    
    #Return the two dataframe
    return world_bank_1, world_bank_2




world_bank_year, world_bank_country = worldbankdata(world_bank, 'Population, total')

 # Loop over the selected indicators
for ind in selected_indicators:
    
    # Get the data for the current indicator for the given countries
    world_bank_year, world_bank_country = worldbankdata(world_bank, ind)
    
    # Loop over the years and plot a swarm plot for each year
    for i in world_bank_year.columns:
        sns.swarmplot(y='Country Name',x=i, data=world_bank_year.loc[countries, :].reset_index())

    # Add a title and axis labels to the plot
    plt.title(ind)
    plt.xlabel('2001-2021')
    
    #Show plot
    plt.show()
    

    
for ind in selected_indicators:
    world_bank_year, world_bank_country = worldbankdata(world_bank, ind)
    for i in countries:
        plt.plot(world_bank_country[i], label=i)

    plt.title(ind)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(rotation=90)
    plt.show()

for i in selected_indicators:
    temp_world_bank = world_bank[world_bank['Indicator Name'] == i]
    temp_world_bank = temp_world_bank.merge(dfc1,left_on='Country Name', right_on='TableName')
    temp_world_bank.groupby(['Region'])['mean'].mean().sort_values(ascending=False).plot(kind='bar')
    plt.title(i)
    plt.show()



world_bank_1, world_bank_2 = (world_bank, 'Arable land (% of land area)')
world_bank_3, world_bank_4 = world_bank_data(world_bank, 'Forest area (% of land area)')

world_bank_1_mean = world_bank_1.mean(axis=1).reset_index().rename({0:'arable land'}, axis=1)
world_bank_3_mean = world_bank_3.mean(axis=1).reset_index().rename({0:'forest land'}, axis=1)

n_world_bank = world_bank_1_mean.merge(df3_mean, on='Country Name')

plt.figure(figsize=(4,3))
sns.scatterplot(x=n_world_bank['arable land'], y=n_world_bank['forest land'])
plt.xlabel('Arable Land')
plt.ylabel('Forest Land')
plt.title('Arable vs Forest Land')
plt.show()

n_world_bank.index = n_world_bank.loc[:, 'Country Name']
n_world_bank.sort_values(by='arable land', ascending=False)[:10].plot(kind='bar',figsize=(4,3))
plt.ylabel('frequency')
plt.title('Top 10 Countries with Max Arable Land')
plt.show()

n_world_bank.sort_values(by='forest land', ascending=False)[:10].plot(kind='bar',figsize=(4,3))
plt.ylabel('frequency')
plt.title('Top 10 Countries with Max Forest Land')
plt.show()



world_bank_1, world_bank_2 = worldbankdata(world_bank, 'Access to electricity (% of population)')
world_bank_3, world_bank_4 = worldbankdata(world_bank, 'Electric power consumption (kWh per capita)')

world_bank_1_mean = world_bank_1.mean(axis=1).reset_index().rename({0:'Access to Electricity'}, axis=1)
world_bank_3_mean = world_bank_3.mean(axis=1).reset_index().rename({0:'Electric Power Consumption'}, axis=1)

n_world_bank = world_bank_1_mean.merge(world_bank_3_mean, on='Country Name')

plt.figure(figsize=(4,3))
sns.scatterplot(x=n_world_bank['Access to Electricity'], y=n_world_bank['Electric Power Consumption'])
plt.xlabel('Access to Electricity')
plt.ylabel('Electric Power Consumption')
plt.title('Access to Electricity vs Electric Power Consumption')
plt.show()

n_world_bank.index = n_world_bank.loc[:, 'Country Name']
n_world_bank.sort_values(by='Access to Electricity', ascending=False)[:10].plot(kind='bar',figsize=(4,3))
plt.ylabel('frequency')
plt.title('Top 10 Countries with Max Access to Electricity')
plt.show()

n_world_bank.sort_values(by='Electric Power Consumption', ascending=False)[:10].plot(kind='bar',figsize=(4,3))
plt.ylabel('frequency')
plt.title('Top 10 Countries with Max Electric Power Consumption')
plt.show()



world_bank_1, world_bank_2 = worldbankdata(world_bank, 'Energy use (kg of oil equivalent) per $1,000 GDP (constant 2017 PPP)')
world_bank_3, world_bank_4 = worldbankdata(world_bank, 'CO2 emissions (kt)')

sns.scatterplot(x=world_bank_1.mean(axis=1), y=world_bank_3.mean(axis=1))
plt.xlabel('Energy use (kg of oil equivalent) per $1,000 GDP (constant 2017 PPP)')
plt.ylabel('CO2 emissions')
plt.title('Energy use vs CO2 emissions')
plt.show()



world_bank_1 = world_bank.groupby(['Country Name','Indicator Name'])['mean'].mean().unstack()

plt.figure(figsize=(10,7))
sns.heatmap(world_bank_1[selected_indicators].corr(), cmap='viridis', linewidths=.5, annot=True)


# Correlation Graph for select Countries using Urban Population Indicator
world_bank_year, world_bank_country = worldbankdata(world_bank, 'Urban population')

plt.figure(figsize=(10,7))
sns.heatmap(world_bank_country[countries].corr(), cmap='viridis', linewidths=.5, annot=True)


# ## Correlation Graph of Countries with the indicator of Arable Land

world_bank_year, world_bank_country = worldbankdata(world_bank, 'Arable land (% of land area)')

plt.figure(figsize=(10,7))
sns.heatmap(world_bank_country[countries].corr(), cmap='viridis', linewidths=.5, annot=True)


# ## Correlation Graph of Countries for "Electric Power Consumption" Indicator


world_bank_year, world_bank_country = worldbankdata(world_bank, 'Electric power consumption (kWh per capita)')

plt.figure(figsize=(10,7))
sns.heatmap(world_bank_country[countries].corr(), cmap='viridis', linewidths=.5, annot=True)

#To find the description of the dataframe, using .describe() method
print(world_bank_1[selected_indicators].describe())

#To find the skewness, using stats module from scipy
print(('Skewness:', stats.skew(world_bank))

#To find the standard deviation in the distribution 
print(world_bank2.std())