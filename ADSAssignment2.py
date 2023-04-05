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
    



# Loop over the selected indicators   
for ind in selected_indicators:
    
    
    # Get the data for the current indicator
    world_bank_year, world_bank_country = worldbankdata(world_bank, ind)
    
    # Loop over the countries and plot the data for each one
    for i in countries:
        plt.plot(world_bank_country[i], label=i)
        
    # Add a title, legend, axis labels, and tick rotation to the plot
    plt.title(ind)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(rotation=90)
    
    
    # Show the plot
    plt.show()

# Loop over the selected indicators
for i in selected_indicators:
    
    
    # Select the data for the current indicator and merge it with the country information dataframe
    temp_world_bank = world_bank[world_bank['Indicator Name'] == i]
    temp_world_bank = temp_world_bank.merge(dfc1,left_on='Country Name', right_on='TableName')
    
    # Compute the mean value of the indicator for each region and plot a bar chart
    temp_world_bank.groupby(['Region'])['mean'].mean().sort_values(ascending=False).plot(kind='bar')
    
    #Add title
    plt.title(i)
    
    
    #Show plot
    plt.show()


# Get the data for the 'Arable land (% of land area)' and 'Forest area (% of land area)' indicators
world_bank_1, world_bank_2 = (world_bank, 'Arable land (% of land area)')
world_bank_3, world_bank_4 = worldbankdata(world_bank, 'Forest area (% of land area)')

# Compute the mean values of the indicators for each country and merge them into a single dataframe
world_bank_1_mean = world_bank_1.mean(axis=1).reset_index().rename({0:'arable land'}, axis=1)
world_bank_3_mean = world_bank_3.mean(axis=1).reset_index().rename({0:'forest land'}, axis=1)
n_world_bank = world_bank_1_mean.merge(world_bank_3_mean, on='Country Name')


# Plot a scatter plot of arable land vs. forest land
plt.figure(figsize=(4,3))
sns.scatterplot(x=n_world_bank['arable land'], y=n_world_bank['forest land'])


# Add axis labels and a title to the plot
plt.xlabel('Arable Land')
plt.ylabel('Forest Land')
plt.title('Arable vs Forest Land')


# Show the plot
plt.show()



# Get the data for the 'Arable land (% of land area)' and 'Forest area (% of land area)' indicators
# Set the index of the dataframe to 'Country Name'
n_world_bank.index = n_world_bank.loc[:, 'Country Name']



# Create a bar chart of the top 10 countries with the highest arable land percentage
n_world_bank.sort_values(by='arable land', ascending=False)[:10].plot(kind='bar',figsize=(4,3))


# Add axis labels and a title to the plot
plt.ylabel('frequency')
plt.title('Top 10 Countries with Max Arable Land')


# Show the plot
plt.show()


# Create a bar chart of the top 10 countries with the highest forest land percentage
n_world_bank.sort_values(by='forest land', ascending=False)[:10].plot(kind='bar',figsize=(4,3))

# Add axis labels and a title to the plot
plt.ylabel('frequency')
plt.title('Top 10 Countries with Max Forest Land')


#Show the plot
plt.show()


# Get the data for the 'Access to electricity (% of population)' and 'Electric power consumption (kWh per capita)' indicators
world_bank_1, world_bank_2 = worldbankdata(world_bank, 'Access to electricity (% of population)')
world_bank_3, world_bank_4 = worldbankdata(world_bank, 'Electric power consumption (kWh per capita)')


# Compute the mean values of the indicators for each country and merge them into a single dataframe
world_bank_1_mean = world_bank_1.mean(axis=1).reset_index().rename({0:'Access to Electricity'}, axis=1)
world_bank_3_mean = world_bank_3.mean(axis=1).reset_index().rename({0:'Electric Power Consumption'}, axis=1)
n_world_bank = world_bank_1_mean.merge(world_bank_3_mean, on='Country Name')


# Plot a scatter plot of access to electricity vs. electric power consumption
plt.figure(figsize=(4,3))
sns.scatterplot(x=n_world_bank['Access to Electricity'], y=n_world_bank['Electric Power Consumption'])


# Add axis labels and a title to the plot
plt.xlabel('Access to Electricity')
plt.ylabel('Electric Power Consumption')
plt.title('Access to Electricity vs Electric Power Consumption')


# Show the plot
plt.show()


# Set the index of the dataframe to 'Country Name
n_world_bank.index = n_world_bank.loc[:, 'Country Name']


# Create a bar chart of the top 10 countries with the highest access to electricity percentage
n_world_bank.sort_values(by='Access to Electricity', ascending=False)[:10].plot(kind='bar',figsize=(4,3))


# Add axis labels and a title to the plot
plt.ylabel('frequency')
plt.title('Top 10 Countries with Max Access to Electricity')


# SHow the plot
plt.show()


# Create a bar chart of the top 10 countries with the highest electric power consumption per capita
n_world_bank.sort_values(by='Electric Power Consumption', ascending=False)[:10].plot(kind='bar',figsize=(4,3))


# Add axis labels and a title to the plot
plt.ylabel('frequency')
plt.title('Top 10 Countries with Max Electric Power Consumption')


# Show the plot
plt.show()


# Get the data for the 'Energy use (kg of oil equivalent) per $1,000 GDP 
# (constant 2017 PPP)' and 'CO2 emissions (kt)' indicators
world_bank_1, world_bank_2 = worldbankdata(world_bank, 'Energy use (kg of oil equivalent) per $1,000 GDP (constant 2017 PPP)')
world_bank_3, world_bank_4 = worldbankdata(world_bank, 'CO2 emissions (kt)')


# Create a scatter plot of energy use vs. CO2 emissions
sns.scatterplot(x=world_bank_1.mean(axis=1), y=world_bank_3.mean(axis=1))

# Add axis labels and a title to the plot
plt.xlabel('Energy use (kg of oil equivalent) per $1,000 GDP (constant 2017 PPP)')
plt.ylabel('CO2 emissions')
plt.title('Energy use vs CO2 emissions')


# Show the plot 
plt.show()


# Compute the mean value of each indicator for each country, and reshape the data into a pivot table
world_bank_1 = world_bank.groupby(['Country Name','Indicator Name'])['mean'].mean().unstack()


# Create a heatmap of the correlation matrix for the selected indicators
plt.figure(figsize=(10,7))
sns.heatmap(world_bank_1[selected_indicators].corr(), cmap='viridis', linewidths=.5, annot=True)


# Display the heatmap
plt.show()


# Get the data for the 'Urban population' indicator
world_bank_year, world_bank_country = worldbankdata(world_bank, 'Urban population')


# Create a heatmap of the correlation matrix for the selected countries
plt.figure(figsize=(10,7))
sns.heatmap(world_bank_country[countries].corr(), cmap='viridis', linewidths=.5, annot=True)


# Show the heatmap
plt.show()


# Retrieve the data for the 'Arable land (% of land area)' indicator
world_bank_year, world_bank_country = worldbankdata(world_bank, 'Arable land (% of land area)')


# Create a heatmap of the correlation matrix for the selected countries
plt.figure(figsize=(10,7))
sns.heatmap(world_bank_country[countries].corr(), cmap='viridis', linewidths=.5, annot=True)


# Display the heatmap
plt.show()


# Retrieve the data for the 'Electric power consumption (kWh per capita)' indicator
world_bank_year, world_bank_country = worldbankdata(world_bank, 'Electric power consumption (kWh per capita)')


# Create a heatmap of the correlation matrix for the selected countries
plt.figure(figsize=(10,7))
sns.heatmap(world_bank_country[countries].corr(), cmap='viridis', linewidths=.5, annot=True)


# Display the heatmap
plt.show()


# Retrieve the data for the selected indicators and display a summary of the data using the .describe() method
world_bank_summary = world_bank_1[selected_indicators].describe()
print(world_bank_summary)

# Compute the skewness of the data using the stats module from scipy and display the result
skewness = stats.skew(world_bank)
print(('Skewness:', skewness))

# Compute the standard deviation of the data in world_bank2 and display the result
standard_deviation = world_bank2.std()
print(standard_deviation)
