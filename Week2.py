#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 06:26:33 2020

@author: ggonecrane
"""

import numpy 
import pandas 
#import statsmodels.api
import statsmodels.formula.api as smf
import seaborn
import matplotlib.pyplot as plt


def printTableLabel(label):
    print('\n')
    print('-------------------------------------------------------------------------------------')
    print(f'\t\t\t\t{label}')
    print('-------------------------------------------------------------------------------------')
    
    
# bug fix for display formats to avoid run time errors
pandas.set_option('display.float_format', lambda x:'%.2f'%x)

#call in data set
data = pandas.read_csv('OutlookLife.csv')

# map table names to user friendly variable names
race_col = 'PPETHM'
income_col = 'W1_P20'
bw_race_col = 'BW_PPETHM'
race_legend = '0: Non-Black, 1: Black'
exp_var = bw_race_col
res_var = income_col


# convert variables to numeric format using convert_objects function
data[income_col] = pandas.to_numeric(data[income_col], errors='coerce')
data[income_col] = data[income_col].replace(-1,numpy.nan)

data[race_col] = pandas.to_numeric(data[race_col], errors='coerce')
bwRaceMap = {1:0, 2:1, 3:0, 4:0, 5:0}
data[bw_race_col] = data[race_col].map(bwRaceMap)

############################################################################################
# Frequency Table
############################################################################################

bw_countsT = data[exp_var].value_counts().sort_index()
bw_percT = data[exp_var].value_counts(normalize=True).sort_index()
sub1 = bw_countsT.to_frame()
sub1['perc.(%)'] = bw_percT.apply(lambda x: '{:.2f}'.format(round(x * 100, 2)))
sub1['cum_sum'] = bw_countsT.cumsum()
sub1['cum_perc'] = bw_percT.cumsum().apply(lambda x: '{:.2f}'.format(round(x * 100, 2)))
sub1.columns = ['Frequency', 'Percentage(%)', 'Cumulative Frequency', 'Cumulative Percentage(%)']
printTableLabel(f'Frequency Table ({race_legend})')
print(sub1)

############################################################################################
# BASIC LINEAR REGRESSION
############################################################################################


printTableLabel("OLS regression model for the association between Ethnicity and Income Level")
reg1 = smf.ols(f'{res_var} ~ {exp_var}', data=data).fit()
print (reg1.summary())

sub2 = data[[exp_var, res_var]].copy().dropna()

printTableLabel(f'Mean - Income Level ({race_legend})')
ds1 = sub2.groupby(exp_var).mean()
print(ds1)

printTableLabel(f'Standard Deviation - Income Level ({race_legend})')
ds2 = sub2.groupby(exp_var).std()
print(ds2)

seaborn.factorplot(x=exp_var, y=res_var, data=sub2, kind='bar', ci=None)
plt.xlabel(f'Ethnicity ({race_legend})')
plt.ylabel(f'Mean Number Income Level ({race_legend})')