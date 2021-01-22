#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 07:09:35 2021

@author: ggonecrane
"""

import pandas as pd
import numpy
import statsmodels.formula.api as smf 

#import seaborn
#import matplotlib.pyplot as plt

g_raceCode1 = {1: 'White', 2:'Black', 3: 'Other', 4: 'Hispanic', 5: 'Mixed'}

g_incomeCode1 = {1: '<5K', 2:'5K-7.5K', 3: '7.5K-10K', 4: '10K-12.5K', 
                  5: '12.5K-15K', 6:'15K-20K', 7:'20K-25K', 8:'25K-30K',
                  9: '30K-35K', 10:'35K-40K', 11:'40K-50K', 12:'50K-60K',
                  13: '60K-75K', 14:'75K-85K', 15:'85K-100K', 16:'100K-125K',
                  17: '125K-150K', 18:'150K-175K', 19:'>175K'}

g_difficultyCode1 = {1: 'Very Hard', 2:'Somewhat Hard', 3: 'Somewhat Easy', 4: 'Very Easy'}

def printTableLabel(label):
    print('\n')
    print('-------------------------------------------------------------------------------------')
    print(f'\t\t\t\t{label}')
    print('-------------------------------------------------------------------------------------')
    
def printEthnicityTable(data, decimal_value=True, original_key=False):
    raceCode1 = g_raceCode1
    
    for index, value in data.iteritems():
        key = index if original_key==True else raceCode1[index]  
        value_formatted = value if decimal_value==True else round(value * 100, 3)
        print('{: <10} {: >5}'.format(key, value_formatted))
        
        
def printIncomeTable(data, decimal_value=True, original_key=False):
    incomeCode1 = g_incomeCode1
    
#    sortedByIndex = data.sort_index()
    for index, value in data.iteritems():
        if original_key==True:
            key = f'{int(index)} ({incomeCode1[index]})' 
        else: 
            key = incomeCode1[index] 
        value_formatted = value if decimal_value==True else round(value * 100, 3)
        # print('{: <10} {: >10}'.format(key, value_formatted))
        print(f'{key: <15} {value_formatted}')
        
def collapseRaceCode(ethnicCode):
    collapsed_value = -1
    if ethnicCode == 1:  # white ethnicity -> 0
        collapsed_value = 0
    elif ethnicCode == 2: # black ethnicity -> 1
        collapsed_value = 1
        
    return collapsed_value


pd.set_option('display.float_format', lambda x:'%f'%x)

# load Outlook Life data set
data = pd.read_csv('OutlookLife.csv', low_memory=False)

df = pd.DataFrame(data)

# map table names to user friendly variable names
race_col = 'PPETHM'
bw_race_col = 'BW_ETHN'
recoded_race_col = 'REC_ETHM'
income_col = 'W1_P20'
wealthy_col = 'W1_F4_D'
bi_income_col = 'BI_INCOME_LEVEL'
mapped_income = 'MAPPED_INCOME'
age_col = 'PPAGE'
age_c_col = 'AGE_C'
gender_col = 'PPGENDER'
edu_col = 'PPEDUCAT'

# Median Income Calculation
approx_income = {1: 2500, 2: 6250, 3: 8750, 4: 11250, 5: 13750, 6: 17500, 7: 22500,
                 8: 27500, 9:32500, 10: 37500, 11: 45000, 12: 55000, 13: 67500, 14: 80000,
                 15: 92500, 16: 112500, 17: 137500, 18: 162500, 19: 200000}
df[mapped_income]= df[income_col].map(approx_income)
# printTableLabel('Approximated Income Map')
#print(df['mapped_income'])

recoded_ethnicity = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
df[recoded_race_col] = df[race_col].map(recoded_ethnicity)

def printRecoded(x):
    if x > 1:
        print(x)
    else:
        pass
    
# df[recoded_race_col].apply(lambda x: printRecoded(x))

median_inc = df[mapped_income].median()
printTableLabel('Median Income of All Population')
print(f'{median_inc}')

df[income_col] = df[income_col].replace(-1,numpy.nan)

mean_age = df[age_col].mean()

# re-code annual income to bi-level (higher or lower than median value)
df[bi_income_col] = df[mapped_income].apply(lambda x: 1 if x > median_inc else 0)

sub1 = df.copy()

# centralize age 
sub1[age_c_col] = (sub1[age_col] - mean_age)

# adding 4 category ethnicity/race. Reference group coding is called "Treatment" coding in python
# and the default reference catergory is the group with a value = 0 (White)
reg1 = smf.ols(f'{mapped_income} ~ {age_c_col} + {gender_col} + {edu_col} + C({recoded_race_col})', 
               data=sub1).fit()
print (reg1.summary())

# can override the default ad specify a different reference group
# non-Hispanic Black as reference group 
reg2 = smf.ols(f'{mapped_income} ~ {age_c_col} + {gender_col} + {edu_col} + C({recoded_race_col}, Treatment(reference=1))', 
               data=sub1).fit()
print (reg2.summary())


##############################################################################
# LOGISTIC REGRESSION
##############################################################################

df[bw_race_col] = df[race_col].apply(lambda x: (collapseRaceCode(x)))
df[bw_race_col] = df[bw_race_col].replace(-1,numpy.nan)

sub2 = df.dropna().copy()

# make sure BW_ETHN to be categorical
sub2[bw_race_col] = sub2[bw_race_col].astype(int)
#sub2[bw_race_col] = sub2[bw_race_col].astype('category')


bw_mean_age = sub2[age_col].mean()
printTableLabel('Mean Age of Black and White Population')
print(f'{bw_mean_age}')

exp_var1 = 'PPMSACAT' #'W2_QF8' # 'W1_A4' #bw_race_col
exp_col1 = 'Non-Metro'
exp_col2 = 'Metro'

exp_freq = sub2[exp_var1].value_counts().sort_index()
exp_freqT = exp_freq.to_frame()
exp_freqT['cum_sum'] = exp_freqT.cumsum()
exp_freqT['cum_perc'] = sub2[exp_var1].value_counts(normalize=True).sort_index()

printTableLabel (f'{exp_var1} Frequency Table')
exp_freqT.columns = ['Counts', 'Cumulative Sum', 'Cumulative Perc.(%)']
exp_freqT.index = [exp_col1, exp_col2]
print(f'{exp_freqT}')

# logistic regression with ethnicity
lreg1 = smf.logit(formula = f'{bi_income_col} ~ {exp_var1}', data = sub2).fit()
printTableLabel('Logistic Regression Summary')
print (lreg1.summary())
# odds ratios
printTableLabel ("Odds Ratios")
print (numpy.exp(lreg1.params))

# odd ratios with 95% confidence intervals
params = lreg1.params
conf = lreg1.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
printTableLabel ("Odds Ratios with 95% Confidence Intervals")
print (numpy.exp(conf))

# logistic regression with social phobia and depression
lreg2 = smf.logit(formula = f'{bi_income_col} ~ {exp_var1} + PPEDUCAT', data = sub2).fit()
printTableLabel('Logistic Regression Summary')
print (lreg2.summary())

# odd ratios with 95% confidence intervals
params = lreg2.params
conf = lreg2.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
printTableLabel('Logistic Regression Summary with 95% Confidence Intervals')
print (numpy.exp(conf))

