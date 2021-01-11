#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 06:26:33 2020

@author: ggonecrane
"""

import pandas 
import statsmodels.api as sm
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

data = pandas.read_csv('gapminder.csv')

exp_var1 = 'incomeperperson' 
exp_var1_c = 'incomeperperson_c' 
resp_var1 = 'lifeexpectancy' 
resp_var1_c = 'lifeexpectancy_c' 
conf_var1 = 'urbanrate' 
conf_var1_c = 'urbanrate_c' 
xLabel = 'Income per Person Rate' 
yLabel = 'Urban Rate' 

# convert to numeric format
data[resp_var1] = pandas.to_numeric(data[resp_var1], errors='coerce')
data[exp_var1] = pandas.to_numeric(data[exp_var1], errors='coerce')
data[conf_var1] = pandas.to_numeric(data[conf_var1], errors='coerce')

# listwise deletion of missing values
sub1 = data[[exp_var1, conf_var1, resp_var1]].dropna()

####################################################################################
# LINEAR REGRESSION
####################################################################################

# linear regression analysis
printTableLabel(f'Linear Regression Analysis for {xLabel} and Life Expectancy')
reg0 = smf.ols(f'{resp_var1} ~ {exp_var1}', data=sub1).fit()
print (reg0.summary())

# linear regression analysis
printTableLabel('Linear Regression Analysis for Suicide Rate and Life Expectancy')
reg1 = smf.ols(f'{resp_var1} ~ {conf_var1}', data=sub1).fit()
print (reg1.summary())

scat0 = seaborn.regplot(x=exp_var1, y=resp_var1, scatter=True, data=sub1)
plt.xlabel('GDP per Capita (a.k.a Personal Income)')
plt.ylabel("Life Expectancy (years)")
plt.show()

scat1 = seaborn.regplot(x=exp_var1, y=resp_var1, scatter=True, order=2, data=sub1)
plt.xlabel('GDP per Capita (a.k.a Personal Income)')
plt.ylabel('Life Expectancy (years)')
plt.show()

# center quantitative IVs for regression analysis
sub1[exp_var1_c] = (sub1[exp_var1] - sub1[exp_var1].mean())
printTableLabel('Mean of Centered Variable')
print(sub1[exp_var1_c].mean())
sub1[resp_var1_c] = (sub1[resp_var1] - sub1[resp_var1].mean())
sub1[[exp_var1_c, resp_var1_c]].describe()
sub1[conf_var1_c] = (sub1[conf_var1] - sub1[conf_var1].mean())


# linear regression analysis with centralized explanatory variable
printTableLabel('Linear Regression Analysis with Personal Income')
reg1 = smf.ols(f'{resp_var1} ~ {exp_var1_c}', data=sub1).fit()
print (reg1.summary())

####################################################################################
# POLYNOMIAL REGRESSION
####################################################################################
#
# quadratic (polynomial) regression analysis
# run following line of code if you get PatsyError 'ImaginaryUnit' object is not callable
printTableLabel('Qudratic Regression Analysis with Personal Income')
reg2 = smf.ols(f'{resp_var1} ~ {exp_var1_c} + I({exp_var1_c}**2)', data=sub1).fit()
print (reg2.summary())

 first order (linear) scatterplot
scat2 = seaborn.regplot(x=conf_var1, y=resp_var1, scatter=True, data=sub1)
plt.xlabel('Suicide Rate (100th)')
plt.ylabel("Life Expectancy (years")
plt.show()

# fit second order polynomial
# run the 2 scatterplots together to get both linear and second order fit lines
scat3 = seaborn.regplot(x=conf_var1, y=resp_var1, scatter=True, order=2, data=sub1)
plt.xlabel('Suicide Rate (100th)')
plt.ylabel('Life Expectancy (years)')
plt.show()


####################################################################################
# EVALUATING MODEL FIT
####################################################################################

# adding income per person
reg3 = smf.ols(f'{resp_var1}  ~ {exp_var1_c} + I({exp_var1_c}**2) + {conf_var1_c}', data=sub1).fit()
printTableLabel('Model Fit Evaluation with Addition of Urban Rate')
print (reg3.summary())

#Q-Q plot for normality
fig1=sm.qqplot(reg3.resid, line='r')
plt.show()

# simple plot of residuals
stdres=pandas.DataFrame(reg3.resid_pearson)
print(stdres)
plt.plot(stdres, 'o', ls='None')
l = plt.axhline(y=0, color='r')
plt.ylabel('Standardized Residual')
plt.xlabel('Observation Number')
plt.show()

# additional regression diagnostic plots
fig2 = plt.figure(figsize=(12,8))
fig2 = sm.graphics.plot_regress_exog(reg3,  conf_var1_c, fig=fig2)
plt.show()

# leverage plot
fig3=sm.graphics.influence_plot(reg3, size=8)
plt.show()

