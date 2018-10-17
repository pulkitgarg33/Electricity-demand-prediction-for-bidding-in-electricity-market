# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 01:42:51 2018

@author: pulki
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import datetime

#preparing weather data****************************************************************************************  

def prepare_weather (weather1):
    weather1 =weather1.drop(['Data Quality' ,'Max Temp Flag', 'Min Temp Flag',
                             'Mean Temp Flag', 'Heat Deg Days Flag' ,
                             'Cool Deg Days Flag', 'Total Rain (mm)',
                             'Total Precip Flag' , 'Total Snow (cm)' ,
                             'Snow on Grnd Flag', 'Dir of Max Gust Flag',
                             'Spd of Max Gust Flag','Total Rain Flag',
                             'Dir of Max Gust (10s deg)', 'Total Snow Flag'] , axis = 1)
    
    weather1 = weather1.drop(weather1.columns[7] , axis=1)
    weather1 = weather1.drop(weather1.columns[7] , axis=1)
    year = weather1.iloc[: , 1].values
    weather1 = weather1.drop(weather1.columns[1] , axis=1)
    weather1 = weather1.drop(weather1.columns[0] , axis=1)
    
    weather1.iloc[: , 2] = (weather1.iloc[:,2]) + 273
    weather1.iloc[: , 3] = (weather1.iloc[:,3]) + 273
    weather1.iloc[: , 4] = (weather1.iloc[:,4]) + 273
    
    weather1.iloc[:,6].fillna(0 , inplace = True)
    weather1.iloc[:,2].fillna(method ='pad' ,limit = 18, inplace = True)
    weather1.iloc[:,3].fillna(method ='pad' ,limit = 16, inplace = True)
    weather1.iloc[:,4].fillna(method ='pad' ,limit = 18, inplace = True)
    weather1.iloc[:,5].fillna(0 , inplace = True)
    weather1.iloc[:,7].fillna('<31' , inplace = True)
    weather1[weather1.columns[7]][weather1[weather1.columns[7]] == '<31']  = np.random.randint(1 , 30, len(weather1[weather1.columns[7]][weather1[weather1.columns[7]] == '<31']))
    weather1['Year'] = year
    return weather1

weather = pd.read_excel('weather data 2012-2018.xlsx')
weather = prepare_weather(weather)

#preparing price data****************************************************************************************  


def dow(date):
    days=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    dayNumber=date.weekday()
    return days[dayNumber]
def preprocess_price(price1):
    price1 = price1.drop('Hour' , axis =1)
    price1 = price1.groupby('Date').max()
    ind = np.arange(0 , len(price1))
    price1['Date'] = price1.index
    price1.index = ind
    for i in range(len(price1)): 
       price1.loc[i,'Day'] = int(price1.loc[i,'Date'].split('-')[0])    
       price1.loc[i,'Month'] = int(price1.loc[i,'Date'].split('-')[1])    
       price1.loc[i,'Year'] = int(price1.loc[i,'Date'].split('-')[2])    
       price1.loc[i , 'weekday'] = dow(datetime.date(int(price1.loc[i,'Year']) , int(price1.loc[i,'Month']) , int(price1.loc[i,'Day'])))
    price1 = price1.drop('Date' , axis =1)
    price1.drop
    return price1

price1 = pd.read_csv('nodal price 2015.csv')
price1 = preprocess_price(price1)
price2 = pd.read_csv('nodal price 2016.csv')
price2 = preprocess_price(price2)
price3 = pd.read_csv('nodal price 2017.csv')
price3 = preprocess_price(price3)
price4 = pd.read_csv('nodal price 2018.csv')
price4 = preprocess_price(price4)
price5 = pd.read_csv('nodal price 2012.csv')
price5 = preprocess_price(price5)
price6 = pd.read_csv('nodal price 2013.csv')
price6 = preprocess_price(price6)
price7 = pd.read_csv('nodal price 2014.csv')
price7 = preprocess_price(price7)
frames = [price1 , price2 , price3, price4 , price5 , price6 , price7]
price = pd.concat(frames)

price1 = pd.read_csv('elec_price_2012.csv')
price1 = preprocess_price(price1)
price2 = pd.read_csv('elec_price_2013.csv')
price2 = preprocess_price(price2)
price3 = pd.read_csv('elec_price_2014.csv')
price3 = preprocess_price(price3)
price4 = pd.read_csv('elec_price_2015.csv')
price4 = preprocess_price(price4)
price5 = pd.read_csv('elec_price_2016.csv')
price5 = preprocess_price(price5)
price6 = pd.read_csv('elec_price_2017.csv')
price6 = preprocess_price(price6)
price7 = pd.read_csv('elec_price_2018.csv')
price7 = preprocess_price(price7)
frames = [price1 , price2 , price3, price4 , price5 , price6 , price7]
hoep_price = pd.concat(frames)

hoep_price['HOEP'] = pd.to_numeric(hoep_price['HOEP'])

#preparing the demand data----------------------------------------------------------------------------------

price1 = pd.read_csv('demand 2012.csv')
price1 = preprocess_price(price1)
price2 = pd.read_csv('demand 2013.csv')
price2 = preprocess_price(price2)
price3 = pd.read_csv('demand 2014.csv')
price3 = preprocess_price(price3)
price4 = pd.read_csv('demand 2015.csv')
price4 = preprocess_price(price4)
price5 = pd.read_csv('demand 2016.csv')
price5 = preprocess_price(price5)
price6 = pd.read_csv('demand 2017.csv')
price6 = preprocess_price(price6)
price7 = pd.read_csv('demand 2018.csv')
price7 = preprocess_price(price7)
frames = [price1 , price2 , price3, price4 , price5 , price6 , price7]
demand = pd.concat(frames)
#combining all the data----------------------------------------------------------------------------------------------------

price_demand_combined = pd.merge(price,hoep_price , on = ['Day' , 'Month' ,'Year', 'weekday'])
dataset = pd.merge(price_demand_combined , weather , on = ['Day' , 'Month' , 'Year'] )
dataset = pd.merge(dataset , demand , on = ['Day' , 'Month' , 'Year' , 'weekday'])


dataset.columns = ['nodal price' , 'Day' , 'Month' ,'Year' 
                                          ,'weekday','hoep','max temp' , 'min temp',
                                         'mean temp' , 'precipitation','snow', 'wind speed' , 
                                         'ontario demand']

dataset['wind speed'] = pd.to_numeric(dataset['wind speed'])


from sklearn.preprocessing import StandardScaler
sc_temp = StandardScaler()
dataset.iloc[: , 6:9] = sc_temp.fit_transform(dataset.iloc[: , 6:9])

sc_wind = StandardScaler()
dataset.iloc[: , 11:12] = sc_wind.fit_transform(dataset.iloc[: , 11:12])

sc_price = StandardScaler()
dataset.iloc[: , [0,5]] = sc_price.fit_transform(dataset.iloc[: , [0,5]])

from sklearn.decomposition import PCA
Pca_temp = PCA(n_components = 1)
final = Pca_temp.fit_transform(dataset.iloc[: , 6:9])
explained_variance = Pca_temp.explained_variance_ratio_

dataset = dataset.drop('max temp' , axis =1)
dataset = dataset.drop('min temp' , axis =1)
dataset = dataset.drop('mean temp' , axis =1)
dataset['temp'] = final


for i in range(len(dataset)):
    if(dataset.iloc[i , 6]  > 3 ):
        dataset.iloc[i , 6] = 1
    else:
        dataset.iloc[i , 6] = 0
    
    if( dataset.iloc[i , 7] > 2):
        dataset.iloc[i , 7] = 1
    else:
        dataset.iloc[i , 7] = 0
        
dataset.drop('precipitation' , axis =1, inplace = True)

season = np.zeros(len(dataset))
summer = [7 , 8]
winter = [12 ,1 ,2 ,3]
autumn = [9 , 10 , 11]
spring = [4 , 5 , 6]
for i in range(len(dataset)):
    if (dataset.iloc[i , dataset.columns.get_loc('Month')] in winter):
        season[i] = 1
    if (dataset.iloc[i , dataset.columns.get_loc('Month')] in spring):
        season[i] = 2
    if (dataset.iloc[i , dataset.columns.get_loc('Month')] in summer):
        season[i] = 3
    if (dataset.iloc[i , dataset.columns.get_loc('Month')] in autumn):
        season[i] = 4
        
        
dataset['season'] = season


pca_price = PCA ( n_components=1)
final = pca_price.fit_transform(dataset.iloc[: , [0,5]])

dataset.drop(['nodal price' , 'hoep'] , axis = 1 , inplace = True)

dataset['price'] = final

working_day = np.zeros(len(dataset))
for i in range(len(dataset)):
    if (dataset.iloc[i , dataset.columns.get_loc('weekday')] in ['Saturday' , 'Sunday']):
        working_day[i] = 1
        
dataset['working day'] = working_day

dataset.drop('weekday' , axis =1 , inplace = True)
dataset.drop(['Month' , 'Day'] , axis = 1 , inplace = True)

sc_demand = StandardScaler()
dataset.iloc[: , 3:4] = sc_demand.fit_transform(dataset.iloc[: , 3:4])

train_data  = dataset[dataset['Year'] != 2018]
train_data.index = np.arange(len(train_data))

test_data = dataset[dataset['Year'] == 2018]
test_data.index = np.arange(len(test_data))


y_train = train_data.iloc[: , 3]
train_data.drop('ontario demand' , axis =1 , inplace = True)
y_test = test_data.iloc[: , 3]
test_data.drop('ontario demand' , axis =1 , inplace = True)

train_data.drop('Year' , axis =1 , inplace = True)
test_data.drop('Year' , axis =1 , inplace = True)






