# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 01:42:51 2018

@author: pulkit
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


from sklearn.preprocessing import MinMaxScaler
sc_temp = MinMaxScaler()
dataset.iloc[: , 6:9] = sc_temp.fit_transform(dataset.iloc[: , 6:9])

sc_wind = MinMaxScaler()
dataset.iloc[: , 11:12] = sc_wind.fit_transform(dataset.iloc[: , 11:12])

sc_price = MinMaxScaler()
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

sc_demand = MinMaxScaler()
dataset.iloc[: , 3:4] = sc_demand.fit_transform(dataset.iloc[: , 3:4])

train_data  = dataset[dataset['Year'] != 2018]
train_data.index = np.arange(len(train_data))

test_data = dataset[dataset['Year'] == 2018]
test_data.index = np.arange(len(test_data))


y_train = train_data.iloc[: , 3]
y_train = np.array(y_train).reshape((len(y_train) , 1))
train_data.drop('ontario demand' , axis =1 , inplace = True)
y_test = test_data.iloc[: , 3]
y_test = np.array(y_test).reshape((len(y_test) , 1))
test_data.drop('ontario demand' , axis =1 , inplace = True)

train_data.drop('Year' , axis =1 , inplace = True)
test_data.drop('Year' , axis =1 , inplace = True)

#**********************************************GENETIC ALGORITHM**************************************************************



from math import floor
from random import random, sample ,choice
from tqdm import tqdm
from numpy.linalg import pinv
from numpy import array, dot, mean



def multiple_linear_regression(inputs, outputs):
    X, Y = np.array(inputs), np.array(outputs)
    X_t, Y_t = X.transpose(), Y.transpose()
    coeff = np.dot((pinv((np.dot(X_t, X)))), (np.dot(X_t, Y)))
    Y_p = np.dot(X, coeff)
    Y_mean = np.mean(Y)
    SST = np.array([(i - Y_mean) ** 2 for i in Y]).sum()
    SSR = np.array([(i - j) ** 2 for i, j in zip(Y, Y_p)]).sum()
    COD = (1 - (SSR / SST)) * 100.0
    av_error = (SSR / len(Y))
    return {'COD': COD, 'coeff': coeff, 'error': av_error}


def check_termination_condition(best_individual , generation_count ,max_generations):
    if ((best_individual['COD'] >= 96.0)
            or (generation_count == max_generations)):
        return True
    else:
        return False
    
    
def create_individual(individual_size):
    return [random() for i in range(individual_size)]   


def create_population(individual_size, population_size):
    return [create_individual(individual_size) for i in range(population_size)]

def get_fitness(individual, inputs ,outputs):
    predicted_outputs = dot(array(inputs), array(individual))
    output_mean = mean(outputs)
    SST = array([(i - output_mean) ** 2 for i in outputs]).sum()
    SSR = array([(i - j) ** 2 for i, j in zip(outputs, predicted_outputs)]).sum()
    COD = (1 - (SSR / SST)) * 100.0
    average_error = (SSR / len(outputs))
    return {'COD': COD, 'error': average_error, 'coeff': individual}


def evaluate_population(population , inputs , outputs , selection_size , best_individuals_stash):
    fitness_list = [get_fitness(individual, inputs , outputs)
                    for individual in tqdm(population)]
    error_list = sorted(fitness_list, key=lambda i: i['error'])
    best_individuals = error_list[: selection_size]
    best_individuals_stash.append(best_individuals[0]['coeff'])
    print('Error: ', best_individuals[0]['error'],
          'COD: ', best_individuals[0]['COD'])
    return best_individuals



def crossover(parent_1, parent_2 , individual_size):
    child = {}
    loci = [i for i in range(0, individual_size)]
    loci_1 = sample(loci, floor(0.5*(individual_size)))
    loci_2 = [i for i in loci if i not in loci_1]
    chromosome_1 = [[i, parent_1['coeff'][i]] for i in loci_1]
    chromosome_2 = [[i, parent_2['coeff'][i]] for i in loci_2]
    child.update({key: value for (key, value) in chromosome_1})
    child.update({key: value for (key, value) in chromosome_2})
    return [child[i] for i in loci]


def mutate(individual , individual_size , probability_of_gene_mutating):
    loci = [i for i in range(0, individual_size)]
    no_of_genes_mutated = floor(probability_of_gene_mutating*individual_size)
    loci_to_mutate = sample(loci, no_of_genes_mutated)
    for locus in loci_to_mutate:
        gene_transform = choice([-1, 1])
        change = gene_transform*random()
        individual[locus] = individual[locus] + change
    return individual

def get_new_generation(selected_individuals , population_size , individual_size , probability_of_individual_mutating , probability_of_gene_mutating):
    parent_pairs = [sample(selected_individuals, 2)
                    for i in range(population_size)]
    offspring = [crossover(pair[0], pair[1] , individual_size) for pair in parent_pairs]
    offspring_indices = [i for i in range(population_size)]
    offspring_to_mutate = sample(
        offspring_indices,
        floor(probability_of_individual_mutating*population_size)
    )
    mutated_offspring = [[i, mutate(offspring[i] , individual_size,probability_of_gene_mutating)]
                         for i in offspring_to_mutate]
    for child in mutated_offspring:
        offspring[child[0]] = child[1]
    return offspring


def genetic_regression(inputs , outputs):
    print(multiple_linear_regression(inputs , outputs))
    
    individual_size = len(inputs[0])
    population_size = 1000
    selection_size = floor(0.1*population_size)
    max_generations = 7
    probability_of_individual_mutating = 0.1
    probability_of_gene_mutating = 0.25
    best_possible = multiple_linear_regression(inputs, outputs)
    best_individuals_stash = [create_individual(individual_size)]
    initial_population = create_population(individual_size, 1000)
    current_population = initial_population
    termination = False
    generation_count = 0
    
    while termination is False:
        current_best_individual = get_fitness(best_individuals_stash[-1], inputs ,outputs)
        print('Generation: ', generation_count)
        best_individuals = evaluate_population(current_population , inputs , outputs , selection_size, best_individuals_stash)
        current_population = get_new_generation(best_individuals , population_size , individual_size ,probability_of_individual_mutating , probability_of_gene_mutating)
        termination = check_termination_condition(current_best_individual , generation_count ,max_generations)
        generation_count += 1
    
    else:
        print(get_fitness(best_individuals_stash[-1], inputs , outputs))
    
    best = get_fitness(best_individuals_stash[-1], inputs , outputs)
    weight = np.array(best['coeff'])
    weight = weight.reshape((3,1))
    return weight

#**************************************************************************************************************************************


train_set = train_data.groupby(['season' , 'snow' , 'working day']).size().reset_index()
train_set = train_set.iloc[: , :-1]

train_data['demand'] = y_train
#
##creating the training structure 

#genetic algorithm
weight_set = []
for i in range(len(train_set)):
    part = train_data[train_data['season'] == train_set.iloc[i , 0]][train_data['snow'] == train_set.iloc[i , 1]][train_data['working day'] == train_set.iloc[i , 2]]
    part.drop( ['season' , 'snow' , 'working day'] , axis =1 , inplace = True)
    part.index = np.arange(len(part))
    weights = genetic_regression(part.iloc[: , :3].values , part.iloc[: , -1:].values)
    weight_set.append(weights)
    

weight_set = np.array(weight_set)
weight_set = weight_set.reshape((12,3))
y_pred_gen = []
for i in range(len(test_data)):
    rule_select = train_set[train_set['season'] == test_data.iloc[i , 3]][train_set['snow'] == test_data.iloc[i , 0]][train_set['working day'] == test_data.iloc[i , -1]]
    index = int(rule_select.index[0])
    y_pred_gen.append(np.array(test_data.iloc[0 , [1 , 2 , 4]]).reshape((1 , 3)) @ weight_set[index])
    
#gradient descent

def computeCost(X,y,theta):
    tobesummed = np.power(((X @ theta.T)-y),2)
    return np.sum(tobesummed)/(2 * len(X))
                  

#gradient descent
def gradientDescent(X,y,theta,iters,alpha):
    cost = np.zeros(iters)
    for i in range(iters):
        theta = theta - (alpha/len(X)) * np.sum(X * (X @ theta.T - y), axis=0)
        cost[i] = computeCost(X, y, theta)
    return theta,cost



    
weight_set = []
for i in range(len(train_set)):
    part = train_data[train_data['season'] == train_set.iloc[i , 0]][train_data['snow'] == train_set.iloc[i , 1]][train_data['working day'] == train_set.iloc[i , 2]]
    part.drop( ['season' , 'snow' , 'working day'] , axis =1 , inplace = True)
    part.index = np.arange(len(part))
    weights = np.random.rand(1,2)
    weights , cost = gradientDescent(part.iloc[: , 1:3].values , part.iloc[: , -1:].values , weights , 1000 , 0.01)
    weight_set.append(weights)
    

weight_set = np.array(weight_set)
weight_set = weight_set.reshape((12,2))
y_pred_grad = []
for i in range(len(test_data)):
    rule_select = train_set[train_set['season'] == test_data.iloc[i , 3]][train_set['snow'] == test_data.iloc[i , 0]][train_set['working day'] == test_data.iloc[i , -1]]
    index = int(rule_select.index[0])
    print(index)
    y_pred_grad.append(np.array(test_data.iloc[0 , [ 2 , 4]]).reshape((1 , 2)) @ weight_set[index])
    
y_pred_grad = np.array(y_pred_grad)



#support vector regression
# Fitting decision tree regressor to the dataset
from sklearn.tree import DecisionTreeRegressor
dtr = []

weight_set = []
for i in range(len(train_set)):
    part = train_data[train_data['season'] == train_set.iloc[i , 0]][train_data['snow'] == train_set.iloc[i , 1]][train_data['working day'] == train_set.iloc[i , 2]]
    part.drop( ['season' , 'snow' , 'working day'] , axis =1 , inplace = True)
    part.index = np.arange(len(part))
    regressor = DecisionTreeRegressor()
    regressor.fit(part.iloc[: , 1:3].values , part.iloc[: , -1:].values )
    dtr.append(regressor)
    


y_pred_dtr = []
for i in range(len(test_data)):
    rule_select = train_set[train_set['season'] == test_data.iloc[i , 3]][train_set['snow'] == test_data.iloc[i , 0]][train_set['working day'] == test_data.iloc[i , -1]]
    index = int(rule_select.index[0])
    y_pred_dtr.append(   dtr[index].predict(np.array(test_data.iloc[0 , [ 2 , 4]] ).reshape((1 , 2))) )
    
y_pred_grad = np.array(y_pred_grad)
y_pred_grad = np.array(y_pred_grad)

#APPLYING THE ANN
import keras
from keras.models import Sequential      # this is used to iinitialise pur neural network
from keras.layers import Dense        # this is used to make the different layers ofour nueral network
ann = []

weight_set = []
for i in range(len(train_set)):
    part = train_data[train_data['season'] == train_set.iloc[i , 0]][train_data['snow'] == train_set.iloc[i , 1]][train_data['working day'] == train_set.iloc[i , 2]]
    part.drop( ['season' , 'snow' , 'working day'] , axis =1 , inplace = True)
    part.index = np.arange(len(part))
    model = Sequential()
    model.add(Dense(4, input_dim=2, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    model.fit(part.iloc[: , 1:3].values , part.iloc[: , -1:].values , epochs = 100, verbose=0)
    ann.append(model)
    


y_pred_ann = []
for i in range(len(test_data)):
    rule_select = train_set[train_set['season'] == test_data.iloc[i , 3]][train_set['snow'] == test_data.iloc[i , 0]][train_set['working day'] == test_data.iloc[i , -1]]
    index = int(rule_select.index[0])
    y_pred_ann.append(   ann[index].predict(np.array(test_data.iloc[0 , [ 2 , 4]] ).reshape((1 , 2))) )
    
y_pred_ann = np.array(y_pred_ann)
y_pred_ann = y_pred_ann.reshape((284 , 1))

y_pred_ann = sc_demand.inverse_transform(y_pred_ann)
y_test = sc_demand.inverse_transform(y_test)

y_pred_grad = y_pred_grad.reshape((284 , 1))
y_pred_grad = sc_demand.inverse_transform(y_pred_grad)
plt.plot(y_pred_grad[:50] , 'red')
plt.plot(y_test[:50] , 'blue')
plt.show()

plt.plot(y_pred_ann[:50] , 'red')
plt.plot(y_test[:50] , 'blue')
plt.show()

