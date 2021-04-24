#Universidade de Brasilia - UnB
#Introducao a inteligencia artificial
#Professor: Dibio Leandro Borges
#Aluno: William Coelho da Silva - 180029274

#Necessario executar:
#pip install xlrd 
#pip install openpyxl

#Importando bibliotecas necessarias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#Le a base de dados
dataset = pd.read_excel('dataset.xlsx', engine='openpyxl')

#Converter dados
labels = np.array(dataset['resultado do exame'])
datasetList = list(dataset.columns)
dataset = dataset.drop('resultado do exame', axis=1)
dataset = np.array(dataset)

print(dataset)
print(labels)

#Treinamento dos dados
trainDataset, testDataset, trainLabels, testLabels = train_test_split(dataset, labels, test_size=0.25, random_state=42)

print ('Training Features Shape:', trainDataset.shape) 
print ('Training Labels Shape:', testDataset.shape) 
print ('Testing Features Shape:', trainLabels.shape) 
print ('Testing Labels Shape:', testLabels.shape)

#Deixando resultado testLabels binário
for i, x in enumerate(testLabels):
    if(x == 'negative'):
        testLabels[i] = 0
    else:
        testLabels[i] = 1


#Previsoes
baselinePreds = testDataset[:, datasetList.index('resultado do exame')]
baselineErrors = abs(baselinePreds - testLabels)

print(baselinePreds)
print('Average baseline error: ', round(np.mean(baselineErrors), 2))






