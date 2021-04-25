#Universidade de Brasilia - UnB
#Introducao a inteligencia artificial
#Professor: Dibio Leandro Borges
#Aluno: William Coelho da Silva - 180029274

#Importando bibliotecas necessarias
import pandas as pd
from chefboost import Chefboost as chef
import gc

#Le a base de dados
dataset = pd.read_excel('dataset.xlsx', engine='openpyxl')

config = {'algorithm': 'C4.5'}
model = chef.fit(dataset.copy(), config)

for ind, istance in dataset.iterrows():
    prediction = chef.predict(model, dataset.iloc[0])
    actual = istance['Decison']
    if actual == prediction:
        classified = True
    else:
        cclassified = False
        print("x", end='')
    
    print(actual, " - ", prediction)



#gc.collect()







