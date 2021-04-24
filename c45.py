#Universidade de Brasilia - UnB
#Introducao a inteligencia artificial
#Professor: Dibio Leandro Borges
#Aluno: William Coelho da Silva - 180029274

#Importando bibliotecas necessarias
import pandas as pd
from chefboost import Chefboost as chef
import gc

df = pd.read_csv('HC_PACIENTES_1.csv')
df.head()

df.info()
config = {'algorithm': 'C4.5'}
chef.fit(pd.read_csv("HC_PACIENTES_1.csv"), config)

gc.collect()







