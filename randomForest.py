#Universidade de Brasilia - UnB
#Introducao a inteligencia artificial
#Professor: Dibio Leandro Borges
#Aluno: William Coelho da Silva - 180029274

#Necessario executar:
#pip install xlrd 
#pip install openpyxl

#Importando bibliotecas necessarias
import pandas as pd

#Le a base de dados

dataset = pd.read_excel('dataset.xlsx', engine='openpyxl')

dataset.head()
dataset.info()






