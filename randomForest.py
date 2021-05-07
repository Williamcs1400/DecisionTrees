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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
#import pydot


print('\nRandom Forest:\n\n')

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
print('Margem de erro: ', round(np.mean(baselineErrors), 2))


#analise exploratoria
def analise():

    print('\n\n Análise exploratória: \n\n')

    dataset = pd.read_excel('dataset.xlsx', engine='openpyxl')
    dataset.columns

    dataset[['resultado do exame', 'Patient addmited to regular ward (1=yes, 0=no)', 
    'Patient addmited to semi-intensive unit (1=yes, 0=no)', 
    'Patient addmited to intensive care unit (1=yes, 0=no)', 'Hematocrit',
    'Hemoglobin', 'Platelets', 'Red blood Cells', 'Lymphocytes', 'Leukocytes',
    'Basophils', 'Eosinophils', 'Mean corpuscular volume (MCV)', 'Monocytes', 'Red blood cell distribution width (RDW)',
    'Influenza A', 'Influenza B', 'Parainfluenza 1', 'CoronavirusNL63']].hist(bins=20, figsize=(15,15))
    plt.show()

    Continous_var = ['resultado do exame', 'Patient addmited to regular ward (1=yes, 0=no)', 
    'Patient addmited to semi-intensive unit (1=yes, 0=no)', 
    'Patient addmited to intensive care unit (1=yes, 0=no)', 'Hematocrit',
    'Hemoglobin', 'Platelets', 'Red blood Cells', 'Lymphocytes', 'Leukocytes',
    'Basophils', 'Eosinophils', 'Mean corpuscular volume (MCV)', 'Monocytes', 'Red blood cell distribution width (RDW)',
    'Influenza A', 'Influenza B', 'Parainfluenza 1', 'CoronavirusNL63']

    dataset[Continous_var].describe()

    dataset['result'] = dataset['resultado do exame'].replace({0: "negative", 1:"positive"})
    dataset['resp'] = dataset['Respiratory Syncytial Virus'].replace({0: "not_detected", 1:"detected"})


analise()

#Otimizacao de parametros
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy


n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
base_model.fit(trainDataset, trainLabels)
base_accuracy = evaluate(base_model, testDataset, testLabels)
best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, test_features, test_labels)


