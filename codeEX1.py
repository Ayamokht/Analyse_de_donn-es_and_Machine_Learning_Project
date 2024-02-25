# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 17:58:27 2023

@author: norac

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


# Chargement de la base de données, séparateur espace et index première ligne
data = pd.read_csv('simu.txt', sep=' ', header=0, names=['X1', 'X2', 'Y'])
X = data[['X1', 'X2']]
Y = data['Y']

# Ensembles de test et ensembles de validation
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Données qu'on va utiliser pour prédire
test_data = pd.read_csv('xsimutest.txt', sep=' ', header=0, names=['X1', 'X2'])


# 1er essai : modèles sans termes quadratiques
models_sans_poly = [
    ('Régression Logistique', LogisticRegression()),
    ('Forêt aléatoire', RandomForestClassifier()),
    ('Gradient Boosting', GradientBoostingClassifier()),
    ('Arbre de décision', DecisionTreeClassifier())]

best_model_sans_poly = None
best_accuracy_sans_poly = 0
predictions_sans_poly = pd.DataFrame()

for nom, model in models_sans_poly:
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)
    print(f'{nom} Précision : {accuracy}')
    
    if accuracy > best_accuracy_sans_poly:
        best_accuracy_sans_poly = accuracy
        best_model_sans_poly = model

    predictions_sans_poly[nom] = predictions

# Prédiction du meilleur modèle sans poly
best_model_predictions_sans_poly = best_model_sans_poly.predict(test_data)

# 2ème essai : modèles avec des termes quadratiques (degré 2, choix arbitraire)
models_avec_poly = [
    ('Régression Logistique (Poly)', make_pipeline(PolynomialFeatures(degree=2), LogisticRegression())),
    ('Forêt aléatoire (Poly)', make_pipeline(PolynomialFeatures(degree=2), RandomForestClassifier())),
    ('Gradient Boosting (Poly)', make_pipeline(PolynomialFeatures(degree=2), GradientBoostingClassifier())),
    ('Arbre de décision (Poly)', make_pipeline(PolynomialFeatures(degree=2), DecisionTreeClassifier()))]

best_model_avec_poly = None
best_accuracy_avec_poly = 0

for model_name, model in models_avec_poly:
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)
    print(f'{model_name} Précision : {accuracy}')
    
    if accuracy > best_accuracy_avec_poly:
        best_accuracy_avec_poly = accuracy
        best_model_avec_poly = model

# Prédiction du meilleur modèle avec poly
best_model_predictions_avec_poly = best_model_avec_poly.predict(test_data)

# Comparer les meilleures précisions des meilleurs modèles sans et avec poly
if best_accuracy_avec_poly > best_accuracy_sans_poly:
    best_model_global = best_model_avec_poly
    best_model_type = 'Avec Poly'
else:
    best_model_global = best_model_sans_poly
    best_model_type = 'Sans Poly'

# Enregistrement des prédictions dans le fichier txt
with open('predictions_global.txt', 'w') as f:
    f.write(f'Meilleur modèle global ({best_model_type}): {best_model_global}\n\n')
    f.write('Prédictions du meilleur modèle global avec X1 et X2 :\n')
    
    # Meilleur modèle global pour prédire
    best_model_predictions_global = best_model_global.predict(test_data)
    for i, prediction in enumerate(best_model_predictions_global):
        x1, x2 = test_data.iloc[i]['X1'], test_data.iloc[i]['X2']
        f.write(f'X1={x1}, X2={x2}, Prédiction={prediction}\n') 

    # Prédictions des autres modèles sans poly
    f.write('\nPrédictions des autres modèles sans poly :\n')
    for nom, model in models_sans_poly:
        predictions = model.predict(test_data)
        f.write(f'{nom} (Sans Poly) : {predictions}\n')

    # Prédictions des autres modèles avec poly
    f.write('\nPrédictions des autres modèles avec poly :\n')
    for model_name, model in models_avec_poly:
        if model != best_model_avec_poly:
            predictions = model.predict(test_data)
            f.write(f'{model_name} : {predictions}\n')
