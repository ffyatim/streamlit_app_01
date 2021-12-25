P7-Scoring - Implementez & Deployez un scoring modèle
Part-3. Dashboard

1. Objectif du projet: (voir aussi https://openclassrooms.com/fr/paths/164/projects/632/assignment)

2. Spécifications du dashboard: Permettre de visualiser le score et l’interprétation de ce score pour chaque client de façon intelligible pour une personne non experte en data science. Permettre de visualiser des informations descriptives relatives à un client (via un système de filtre). Permettre de comparer les informations descriptives relatives à un client à l’ensemble des clients ou à un groupe de clients similaires.

3. Code: streamlit_app_01.py. Cette partie du code concerne la presentation des donnees sous forme de Dashboard repondant aus specifications demandees. Deux autres parties (modelisation - ipynb) et (api - Flask) sont presentées dans des repertoires indépendants.

4. Données en entrée:

Fichiers csv resultats de la modelisation:
  application_sample.csv : Fichier nettoyé et pre-traité (EAD) 
  X_train_sample.csv: Fichier csv d'entrainement du modele 
  XGB_clf_model_f.pkl : Le XGBoostClassifier modele

5. Données en sortie:
Données & graphiques affichées sur écran selon demande utilisateur

