# 📊 FT_LINEAR_REGRESSION  
## Implémentation d'une régression linéaire simple

---

## 📌 Objectif du projet  
Ce projet implémente un modèle de **régression linéaire simple** afin de **prédire le prix d'un véhicule en fonction de son kilométrage**.  
L'objectif est de mettre en pratique les **fondamentaux du machine learning**, à travers l'implémentation de **la descente de gradient**, sans utiliser de bibliothèques spécialisées dans la régression.

L'algorithme ajuste les paramètres **θ₀** et **θ₁** pour minimiser la fonction de coût **J(θ)**, permettant ainsi d'obtenir des **prédictions précises**.

---

## 📚 Qu'est-ce que la régression linéaire simple ?

La **régression linéaire simple** est une méthode statistique qui permet de modéliser la relation entre deux variables :
- **Une variable indépendante (feature)** ➔ ici le **kilométrage**
- **Une variable dépendante (target)** ➔ ici le **prix de la voiture**

Elle cherche à ajuster une droite (appelée **droite de régression**) qui prédit au mieux la valeur de la variable dépendante en fonction de la variable indépendante. Cette droite est définie par l'équation :

```
estimatePrice(mileage) = θ₀ + (θ₁ * mileage)
```

- **θ₀ (theta zero)** est l'ordonnée à l'origine ➔ c'est la valeur prédite lorsque le kilométrage est nul.
- **θ₁ (theta un)** est la pente ➔ il détermine comment le prix évolue avec le kilométrage.

L'objectif est de trouver les meilleurs **θ₀** et **θ₁** qui minimisent l'erreur entre les valeurs prédites et les valeurs réelles. Cette erreur est mesurée via la **fonction de coût J(θ)**, généralement l'**erreur quadratique moyenne** (**Mean Squared Error - MSE**).

### ✏️ Pour aller plus loin :
- [Régression Linéaire Simple - Wikipédia](https://fr.wikipedia.org/wiki/R%C3%A9gression_lin%C3%A9aire)
- [Linear Regression Explained - Towards Data Science](https://towardsdatascience.com/linear-regression-detailed-view-ea73175f6e86)
- [Gradient Descent Algorithm - GeeksforGeeks](https://www.geeksforgeeks.org/gradient-descent-in-linear-regression/)
- [Machine Learning Crash Course - Google](https://developers.google.com/machine-learning/crash-course)

---

## ⚙️ Fonctionnalités principales

✅ **Entraînement du modèle par descente de gradient**  
✅ **Sauvegarde et chargement des paramètres (θ₀, θ₁) en JSON**  
✅ **Normalisation / Dénormalisation des données d'entrée et de sortie**  
✅ **Gestion complète des erreurs et contrôle des limites de prédiction**  
✅ **Prédictions en temps réel ou en batch sur de nouveaux kilométrages**

---

## 🧠 Concepts abordés

- Régression linéaire simple  
- Fonction de coût **J(θ)** (Erreur quadratique moyenne)  
- Optimisation par **descente de gradient**  
- Normalisation des données pour améliorer la convergence  
- Évaluation des performances (MSE et R²)

---

## ✨ Bonus implémentés

| Bonus | Description |
|-------|-------------|
| ✅ Sauvegarde et chargement du modèle en **JSON** | Facilite la lecture et la réutilisation des paramètres θ₀, θ₁ et des valeurs de normalisation |
| ✅ **Visualisation de la droite de régression** | Graphique `regression_plot.png` superposant la droite de régression aux données initiales |
| ✅ **Courbe de coût J(θ)** | Graphique `cost_function_plot.png` affichant l'évolution du coût durant l'entraînement |
| ✅ **Graphique des résidus** | `residuals_plot.png` visualise la différence entre valeurs prédites et réelles |
| ✅ **Histogramme des erreurs de prédiction** | `error_distribution.png` montrant la répartition des erreurs |
| ✅ **Boxplot de la distribution initiale des données** | `data_distribution.png` pour visualiser la dispersion des valeurs |
| ✅ **Histogramme de la répartition du kilométrage** | `mileage_distribution.png` |
| ✅ **Comparaison graphique entre MSE et R²** | `metrics_comparison.png` comparant les performances du modèle |
| ✅ **Prédictions en batch avec extrapolation modérée (±20%)** | Limite la prédiction à des valeurs réalistes et contrôlées |
| ✅ **Sauvegarde régulière des checkpoints pendant l'entraînement** | Fichiers `model_epoch_{epoch}.json` pour reprendre ou analyser la progression |
| ✅ **Calcul du coefficient R²** | Fournit une métrique alternative et complémentaire au MSE |
| ✅ **Gestion robuste des erreurs** | Contrôle des entrées utilisateur, vérification des fichiers, gestion des prédictions hors limites |

---

## 🚀 Visualisation des résultats graphiques

Les graphiques générés sont automatiquement **sauvegardés** dans le répertoire `app/` ou le répertoire courant selon la configuration Docker.  
### 📂 Liste des graphiques :
- `regression_plot.png` ➔ Données initiales et droite de régression  
- `cost_function_plot.png` ➔ Courbe de coût J(θ)  
- `residuals_plot.png` ➔ Graphique des résidus  
- `error_distribution.png` ➔ Histogramme des erreurs de prédiction  
- `data_distribution.png` ➔ Boxplot des distributions kilométrage / prix  
- `mileage_distribution.png` ➔ Histogramme des kilométrages  
- `metrics_comparison.png` ➔ Comparaison directe MSE / R²  

---

## 📌 Prédictions réalistes et contrôlées

Le modèle **prévient les extrapolations extrêmes** :  
- Les prédictions sont limitées à une **plage de confiance** basée sur le dataset d’origine (+/- 20%).  
- Si l'utilisateur saisit un kilométrage en dehors de ces bornes, un **message clair** est retourné, indiquant les limites valides.

---

## 📈 Métriques de performance

Les performances du modèle sont mesurées à l’aide des indicateurs suivants :  
| **Métrique**        | **Valeur**        |
|---------------------|-------------------|
| **MSE (Mean Squared Error)** | `445645.25` |
| **R² (Coefficient de détermination)** | `0.7330` ➔ Le modèle explique **73 %** de la variance |

Ces métriques sont affichées lors de l'exécution de `evaluate.py` et visualisées dans `metrics_comparison.png`.

---

## 🛠️ Commandes rapides

Le projet est entièrement **dockerisé** pour une exécution simple et reproductible.

```bash
make build      # Construire l'image Docker
make train      # Entraîner le modèle (lance train.py)
make predict    # Faire une prédiction (lance predict.py)
make evaluate   # Évaluer le modèle et générer les graphiques bonus (lance evaluate.py)
make test       # Lancer les tests unitaires
make clean      # Supprimer les ressources Docker inutilisées
make purge      # Supprimer complètement l'image Docker
```

---

## ⚙️ Structure des fichiers
```
.
├── data.csv                      # Dataset utilisé pour l'entraînement
├── train.py                      # Entraînement du modèle par descente de gradient
├── predict.py                    # Prédiction du prix d'un véhicule à partir du kilométrage
├── evaluate.py                   # Évaluation du modèle (MSE, R²) et génération des graphiques
├── model.json                    # Sauvegarde des paramètres θ et des normalisations
├── regression_plot.png           # Graphique de la droite de régression
├── cost_function_plot.png        # Courbe de coût J(θ)
├── residuals_plot.png            # Graphique des résidus
├── mileage_distribution.png      # Histogramme du kilométrage
├── data_distribution.png         # Boxplot des données
├── error_distribution.png        # Histogramme des erreurs
├── metrics_comparison.png        # Comparaison MSE / R²
├── Dockerfile                    # Dockerfile pour l'environnement d'exécution
├── Makefile                      # Automatisation des commandes courantes
└── requirements.txt              # Dépendances Python (numpy, matplotlib, etc.)
```

---

## 🎯 Conclusion

Ce projet **FT_LINEAR_REGRESSION** propose une **expérience complète et didactique** sur les concepts fondamentaux de la **régression linéaire** et de la **descente de gradient**, enrichie par des **visualisations**, des **métriques détaillées** et une **gestion robuste des données**.  

Grâce à une **interface utilisateur simple** et une **dockerisation complète**, il permet de **visualiser facilement** les performances du modèle, tout en respectant les contraintes pédagogiques du sujet **42**.

---
