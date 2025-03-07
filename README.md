# 📊 FT_LINEAR_REGRESSION - Implémentation d'une régression linéaire simple

## 📌 Objectif du projet
Ce projet implémente un modèle de **régression linéaire simple** afin de **prédire le prix d'un véhicule en fonction de son kilométrage**.  
L'objectif est d'utiliser **la descente de gradient** pour ajuster les paramètres du modèle et obtenir **une prédiction fiable**.

---

## 📖 Bonus implémentés

✅ **Sauvegarde et chargement du modèle en JSON**  
✅ **Visualisation graphique des données et de la régression linéaire** (`regression_plot.png`)  
✅ **Visualisation graphique des résidus** (`residuals_plot.png`)  
✅ **Histogramme des erreurs de prédiction** (`error_distribution.png`)  
✅ **Boxplot de la distribution initiale des données** (`data_distribution.png`)  
✅ **Histogramme de la répartition du kilométrage** (`mileage_distribution.png`)  
✅ **Comparaison graphique entre MSE et R²** (`metrics_comparison.png`)  
✅ **Prédictions en batch** avec extrapolation modérée (±20%)  
✅ **Sauvegarde régulière des checkpoints durant l'entraînement**  
✅ **Calcul du coefficient R² pour l'évaluation alternative**  
✅ **Normalisation et dénormalisation des données pour éviter la divergence**  
✅ **Gestion complète des erreurs et des limites de prédictions**  

---

## 🚀 Comment visualiser les résultats graphiques ?
Les graphiques générés (format `.png`) sont sauvegardés automatiquement dans le dossier `app/` lors de l’exécution des scripts. Ils incluent :
- `regression_plot.png` : données initiales et droite de régression.
- `residuals_plot.png` : résidus du modèle.
- `error_distribution.png` : histogramme des erreurs.
- `data_distribution.png` : boxplot du kilométrage et prix.
- `mileage_distribution.png` : histogramme du kilométrage.
- `metrics_comparison.png` : comparaison directe des métriques MSE et R².

---

## 📌 Prédictions réalistes
Le modèle bloque automatiquement les prédictions non réalistes, en limitant les prédictions aux kilométrages observés dans le dataset initial, avec une extrapolation modérée de ±20%. En dehors de ces limites, une erreur informative est affichée, indiquant clairement les bornes réalistes définies.

---

## 📈 Métriques de performance

- **Erreur quadratique moyenne (MSE)** : mesure précise des erreurs de prédiction (445645.25).
- **Coefficient de détermination (R²)** : mesure alternative de précision (0.7330), indiquant que le modèle explique environ **73% de la variance** des données.

Ces métriques sont clairement affichées à chaque exécution du script `evaluate.py`.

---

## 🛠️ Commandes rapides pour utiliser le projet

```bash
make build      # Construire l'image Docker
make train      # Entraîner le modèle
make predict    # Faire une prédiction
make evaluate   # Évaluer le modèle (génère tous les graphiques bonus)
make test       # Exécuter les tests unitaires
make clean      # Nettoyer les ressources Docker inutilisées
make purge      # Supprimer complètement l'image Docker
```

---

## 🎯 Conclusion
Avec ces bonus clairement intégrés et documentés, ce projet FT_LINEAR_REGRESSION permet une compréhension complète, visuelle et interactive des principes fondamentaux de la régression linéaire et de ses performances, facilitant ainsi une meilleure appréhension des concepts statistiques et de machine learning de base.