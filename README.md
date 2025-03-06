# 📊 FT_LINEAR_REGRESSION - Implémentation d'une régression linéaire simple

## 📌 Objectif du projet
Ce projet implémente un modèle de **régression linéaire simple** afin de **prédire le prix d'un véhicule en fonction de son kilométrage**.  
L'objectif est d'utiliser **la descente de gradient** pour ajuster les paramètres du modèle et obtenir **une prédiction fiable**.

---

## 📖 Qu'est-ce que la régression linéaire ?

### 🔹 Définition
La **régression linéaire** est une méthode statistique permettant **de modéliser la relation entre une variable dépendante (Y) et une ou plusieurs variables indépendantes (X)** à l’aide d’une **droite d’équation** :

\[
Y = \theta_0 + \theta_1 X + \epsilon
\]

où :
- **\( Y \)** est la variable cible (ici, le prix d’un véhicule),
- **\( X \)** est la variable explicative (ici, le kilométrage du véhicule),
- **\( \theta_0 \)** est l’ordonnée à l’origine (le prix estimé quand \( X = 0 \)),
- **\( \theta_1 \)** est la pente (la variation du prix en fonction du kilométrage),
- **\( \epsilon \)** représente l’erreur de modélisation.

La régression linéaire trouve **les valeurs optimales de \( \theta_0 \) et \( \theta_1 \)** qui **minimisent l’erreur** entre les prédictions et les valeurs réelles.

---

## 📌 À quoi sert la régression linéaire ?

### 🔹 Exemples d’utilisation
La régression linéaire est **utilisée dans de nombreux domaines** pour **prédire des valeurs futures** ou **analyser l’impact d’une variable sur une autre**.

🔹 **Économie et finance**  
- Prédiction des **ventes** d’un produit en fonction de la publicité.
- Estimation de **l’évolution des prix immobiliers** selon la surface.

🔹 **Marketing & Business Intelligence**  
- Prédiction du **chiffre d’affaires** d’une entreprise en fonction des tendances du marché.
- Analyse de l’impact d’une **campagne publicitaire** sur les ventes.

🔹 **Science et environnement**  
- Prédiction des **températures futures** en fonction des années.
- Estimation du **taux de pollution** en fonction du trafic routier.

🔹 **Domaine médical**  
- Prédiction du **risque de maladies** en fonction du mode de vie (ex : poids, tabagisme).
- Estimation de l’espérance de vie en fonction de critères socio-économiques.

### 🔹 Pourquoi utiliser la régression linéaire ?
✅ **Facile à comprendre et interpréter** 📊  
✅ **Modèle rapide et efficace** 🚀  
✅ **Exige peu de données** 📉  
✅ **Applicable à de nombreux domaines** 🌍  

---

## 🏗️ Structure du projet

```bash
. ├── app/ 
  │ ├── train.py # Script d'entraînement du modèle 
  │ ├── predict.py # Script de prédiction du prix
  │ ├── evaluate.py # Évaluation de la précision du modèle
  │ ├── data.csv # Dataset d'entraînement
  │ ├── model.json # Modèle sauvegardé après entraînement
  │ ├── requirements.txt # Dépendances Python
  │ ├── tests/ # Dossier des tests unitaires
  │ │ ├── test_train.py # Tests pour train.py
  │ │ ├── test_predict.py # Tests pour predict.py
  │ │ ├── test_evaluate.py # Tests pour evaluate.py
  ├── Dockerfile # Image Docker du projet
  ├── Makefile # Commandes automatisées
  ├── README.md # Documentation du projet
```

---

## 🚀 Installation & Exécution

### **1️⃣ Prérequis**
- **Python 3.10+**
- **Docker & Docker Compose** installé sur votre machine

### **2️⃣ Cloner le projet**
```sh
git clone https://github.com/votre-repo/42_ft_linear_regression.git
cd 42_ft_linear_regression
```

### **3️⃣ Construire l'image Docker**

```bash
make build
```

### **4️⃣ Entraîner le modèle**

```bash
make train
```

📊 Entraîne le modèle et sauvegarde les paramètres θ0 et θ1.

### **5️⃣ Faire une prédiction**

```bash
make predict
```

🚗 Demande un kilométrage et retourne le prix estimé.

### **6️⃣ Évaluer la précision du modèle**

```bash
make evaluate
```

📏 Affiche l'erreur quadratique moyenne (MSE) pour mesurer la précision.

### **7️⃣ Exécuter les tests unitaires**

```bash
make test
```

✅ Vérifie que tous les scripts fonctionnent correctement.

---

## 📊 Explication du modèle

La régression linéaire modélise la relation entre **le kilométrage d’un véhicule et son prix de revente** :

\[
\text{estimatePrice} = \theta_0 + \theta_1 \times \text{mileage}
\]

Le modèle ajuste **θ₀** et **θ₁** grâce à **l’algorithme de descente de gradient** :

\[
\theta_0 = \theta_0 - \alpha \times \frac{1}{m} \sum_{i=1}^{m} (\text{estimatePrice} - \text{price})
\]

\[
\theta_1 = \theta_1 - \alpha \times \frac{1}{m} \sum_{i=1}^{m} (\text{estimatePrice} - \text{price}) \times \text{mileage}
\]

💡 **Explication :**
- **`θ₀`** représente l'ordonnée à l'origine (**le prix de base du véhicule**).
- **`θ₁`** est la pente de la régression (**l’impact du kilométrage sur le prix**).
- **La descente de gradient** met à jour `θ₀` et `θ₁` à chaque itération pour minimiser l'erreur.

🔢 **Objectif :** Trouver `θ₀` et `θ₁` de manière à réduire l'erreur entre les prix réels et les prix prédits.

---

## ✅ Résultats & Performance

### 📌 Paramètres entraînés :
```bash
θ0 = 8499.55
θ1 = -0.021448
```

### 📌 Erreur quadratique moyenne (MSE) après entraînement :
```bash
MSE = 445645.25
```

---

## 🚀 Pour aller plus loin

### 🔹 Limites de la régression linéaire
Bien que la **régression linéaire** soit un modèle simple et efficace, elle présente **certaines limites** :
- **Hypothèse de linéarité** : elle suppose que la relation entre les variables est **toujours une droite**, ce qui **n'est pas toujours vrai**.
- **Sensibilité aux valeurs aberrantes** : quelques points extrêmes peuvent **fortement influencer** le modèle.
- **Ne prend pas en compte les interactions complexes** entre plusieurs variables.

Pour pallier ces limites, **d'autres modèles existent** en Machine Learning.

---

## 📌 Autres modèles de régression

### 1️⃣ **Régression polynomiale**  
💡 **Permet d'ajuster des courbes au lieu de droites**.  
Elle ajoute des **termes quadratiques, cubiques, etc.** pour mieux **capturer des relations non linéaires**.

\[
Y = \theta_0 + \theta_1 X + \theta_2 X^2 + \theta_3 X^3 + ... + \epsilon
\]

✅ **Avantage** : fonctionne mieux pour des relations courbes.  
❌ **Inconvénient** : risque d'**overfitting** si trop de termes sont ajoutés.

---

### 2️⃣ **Régression logarithmique et exponentielle**  
💡 **Utile lorsque les données suivent une croissance ou décroissance exponentielle**.  
- **Régression exponentielle** : modèle la croissance rapide d'une valeur  
  \[
  Y = a \times e^{bX}
  \]
- **Régression logarithmique** : adaptée aux phénomènes où la croissance ralentit avec le temps  
  \[
  Y = a + b \times \log(X)
  \]

✅ **Avantage** : s'adapte aux tendances exponentielles.  
❌ **Inconvénient** : moins intuitif à interpréter.

---

### 3️⃣ **Régression multiple**  
💡 **Utilise plusieurs variables pour améliorer les prédictions**.  
Exemple : prédire le prix d’un appartement en fonction de **sa surface, son quartier et son année de construction**.

\[
Y = \theta_0 + \theta_1 X_1 + \theta_2 X_2 + \theta_3 X_3 + ... + \epsilon
\]

✅ **Avantage** : prend en compte plusieurs facteurs.  
❌ **Inconvénient** : nécessite un dataset plus grand et un bon prétraitement des données.

---

## 📌 Modèles plus avancés (Machine Learning)

### 4️⃣ **Régression Ridge & Lasso** (Regularization)  
💡 **Ajoutent une pénalité pour éviter le sur-ajustement (overfitting).**  
- **Régression Ridge** : réduit l'importance des coefficients avec une pénalité **L2**.
- **Régression Lasso** : force certains coefficients à **zéro** pour sélectionner automatiquement les variables les plus importantes.

✅ **Avantage** : empêche les modèles trop complexes d’overfitter.  
❌ **Inconvénient** : plus difficile à interpréter qu'une simple régression linéaire.

---

### 5️⃣ **Arbres de décision et forêts aléatoires** 🌳  
💡 **Utilisent des divisions successives pour prédire une valeur**.  
Ils fonctionnent bien quand la relation entre les variables est **très non linéaire**.

✅ **Avantage** : fonctionne sur des relations complexes sans besoin de transformation des données.  
❌ **Inconvénient** : peut être instable si l’on ne règle pas bien les hyperparamètres.

---

### 6️⃣ **Réseaux de neurones (Deep Learning) 🧠**  
💡 **Utilisés pour des prédictions ultra précises** grâce à plusieurs couches de neurones artificiels.

✅ **Avantage** : excellent pour les **grands volumes de données** et les **relations très complexes**.  
❌ **Inconvénient** : demande **beaucoup de données et de puissance de calcul**.

---

## 📚 Sources & Documentation

- 📖 [Cours OpenClassrooms - Régression linéaire](https://openclassrooms.com/fr/courses/4525326)
- 📖 [Descente de gradient - Wikipédia](https://fr.wikipedia.org/wiki/Descente_de_gradient)
- 📖 [Documentation officielle `numpy`](https://numpy.org/doc/)
- 📖 [Documentation officielle `pytest`](https://docs.pytest.org/en/stable/)
- 📖 [Cours OpenClassrooms - Machine Learning](https://openclassrooms.com/fr/courses/4525326)
- 📖 [Comparaison des modèles de régression](https://scikit-learn.org/stable/supervised_learning.html)
- 📖 [Documentation officielle `scikit-learn`](https://scikit-learn.org/stable/)
- 📖 [Tutoriel sur les arbres de décision](https://towardsdatascience.com/decision-trees-in-machine-learning-641b9c4e8052)

