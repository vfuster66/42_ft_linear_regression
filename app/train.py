import numpy as np
import json
import matplotlib.pyplot as plt
from colorama import Fore, Style, init

# Initialisation de colorama pour Windows et autres plateformes
init(autoreset=True)

# Hyperparamètres
LEARNING_RATE = 0.0001
EPOCHS = 100000

MAX_THETA = 1e6


def load_data(filename="data.csv"):
    """Charge les données depuis un fichier CSV."""
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    mileage = data[:, 0]  # Kilométrage
    price = data[:, 1]  # Prix
    return mileage, price


def normalize_data(mileage, price):
    """Normalise les données pour éviter la divergence
    de la descente de gradient."""
    mileage_mean, mileage_std = np.mean(mileage), np.std(mileage)
    price_mean, price_std = np.mean(price), np.std(price)

    mileage_norm = (mileage - mileage_mean) / mileage_std
    price_norm = (price - price_mean) / price_std

    return (mileage_norm, price_norm, mileage_mean,
            mileage_std, price_mean, price_std)


def estimate_price(mileage, theta0, theta1):
    """Calcule le prix estimé en fonction du kilométrage."""
    return theta0 + (theta1 * mileage)


def train_model(mileage, price, learning_rate, epochs):
    """Entraîne un modèle de régression linéaire par descente de gradient."""
    theta0 = np.random.uniform(-1, 1)  # 🔵 Initialisation aléatoire autour de 0
    theta1 = np.random.uniform(-1, 1)

    m = len(mileage)

    for epoch in range(epochs):
        error = estimate_price(mileage, theta0, theta1) - price

        gradient0 = (1 / m) * np.sum(error)
        gradient1 = (1 / m) * np.sum(error * mileage)

        theta0 -= learning_rate * gradient0
        theta1 -= learning_rate * gradient1

        # ✅ Ajout d'une borne pour éviter l'explosion des valeurs
        theta0 = np.clip(theta0, -MAX_THETA, MAX_THETA)
        theta1 = np.clip(theta1, -MAX_THETA, MAX_THETA)

        # Afficher l'évolution toutes les 10 000 itérations
        if epoch % 10000 == 0:
            print(f"{Fore.YELLOW}📊 Epoch {epoch} - θ0: {theta0:.4f}, θ1: {theta1:.6f}{Style.RESET_ALL}")

    return theta0, theta1


def save_model(theta0, theta1, mileage_mean, mileage_std,
               price_mean, price_std, filename="model.json"):
    """Sauvegarde les paramètres du modèle et des valeurs de normalisation."""
    with open(filename, "w") as f:
        json.dump({
            "theta0": theta0,
            "theta1": theta1,
            "mileage_mean": mileage_mean,
            "mileage_std": mileage_std,
            "price_mean": price_mean,
            "price_std": price_std
        }, f)


def plot_regression(mileage, price, theta0, theta1,
                    filename="regression_plot.png"):
    """Sauvegarde le graphique au lieu
    de l'afficher pour éviter les problèmes avec Docker."""
    plt.figure(figsize=(8, 6))  # Taille du graphique
    plt.scatter(mileage, price, color='blue', label='Données réelles')

    # Tracer la droite de régression
    min_x, max_x = min(mileage), max(mileage)
    x_values = np.linspace(min_x, max_x, 100)
    y_values = theta0 + theta1 * x_values

    plt.plot(x_values, y_values, color='red', label='Régression linéaire')
    plt.xlabel('Kilométrage (km)')
    plt.ylabel('Prix (€)')
    plt.title('Régression Linéaire - Prédiction du prix des voitures')
    plt.legend()
    plt.grid(True)

    # Sauvegarde du graphique au lieu de l'afficher
    plt.savefig(filename)
    print(f"{Fore.CYAN}📊 Le graphique a été sauvegardé sous '{filename}'"
          f"{Style.RESET_ALL}")


def mean_squared_error(mileage, price, theta0, theta1):
    """Calcule l'erreur quadratique moyenne (MSE)
    entre les prix réels et estimés."""
    predictions = theta0 + theta1 * mileage
    mse = np.mean((predictions - price) ** 2)
    return mse


if __name__ == "__main__":
    print(f"{Fore.CYAN}📊 Chargement des données...{Style.RESET_ALL}")
    mileage, price = load_data()
    mileage_norm, price_norm, mileage_mean, mileage_std, \
        price_mean, price_std = normalize_data(mileage, price)

    print(f"{Fore.YELLOW}🔄 Entraînement du modèle en cours..."
          f"{Style.RESET_ALL}")
    theta0_norm, theta1_norm = train_model(
        mileage_norm, price_norm, LEARNING_RATE, EPOCHS)

    # Conversion des paramètres normalisés en valeurs réelles
    theta1 = theta1_norm * (price_std / mileage_std)
    theta0 = price_mean - (theta1 * mileage_mean)

    # Sauvegarde des paramètres du modèle
    save_model(theta0, theta1, mileage_mean, mileage_std,
               price_mean, price_std)

    print(f"{Fore.GREEN}✅ Modèle entraîné et sauvegardé !{Style.RESET_ALL}")
    print(f"{Fore.BLUE}📌 Paramètres appris : θ0 = {Fore.MAGENTA}{theta0:.2f}"
          f"{Style.RESET_ALL}, θ1 = {Fore.MAGENTA}{theta1:.6f}"
          f"{Style.RESET_ALL}")

    mse = mean_squared_error(mileage, price, theta0, theta1)

    # Affichage du graphique avec la régression linéaire
    plot_regression(mileage, price, theta0, theta1)
