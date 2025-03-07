import json
import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore, Style, init

# Initialisation de colorama
init(autoreset=True)


def load_model(filename="model.json"):
    """Charge les paramètres du modèle et les valeurs de normalisation."""
    try:
        with open(filename, "r") as f:
            model = json.load(f)
        return model
    except FileNotFoundError:
        print(f"{Fore.RED}❌ Erreur : modèle non trouvé. "
              f"Exécutez d'abord `train.py`." f"{Style.RESET_ALL}")
        exit(1)


def load_data(filename="data.csv"):
    """Charge les données depuis un fichier CSV."""
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    mileage, price = data[:, 0], data[:, 1]
    return mileage, price


def mean_squared_error(mileage, price, theta0, theta1):
    """Calcule l'erreur quadratique moyenne (MSE)
    entre les prix réels et estimés."""
    predictions = theta0 + theta1 * mileage
    mse = np.mean((predictions - price) ** 2)
    return mse


def plot_residuals(mileage, price, theta0, theta1):
    """Génère et sauvegarde un graphique des résidus."""
    predictions = theta0 + theta1 * mileage
    residuals = price - predictions

    plt.figure(figsize=(8, 6))
    plt.scatter(predictions, residuals, color='purple')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.xlabel('Prix prédits (€)')
    plt.ylabel('Résidus (€)')
    plt.title('Graphique des résidus')
    plt.grid(True)
    plt.savefig('residuals_plot.png')


def r_squared(mileage, price, theta0, theta1):
    """Calcule le coefficient de détermination (R²)."""
    predictions = theta0 + theta1 * mileage
    ss_total = np.sum((price - np.mean(price)) ** 2)
    ss_residual = np.sum((price - predictions) ** 2)
    r_squared = 1 - (ss_residual / ss_total)
    return r_squared


def plot_data_distribution(mileage, price):
    """Boxplot des données initiales (kilométrage et prix)."""
    plt.figure(figsize=(8, 6))
    plt.boxplot(
        [mileage, price],
        tick_labels=['Kilométrage (km)', 'Prix (€)'],
        vert=False
    )
    plt.title('Distribution initiale des données (Boxplot)')
    plt.grid(True)
    plt.savefig('data_distribution.png')
    print(f"{Fore.CYAN}📦 Distribution initiale sauvegardée sous "
          f"'data_distribution.png'{Style.RESET_ALL}")


def plot_error_distribution(mileage, price, theta0, theta1):
    """Affiche la distribution des erreurs de prédiction."""
    predictions = theta0 + theta1 * mileage
    errors = price - predictions

    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=10, color='purple', edgecolor='black')
    plt.xlabel('Erreur de prédiction (€)')
    plt.ylabel('Nombre de véhicules')
    plt.title('Distribution des erreurs de prédiction')
    plt.grid(True)
    plt.savefig('error_distribution.png')


def plot_mileage_distribution(mileage):
    """Histogramme montrant la répartition du kilométrage."""
    plt.figure(figsize=(8, 6))
    plt.hist(mileage, bins=10, color='green', edgecolor='black')
    plt.xlabel('Kilométrage (km)')
    plt.ylabel('Nombre de véhicules')
    plt.title('Distribution du kilométrage')
    plt.grid(True)
    plt.savefig('mileage_distribution.png')


if __name__ == "__main__":
    print(f"{Fore.CYAN}📂 Chargement du modèle et des données..."
          f"{Style.RESET_ALL}")

    model = load_model()
    mileage, price = load_data()

    theta0 = model["theta0"]
    theta1 = model["theta1"]

    mse = mean_squared_error(mileage, price, theta0, theta1)

    print(f"{Fore.YELLOW}📏 Erreur quadratique moyenne (MSE) : "
          f"{Fore.MAGENTA}{mse:.2f}{Style.RESET_ALL}")

    # Visualisation des résidus
    plot_residuals(mileage, price, theta0, theta1)
    print(f"{Fore.CYAN}📉 Graphique des résidus sauvegardé "
          f"(residuals_plot.png){Style.RESET_ALL}")

    # Distribution des erreurs
    plot_error_distribution(mileage, price, theta0, theta1)
    print(f"{Fore.CYAN}📊 Distribution des erreurs sauvegardée sous "
          f"'error_distribution.png'{Style.RESET_ALL}")

    # Graphiques bonus supplémentaires
    plot_data_distribution(mileage, price)
    print(f"{Fore.CYAN}📈 Distribution initiale des données sauvegardée sous "
          f"'data_distribution.png'{Style.RESET_ALL}")

    plot_mileage_distribution(mileage)
    print(f"{Fore.CYAN}🚗 Distribution du kilométrage sauvegardée sous "
          f"'mileage_distribution.png'{Style.RESET_ALL}")

    # Calcul du R²
    r2 = r_squared(mileage, price, theta0, theta1)
    print(f"{Fore.YELLOW}🔍 R² : {Fore.MAGENTA}{r2:.4f}{Style.RESET_ALL}")
