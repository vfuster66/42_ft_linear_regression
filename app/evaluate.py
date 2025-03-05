import json
import numpy as np
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
              f"Exécutez d'abord `train.py`.{Style.RESET_ALL}")
        exit(1)


def load_data(filename="data.csv"):
    """Charge les données depuis un fichier CSV."""
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    mileage = data[:, 0]  # Kilométrage
    price = data[:, 1]  # Prix
    return mileage, price


def mean_squared_error(mileage, price, theta0, theta1):
    """Calcule l'erreur quadratique moyenne (MSE) entre les prix réels
    et estimés."""
    predictions = theta0 + theta1 * mileage
    mse = np.mean((predictions - price) ** 2)
    return mse


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
