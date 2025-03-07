import json
import numpy as np
from colorama import Fore, Style, init


init(autoreset=True)


def load_model(filename="model.json"):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"{Fore.RED}❌ Modèle non trouvé, exécutez 'train.py'."
              f"{Style.RESET_ALL}")
        exit(1)


def load_data(filename="data.csv"):
    """Charge les données depuis un fichier CSV."""
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    mileage, price = data[:, 0], data[:, 1]
    return mileage, price


def batch_predict(mileage_list, theta0, theta1, min_limit=0, max_limit=300000):
    """Prédit les prix pour une liste de kilométrages avec validation des
    limites réalistes."""
    mileage_array = np.array(mileage_list)

    if np.any(mileage_array < min_limit) or np.any(mileage_array > max_limit):
        print(f"{Fore.RED}Kilométrage hors limites réalistes "
              f"({min_limit} km - {max_limit} km).{Style.RESET_ALL}")
        return None

    predicted_prices = theta0 + theta1 * mileage_array
    predicted_prices = np.maximum(predicted_prices, 0)

    return predicted_prices


if __name__ == "__main__":
    model = load_model()
    mileage_data, _ = load_data()

    min_real, max_real = np.min(mileage_data), np.max(mileage_data)

    margin = 0.20
    min_limit = max(0, min_real * (1 - margin))
    max_limit = max_real * (1 + margin)

    print(f"{Fore.CYAN}📂 Chargement du modèle terminé.{Style.RESET_ALL}")
    mileage_input = input(
        f"{Fore.YELLOW}🚗 Entrez un ou plusieurs kilométrages "
        f"(séparés par des virgules) : {Style.RESET_ALL}"
    )
    mileage_list = [float(x.strip()) for x in mileage_input.split(',')]

    predictions = batch_predict(
        mileage_list, model["theta0"], model["theta1"], min_limit, max_limit
    )

    if predictions is not None:
        for mileage, predicted_price in zip(mileage_list, predictions):
            print(f"{Fore.GREEN}💰 Prix estimé pour {Fore.BLUE}{mileage:.0f} km"
                  f"{Fore.GREEN} : {Fore.MAGENTA}{predicted_price:.2f} €"
                  f"{Style.RESET_ALL}")
