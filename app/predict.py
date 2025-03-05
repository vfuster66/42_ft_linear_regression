import json
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


def estimate_price(mileage, theta0, theta1):
    """Prédit le prix en fonction du kilométrage."""
    return theta0 + (theta1 * mileage)


if __name__ == "__main__":
    print(f"{Fore.CYAN}📂 Chargement du modèle...{Style.RESET_ALL}")
    model = load_model()

    theta0 = model["theta0"]
    theta1 = model["theta1"]
    mileage_mean = model["mileage_mean"]
    mileage_std = model["mileage_std"]
    price_mean = model["price_mean"]
    price_std = model["price_std"]

    mileage_input = f"{Fore.YELLOW}🚗 Entrez le kilométrage : {Style.RESET_ALL}"
    mileage = float(input(mileage_input))

    # Normalisation correcte du kilométrage
    mileage_norm = (mileage - mileage_mean) / mileage_std

    # Prédiction du prix
    predicted_price = estimate_price(mileage, theta0, theta1)

    print(f"{Fore.GREEN}💰 Prix estimé pour {Fore.BLUE}{mileage:.0f} km"
          f"{Fore.GREEN} : {Fore.MAGENTA}{predicted_price:.2f} €"
          f"{Style.RESET_ALL}")
