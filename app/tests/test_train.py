import sys
import os
import numpy as np
from colorama import Fore, Style, init
from train import train_model, estimate_price, mean_squared_error
from predict import estimate_price, load_model

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


init(autoreset=True)

# Définition des données de test
mileage_test = np.array([10000, 50000, 100000, 200000])
price_test = np.array([8000, 7000, 5000, 3000])


def test_train_model():
    """🔄 Test de l'entraînement du modèle avec des données factices."""
    print(f"{Fore.YELLOW}🔄 Test: Entraînement du modèle...{Style.RESET_ALL}")
    theta0, theta1 = train_model(mileage_test, price_test, learning_rate=0.0001, epochs=10000)
    assert theta0 != 0, "❌ θ0 ne doit pas être 0 après l'entraînement"
    assert theta1 != 0, "❌ θ1 ne doit pas être 0 après l'entraînement"
    print(f"{Fore.GREEN}✅ Entraînement validé !{Style.RESET_ALL}")


def test_estimate_price():
    """💰 Test de l'estimation du prix avec des valeurs arbitraires."""
    print(f"{Fore.YELLOW}💰 Test: Estimation de prix...{Style.RESET_ALL}")
    theta0, theta1 = 8499.50, -0.021448
    price = estimate_price(50000, theta0, theta1)
    assert 3000 < price < 9000, "❌ Le prix estimé doit être raisonnable"
    print(f"{Fore.GREEN}✅ Estimation correcte !{Style.RESET_ALL}")


def test_mean_squared_error():
    """📏 Test du calcul de l'erreur quadratique moyenne."""
    print(f"{Fore.YELLOW}📏 Test: Calcul de la MSE...{Style.RESET_ALL}")
    theta0, theta1 = 8499.50, -0.021448
    mse = mean_squared_error(mileage_test, price_test, theta0, theta1)
    assert mse >= 0, "❌ La MSE ne peut pas être négative"
    print(f"{Fore.GREEN}✅ MSE calculée avec succès !{Style.RESET_ALL}")

def test_load_model():
    """📂 Test du chargement du modèle."""
    print(f"{Fore.YELLOW}📂 Test: Chargement du modèle...{Style.RESET_ALL}")
    model = load_model("model.json")
    assert "theta0" in model, "❌ Le modèle doit contenir θ0"
    assert "theta1" in model, "❌ Le modèle doit contenir θ1"
    print(f"{Fore.GREEN}✅ Modèle chargé avec succès !{Style.RESET_ALL}")


def test_estimate_price():
    """💰 Test de la prédiction avec des valeurs connues."""
    print(f"{Fore.YELLOW}💰 Test: Prédiction de prix...{Style.RESET_ALL}")
    theta0, theta1 = 8499.50, -0.021448
    price_low = estimate_price(5000, theta0, theta1)
    price_high = estimate_price(300000, theta0, theta1)

    assert price_low > price_high, "❌ Une voiture avec moins de km doit coûter plus cher"
    assert price_low > 7000, "❌ Le prix d'une voiture récente doit être raisonnable"
    assert price_high < 3000, "❌ Le prix d'une voiture très kilométrée doit être faible"

    print(f"{Fore.GREEN}✅ Prédiction correcte !{Style.RESET_ALL}")