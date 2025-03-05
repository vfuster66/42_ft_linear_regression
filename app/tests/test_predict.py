import sys
import os
from predict import estimate_price, load_model
from colorama import Fore, Style, init

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


init(autoreset=True)


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
