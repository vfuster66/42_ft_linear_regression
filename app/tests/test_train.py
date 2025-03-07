import sys
import os
import numpy as np
import json
from colorama import Fore, Style, init
from train import (
    train_model,
    estimate_price,
    mean_squared_error,
    normalize_data,
    save_model_checkpoint,
)

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

init(autoreset=True)

mileage_test = np.array([10000, 50000, 100000, 200000])
price_test = np.array([8000, 7000, 5000, 3000])


def test_train_model():
    """🔄 Test de l'entraînement du modèle avec des données factices."""
    print(f"{Fore.YELLOW}🔄 Test: Entraînement du modèle...{Style.RESET_ALL}")

    mileage_mean, mileage_std = np.mean(mileage_test), np.std(mileage_test)
    price_mean, price_std = np.mean(price_test), np.std(price_test)

    mileage_norm = (mileage_test - mileage_mean) / mileage_std
    price_norm = (price_test - price_mean) / price_std

    theta0, theta1 = train_model(
        mileage_norm, price_norm, learning_rate=0.01, epochs=1000,
        mileage_mean=mileage_mean, mileage_std=mileage_std,
        price_mean=price_mean, price_std=price_std
    )

    assert theta0 != 0, "❌ θ0 ne doit pas être 0 après l'entraînement"
    assert theta1 != 0, "❌ θ1 ne doit pas être 0 après l'entraînement"

    print(f"{Fore.GREEN}✅ Entraînement validé !{Style.RESET_ALL}")


def test_mean_squared_error():
    """📏 Test du calcul de l'erreur quadratique moyenne."""
    print(f"{Fore.YELLOW}📏 Test: Calcul de la MSE...{Style.RESET_ALL}")
    theta0, theta1 = 8499.50, -0.021448
    mse = mean_squared_error(mileage_test, price_test, theta0, theta1)
    assert mse >= 0, "❌ La MSE ne peut pas être négative"
    print(f"{Fore.GREEN}✅ MSE calculée avec succès !{Style.RESET_ALL}")


def test_estimate_price():
    """💰 Test de la prédiction avec des valeurs connues."""
    print(f"{Fore.YELLOW}💰 Test: Prédiction de prix...{Style.RESET_ALL}")
    theta0, theta1 = 8499.50, -0.021448
    price_low = estimate_price(5000, theta0, theta1)
    price_high = estimate_price(300000, theta0, theta1)

    assert price_low > price_high, (
        "❌ Une voiture avec moins de km doit coûter plus cher"
    )
    assert price_low > 7000, (
        "❌ Le prix d'une voiture récente doit être raisonnable"
    )
    assert price_high < 3000, (
        "❌ Le prix d'une voiture très kilométrée doit être faible"
    )
    print(f"{Fore.GREEN}✅ Prédiction correcte !{Style.RESET_ALL}")


def test_normalize_denormalize():
    """🔄 Test: Normalisation et dénormalisation."""
    mileage_norm, price_norm, mileage_mean, \
        mileage_std, price_mean, price_std = \
        normalize_data(mileage_test, price_test)

    mileage_denorm = mileage_norm * mileage_std + mileage_mean
    price_denorm = price_norm * price_std + price_mean

    assert np.allclose(mileage_test, mileage_denorm), (
        "❌ Erreur de dénormalisation (kilométrage)"
    )
    assert np.allclose(price_test, price_denorm), (
        "❌ Erreur de dénormalisation (prix)"
    )

    print(f"{Fore.GREEN}✅ Normalisation/dénormalisation validée !"
          f"{Style.RESET_ALL}")


def test_checkpoint_saving():
    """📁 Test: Sauvegarde des checkpoints."""
    epoch_test = 12345
    theta0_test, theta1_test = 123, -0.045
    mileage_mean, mileage_std = 100000, 50000
    price_mean, price_std = 7000, 1000

    save_model_checkpoint(epoch_test, theta0_test, theta1_test,
                          mileage_mean, mileage_std, price_mean, price_std)

    checkpoint_filename = f"model_epoch_{epoch_test}.json"
    assert os.path.exists(checkpoint_filename), "❌ Checkpoint non sauvegardé"

    with open(checkpoint_filename, "r") as f:
        checkpoint = json.load(f)

    assert checkpoint["theta0"] == theta0_test, (
        "❌ θ0 incorrect dans le checkpoint"
    )
    assert checkpoint["theta1"] == theta1_test, (
        "❌ θ1 incorrect dans le checkpoint"
    )

    os.remove(checkpoint_filename)
    print(f"{Fore.GREEN}✅ Sauvegarde des checkpoints validée !"
          f"{Style.RESET_ALL}")
