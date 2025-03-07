import sys
import os
from colorama import Fore, Style, init
from predict import load_model, batch_predict

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
)

init(autoreset=True)


def test_load_model():
    """📂 Test du chargement du modèle."""
    print(f"{Fore.YELLOW}📂 Test: Chargement du modèle...{Style.RESET_ALL}")
    model = load_model("model.json")
    assert "theta0" in model, "❌ Le modèle doit contenir θ0"
    assert "theta1" in model, "❌ Le modèle doit contenir θ1"
    print(f"{Fore.GREEN}✅ Modèle chargé avec succès !{Style.RESET_ALL}")


def test_batch_predict_limits():
    """⚠️ Test: Vérification des limites batch."""
    model = load_model()
    theta0, theta1 = model["theta0"], model["theta1"]

    valid_mileages = [5000, 150000, 250000]
    invalid_mileages = [-100, 350000]

    predictions_valid = batch_predict(valid_mileages, theta0, theta1)
    assert predictions_valid is not None, \
        "❌ Batch valide ne doit pas retourner None"
    assert len(predictions_valid) == len(valid_mileages), \
        "❌ Mauvais nombre de prédictions retournées"

    predictions_invalid = batch_predict(invalid_mileages, theta0, theta1)
    assert predictions_invalid is None, "❌ Batch invalide doit retourner None"

    print(f"{Fore.GREEN}✅ Limites des prédictions batch validées !"
          f"{Style.RESET_ALL}")
