import sys
import os
import numpy as np
import pytest
from colorama import Fore, Style, init

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluate import mean_squared_error, load_model, load_data

init(autoreset=True)


def test_evaluate_mse():
    """📏 Test du calcul de la MSE depuis evaluate.py"""
    print(f"{Fore.YELLOW}📏 Test: Calcul de la MSE depuis evaluate.py...{Style.RESET_ALL}")
    model = load_model()
    mileage, price = load_data()

    theta0 = model["theta0"]
    theta1 = model["theta1"]

    mse = mean_squared_error(mileage, price, theta0, theta1)
    assert mse >= 0, "❌ La MSE ne peut pas être négative"

    print(f"{Fore.GREEN}✅ MSE correcte dans evaluate.py !{Style.RESET_ALL}")
