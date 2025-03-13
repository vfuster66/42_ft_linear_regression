import numpy as np
import json
import matplotlib.pyplot as plt
from colorama import Fore, Style, init

# Initialisation de colorama
init(autoreset=True)

# Hyperparamètres
LEARNING_RATE = 0.0001
EPOCHS = 100000
MAX_THETA = 1e6


def load_data(filename="data.csv"):
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    mileage = data[:, 0]
    price = data[:, 1]
    return mileage, price


def normalize_data(mileage, price):
    mileage_mean, mileage_std = np.mean(mileage), np.std(mileage)
    price_mean, price_std = np.mean(price), np.std(price)

    mileage_norm = (mileage - mileage_mean) / mileage_std
    price_norm = (price - price_mean) / price_std

    return (mileage_norm, price_norm, mileage_mean,
            mileage_std, price_mean, price_std)


def estimate_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)


def train_model(mileage, price, learning_rate, epochs, mileage_mean,
                mileage_std, price_mean, price_std):
    theta0 = np.random.uniform(-1, 1)
    theta1 = np.random.uniform(-1, 1)

    m = len(mileage)
    cost_history = []  # 🔵 On initialise la liste pour stocker J(θ)

    for epoch in range(epochs):
        predictions = estimate_price(mileage, theta0, theta1)
        error = predictions - price

        gradient0 = (1 / m) * np.sum(error)
        gradient1 = (1 / m) * np.sum(error * mileage)

        theta0 -= learning_rate * gradient0
        theta1 -= learning_rate * gradient1

        theta0 = np.clip(theta0, -MAX_THETA, MAX_THETA)
        theta1 = np.clip(theta1, -MAX_THETA, MAX_THETA)

        # ✅ Calcul du coût J(θ) à chaque epoch
        cost = (1 / (2 * m)) * np.sum(error ** 2)
        cost_history.append(cost)

        if epoch % 10000 == 0:
            print(f"{Fore.YELLOW}📊 Epoch {epoch} - θ0: {theta0:.4f}, "
                  f"θ1: {theta1:.6f} - Coût J(θ): {cost:.6f}{Style.RESET_ALL}")

            save_model_checkpoint(epoch, theta0, theta1,
                                  mileage_mean, mileage_std,
                                  price_mean, price_std)

    return theta0, theta1, cost_history


def save_model(theta0, theta1, mileage_mean, mileage_std,
               price_mean, price_std, filename="model.json"):
    with open(filename, "w") as f:
        json.dump({
            "theta0": theta0,
            "theta1": theta1,
            "mileage_mean": mileage_mean,
            "mileage_std": mileage_std,
            "price_mean": price_mean,
            "price_std": price_std
        }, f)


def save_model_checkpoint(epoch, theta0, theta1, mileage_mean, mileage_std,
                          price_mean, price_std):
    checkpoint_filename = f"model_epoch_{epoch}.json"
    with open(checkpoint_filename, "w") as f:
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
    plt.figure(figsize=(8, 6))
    plt.scatter(mileage, price, color='blue', label='Données réelles')

    min_x, max_x = min(mileage), max(mileage)
    x_values = np.linspace(min_x, max_x, 100)
    y_values = theta0 + theta1 * x_values

    plt.plot(x_values, y_values, color='red', label='Régression linéaire')
    plt.xlabel('Kilométrage (km)')
    plt.ylabel('Prix (€)')
    plt.title('Régression Linéaire - Prédiction du prix des voitures')
    plt.legend()
    plt.grid(True)

    plt.savefig(filename)
    print(f"{Fore.CYAN}📊 Le graphique a été sauvegardé sous '{filename}'"
          f"{Style.RESET_ALL}")


def plot_cost_function(cost_history, filename="cost_function_plot.png"):
    """Trace la courbe de coût J(θ) en fonction des itérations."""
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(cost_history)), cost_history, color='green')
    plt.title("Courbe de coût J(θ) pendant l'apprentissage")
    plt.xlabel("Itérations")
    plt.ylabel("Coût J(θ)")
    plt.grid(True)
    plt.savefig(filename)
    print(f"{Fore.CYAN}📊 Courbe de coût sauvegardée sous '{filename}'"
          f"{Style.RESET_ALL}")


def mean_squared_error(mileage, price, theta0, theta1):
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
    theta0_norm, theta1_norm, cost_history = train_model(
        mileage_norm, price_norm, LEARNING_RATE, EPOCHS,
        mileage_mean, mileage_std, price_mean, price_std)

    # Conversion des paramètres normalisés en valeurs réelles
    theta1 = theta1_norm * (price_std / mileage_std)
    theta0 = price_mean - (theta1 * mileage_mean)

    save_model(theta0, theta1, mileage_mean, mileage_std,
               price_mean, price_std)

    print(f"{Fore.GREEN}✅ Modèle entraîné et sauvegardé !{Style.RESET_ALL}")
    print(f"{Fore.BLUE}📌 Paramètres appris : θ0 = {Fore.MAGENTA}{theta0:.2f}"
          f"{Style.RESET_ALL}, θ1 = {Fore.MAGENTA}{theta1:.6f}"
          f"{Style.RESET_ALL}")

    mse = mean_squared_error(mileage, price, theta0, theta1)
    print(f"{Fore.BLUE}📏 MSE sur l'ensemble des données : "
          f"{Fore.MAGENTA}{mse:.2f}{Style.RESET_ALL}")

    plot_regression(mileage, price, theta0, theta1)
    plot_cost_function(cost_history)
