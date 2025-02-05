import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse

def display_movement(theta0_n, theta1_n, mileages, prices):
    theta0, theta1 = denormalize(mileages, prices, theta0_n, theta1_n)
    x_line = [min(mileages), max(mileages)]
    y_line = [theta0 + theta1 * x for x in x_line]
    plt.clf()
    plt.scatter(mileages, prices, color='blue', label='Data points')
    plt.plot(x_line, y_line, color='red', label='Regression line')
    plt.xlabel('Mileage')
    plt.ylabel('Price')
    plt.title('Linear Regression Movement')
    plt.legend()
    plt.pause(0.001)

def gradient_descent(x, y, record_cost_history=False, plot_enabled=False):
    theta0_n = 0
    theta1_n = 0
    learning_rate = 0.1
    num_iterations = 2000
    m = len(x)

    x_numpy = np.array(x)
    y_numpy = np.array(y)
    cost_history = []

    for i in range(num_iterations):
        estimated_y = theta0_n + theta1_n * x_numpy
        errors = estimated_y - y_numpy 
        if record_cost_history:
            cost = np.sum(errors ** 2) / (2 * m)
            cost_history.append(cost)
        theta0_n_gradient = np.sum(errors) / m
        theta1_n_gradient = np.sum(errors * x_numpy) / m
        theta0_n -= learning_rate * theta0_n_gradient
        theta1_n -= learning_rate * theta1_n_gradient
        if plot_enabled and i % 10 == 0:
            display_movement(theta0_n, theta1_n, x, y)
    
    plt.clf()
    if record_cost_history:
        return theta0_n, theta1_n, cost_history
    else:
        return theta0_n, theta1_n

def load_data(filename):
    mileages = []
    prices = []
    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            mileages.append(float(row[0]))
            prices.append(float(row[1]))
    return mileages, prices

def denormalize(mileages, prices, theta0_n, theta1_n):
    mileage_min = min(mileages)
    mileage_max = max(mileages)
    mileage_range = mileage_max - mileage_min
    price_min = min(prices)
    price_max = max(prices)
    price_range = price_max - price_min

    theta0 = price_min + price_range * theta0_n - (price_range * theta1_n * mileage_min) / mileage_range
    theta1 = (price_range / mileage_range) * theta1_n
    return theta0, theta1

def normalize(mileages, prices):
    mileage_min = min(mileages)
    mileage_max = max(mileages)
    mileage_range = mileage_max - mileage_min
    price_min = min(prices)
    price_max = max(prices)
    price_range = price_max - price_min

    if mileage_range == 0 or price_range == 0:
        print("Cannot be normalized, division by 0.")
        sys.exit(1)

    mileages_n = [(m - mileage_min) / mileage_range for m in mileages]
    prices_n = [(p - price_min) / price_range for p in prices]
    return mileages_n, prices_n

def safe_results(theta0, theta1):
    with open("results.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["theta0", "theta1"])
        writer.writerow([theta0, theta1])

def coefficient_of_determination(mileages, prices, theta0, theta1):
    predictions = [theta0 + theta1 * m for m in mileages]
    mean_price = sum(prices) / len(prices)
    ss_tot = sum((p - mean_price) ** 2 for p in prices)
    ss_res = sum((p - pred) ** 2 for p, pred in zip(prices, predictions))
    if ss_tot == 0:
        return 1.0
    return 1 - ss_res / ss_tot

# Calculates the average of the squared differences between actual values (prices) and predicted values. Squaring emphasizes larger errors and gives a smooth gradient for optimization.
def mean_squared_error(prices, predictions):
    m = len(prices)
    return sum((p - pred)**2 for p, pred in zip(prices, predictions)) / m

# Takes the square root of mean squared error, returning the error in the same unit as the data. It's useful for intuitive understanding of prediction error size.
def root_mean_squared_error(prices, predictions):
    mse = mean_squared_error(prices, predictions)
    return mse ** 0.5

# Computes the average of the absolute differences between prices and predictions. It provides a straightforward measure of average error without squaring, making it less sensitive to outliers.
def mean_absolute_error(prices, predictions):
    m = len(prices)
    return sum(abs(p - pred) for p, pred in zip(prices, predictions)) / m

def main():
    parser = argparse.ArgumentParser(
        description="Train a linear regression model and display cost history, coefficient of determination (R^2), and plotting."
    )
    parser.add_argument("--cost-history", "-ch", action="store_true",
                        help="Display cost history plot in a separate window")
    parser.add_argument("--coefficient-determenation", "-cd", action="store_true",
                        help="Display the coefficient of determination (R^2) for the model")
    parser.add_argument("--plot", "-p", action="store_true",
                        help="Enable display of movement during training and final regression plot (enabled by default)")
    parser.add_argument("--save-pictures", "-sp", action="store_true",
                        help="Save the regression model and cost history plots as files")
    parser.add_argument("--mean_squared_error", "-ms", action="store_true",
                        help="Display MSE, RMSE and MAE calculated")
    args = parser.parse_args()
    
    if args.plot:
        plt.ion()

    mileages, prices = load_data("data.csv")
    mileages_n, prices_n = normalize(mileages, prices)
    
    if args.cost_history:
        theta0_n, theta1_n, cost_history = gradient_descent(mileages_n, prices_n, record_cost_history=True, plot_enabled=args.plot)
    else:
        theta0_n, theta1_n = gradient_descent(mileages_n, prices_n, record_cost_history=False, plot_enabled=args.plot)
    
    theta0, theta1 = denormalize(mileages, prices, theta0_n, theta1_n)
    safe_results(theta0, theta1)

    if args.mean_squared_error:
        predictions = [theta0 + theta1 * m for m in mileages]
        mse = mean_squared_error(prices, predictions)
        rmse = root_mean_squared_error(prices, predictions)
        mae = mean_absolute_error(prices, predictions)
        print(f"MSE: {mse:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"MAE: {mae:.3f}")

    if args.plot:
        plt.scatter(mileages, prices, color="blue", label="Data points")
        x_line = [min(mileages), max(mileages)]
        y_line = [theta0 + theta1 * x for x in x_line]
        plt.plot(x_line, y_line, color="green", label="Regression line")
        if args.save_pictures:
            plt.savefig("regression_model.png")
        plt.xlabel("Mileage")
        plt.ylabel("Price")
        plt.title("Linear Regression Model")
        plt.legend()
        plt.ioff()
        plt.show()

    if args.cost_history:
        plt.plot(cost_history)
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.title("Cost History")
        if args.save_pictures:
            plt.savefig("cost_history.png")
        plt.show()

    if args.coefficient_determenation:
        r2 = coefficient_of_determination(mileages, prices, theta0, theta1)
        print(f"Coefficient of Determination (R^2): {r2:.3f}")
    
    print(f"Theta0 = {theta0:.3f}, Theta1 = {theta1:.4f}")

if __name__ == "__main__":
    main()