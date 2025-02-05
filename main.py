import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse

def display_movement(theta0_n, theta1_n, mileages, prices):
    # Denormalize parameters using our denormalize function
    theta0, theta1 = denormalize(mileages, prices, theta0_n, theta1_n)
    # Prepare the regression line using the extreme mileage values
    x_line = [min(mileages), max(mileages)]
    y_line = [theta0 + theta1 * x for x in x_line]
    # Plot data and current regression line
    plt.clf()  # clear current figure
    plt.scatter(mileages, prices, color='blue', label='Data points')
    plt.plot(x_line, y_line, color='red', label='Regression line')
    plt.xlabel('Mileage')
    plt.ylabel('Price')
    plt.title('Linear Regression Movement')
    plt.legend()
    plt.pause(0.001)  # pause to update plot view

# returns us normalized thetas.
def gradient_descent(x, y, record_cost_history=False):
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
        # Record cost if flag is enabled: cost = (1/2m) * sum(errors^2)
        if record_cost_history:
            cost = np.sum(errors ** 2) / (2 * m)
            cost_history.append(cost)
        theta0_n_gradient = np.sum(errors) / m
        theta1_n_gradient = np.sum(errors * x_numpy) / m
        theta0_n -= learning_rate * theta0_n_gradient
        theta1_n -= learning_rate * theta1_n_gradient
        if i % 10 == 0:
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost-history", "-ch", action="store_true", help="Display cost history plot in a separate window")
    args = parser.parse_args()
    
    plt.ion()
    mileages, prices = load_data("data.csv")
    mileages_n, prices_n = normalize(mileages, prices)
    
    
    if args.cost_history:
        theta0_n, theta1_n, cost_history = gradient_descent(mileages_n, prices_n, record_cost_history=True)
    else:
        theta0_n, theta1_n = gradient_descent(mileages_n, prices_n, record_cost_history=False)
    
    theta0, theta1 = denormalize(mileages, prices, theta0_n, theta1_n)
    safe_results(theta0, theta1)
    
    plt.scatter(mileages, prices, color="blue", label="Data points")
    x_line = [min(mileages), max(mileages)]
    y_line = [theta0 + theta1 * x for x in x_line]
    plt.plot(x_line, y_line, color="green", label="Regression line")
    plt.xlabel("Mileage")
    plt.ylabel("Price")
    plt.title("Linear Regression Model")
    plt.legend()
    plt.ioff()
    plt.show()
    
    if args.cost_history:
        plt.figure()
        plt.plot(cost_history)
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.title("Cost History")
        plt.show()

if __name__ == "__main__":
    main()