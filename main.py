import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse

def display_movement(theta0_n, theta1_n, mileages, prices):
    try:
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
    except Exception as e:
        print(f"Error in display: {str(e)}")
        plt.close('all')
    

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
        if record_cost_history: # Cost history tracks how the error (cost/loss) of the model changes during training.    Measures how well the model fits the data
            cost = np.sum(errors ** 2) / (2 * m)  # Mean squared error calculation
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
    try:
        with open(filename) as f:
            reader = csv.reader(f)  # Creates an iterator for the CSV file
            next(reader)            # Moves to next row using __next__() method
            for row in reader:
                mileages.append(float(row[0]))
                prices.append(float(row[1]))
        if not mileages or not prices:
            raise ValueError("Empty data lists")
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        sys.exit(1)
    except StopIteration:
        print("Error: Empty file or missing data")
        sys.exit(1)
    except (ValueError, IndexError) as e:
        print(f"Error: Invalid data format - {str(e)}")
        sys.exit(1)
        
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

    EPSILON = 1e-10  # Small value to prevent division by very small numbers
    if abs(mileage_range) < EPSILON or abs(price_range) < EPSILON:
        print("Error: Data range too small for normalization")
        sys.exit(1)

    mileages_n = [(m - mileage_min) / mileage_range for m in mileages]
    prices_n = [(p - price_min) / price_range for p in prices]
    return mileages_n, prices_n

def safe_results(theta0, theta1):
    try:
        with open("results.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["theta0", "theta1"])
            writer.writerow([theta0, theta1])
    except (IOError, OSError) as e:
        print(f"Error saving results: {str(e)}")
        sys.exit(1)

# The coefficient of determination (R²) measures how well a regression model fits the data
# R² ranges from 0 to 1
# R² = 1: perfect fit
# R² = 0: model explains none of the variability
def coefficient_of_determination(mileages, prices, theta0, theta1):
    predictions = [theta0 + theta1 * m for m in mileages]
    mean_price = sum(prices) / len(prices)
    ss_tot = sum((p - mean_price) ** 2 for p in prices) # Total variance
    ss_res = sum((p - pred) ** 2 for p, pred in zip(prices, predictions)) # Residual variance
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
        plt.ion() # for real-time plot updates and showing plot without blocking programm execution

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
    try:
        main()
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)

 # Bonuses implemented:
# 1. Normalization: Enables handling data of any scale without adjusting learning rate
# 2. Learning visualization: Displays regression line movement during training 
# 3. Final plot: Shows regression line with denormalized values
# 4. Cost history: Tracks and displays model fit improvement over iterations
# 5. Coefficient of determination (R²): Measures goodness of fit
# 6. Optional visualization: Enable/disable plots with -p or -ch flags
# 7. Save plots: Export regression and cost history plots with -sp flag
# 8. Additional metrics: MSE, RMSE, MAE for model evaluation

# bonuses
# 1. Normalization. allows you to work with big or small data without adjusting the learning rate.
# 2. Learning visualization. Shows you how the red curve is moving in the plot while learning.
# 3. Final grapth. Displayed final green line regression with denormalized walues.
# 4. cost history calculation and display. Allows you to see the cost history, which tells you how well the model fits the data
# 5. Coefficient of determmination. 
# 6. ability to turn on and turn off visualisation with flags -p or -ch
# 7. ability to safe final pictures of regression and cost function to the working folder automaticly whth flag -sp
# 8. ability to calculate additional statistical measures like mean_squared_error, root_mean_squared_error, mean_absolute_error
 