import csv
import numpy as np
import matplotlib.pyplot as plt

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
def gradient_descent(x, y):
    theta0_n = 0
    theta1_n = 0
    learning_rate = 0.1
    num_iterations = 2000
    m = len(x)

    # np => easier to multiply array on one number for estimation.
    x_numpy = np.array(x)
    y_numpy = np.array(y)

    for i in range(num_iterations):
        estimated_y = theta0_n + theta1_n * x_numpy
        errors = estimated_y - y_numpy # we will get array of errors, how strongly predicted values diviates from the actual one
        theta0_n_gradient = np.sum(errors) / m
        theta1_n_gradient = np.sum(errors * x_numpy) / m
        theta0_n -= learning_rate * theta0_n_gradient
        theta1_n -= learning_rate * theta1_n_gradient
        if i % 10 == 0:
            display_movement(theta0_n, theta1_n, x, y)

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

    if (mileage_range == 0 or price_range == 0):
        print("Cannot be normalized, division by 0.")
        exit(-1)

    # normalization of data. now in list we will have values from 0 to 1 (from min value to max value)
    mileages_n = []
    for mileage in mileages:
        mileages_n.append((mileage - mileage_min) / mileage_range)
    
    prices_n = []
    for price in prices:
        prices_n.append((price - price_min) / price_range)
    
    return mileages_n, prices_n

def main():
    plt.ion()
    mileages, prices = load_data("data.csv")
    mileages_n, prices_n = normalize(mileages, prices)
    theta0_n, theta1_n = gradient_descent(mileages_n, prices_n)
    plt.clf()
    theta0, theta1 = denormalize(mileages, prices, theta0_n, theta1_n)


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

if __name__ == "__main__":
    main()