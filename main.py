import pandas as pd

def estimate_price(mileage, theta0, theta1):
    """Estimate the price using the linear regression hypothesis."""
    return theta0 + (theta1 * mileage)

def gradient_descent(data, learning_rate, num_iterations):
    """Perform gradient descent to learn theta0 and theta1."""
    theta0, theta1 = 0, 0  # Initialize parameters
    m = len(data)  # Number of data points

    for _ in range(num_iterations):
        tmp_theta0 = 0
        tmp_theta1 = 0

        # Calculate gradients
        for i in range(m):
            mileage = data.iloc[i, 0]
            price = data.iloc[i, 1]
            predicted_price = estimate_price(mileage, theta0, theta1)
            tmp_theta0 += (predicted_price - price)
            tmp_theta1 += (predicted_price - price) * mileage

        # Update theta0 and theta1 simultaneously
        theta0 -= (learning_rate * (1 / m) * tmp_theta0)
        theta1 -= (learning_rate * (1 / m) * tmp_theta1)

    return theta0, theta1

def main():
    # Load the dataset
    data = pd.read_csv("data.csv")

    # Hyperparameters
    learning_rate = 0.01
    num_iterations = 1000

    # Perform gradient descent
    theta0, theta1 = gradient_descent(data, learning_rate, num_iterations)

    # Save the results to result.csv
    results = pd.DataFrame({"theta0": [theta0], "theta1": [theta1]})
    results.to_csv("result.csv", index=False)

    print(f"Training complete! theta0 = {theta0}, theta1 = {theta1}")
    print("Results saved to result.csv")

if __name__ == "__main__":
    main()