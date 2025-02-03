import csv

def estimate_price(mileage, theta0, theta1):
    """Estimate the price using the linear regression hypothesis."""
    return theta0 + (theta1 * mileage)

def train_model(mileage, price, learning_rate=0.01, epochs=1000):
    theta0, theta1 = 0, 0
    m = len(mileage)
    
    for _ in range(epochs):
        error_sum0 = 0
        error_sum1 = 0
        
        for i in range(m):
            prediction = estimate_price(mileage[i], theta0, theta1)
            error = prediction - price[i]
            error_sum0 += error
            error_sum1 += error * mileage[i]
        
        tmp_theta0 = theta0 - (learning_rate * error_sum0 / m)
        tmp_theta1 = theta1 - (learning_rate * error_sum1 / m)
        
        theta0, theta1 = tmp_theta0, tmp_theta1

    return theta0, theta1
    


def load_data(filename):
    mileage = []
    price = []
    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header
        for row in reader:
            mileage.append(int(row[0]))
            price.append(int(row[1]))
    return mileage, price

def save_results(filename, theta0, theta1):
    """Save the results to a CSV file."""
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["theta0", "theta1"])
        writer.writerow([theta0, theta1])

def main():
    mileage, price = load_data("data.csv")
    # Perform gradient descent
    theta0, theta1 = train_model(mileage, price)

    # Save the results to result.csv
    save_results("result.csv", theta0, theta1)

    print(f"Training complete! theta0 = {theta0}, theta1 = {theta1}")
    print("Results saved to result.csv")

if __name__ == "__main__":
    main()