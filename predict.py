import csv
import argparse
import sys

def read_results():
    try:
        with open("results.csv") as f:
            reader = csv.reader(f)
            next(reader)
            row = next(reader)
            theta0 = float(row[0])
            theta1 = float(row[1])
    except FileNotFoundError:
        print("Error: 'results.csv' file not found.")
        sys.exit(1)
    except StopIteration:
        print("Error: 'results.csv' does not contain the expected number of rows.")
        sys.exit(1)
    except (ValueError, IndexError):
        print("Error: Could not convert the CSV values to floats.")
        sys.exit(1)

    return theta0, theta1

def main():
    parser = argparse.ArgumentParser(description="Predict car price from mileage using stored regression parameters.")
    parser.add_argument("mileage", nargs="?", type=float, help="Mileage of car")
    args = parser.parse_args()

    if args.mileage is None:
        try:
            mileage = float(input("Enter the mileage: "))
        except ValueError:
            print("Invalid input. Please enter a valid number.")
            return
    else:
        mileage = args.mileage
    if mileage < 0:
        print("Error: Mileage cannot be negative")
        sys.exit(1)
    if mileage > sys.float_info.max:
        print("Error: Mileage value too large")
        sys.exit(1)
    
    theta0, theta1 = read_results()
    predict_price = theta0 + theta1 * mileage
    print(f"Predicted price: {predict_price:.3f}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)()