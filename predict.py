# Загрузка обученных параметров
def load_model(filename):
    with open(filename, 'r') as f:
        theta0, theta1 = map(float, f.read().split(','))
    return theta0, theta1

# Функция для предсказания
def estimate_price(mileage, theta0, theta1):
    return theta0 + theta1 * mileage

# Основная программа
theta0, theta1 = load_model('model.txt')

mileage = float(input("Введите пробег машины (в км): "))
predicted_price = estimate_price(mileage, theta0, theta1)

print(f"Предсказанная цена для пробега {mileage} км: {predicted_price:.2f} у.е.")
