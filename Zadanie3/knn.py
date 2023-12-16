from collections import Counter
from math import sqrt
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split


def euclidean_distance(x, y):
    return sqrt(sum((a - b) ** 2 for a, b in zip(x, y)))


def manhattan_distance(x, y):
    return sum(abs(a - b) for a, b in zip(x, y))


def kNN_predict(train_data, train_labels, test_instance, k, distance_metric):
    distances = [(index, distance_metric(test_instance, train_data[index]))
                 for index in range(len(train_data))]

    distances.sort(key=lambda x: x[1])

    k_nearest_labels = [train_labels[index] for index, _ in distances[:k]]

    most_common_label = Counter(k_nearest_labels).most_common(1)

    return most_common_label[0][0]


def kNN_evaluate(train_data, train_labels, test_data, test_labels, k, distance_metric):
    correct_predictions = 0

    for i in range(len(test_data)):
        predicted_label = kNN_predict(train_data, train_labels, test_data[i], k, distance_metric)
        if predicted_label == test_labels[i]:
            correct_predictions += 1

    accuracy = correct_predictions / len(test_data)
    return accuracy


# Załaduj dane Iris
iris = load_iris()
iris_data = iris.data
iris_labels = iris.target

# Podziel dane Iris na zbiór treningowy i testowy
iris_train_data, iris_test_data, iris_train_labels, iris_test_labels = train_test_split(
    iris_data, iris_labels, test_size=0.2, random_state=42
)

# Przetestuj algorytm dla różnych wartości k i miar odległości na zbiorze Iris
k_values = [3, 5, 7]
distance_metrics = [euclidean_distance, manhattan_distance]

for k in k_values:
    for distance_metric in distance_metrics:
        accuracy = kNN_evaluate(
            iris_train_data, iris_train_labels, iris_test_data, iris_test_labels, k, distance_metric
        )
        print(f'Iris - k={k}, Distance Metric={distance_metric.__name__}, Accuracy={accuracy:.2f}')

# Załaduj dane Wine
wine = load_wine()
wine_data = wine.data
wine_labels = wine.target

# Podziel dane Wine na zbiór treningowy i testowy
wine_train_data, wine_test_data, wine_train_labels, wine_test_labels = train_test_split(
    wine_data, wine_labels, test_size=0.2, random_state=42
)

# Przetestuj algorytm dla różnych wartości k i miar odległości na zbiorze Wine
for k in k_values:
    for distance_metric in distance_metrics:
        accuracy = kNN_evaluate(
            wine_train_data, wine_train_labels, wine_test_data, wine_test_labels, k, distance_metric
        )
        print(f'Wine - k={k}, Distance Metric={distance_metric.__name__}, Accuracy={accuracy:.2f}')
