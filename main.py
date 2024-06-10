from collections import Counter

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, balanced_accuracy_score


# 1
file_path = 'dataset3_l4.csv'
data = pd.read_csv(file_path)

# 2, 3
print(f'2. к-кість записів: {data.shape[0]}')
print(f'3. атрибути набору даних: {",".join(data.columns)}')

# 4
n_splits = int(input("введіть к-кість варіантів перемішування (>=3): "))
if n_splits < 3:
    raise ValueError("!!! к-кість варіантів перемішування >= 3.")

shuffle_split = ShuffleSplit(n_splits=n_splits, test_size=.25, random_state=0)
print(shuffle_split)
np.set_printoptions(threshold=9)

option = 2

for i, (train_index, test_index) in enumerate(shuffle_split.split(data), start=1):
    print(f"Варіант перемішування {i}:")
    print(f"  Train: {train_index}, length={len(train_index)}")
    print(f"  Test:  {test_index}, length={len(test_index)}\n")

    if i == option - 1:
        train, test = train_index, test_index


splits = list(shuffle_split.split(data))
train_indices, test_indices = splits[1]

train_data = data.iloc[train_indices]
test_data = data.iloc[test_indices]

column_to_check = data.columns[0]
train_column_counts = train_data[column_to_check].value_counts()
test_column_counts = test_data[column_to_check].value_counts()

# для перевірки збалансованості
print(f"розподіл значень у стовпці '{column_to_check}' у навчальній вибірці:")
print(train_column_counts)

print(f"розподіл значень у стовпці '{column_to_check}' у тестовій вибірці:")
print(test_column_counts)

# 5
k_neighbors_classifier = KNeighborsClassifier()

X_train = train_data.drop(columns=['NObeyesdad'])
y_train = train_data['NObeyesdad']
X_test = test_data.drop(columns=['NObeyesdad'])
y_test = test_data['NObeyesdad']

# категоріальні змінні -> числові
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"точність моделі на тестовій вибірці: {accuracy:.2f}")

# Виведення кількох прогнозованих та реальних значень для перевірки
print("\nпрогнозовані значення:")
print(y_pred[:10])

print("\nреальні значення:")
print(y_test.values[:10])

# 6
def calculate_metrics(model, x_cord, y_cord):
    all_metrics = {'accuracy': 0,
                   'precision': 0,
                   'recall': 0,
                   'f_scores': 0,
                   'MCC': 0,
                   'BA': 0}

    model_predictions = model.predict(x_cord)
    all_metrics['accuracy'] = accuracy_score(y_cord, model_predictions)
    all_metrics['precision'] = precision_score(y_cord, model_predictions, average='weighted')
    all_metrics['recall'] = recall_score(y_cord, model_predictions, average='weighted')
    all_metrics['f_scores'] = f1_score(y_cord, model_predictions, average='weighted')
    all_metrics['MCC'] = matthews_corrcoef(y_cord, model_predictions)
    all_metrics['BA'] = balanced_accuracy_score(y_cord, model_predictions)

    return all_metrics


metrics_test_df = calculate_metrics(knn, X_test, y_test)
# metrics_train_df = calculate_metrics(knn, X_train, y_train)

df_test_train_graph = pd.DataFrame({'Test': metrics_test_df})
df_test_train_graph.plot(kind='bar', figsize=(10, 6))
plt.title('метрики для тестової вибірки')
plt.ylabel('значення')
plt.xlabel('метрика')
plt.xticks(rotation=45)
plt.savefig(f'metrics.png')

# 7
p_values = list(range(1, 21))
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
mcc_scores = []
ba_scores = []

for p in p_values:
    knn = KNeighborsClassifier(n_neighbors=3, p=p)
    knn.fit(X_train, y_train)
    metrics = calculate_metrics(knn, X_test, y_test)

    accuracy_scores.append(metrics['accuracy'])
    precision_scores.append(metrics['precision'])
    recall_scores.append(metrics['recall'])
    f1_scores.append(metrics['f_scores'])
    mcc_scores.append(metrics['MCC'])
    ba_scores.append(metrics['BA'])

plt.figure(figsize=(12, 8))
plt.plot(p_values, accuracy_scores, label='Accuracy')
plt.plot(p_values, precision_scores, label='Precision')
plt.plot(p_values, recall_scores, label='Recall')
plt.plot(p_values, f1_scores, label='F1 Score')
plt.plot(p_values, mcc_scores, label='MCC')
plt.plot(p_values, ba_scores, label='Balanced Accuracy')
plt.xlabel('степінь метрики мінковського')
plt.ylabel('значення метрик')
plt.title('вплив степеня метрики мінковського на результати класифікації')
plt.legend()
plt.grid(True)
plt.savefig('minkowski_metrics.png')
plt.show()


