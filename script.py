import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve


# Charger les données depuis les fichiers CSV
labels_df = pd.read_csv('labels.csv')
data_df = pd.read_csv('data.csv')

# Fusionner les deux dataframes sur la colonne 'Sample_ID'
merged_df = pd.merge(labels_df, data_df, on='Sample_ID')

# Convertir les étiquettes en numériques avec LabelEncoder
label_encoder = LabelEncoder()
merged_df['Class'] = label_encoder.fit_transform(merged_df['Class'])

# Séparer les données en ensemble d'entraînement et ensemble de test
X_train, X_test, y_train, y_test = train_test_split(merged_df.drop(['Sample_ID', 'Class'], axis=1), merged_df['Class'], test_size=0.2, random_state=42)

# Modèle 1 (par exemple, RandomForestClassifier)
model1 = RandomForestClassifier(random_state=42)
model1.fit(X_train, y_train)
predictions1 = model1.predict(X_test)

# Modèle 2 (SVM)
model2 = SVC(random_state=42)
model2.fit(X_train, y_train)
predictions2 = model2.predict(X_test)

# Mesures de performance
accuracy1 = accuracy_score(y_test, predictions1)
precision1 = precision_score(y_test, predictions1, average='weighted')
recall1 = recall_score(y_test, predictions1, average='weighted')
f1_score1 = f1_score(y_test, predictions1, average='weighted')
conf_matrix1 = confusion_matrix(y_test, predictions1)

accuracy2 = accuracy_score(y_test, predictions2)
precision2 = precision_score(y_test, predictions2, average='weighted')
recall2 = recall_score(y_test, predictions2, average='weighted')
f1_score2 = f1_score(y_test, predictions2, average='weighted')
conf_matrix2 = confusion_matrix(y_test, predictions2)

# Afficher les résultats
print("Modèle 1: RFC")
print("Accuracy:", accuracy1)
print("Precision:", precision1)
print("Recall:", recall1)
print("F1 Score:", f1_score1)
print("Confusion Matrix:\n", conf_matrix1)

print("\nModèle 2: SVM")
print("Accuracy:", accuracy2)
print("Precision:", precision2)
print("Recall:", recall2)
print("F1 Score:", f1_score2)
print("Confusion Matrix:\n", conf_matrix2)

# Calcule les courbes d'apprentissage pour chaque modèle
train_sizes1, train_scores1, test_scores1 = learning_curve(model1, X_train, y_train, cv=5)
train_sizes2, train_scores2, test_scores2 = learning_curve(model2, X_train, y_train, cv=5)

# Calcul des moyennes et écart-types des scores pour chaque modèle
train_mean1 = np.mean(train_scores1, axis=1)
train_std1 = np.std(train_scores1, axis=1)
test_mean1 = np.mean(test_scores1, axis=1)
test_std1 = np.std(test_scores1, axis=1)

train_mean2 = np.mean(train_scores2, axis=1)
train_std2 = np.std(train_scores2, axis=1)
test_mean2 = np.mean(test_scores2, axis=1)
test_std2 = np.std(test_scores2, axis=1)

# Tracez les courbes d'apprentissage pour chaque modèle
plt.figure(figsize=(12, 8))

# Courbes pour le modèle 1 (RandomForestClassifier)
plt.subplot(2, 1, 1)
plt.plot(train_sizes1, train_mean1, color='blue', marker='o', markersize=5, label='Training Accuracy (RF)')
plt.fill_between(train_sizes1, train_mean1 - train_std1, train_mean1 + train_std1, alpha=0.15, color='blue')
plt.plot(train_sizes1, test_mean1, color='green', linestyle='--', marker='s', markersize=5, label='Validation Accuracy (RF)')
plt.fill_between(train_sizes1, test_mean1 - test_std1, test_mean1 + test_std1, alpha=0.15, color='green')
plt.xlabel('Number of Training Samples')
plt.ylabel('Accuracy')
plt.title('Learning Curve - Model 1 (RandomForestClassifier)')
plt.legend(loc='lower right')

# Courbes pour le modèle 2 (SVM)
plt.subplot(2, 1, 2)
plt.plot(train_sizes2, train_mean2, color='blue', marker='o', markersize=5, label='Training Accuracy (SVM)')
plt.fill_between(train_sizes2, train_mean2 - train_std2, train_mean2 + train_std2, alpha=0.15, color='blue')
plt.plot(train_sizes2, test_mean2, color='green', linestyle='--', marker='s', markersize=5, label='Validation Accuracy (SVM)')
plt.fill_between(train_sizes2, test_mean2 - test_std2, test_mean2 + test_std2, alpha=0.15, color='green')
plt.xlabel('Number of Training Samples')
plt.ylabel('Accuracy')
plt.title('Learning Curve - Model 2 (SVM)')
plt.legend(loc='lower right')

plt.tight_layout()
plt.show()