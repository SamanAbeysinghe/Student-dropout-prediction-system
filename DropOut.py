import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

data = pd.read_csv('D:/Level 6 & 7/Neural Network/Mini Project/Implementation/Data_set_2023.csv')
X = data.drop('Target', axis=1)  
y = data['Target'] 

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

data_encoded = pd.get_dummies(data, columns=['Target'])

X_train_temp, X_temp, y_train_temp, y_temp = train_test_split(X, y_encoded, test_size=0.8, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
X_train, X_validate, y_train, y_validate = train_test_split(X_train_temp, y_train_temp, test_size=0.25, random_state=42)

scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

model = Sequential()
model.add(Dense(256, input_dim=X_train.shape[1], activation='relu', kernel_regularizer='l2'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=300, batch_size=64, validation_split=0.2)

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
#print(f'Confusion Matrix:\n{conf_matrix}')
#print(f'Classification Report:\n{classification_rep}')

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')


plt.subplot(2, 1, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.tight_layout()

overall_accuracy = accuracy_score(y_test, y_pred)
print(f'Overall Accuracy: {overall_accuracy}')

plt.figure(figsize=(12, 8))
sns.boxplot(data=pd.DataFrame(X_train_normalized, columns=X_train.columns), palette='viridis')
plt.title('Boxplot of Normalized Features')
plt.xticks(rotation=45, ha='right')