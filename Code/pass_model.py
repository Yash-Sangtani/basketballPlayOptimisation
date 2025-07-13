import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from tensorflow.keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

dates, aways, homes = [], [], []
with open('./allgames.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        components = line.split('.')
        dates.append('.'.join(components[:3]))
        aways.append(components[3])
        homes.append(components[5].split()[0])
NOO = ['01.01.2016.TOR.CHA.7z','01.01.2016.WAS.ORL.7z','01.02.2016.BOS.BKN.7z','01.02.2016.IND.DET.7z','01.02.2016.UTA.MEM.7z',
       '01.02.2016.MIN.MIL.7z','01.02.2016.CLE.ORL.7z']
def load_data(lists):
    X, y = [], []
    directory = './../Tracking_data/pass_features'
    for date, away, home in tqdm(list(lists)):  # Convert to list for tqdm
        file_name = f'{date}.{away}.{home}.pkl'
        if file_name in NOO:
            continue
        data_path = os.path.join(directory, file_name)
        df = pd.read_pickle(data_path)
        X.append(df)
    return pd.concat(X, ignore_index=True).astype('float32')

# Load and prepare data
data_lists = zip(dates, aways, homes)
X = load_data(data_lists)
X = X.loc[X['backcourt'] == 0]
print(len(X))
X.dropna(inplace=True)
print(len(X))
y = X['pass_successful']
X = X.drop('pass_successful', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

feature_names = X.columns.tolist()

# Building and compiling the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the model
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.4)

#y_pred = (model.predict(X_test) > 0.5).astype("int32")
y_pred = (model.predict(X_test) > 0.5).astype("int32")
y_pred_prob = model.predict(X_test).ravel()  # Prediction probabilities needed for ROC AUC

weights = model.layers[0].get_weights()[0]
print(weights)
print(feature_names)
print(y_pred_prob)
print("F1 Score: ", f1_score(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("ROC AUC Score: ", roc_auc_score(y_test, y_pred_prob))

#Plotting the ROC curve.
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc_score(y_test, y_pred_prob))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
#Plotting the Losses and Val Losses.
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

# Plotting the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss, 'bo', label='Training loss') 
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
