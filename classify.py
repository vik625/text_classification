from tensorflow.keras.models import load_model
import csv
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, GlobalMaxPooling1D
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

data = pd.read_csv('LabelledData.csv',delimiter='\t',header=None)
#data[1].replace({"  who": "who", "  what": "what"}, inplace=True)
data[1] = data[1].str.strip()
# print(data)
x = data.iloc[:,0]
y = data.iloc[:, 1]
# print(x)
data[1].value_counts()

labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)

onehotencoder = OneHotEncoder()
y = onehotencoder.fit_transform(y.reshape(-1,1)).toarray()
# print(y)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.6, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42) 
print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)

loaded_obj = 'https://tfhub.dev/google/universal-sentence-encoder/4'

model = Sequential()
model.add(hub.KerasLayer(loaded_obj, input_shape=[], dtype=tf.string, trainable=True))
#model.add(Dense(64, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(5, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
checkpoint = tf.keras.callbacks.ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True, verbose=1, save_weights_only=False)
history = model.fit(X_train, y_train, batch_size=16, epochs=6, shuffle=True, validation_data=(X_val,y_val),callbacks=[checkpoint])

model.save('model.h5')
model = load_model('model.h5', custom_objects={'KerasLayer':hub.KerasLayer})
model.summary()

# print test accuracy
model.evaluate(X_test, y_test)
# Print classification report
classes = data[1].unique().tolist()
print(classes)
y_pred = model.predict(X_test).argmax(axis=-1)
#print(y_pred)
y_pred = onehotencoder.fit_transform(y_pred.reshape(-1,1)).toarray()
#y_pred.shape
print(classification_report(y_test, y_pred, target_names=classes))