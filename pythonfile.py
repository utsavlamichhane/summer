import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#rumen_and_fecaldata
df = pd.read_excel("entire_data.xlsx")

#tar
X = df.iloc[:, 1:]  # Features: all columns except the first one
y = df.iloc[:, 0]  # Target: the first column

#encode_categorical_as we have the RFI class as categorical
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# trainnn_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=250)

#arch
model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))  # IIILL
model.add(Dense(16, activation='relu'))  #HHH
model.add(Dense(len(set(y)), activation='softmax'))  

#compil
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#fitting_model
model.fit(X_train, y_train, epochs=100, batch_size=15)

#accuracy
_, accuracy = model.evaluate(X_test, y_test)
print(f"Model accuracy: {accuracy}")
