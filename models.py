
import tensorflow as tf
print(tf.__version__)

import numpy as np

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

def create_sequential_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=input_shape),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model





def apply_k_fold_on_model(X,y,n_splits=5, epochs=20, batch_size=32, random_state=42):
    X = np.array([df.to_numpy() for df in X])  # Assumes each DataFrame has a consistent shape - this is our Xs
    y = np.array(y)  # Convert list of booleans to NumPy array - these are labels
     
    # Reshape y to match the output of the model (e.g., (None, 1) if it's binary classification)
    y = y.reshape(-1, 1)
    
    # Initialize KFold
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    histories = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"Training on fold {fold + 1}...")
        
        # Splitting into K-Fold train and validation sets
        X_fold_train = X[train_idx]
        X_fold_val = X[val_idx]
        y_fold_train = y[train_idx]
        y_fold_val = y[val_idx]
        
        # Define the model using the provided create_model function
        model = create_sequential_model(input_shape=(X_fold_train.shape[1], X_fold_train.shape[2]))
        
        # Train the model
        history = model.fit(X_fold_train, y_fold_train, epochs=epochs, batch_size=batch_size,
                            validation_data=(X_fold_val, y_fold_val), verbose=1)
        
        histories.append(history)
    
    return model, histories

def apply_strapified_k_fold_on_model(X,y,n_splits=5, epochs=20, batch_size=32, random_state=42):
    X = np.array([df.to_numpy() for df in X])  # Assumes each DataFrame has a consistent shape - this is our Xs
    y = np.array(y)  # Convert list of booleans to NumPy array - these are labels
     
    # Reshape y to match the output of the model (e.g., (None, 1) if it's binary classification)
    y = y.reshape(-1, 1)
    
    # Initialize strapified KFold
    stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    histories = []
    
    for fold, (train_idx, val_idx) in enumerate(stratified_kfold.split(X, y.flatten())):
        print(f"Training on fold {fold + 1}...")
        
        # Splitting into K-Fold train and validation sets
        X_fold_train = X[train_idx]
        X_fold_val = X[val_idx]
        y_fold_train = y[train_idx]
        y_fold_val = y[val_idx]
        
        # Define the model using the provided create_model function
        model = create_sequential_model(input_shape=(X_fold_train.shape[1], X_fold_train.shape[2]))
        
        # Train the model
        history = model.fit(X_fold_train, y_fold_train, epochs=epochs, batch_size=batch_size,
                            validation_data=(X_fold_val, y_fold_val), verbose=1)
        
        histories.append(history)
    
    return model, histories