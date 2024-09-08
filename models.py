
import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.model_selection import KFold

def create_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=input_shape),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model





def apply_k_fold_on_model(X_train,y_train,n_splits=5, epochs=20, batch_size=32, random_state=42):
    kfold = KFold(n_splits = 5, shuffle = True, random_state = 42)
    histories = []

    # Initialize KFold
    kfold = KFold(n_splits = n_splits, shuffle = True, random_state = random_state)
    histories = []

    # Perform K-Fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"Training on fold {fold + 1}...")

        # Splitting into K-Fold train and validation sets
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]

        # Define the model
        model = create_model(input_shape = (X_fold_train.shape[1], X_fold_train.shape[2]))

        # Train the model
        history = model.fit(X_fold_train, y_fold_train, epochs = epochs, batch_size = batch_size,
                            validation_data = (X_fold_val, y_fold_val), verbose = 1)

        histories.append(history)

    # After training all folds, evaluate the final model on the full dataset
    final_model = create_model(input_shape = (X.shape[1], X.shape[2]))
    final_model.fit(X, y, epochs = epochs, batch_size = batch_size, verbose = 1)
    final_eval = final_model.evaluate(X, y)

    return histories, final_model, final_eval