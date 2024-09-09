
import tensorflow as tf
print(tf.__version__)

from sklearn.model_selection import KFold
import tensorflow as tf
print("Tensorflow version",tf.__version__)
from tensorflow.keras.layers import GRU
GRU(128, recurrent_dropout=0.2, unroll=True)

from tensorflow.keras.models import Sequential

from sklearn.model_selection import KFold

from tensorflow.keras.utils import plot_model as keras_plot_model  # Explicit import to avoid conflict
from IPython.display import Image, display

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Conv1D, GlobalAveragePooling1D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
import numpy as np
from sklearn.model_selection import StratifiedKFold

from tensorflow.keras.layers import Input, LSTM, Dense, Attention, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import SimpleRNN
    

import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Attention, Concatenate, Dropout
from tensorflow.keras.models import Model

def create_attention_model(input_shape):
    inputs = Input(shape=input_shape)
    model = AttentionModel(input_shape=input_shape)(inputs)
    model = Model(inputs=inputs, outputs=model)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_lstm_attention_model(input_shape):
    inputs = Input(shape=input_shape)
    lstm_out = LSTM(64, return_sequences=True)(inputs)
    # Use only LSTM output without attention
    context_vector = tf.reduce_mean(lstm_out, axis=1)
    combined = Concatenate()([context_vector, context_vector])
    dense_out = Dense(64, activation='relu')(combined)
    dropout = Dropout(0.5)(dense_out)
    output = Dense(1, activation='sigmoid')(dropout)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


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

def create_functinal_api_model(input_shape):
    inputs = Input(shape=(input_shape,))

    # Create the hidden layers
    x = Dense(64, activation='relu')(inputs)
    x = Dense(32, activation='relu')(x)

    # Define the output layer
    outputs = Dense(1, activation='sigmoid')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])    

def create_malstm_fcn_model(input_shape):
    import os
    os.environ["TF_USE_CUDNN_RNN"] = "0"

    # Input Layer
    input_seq = Input(shape=input_shape)

    # LSTM Block
    x_lstm = LSTM(128, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0.2, unroll=True)(input_seq)
    x_lstm = Dropout(0.5)(x_lstm)

    # FCN Block
    x_fcn = Conv1D(filters=128, kernel_size=8, padding='same', activation='relu')(input_seq)
    x_fcn = BatchNormalization()(x_fcn)
    x_fcn = Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(x_fcn)
    x_fcn = BatchNormalization()(x_fcn)
    x_fcn = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x_fcn)
    x_fcn = BatchNormalization()(x_fcn)
    x_fcn = GlobalAveragePooling1D()(x_fcn)

    # Combine LSTM and FCN
    combined = tf.keras.layers.concatenate([x_lstm[:, -1, :], x_fcn])

    # Output Layer
    output = Dense(1, activation='sigmoid')(combined)

    # Create Model
    model = Model(inputs=input_seq, outputs=output)

    # Compile the model
    model.compile(optimizer=Adam(), loss=binary_crossentropy, metrics=['accuracy'])

    return model

def apply_k_fold_on_input_model(X,y,model_type,n_splits=5, epochs=40, batch_size=32, random_state=42):
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
        
        # based on input model type select the model, this way, same k-fold can be applied to different models.
        if model_type == "sequential" :
            model = create_sequential_model(input_shape=(X_fold_train.shape[1], X_fold_train.shape[2]))
        elif  model_type == "malstm_fcn":
            model = create_malstm_fcn_model(input_shape=(X_fold_train.shape[1], X_fold_train.shape[2]))
        elif model_type == "attn":
            model = create_attention_model(input_shape=(X_fold_train.shape[1], X_fold_train.shape[2]))
        else:
            model = create_functinal_api_model(input_shape=(X_fold_train.shape[1], X_fold_train.shape[2]))
           
        
        # Train the model
        history = model.fit(X_fold_train, y_fold_train, epochs=epochs, batch_size=batch_size,
                                validation_data=(X_fold_val, y_fold_val), verbose=1)
        histories.append(history)
    
    return model, histories


def apply_strapified_k_fold_on_input_model(X,y,model_type,n_splits=5, epochs=40, batch_size=32, random_state=42):
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
        
    
        # based on input model type select the model, this way, same k-fold can be applied to different models.
        if model_type == "sequential" :
            model = create_sequential_model(input_shape=(X_fold_train.shape[1], X_fold_train.shape[2]))
        elif  model_type == "malstm_fcn":
            model = create_malstm_fcn_model(input_shape=(X_fold_train.shape[1], X_fold_train.shape[2]))
        elif model_type == "attn":
            model = create_attention_model(input_shape=(X_fold_train.shape[1], X_fold_train.shape[2]))
        else:
            model = create_functinal_api_model(input_shape=(X_fold_train.shape[1], X_fold_train.shape[2]))
        

        
        # Train the model
        history = model.fit(X_fold_train, y_fold_train, epochs=epochs, batch_size=batch_size,
                                validation_data=(X_fold_val, y_fold_val), verbose=1)
        histories.append(history)
    
    return model, histories

def plot_model(model,title):
    print(f"Model name {title}")
    keras_plot_model(model, show_shapes=True, show_layer_names=True, to_file='model_diagram.png')
    display(Image(filename='model_diagram.png'))