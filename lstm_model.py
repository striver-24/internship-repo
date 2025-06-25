import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

def train_lstm_model():
    # --- 1) Load your prepared master DataFrame ---
    try:
        master = pd.read_csv('master_final_features.csv')
        print(f"Loaded data with {len(master)} records and {len(master.columns)} features.")
    except FileNotFoundError:
        try:
            master = pd.read_csv('uniform_emissions_data.csv')
            print(f"Loaded uniform emissions data with {len(master)} records and {len(master.columns)} features.")
        except FileNotFoundError:
            print("Please run feature_filtering.py first to generate the final features dataset.")
            return

    # Convert date column to datetime if it exists
    date_col = None
    for col in ['ds', 'date']:
        if col in master.columns:
            master[col] = pd.to_datetime(master[col])
            date_col = col
            break

    # Identify target variable
    target = None
    for col in ['y_vehicle', 'total_vehicle_emissions']:
        if col in master.columns:
            target = col
            break

    if target is None:
        print("Target variable not found. Please check your data.")
        return

    # Dynamically select features based on what's available
    numeric_cols = master.select_dtypes(include=[np.number]).columns.tolist()
    if target in numeric_cols:
        numeric_cols.remove(target)
    for col in ['y_packaging', 'y_waste', 'total_packaging_emissions', 'total_waste_emissions']:
        if col in numeric_cols and col in master.columns:
            numeric_cols.remove(col)
    features = numeric_cols[:min(5, len(numeric_cols))]

    print(f"Using features: {features}")
    print(f"Target variable: {target}")

    # Drop rows with NaN values in features or target
    df = master.dropna(subset=features + [target]).reset_index(drop=True)
    print(f"Dataset after dropping NaN values: {len(df)} records")

    # --- 2) Scale inputs & target separately ---
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(df[features])
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(df[[target]])

    # --- 3) Create sequences ---
    def create_sequences(X, y, window=14):
        Xs, ys = [], []
        for i in range(len(X)-window):
            Xs.append(X[i:i+window])
            ys.append(y[i+window])
        return np.array(Xs), np.array(ys)

    WINDOW_SIZE = 14
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, window=WINDOW_SIZE)

    print(f"Created {len(X_seq)} sequences with window size {WINDOW_SIZE}")
    print(f"Sequence shape: {X_seq.shape}, Target shape: {y_seq.shape}")

    # split into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_seq, y_seq, test_size=0.2, shuffle=False
    )

    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")

    # --- 4) Build and train the final LSTM model with best hyperparameters ---
    n_features = X_train.shape[2]

    model = Sequential()
    model.add(LSTM(
        32,
        return_sequences=True,
        recurrent_dropout=0.1,
        kernel_regularizer=l2(1e-4),
        recurrent_regularizer=l2(1e-4),
        input_shape=(WINDOW_SIZE, n_features)
    ))
    model.add(Dropout(0.2))
    model.add(LSTM(
        32,
        recurrent_dropout=0.1,
        kernel_regularizer=l2(1e-4),
        recurrent_regularizer=l2(1e-4)
    ))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(
        optimizer=Adam(learning_rate=0.00112),
        loss='huber_loss'
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2),
        ModelCheckpoint('best_lstm_model.h5', monitor='val_loss', save_best_only=True)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=64,
        callbacks=callbacks,
        verbose=2
    )

    # --- 5) Save the model ---
    model.save('lstm_emissions_model.h5')
    print("Model saved as 'lstm_emissions_model.h5'")

    # --- 6) Make predictions on validation data ---
    print("\nEvaluating model performance...")
    val_predictions = model.predict(X_val)

    # Inverse transform predictions and actual values
    val_predictions = scaler_y.inverse_transform(val_predictions)
    y_val_inv = scaler_y.inverse_transform(y_val)

    # Calculate metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(y_val_inv, val_predictions)
    rmse = np.sqrt(mean_squared_error(y_val_inv, val_predictions))
    r2 = r2_score(y_val_inv, val_predictions)

    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    # --- 7) Plot predictions vs actual ---
    plt.figure(figsize=(12, 6))

    # Plot training history
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot predictions vs actual
    plt.subplot(1, 2, 2)
    plt.plot(y_val_inv, label='Actual')
    plt.plot(val_predictions, label='Predicted')
    plt.title('LSTM Model: Predicted vs Actual Emissions')
    plt.xlabel('Time Step')
    plt.ylabel('Emissions')
    plt.legend()

    plt.tight_layout()
    plt.savefig('lstm_predictions.png')
    print("Results visualization saved as 'lstm_predictions.png'")

    print("\nLSTM model training complete!")
    return model, history, features

if __name__ == "__main__":
    train_lstm_model()