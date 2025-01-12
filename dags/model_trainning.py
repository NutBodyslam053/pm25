import duckdb
import mlflow
import mlflow.keras
import numpy as np
import pandas as pd
import tensorflow as tf
import boto3
import os
from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
from mlflow.models import infer_signature
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout, Reshape
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical


# Define the focal loss function
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        # Clip y_pred to avoid log(0) error
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

        # Calculate focal loss for each time step using one-hot encoded labels
        # tf.where requires y_true to have the same shape as y_pred

        # Calculate cross-entropy loss
        ce_loss = -y_true * K.log(y_pred)

        # Apply focal weights
        loss = alpha * K.pow(1.0 - y_pred, gamma) * ce_loss

        # Modification: Sum over the last axis to get a single loss value
        loss = K.sum(loss, axis=-1)

        # Average loss over the time dimension
        return K.mean(loss)  # Calculate the mean over the remaining dimensions

    return focal_loss_fixed


# Default arguments for the DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
}


@dag(
    description="A pipeline to train and evaluate an LSTM model for PM2.5 prediction.",
    default_args=default_args,
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=["ml", "pm2.5"],
)
def pm25_ml_pipeline():
    """Airflow pipeline for processing PM2.5 data and preparing it for ML models."""

    @task
    def data_extraction():
        """Load the dataset from DuckDB parquet files."""
        con = duckdb.connect(database=":memory:")
        df = con.read_parquet("data/pm25*.parquet").df()

        # Add 'day' and 'hour' columns
        df.insert(loc=3, column="day", value=df["timestamp"].dt.day)
        df.insert(loc=4, column="hour", value=df["timestamp"].dt.hour)

        # Remove timezone info
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        return df

    @task
    def data_preparation(df: pd.DataFrame):
        """Preprocess the data for ML."""
        features = [
            "year",
            "month",
            "day",
            "hour",
            "temperature",
            "humidity",
            "pressure",
            "wind_direction",
            "wind_speed",
            "pm2_5",
        ]

        df = df[features + ["pollution_level"]].copy()

        # Map pollution levels to numerical values
        pollution_level_mapping = {
            "คุณภาพอากาศดีมาก": 0,  # Very Good
            "คุณภาพอากาศดี": 1,  # Good
            "ปานกลาง": 2,  # Moderate
            "เริ่มมีผลกระทบต่อสุขภาพ": 3,  # Unhealthy for Sensitive Groups
            "มีผลกระทบต่อสุขภาพ": 4,  # Unhealthy
        }

        df["pollution_level_class"] = df["pollution_level"].map(pollution_level_mapping)

        # Sort data and remove duplicates
        df = df.sort_values(by=["year", "month", "day", "hour"], ascending=True)
        df = df.drop_duplicates(subset=features, keep="first")

        train_data = df[df["year"].isin([2021, 2022, 2023])]
        test_data = df[df["year"] == 2024]

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_train_features = scaler.fit_transform(train_data[features])
        scaled_test_features = scaler.transform(test_data[features])

        y_train_encoded = to_categorical(train_data["pollution_level_class"])
        y_test_encoded = to_categorical(test_data["pollution_level_class"])

        def create_sequences(data, target, look_back, predict_ahead):
            X, y = [], []
            for i in range(48, len(data) - look_back - predict_ahead + 1):
                X.append(data[i : i + look_back, :])
                y.append(target[i + look_back : i + look_back + predict_ahead])
            return np.array(X), np.array(y)

        look_back = 48  # Past 20 time steps
        predict_ahead = 12  # Predict the next 12 hours

        X_train, y_train = create_sequences(scaled_train_features, y_train_encoded, look_back, predict_ahead)
        X_test, y_test = create_sequences(scaled_test_features, y_test_encoded, look_back, predict_ahead)

        return X_train, y_train, X_test, y_test

    @task
    def model_training(prepared_data):
        """Train the LSTM model."""
        X_train, y_train, X_test, y_test = prepared_data

        model = Sequential()
        model.add(
            LSTM(
                units=50,
                return_sequences=True,
                input_shape=(X_train.shape[1], X_train.shape[2]),
            )
        )
        model.add(Dropout(0.2))
        model.add(LSTM(units=y_train.shape[1] * y_train.shape[2]))
        model.add(Dropout(0.2))
        model.add(Reshape((y_train.shape[1], y_train.shape[2])))
        model.add(Dense(units=y_train.shape[2], activation="softmax"))

        # Compile the model with focal loss
        model.compile(
            optimizer="adam",
            loss=focal_loss(alpha=0.25, gamma=2),
            metrics=["accuracy", "Precision", "Recall"],
        )

        # Add a model checkpoint callback
        checkpoint_filepath = "best_model.keras"
        model_checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=False,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        )

        # Train the model
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=200,
            batch_size=32,
            callbacks=[model_checkpoint_callback],
        )

        # Configure environment for s3fs
        os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
        os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin'
        os.environ['AWS_S3_ENDPOINT'] = 'http://localhost:9000'

        # Save the model to MinIO (localhost:9000) with bucket named 'mlflow'
        model.save('s3://mlflow/best_model.keras')
        print("Model training completed.")
        return  X_test, y_test

    @task
    def model_evaluation(training_result):
        """Evaluate the trained model and log it to MLflow."""
        X_test, y_test = training_result

        model = load_model(
            "s3://mlflow/best_model.keras",
            custom_objects={"focal_loss_fixed": focal_loss(alpha=0.25, gamma=2)},
        )
        test_loss, test_accuracy, test_precision, test_recall = model.evaluate(X_test, y_test)
        print(f"Test accuracy: {test_accuracy * 100:.2f}%")
        print(f"Test precision: {test_precision * 100:.2f}%")
        print(f"Test recall: {test_recall * 100:.2f}%")
        
        mlflow.set_tracking_uri("http://localhost:5000")
        with mlflow.start_run():
            mlflow.log_metric("test_loss", test_loss)
            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.log_metric("test_precision", test_precision)
            mlflow.log_metric("test_recall", test_recall)
            predictions = model.predict(X_test)
            signature = infer_signature(X_test, predictions)
            mlflow.keras.log_model(model, "model", signature=signature)
            print(f"Model saved in run {mlflow.active_run().info.run_uuid}")

    # Task dependencies
    raw_data = data_extraction()
    prepared_data = data_preparation(raw_data)
    training_result = model_training(prepared_data)
    model_evaluation(training_result)


pipeline = pm25_ml_pipeline()
