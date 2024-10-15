import numpy as np
import matplotlib.pyplot as plt
import random
import math
import sys
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Parameters
TOTAL_POINTS = 1000  # Number of data points to simulate
ANOMALY_FRACTION = 0.05  # Fraction of data points that are anomalies
THRESHOLD = 7.8  # Number of standard deviations for anomaly detection

def simulate_data_stream(total_points=TOTAL_POINTS, anomaly_fraction=ANOMALY_FRACTION):
    """
    Generator function to simulate a data stream with regular patterns, seasonality, and noise.
    Occasionally injects anomalies.

    Yields:
        value (float): The data point value.
        is_anomaly (bool): True if the data point is an anomaly, False otherwise.
    """
    t = 0
    while t < total_points:
        try:
            # Regular pattern: sine wave
            regular = 10 * np.sin(2 * np.pi * t / 50)

            # Trend component
            trend = 0.05 * t

            # Random noise
            noise = np.random.normal(0, 2)

            # Combine components
            value = regular + trend + noise

            # Inject anomaly
            is_anomaly = False
            if random.random() < anomaly_fraction:
                # Anomaly: add a large spike
                anomaly = np.random.normal(20, 5)
                value += anomaly
                is_anomaly = True

            yield value, is_anomaly
            t += 1
        except Exception as e:
            print(f"Error in data stream at point {t}: {e}")
            break

class KalmanFilterAnomalyDetector:
    def __init__(self, process_variance=1e-2, measurement_variance=4, initial_estimate=0.0, initial_error=1.0, threshold=THRESHOLD):
        """
        Initializes the Kalman Filter parameters.
        """
        self.process_variance = process_variance          # Q: Process variance
        self.measurement_variance = measurement_variance  # R: Measurement variance
        self.estimate = initial_estimate                  # Initial state estimate
        self.error_covariance = initial_error             # Initial estimation error covariance
        self.threshold = threshold                        # Threshold for anomaly detection

    def update(self, measurement):
        """
        Updates the Kalman Filter with a new measurement and detects anomalies.
        Returns:
            is_anomaly (bool): Whether the current measurement is an anomaly.
            residual (float): The residual (innovation) value.
            residual_std (float): The standard deviation of the residual.
        """
        # Prediction step
        predicted_estimate = self.estimate
        predicted_error_covariance = self.error_covariance + self.process_variance

        # Kalman Gain
        kalman_gain = predicted_error_covariance / (predicted_error_covariance + self.measurement_variance)

        # Update step
        self.estimate = predicted_estimate + kalman_gain * (measurement - predicted_estimate)
        self.error_covariance = (1 - kalman_gain) * predicted_error_covariance

        # Calculate residual (innovation)
        residual = measurement - predicted_estimate
        residual_std = np.sqrt(predicted_error_covariance + self.measurement_variance)

        # Anomaly detection
        is_anomaly = abs(residual) > self.threshold * residual_std
        return is_anomaly, residual, residual_std

def main():
    # Initialize data structures
    stream = simulate_data_stream()
    detector = KalmanFilterAnomalyDetector()
    data_points = []
    estimate_data = []
    anomaly_x = []
    anomaly_y = []
    true_labels = []
    predicted_labels = []
    residuals = []
    residual_stds = []
    t = 0

    # Set up real-time plotting
    plt.style.use('seaborn-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    plt.ion()
    plt.show()

    x_data = []

    while t < TOTAL_POINTS:
        try:
            value, is_anomaly = next(stream)
            is_detected_anomaly, residual, residual_std = detector.update(value)

            data_points.append(value)
            estimate_data.append(detector.estimate)
            residuals.append(residual)
            residual_stds.append(residual_std)
            x_data.append(t)
            true_labels.append(is_anomaly)
            predicted_labels.append(is_detected_anomaly)

            if is_detected_anomaly:
                anomaly_x.append(t)
                anomaly_y.append(value)

            # Update plots every 10 data points
            if t % 10 == 0 or t == TOTAL_POINTS - 1:
                # Data Stream Plot
                ax1.clear()
                ax1.plot(x_data, data_points, label='Data Stream', color='blue')
                ax1.plot(x_data, estimate_data, label='Kalman Estimate', color='green')
                ax1.scatter(anomaly_x, anomaly_y, color='red', label='Detected Anomalies')
                ax1.set_title('Kalman Filter Anomaly Detection')
                ax1.set_xlabel('Time')
                ax1.set_ylabel('Value')
                ax1.legend()

                # Residuals Plot
                ax2.clear()
                ax2.plot(x_data, residuals, label='Residuals', color='purple')
                threshold_upper = [detector.threshold * std for std in residual_stds]
                threshold_lower = [-detector.threshold * std for std in residual_stds]
                ax2.plot(x_data, threshold_upper, '--', label='Threshold Upper', color='red')
                ax2.plot(x_data, threshold_lower, '--', label='Threshold Lower', color='red')
                ax2.set_title('Residuals Over Time')
                ax2.set_xlabel('Time')
                ax2.set_ylabel('Residual')
                ax2.legend()

                plt.pause(0.001)

            t += 1
        except StopIteration:
            break
        except Exception as e:
            print(f"An error occurred at point {t}: {e}")
            break

    # After processing all data points, calculate and print metrics
    print("\nData processing complete. Calculating metrics...\n")

    y_true = np.array(true_labels)
    y_pred = np.array(predicted_labels)

    # Handle edge cases
    if np.sum(y_pred) == 0 and np.sum(y_true) == 0:
        print("=== Anomaly Detection Evaluation Metrics ===")
        print(f"Total Data Points: {len(y_true)}")
        print("No anomalies detected and no anomalies present in the data.")
        print("============================================\n")
    elif np.sum(y_pred) == 0:
        print("=== Anomaly Detection Evaluation Metrics ===")
        print(f"Total Data Points: {len(y_true)}")
        print("No anomalies detected.")
        print("============================================\n")
    elif np.sum(y_true) == 0:
        print("=== Anomaly Detection Evaluation Metrics ===")
        print(f"Total Data Points: {len(y_true)}")
        print("No anomalies present in the data.")
        print("============================================\n")
    else:
        # Calculate confusion matrix
        try:
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                tn, fp, fn, tp = 0, 0, 0, 0  # Handle unexpected cases

            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            print("=== Anomaly Detection Evaluation Metrics ===")
            print(f"Total Data Points: {len(y_true)}")
            print(f"True Positives (TP): {tp}")
            print(f"False Positives (FP): {fp}")
            print(f"True Negatives (TN): {tn}")
            print(f"False Negatives (FN): {fn}")
            print(f"Precision: {precision:.2f}")
            print(f"Recall: {recall:.2f}")
            print(f"F1 Score: {f1:.2f}")
            print("============================================\n")
        except Exception as e:
            print(f"An error occurred during metrics calculation: {e}\n")

    # Keep the plot open until manually closed
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
