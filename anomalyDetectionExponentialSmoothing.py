import numpy as np
import matplotlib.pyplot as plt
import random
import math
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Parameters for Holt-Winters Exponential Smoothing
ALPHA = 0.5  # Level smoothing factor
BETA = 0.1   # Trend smoothing factor
GAMMA = 0.1  # Seasonality smoothing factor
SEASONAL_PERIOD = 50  # Seasonality period
THRESHOLD = 3  # Number of standard deviations for anomaly detection

# Number of data points to simulate
TOTAL_POINTS = 1000  # Adjust as needed

def data_stream(total_points=TOTAL_POINTS):
    """
    Generator function to simulate a real-time data stream.
    Incorporates regular patterns, seasonal elements, random noise, and injected anomalies.
    
    Yields:
    - value: The generated data point.
    - is_anomaly: Boolean indicating if the data point is an anomaly.
    """
    t = 0
    for _ in range(total_points):
        # Regular pattern: sine wave to simulate seasonality
        seasonal = 10 * math.sin(2 * math.pi * t / SEASONAL_PERIOD)
        # Trend component
        trend = 0.05 * t
        # Random noise
        noise = np.random.normal(0, 2)
        # Simulated anomaly
        anomaly = 0
        is_anomaly = False
        if random.random() < 0.05:  # 5% chance of anomaly
            anomaly = np.random.normal(20, 5)
            is_anomaly = True
        value = seasonal + trend + noise + anomaly
        yield value, is_anomaly
        t += 1

def exponential_smoothing(value, level_prev, trend_prev, seasonals, t):
    """
    Performs one iteration of Holt-Winters Exponential Smoothing.
    
    Parameters:
    - value: The current data point.
    - level_prev: Previous level component.
    - trend_prev: Previous trend component.
    - seasonals: List of seasonal components.
    - t: Current time index.
    
    Returns:
    - level_curr: Updated level component.
    - trend_curr: Updated trend component.
    - seasonals: Updated seasonal components.
    - forecast: Forecasted value.
    """
    seasonal_prev = seasonals[t % SEASONAL_PERIOD]
    # Level component
    level_curr = ALPHA * (value - seasonal_prev) + (1 - ALPHA) * (level_prev + trend_prev)
    # Trend component
    trend_curr = BETA * (level_curr - level_prev) + (1 - BETA) * trend_prev
    # Seasonal component
    seasonal_curr = GAMMA * (value - level_curr) + (1 - GAMMA) * seasonal_prev
    seasonals[t % SEASONAL_PERIOD] = seasonal_curr
    # Forecast
    forecast = level_prev + trend_prev + seasonal_prev
    return level_curr, trend_curr, seasonals, forecast

def main():
    """
    Main function to run the real-time anomaly detection, visualization, and evaluation.
    """
    # Initialize data structures
    stream = data_stream(TOTAL_POINTS)
    level = None
    trend = 0
    seasonals = [0] * SEASONAL_PERIOD
    data_points = []
    forecast_data = []
    anomaly_x = []
    anomaly_y = []
    true_labels = []
    predicted_labels = []
    residuals = []

    # Set up real-time plotting
    plt.style.use('seaborn-darkgrid')
    fig, ax = plt.subplots()
    line_data, = ax.plot([], [], 'b-', label='Data Stream')
    line_forecast, = ax.plot([], [], 'g--', label='Forecast')
    line_anomaly, = ax.plot([], [], 'ro', label='Anomalies')
    ax.set_xlim(0, 100)
    ax.set_ylim(-30, 70)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Real-Time Data Stream with Exponential Smoothing Anomaly Detection')
    ax.legend(loc='upper left')
    plt.ion()  # Turn on interactive mode
    plt.show()

    for t in range(TOTAL_POINTS):
        try:
            value, is_anomaly = next(stream)
        except StopIteration:
            break

        if level is None:
            # Initialize level with the first value
            level = value
            forecast = value
        else:
            level, trend, seasonals, forecast = exponential_smoothing(
                value, level, trend, seasonals, t
            )

        residual = value - forecast
        residuals.append(residual)

        # Calculate rolling standard deviation
        window_size = SEASONAL_PERIOD
        if len(residuals) >= window_size:
            std_dev = np.std(residuals[-window_size:])
        else:
            std_dev = np.std(residuals) if len(residuals) > 1 else 1  # Prevent division by zero

        anomaly = abs(residual) > THRESHOLD * std_dev

        # Collect data for visualization
        data_points.append(value)
        forecast_data.append(forecast)
        true_labels.append(is_anomaly)
        predicted_labels.append(anomaly)

        if anomaly:
            anomaly_x.append(t)
            anomaly_y.append(value)

        # Update plot data
        line_data.set_data(range(len(data_points)), data_points)
        line_forecast.set_data(range(len(forecast_data)), forecast_data)
        line_anomaly.set_data(anomaly_x, anomaly_y)

        # Adjust plot limits dynamically
        if t > ax.get_xlim()[1]:
            ax.set_xlim(0, t + 100)
        ymin = min(data_points + forecast_data) - 10
        ymax = max(data_points + forecast_data) + 10
        ax.set_ylim(ymin, ymax)

        # Redraw the plot
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)  # Pause to allow the plot to update

    # After processing all data points, calculate and print metrics
    print("Data processing complete. Calculating metrics...\n")

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
            elif cm.shape == (1, 1):
                if y_true[0] == 1:
                    tn, fp, fn, tp = 0, 0, 0, cm[0,0]
                else:
                    tn, fp, fn, tp = cm[0,0], 0, 0, 0
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
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
