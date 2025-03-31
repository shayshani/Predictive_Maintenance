import pdb
from pyexpat import features
import pdb
import numpy as np
import matplotlib.pyplot as plt

class Plots:
    def __init__(self, window_size, features, npy_file):
        self.window_size = window_size
        self.features = features
        self.data = np.load(npy_file)

    def plot_average_prediction_one_feature(self, start_time_step, end_time_step, feature):
        #
        real_features = self.data[start_time_step:end_time_step, 0:self.features].reshape(-1)

        predictions = []
        anomalies = []
        #pdb.set_trace()
        for time_step in range(end_time_step-start_time_step):
            sum = 0
            count = 0
            count_of_anomalies = 0
            for pred in range(self.window_size):
                var = self.data[time_step+start_time_step, features + pred*features + feature]
                anomaly = self.data[time_step+start_time_step, features + window_size*features + window_size + pred]
                #print(var)
                if not np.isnan(var):
                    sum += var
                    count += 1
                    if abs(anomaly):
                        count_of_anomalies += anomaly
            anomaly = 0
            if count_of_anomalies > count/2:
                anomaly = 1
            elif count_of_anomalies < -count/2:
                anomaly = -1
            anomalies.append(anomaly)


            predictions.append(sum/count)
        #pdb.set_trace()
        print(anomalies)
        anomalies = np.array(anomalies, dtype=int)
        plt.figure(figsize=(16, 9))
        plt.plot(range(start_time_step, end_time_step), predictions, label=f'Average Prediction')
        plt.plot(range(start_time_step, end_time_step), real_features, label=f'Real Features')

        time_steps = np.arange(start_time_step, end_time_step)

        # Find indices of 1s and -1s
        anomaly_pos_indices = np.where(anomalies == 1)[0]
        anomaly_neg_indices = np.where(anomalies == -1)[0]
        #print(anomaly.count)
        #print(anomaly_neg_indices, anomaly_pos_indices)

        # Plot the 1s (positive anomalies) in red
        plt.scatter(time_steps[anomaly_pos_indices], np.array(predictions)[anomaly_pos_indices],
                    color='red', label='Positive Anomalies', marker='D', s=100, zorder=5)

        # Plot the -1s (negative anomalies) in blue
        plt.scatter(time_steps[anomaly_neg_indices], np.array(predictions)[anomaly_neg_indices],
                    color='blue', label='Negative Anomalies', marker='D', s=100, zorder=5)

        plt.title('Average Prediction Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Average Prediction Value')
        plt.legend()

        plt.savefig('average_one_feature_plot.png')
        #pdb.set_trace()

if __name__ == "__main__":
    # Example parameters
    window_size = 100  # For instance, 3 prediction blocks per time step
    features = 1  # Suppose each block has 5 features
    npy_file = '/home/shays/Projects/Time-Series-Library/data_array.npy'  # Replace with your .npy file path

    # Create an instance of the Plots class
    plotter = Plots(window_size, features, npy_file)

    # Call the plotting method for a specific feature (e.g., feature index 1)
    # from time step 0 to 100.
    plotter.plot_average_prediction_one_feature(start_time_step=4900,
                                                end_time_step=5100,
                                                feature=0)
