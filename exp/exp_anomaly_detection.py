from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import torch.multiprocessing
import matplotlib.pyplot as plt
torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pdb
import random
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, batch_x in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)

                outputs = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_losses = []
        vali_losses = []
        test_losses = []

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, batch_x in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)

                outputs = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                loss = criterion(outputs, batch_x)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            train_losses.append(train_loss)
            vali_losses.append(vali_loss)
            test_losses.append(test_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        self.plot_loss_graph(train_losses, vali_losses, test_losses)

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        attens_energy = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the train set
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                outputs = self.model(batch_x, None, None, None)
                # criterion
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        test_labels = []
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            # reconstruction
            outputs = self.model(batch_x, None, None, None)
            # criterion
            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            #test_labels.append(batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
        print("Threshold :", threshold)

        # (3) evaluation on the test set
        pred = (test_energy > threshold).astype(int)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # (4) detection adjustment
        gt, pred = adjustment(gt, pred)

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))

        f = open("result_anomaly_detection.txt", 'a')
        f.write(setting + "  \n")
        f.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))
        f.write('\n')
        f.write('\n')
        f.close()
        return



    def save_preds(self, setting, test = 1):
        print("starting saving process")
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        attens_energy = []
        folder_path = './save_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)


        time_steps = train_data.train.shape[0]
        features = train_data.train.shape[1] - 1    # the first feature is time step
        amount_of_predictions = train_data.win_size #a mount of times we will predict a value
        '''the table will look like time step(1), real valued features(features), prediction 1(features), ... ,prediction amount_of_predictions(features)'''


        data_array = np.full((time_steps, features + amount_of_predictions * features), np.nan)


        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)
        torch.set_printoptions(threshold=10_000_000)

        # (1) stastic on the train set

        n = 0 # to index right the beginning
        last_index = len(train_loader)-1
        last_batch = None
        time_step = 0
        with torch.no_grad():


            for i, (batch_x, batch_y) in enumerate(train_loader):
                print(i)
                start_time = time.time()
                batch_x = batch_x.float().to(self.device)


                #print(batch_x[:,:5,1])
                # reconstruction

                outputs = self.model(batch_x, None, None, None)
                # criterion


                real_features = batch_x.cpu().numpy()
                if i == last_index:
                    last_window = real_features[-1]

                #print(batch_x[0,:10,1])


                
                for window_index in range(0, batch_x.shape[0]):
                    for time_step_index in range(amount_of_predictions):
                        #print(window_index, time_step_index)

                        prediction_values = outputs[window_index, time_step_index, 1:].cpu().numpy()
                        start_col = features + (min(n+time_step_index,amount_of_predictions-1) - time_step_index)*features

                        end_col = start_col + features
                        #print(start_col//24)
                        #print("fill: (", time_step + time_step_index, start_col // 24, ")")
                        data_array[time_step + time_step_index, start_col:end_col] = prediction_values
                    data_array[time_step, :24] = real_features[window_index, 0, 1:]
                    time_step += 1
                    n += 1
                    #pdb.set_trace()
                    #print("--------")
                #print("----------------")
                #print(data_array[:10,:24])
                #pdb.set_trace()

                '''
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)
                '''
                end_time = time.time()
                batch_time = end_time - start_time
                print(f"Batch {i} processed in {batch_time:.4f} seconds.")

        #filling the last real features
        for i in range(1, amount_of_predictions):
            data_array[time_step, :24] = last_window[i, 1:]
            time_step += 1

        nan_count = np.isnan(data_array).sum()

        print(f"Number of NaNs in the array: {nan_count}")

        pdb.set_trace()
        print("done")
        # Count total NaNs in the entire DataFrame
        output_path = '/home/shays/Projects/Time-Series-Library/data_array.npy'



        np.save(output_path, data_array)

        print("Done saving the results.")

        pdb.set_trace()
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        test_labels = []

        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            # if(i >= 684):
            # print(i)
            # print(batch_x[:,:,0])
            # reconstruction
            outputs = self.model(batch_x, None, None, None)
            # criterion
            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(batch_y)

        pdb.set_trace()
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
        print("Threshold :", threshold)

        # (3) evaluation on the test set
        pred = (test_energy > threshold).astype(int)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # (4) detection adjustment
        gt, pred = adjustment(gt, pred)

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))

        f = open("result_anomaly_detection.txt", 'a')
        f.write(setting + "  \n")
        f.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))
        f.write('\n')
        f.write('\n')
        f.close()
        return




    def plot_loss_graph(self, train_losses, vali_losses, test_losses):
        epochs = range(1, len(train_losses) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, label='Train Loss', color='blue')
        plt.plot(epochs, vali_losses, label='Validation Loss', color='orange')
        plt.plot(epochs, test_losses, label='Test Loss', color='green')

        plt.title('Loss vs Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Save the plot if you want
        plt.savefig('loss_plot.png')

        # Show the plot
        plt.show()

    def random_seq(self, setting, threshold, test = 1):
        print("starting sampling process")
        print(setting)
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')

        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        save_folder = f'/home/shays/Projects/Time-Series-Library/comparison_plots_{setting}'

        # Create the folder if it doesn't exist
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)



        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)
        torch.set_printoptions(threshold=10_000_000)


        n = 0
        with torch.no_grad():
            for i, batch_x in enumerate(test_loader):
                start_time = time.time()
                batch_x = batch_x.float().to(self.device)
                # print(batch_x[:,:5,1])
                # reconstruction

                outputs = self.model(batch_x, None, None, None)
                # criterion
                rand_index = random.randint(0, batch_x.shape[0]-1)
                window = batch_x[rand_index].cpu().numpy()
                prediction = outputs[rand_index].cpu().numpy()


                self.save_comparison_plot(window, prediction, n, save_folder, threshold, test_data)

                n += 1

    def save_comparison_plot(self, window, prediction, index, save_folder, threshold, test_data):
        """
        Saves a graph comparing the input window and its corresponding prediction on the same chart.
        Marks anomalies where the MSE exceeds the threshold.
        Args:
            window: The input window (ground truth).
            prediction: The model's predicted output for the window.
            index: The index of the current window in the test set (used for file naming).
            save_folder: The directory where the comparison graph will be saved.
            threshold: The threshold for anomaly detection based on the MSE.
        """
        # Compute MSE for each point
        mse = np.abs(window - prediction)  # Absolute difference between window and prediction

        anomaly_points = mse > threshold

        # Inverse transform the predictions back to original scale
        prediction = test_data.scaler.inverse_transform(prediction)

        # Inverse transform the real data as well (if needed)
        window = test_data.scaler.inverse_transform(window)

        # Create a figure for the plot
        plt.figure(figsize=(10, 5))

        # Plot both the input window and prediction on the same plot
        plt.plot(window, label='Input Window', color='blue')
        plt.plot(prediction, label='Prediction', color='red')

        # Mark anomalies where MSE exceeds the threshold

        plt.scatter(np.where(anomaly_points)[0], window[anomaly_points], color='black', label='Anomaly', zorder=5)

        # Add labels and title
        plt.title(f"Comparison {index} (Anomalies marked)")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()

        # Save the plot to the specified folder
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f'comparison_{index}.png'))
        plt.close()  # Close the plot to free memory

    def calculate_threshold(self, setting, test=1):

        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        attens_energy = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the train set
        with torch.no_grad():
            for i, batch_x in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                outputs = self.model(batch_x, None, None, None)
                # criterion
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        test_labels = []
        for i, batch_x in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            # reconstruction
            outputs = self.model(batch_x, None, None, None)
            # criterion
            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            # test_labels.append(batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
        print("Threshold :", threshold)
        return threshold



