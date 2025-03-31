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

    '''
    The train method was changed in order to fit data with no labeling. In order to use it with labels set add a batch_y to the enumeration for loop.
    Also, make sure in data_factory that shuffle flag is True
    '''

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

    '''
    The test method needs labeling therefore for unlabeled data it is useless.
    '''

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


    '''
    save_preds used to save in an npy file the following data:
    for each time step- [real features] [prediction 1-feature 1, prediction 1-feature 2,...,prediction 2-feature 1,...] [prediction 1 score, prediction 2 score,...] [is prediction 1 anomaly, is prediction 2 anomaly,...]
    save_preds method gets:
    setting-in order to load the correct model
    threshold-used to calculate if a score of a prediction is indicating an anomaly. Can be calculated with with calculate_threshold(...)
    
    It saves an npy file data_array.npy on the current directory
    
    Make sure that shuffle flag = false in data_factory
    
    It transforms the values back to the original scale
    '''

    def save_preds(self, setting, threshold, test = 1):
        print("starting saving process")

        train_data, train_loader = self._get_data(flag='train')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))


        folder_path = './save_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)


        time_steps = train_data.train.shape[0] #this is specifit for UNI
        #features = train_data.train.shape[1] - 1    # the first feature is time step
        features = 1
        amount_of_predictions = train_data.win_size #amount of times we will predict a value because it is a sliding window of size 1


        #real features --- predictions[pred 1 feature 1, pred 1 feature 2,..., pred 2 feature 1,..., --- score[anomaly_critertion(real,pred1), anomaly_critertion(real,pred2),...] --- anomaly ---
        data_array = np.full((time_steps, features + amount_of_predictions * features + amount_of_predictions + amount_of_predictions), np.nan)


        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)
        torch.set_printoptions(threshold=10_000_000)

        # (1) stastic on the train set

        n = 0 # to index right the beginning
        last_index = len(train_loader)-1
        time_step = 0
        with torch.no_grad():


            for i, batch_x in enumerate(train_loader):


                start_time = time.time()
                batch_x = batch_x.float().to(self.device)


                # reconstruction of the exact time window

                outputs = self.model(batch_x, None, None, None)
                # criterion


                real_features = batch_x.cpu().numpy()
                if i == last_index:
                    last_window = real_features[-1]

                #the score is calculated with mse for each time step, for each feature but then we take the mean over them so each time step gets a score that is one number
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                score = score.detach().cpu().numpy()


                '''
                we will go over each window in the batch(the first for loop), then over each time step in that window.
                then we will the fill in the numpy array.
                '''

                for window_index in range(0, batch_x.shape[0]):
                    for time_step_index in range(amount_of_predictions):
                        #print(window_index, time_step_index)

                        prediction_values = outputs[window_index, time_step_index, 0].cpu().numpy().reshape(-1,1)
                        '''calculating the prediction number-every time step gets a few predictions because it appears in many windows because that is a sliding window.
                        more specifically, except for the first numbers, each time step has [window_size] predictions because the window slides one step at a time-
                        that is being lets say for time step t it will be reconstructed in the window that starts with t-99: [t-99, t-98,...,t] and then with the one
                        that starts with t-98: [t-98, t-97,...,t,t+1] and so on. Notice that for the first time steps there are less predictions that's why we have 
                        the calculation of  min(n+time_step_index,amount_of_predictions-1) - time_step_index that 
                        '''
                        number_of_prediction = min(n+time_step_index,amount_of_predictions-1) - time_step_index

                        #fills in the data of the prediction
                        start_col = features + number_of_prediction*features
                        end_col = start_col + features
                        data_array[time_step + time_step_index, start_col:end_col] = train_data.scaler.inverse_transform(prediction_values)

                        #fills in the data of the score
                        col_for_score = features + features*amount_of_predictions + number_of_prediction
                        data_array[time_step + time_step_index, col_for_score] = score[window_index, time_step_index]

                        #fills in the data of if it is an anomaly, for over anomalies 1, under -1 and no anomaly 0
                        is_anomaly = score[window_index, time_step_index] > threshold
                        type_of_anomaly = 0
                        if is_anomaly:
                            if real_features[window_index, time_step] - outputs[window_index, time_step] > 0:
                                type_of_anomaly = 1
                            else:
                                type_of_anomaly = -1
                        col_for_anomaly = features + features*amount_of_predictions + amount_of_predictions + number_of_prediction
                        data_array[time_step + time_step_index, col_for_anomaly] = type_of_anomaly

                    #fills in the data of the real features, notice it has a completion next in the code below
                    data_array[time_step, :features] = train_data.scaler.inverse_transform(real_features[window_index, 0, 0].reshape(-1,1))

                    time_step += 1
                    n += 1


                end_time = time.time()
                batch_time = end_time - start_time
                print(f"Batch {i} processed in {batch_time:.4f} seconds.")

        #filling the last real features
        for i in range(1, amount_of_predictions):
            data_array[time_step, :features] = train_data.scaler.inverse_transform(last_window[i, 0].reshape(-1,1))
            time_step += 1
        #makes sure the table is filled properly. it should print ((1+window_size)/2 * (features + 2)
        nan_count = np.isnan(data_array).sum()
        print(f"Number of NaNs in the array: {nan_count}")



        # Count total NaNs in the entire DataFrame


        print("Done saving the train set results. Starting test set...")

        '''
        Now it is the same but for the test set
        '''

        print("-------------------------------------------")
        test_data, test_loader = self._get_data(flag='test')
        time_steps = test_data.test.shape[0]
        # features = train_data.train.shape[1] - 1    # the first feature is time step
        features = 1
        amount_of_predictions = test_data.win_size  # a mount of times we will predict a value
        '''the table will look like time step(1), real valued features(features), prediction 1(features), ... ,prediction amount_of_predictions(features)'''

        # real features --- predictions[pred 1 feature 1, pred 1 feature 2,..., pred 2 feature 1,..., --- score[anomaly_critertion(real,pred1), anomaly_critertion(real,pred2),...] --- anomaly ---
        data_array2 = np.full(
            (time_steps, features + amount_of_predictions * features + amount_of_predictions + amount_of_predictions),
            np.nan)

        self.model.eval()


        # (1) stastic on the train set

        n = 0  # to index right the beginning
        last_index = len(test_loader) - 1
        time_step = 0
        with torch.no_grad():

            for i, batch_x in enumerate(test_loader):

                start_time = time.time()
                batch_x = batch_x.float().to(self.device)

                # print(batch_x[:,:5,1])
                # reconstruction

                outputs = self.model(batch_x, None, None, None)
                # criterion

                real_features = batch_x.cpu().numpy()
                if i == last_index:
                    last_window = real_features[-1]

                # print(batch_x[0,:10,1])

                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                score = score.detach().cpu().numpy()

                # pdb.set_trace()

                for window_index in range(0, batch_x.shape[0]):
                    for time_step_index in range(amount_of_predictions):
                        # print(window_index, time_step_index)

                        prediction_values = outputs[window_index, time_step_index, 0].cpu().numpy().reshape(-1, 1)

                        number_of_prediction = min(n + time_step_index, amount_of_predictions - 1) - time_step_index
                        start_col = features + number_of_prediction * features
                        end_col = start_col + features
                        data_array2[time_step + time_step_index,
                        start_col:end_col] = train_data.scaler.inverse_transform(prediction_values)

                        col_for_score = features + features * amount_of_predictions + number_of_prediction
                        data_array2[time_step + time_step_index, col_for_score] = score[window_index, time_step_index]
                        # print(score[window_index, time_step_index], "----", threshold)
                        is_anomaly = score[window_index, time_step_index] > threshold
                        col_for_anomaly = features + features * amount_of_predictions + amount_of_predictions + number_of_prediction
                        data_array2[time_step + time_step_index, col_for_anomaly] = is_anomaly

                    data_array2[time_step, :features] = train_data.scaler.inverse_transform(
                        real_features[window_index, 0, 0].reshape(-1, 1))

                    time_step += 1
                    n += 1
                    # pdb.set_trace()
                    # print("--------")
                # print("----------------")
                # print(data_array[:10,:24])
                # pdb.set_trace()

                '''

                '''
                end_time = time.time()
                batch_time = end_time - start_time
                print(f"Batch {i} processed in {batch_time:.4f} seconds.")

        # filling the last real features
        for i in range(1, amount_of_predictions):
            data_array2[time_step, :features] = train_data.scaler.inverse_transform(last_window[i, 0].reshape(-1, 1))
            time_step += 1

        nan_count = np.isnan(data_array).sum()

        print(f"Number of NaNs in the array2: {nan_count}")

        #concatenating the train and test arrays
        data_array = np.concatenate((data_array, data_array2), axis=0)

        #saving the array
        output_path = '/home/shays/Projects/Time-Series-Library/data_array.npy'

        np.save(output_path, data_array)


    '''
    plots the loss graphs of the losses after the training
    '''
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


    '''
    calculates the threshold by calculating the score for every time step in every window, takes a percentile that is determined by anomaly_ratio parameter in the bashrun.sh file
    calculate_threshold get:
    setting-for loading the model
    '''

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

        for i, batch_x in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            # reconstruction
            outputs = self.model(batch_x, None, None, None)
            # criterion
            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)


        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
        print("Threshold :", threshold)
        return threshold


    '''
    loads the npy file and updates the is anomaly columns in the file 
    update_anomalies get:
    threshold-can be calculated by the calculate_threshold(...)
    window_size-in order to know how many predictions
    '''
    def update_anomalies(self, threshold, window_size, test=1):
        print("starting saving process")

        data_array = np.load('/home/shays/Projects/Time-Series-Library/data_array.npy')

        time_steps = data_array.shape[0]
        # features = train_data.train.shape[1] - 1    # the first feature is time step
        features = 1

        for time_step in range(time_steps):
            for i in range(0, window_size):
                if not np.isnan(data_array[time_step, features + i]):
                    type_of_anomaly = 0
                    is_anomaly = data_array[time_step, features + window_size * features + i] > threshold
                    if is_anomaly:
                        if data_array[time_step, 0] > data_array[time_step, features + i]:
                            type_of_anomaly = 1
                        else:
                            type_of_anomaly = -1

                    data_array[time_step, features + window_size * features + window_size + i] = type_of_anomaly

                else:
                    continue

        nan_count = np.isnan(data_array).sum()

        print(f"Number of NaNs in the array: {nan_count}")

        print("done")
        # Count total NaNs in the entire DataFrame
        output_path = '/home/shays/Projects/Time-Series-Library/data_array.npy'

        np.save(output_path, data_array)

        print("Done saving the results.")



