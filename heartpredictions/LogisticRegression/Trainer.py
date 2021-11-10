import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from torch import nn
import torch

"""
Trainer Class

Class to train, store and evaluate a model.

"""

class Trainer:
    def __init__(self, model, class_weights, save_directory, loss='dl', lr=0.5, label_name = ""):
        """
        Parameters:
            model (Module) : The model (torch.nn.Module).
            class_weights (Tensor) : Weights of each class.
            save_directory (str) : Save directory path.
            loss (str) : String to match a loss function.
            lr (float) : Learning rate.
            label_name (str) : Name of the current label.

        Atributes:
            model (list(str)) : The model (torch.nn.Module).
            criterion (torch) : The loss function.
            optimizer (torch) : The optimizer.
            scheduler (torch) : The reduce on lr plateau scheduler.
            history (dict) : Dictionnary of history.
            max_val_acc (float): Current maximal accuracy.
            save_dir (str): Save directory path.
            label_name (int): NAme of the current label.
        """


        possible_loss = {'nllloss' : nn.NLLLoss(weight=class_weights, reduction='mean'),
                         'cross' : nn.CrossEntropyLoss(weight=class_weights), 'mse' : nn.MSELoss(reduction='mean'),
                         'BCEloss' : nn.BCELoss(), 'BCElogits' : nn.BCEWithLogitsLoss(weight=class_weights)}

        self.model = model
        self.criterion = possible_loss[loss]
        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.6, patience=2, cooldown=2)
        self.history = {'lr': [], 'loss': [], 'acc':[], 'val_loss': [], 'val_acc':[]}
        self.max_val_acc = float('-inf')
        self.save_dir = save_directory
        self.label_name = label_name

    def fit(self, train_dataloader, val_dataloader, nb_epochs):
        """
        Train the model.

        Parameters:
            train_dataloader (Dataloader) : The training dataloader.
            val_dataloader (Dataloader) : The validation dataloader.
            nb_epochs (int) : The number of epochs.
        """

        print(f'==== Training {self.label_name} ====\n')

        for epoch in range(nb_epochs):
            print(f'Epoch {epoch + 1} / {nb_epochs}')
            train_loss = val_loss = train_acc = val_acc = 0.0

            self.model.train()
            pbar = tf.keras.utils.Progbar(target=len(train_dataloader))

            for i, batch in enumerate(train_dataloader):
                inputs, labels = batch

                # Clear gradients w.r.t. parameters
                self.optimizer.zero_grad()

                # Forward pass to get output/logits
                output = self.model(inputs)

                # Calculate Loss
                loss = self.criterion(output, labels)
                train_loss += loss
                train_acc += 1 if np.argmax(labels.detach().numpy()[0]) == np.argmax(output.detach().numpy()[0]) else 0

                # Getting gradients w.r.t. parameters
                loss.backward()

                pbar.update(i + 1, values=
                            [
                                ("loss", train_loss.item()/(i + 1)),
                                ("acc", train_acc/(i + 1)),
                                ("lr", self.scheduler.optimizer.param_groups[0]['lr'])
                            ])

                # Updating parameters
                self.optimizer.step()

            print('Validation')

            self.model.eval()
            pbar = tf.keras.utils.Progbar(target=len(val_dataloader))

            with torch.no_grad():
                for i, batch in enumerate(val_dataloader):
                    inputs, labels = batch
                    output = self.model(inputs)

                    val_loss += loss
                    val_acc += 1 if np.argmax(labels.detach().numpy()[0]) == np.argmax(output.detach().numpy()[0]) else 0

                    pbar.update(i + 1, values=
                            [
                                ("loss", val_loss.item()/(i + 1)),
                                ("acc", val_acc/(i + 1)),
                                ("lr", self.scheduler.optimizer.param_groups[0]['lr'])
                            ])

            train_loss = train_loss / len(train_dataloader)
            train_acc = train_acc / len(train_dataloader)

            val_loss = val_loss / len(val_dataloader)
            val_acc = val_acc / len(val_dataloader)

            lr = self.scheduler.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)

            self.history['lr'].append(lr)
            self.history['loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)

            if val_acc > self.max_val_acc:
                print(f'Model saved. Acc updated: {self.max_val_acc:.3f} -> {val_acc:.3f}')
                self.max_val_acc = val_acc
                torch.save(self.model.state_dict(), f'{self.save_dir}/logistic_regression_{self.label_name}.pt')

    def evaluate(self, test_dataloader, display=True):
        """
        Evaluate the model.

        Parameters:
            test_dataloader (Dataloader) : The test dataloader.
            display (boolean) : If True, display information.

        Returns:
            total_accuracy (float)
        """

        print(f'==== Evaluate {self.label_name} ====\n')
        correct = total_loss = total = 0.0

        self.model.eval()
        with torch.no_grad():

            for i, (inputs, labels) in enumerate(test_dataloader):

                pred = self.model(inputs)

                loss = self.criterion(pred, labels)
                total_loss += loss

                # Total correct predictions
                correct += 1 if np.argmax(labels.detach().numpy()[0]) == np.argmax(pred.detach().numpy()[0]) else 0

            total_accuracy = 100 * correct / len(test_dataloader)

            if display: 
                print('Iteration: {}. Loss: {}. Accuracy: {}. total loss: {}.'.format(len(test_dataloader), loss.item(), total_accuracy, total_loss))

            return total_accuracy

    def display_history(self, accuracy=True, loss=False):
        """
        Display the history loss or accuracy.

        Parameters:
            accuracy (boolean) : If True, plot the accuracy evolution.
            loss (boolean) : If True, plot the loss evolution.
        """

        if loss:
            plt.figure(figsize=(6,6))
            plt.plot(self.history['loss'], label="Loss")
            plt.plot(self.history['val_loss'], label="Validation loss")
            plt.ylabel('Loss', fontsize=10)
            plt.xlabel('Epochs', fontsize=10)
            plt.legend()
            plt.show()

        if accuracy:
            plt.figure(figsize=(6,6))
            plt.plot(self.history['acc'], label="Accuracy")
            plt.plot(self.history['val_acc'], label="Validation accuracy")
            plt.ylabel('Accuracy', fontsize=10)
            plt.xlabel('Epochs', fontsize=10)
            plt.legend()
            plt.show()

    def load_weights(self, path):
        """
        Load weights from filepath.

        Parameters:
            path (str) : Weights path.
        """

        self.model.load_state_dict(torch.load(path))
        self.model.eval()
