import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
import matplotlib.pyplot as plt
import pickle

from PreResCnn118 import *
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score


class Trainer(object):
    def __init__(self, use_cuda=None):
        if use_cuda is None:
            # check if GPU exists
            use_cuda = torch.cuda.is_available()

        # use CPU or GPU
        self.device = torch.device("cuda" if use_cuda else "cpu")

        # set path
        self.data_path = "./jpegs_256/"  # define UCF-101 RGB data path
        # action_name_path = './UCF101actions.pkl'
        self.save_model_path = "./CRNN_ckpt/"

        # EncoderCNN architecture
        self.CNN_fc_hidden1 = 1024
        self.CNN_fc_hidden2 = 768
        self.CNN_embed_dim = 512  # latent dim extracted by 2D CNN\

        self.res_size = 224
        self.img_x = 256
        self.img_y = 342  # resize video 2d frame size

        self.dropout_p = 0.0  # dropout probability

        # DecoderRNN architecture
        self.RNN_hidden_layers = 3
        self.RNN_hidden_nodes = 512
        self.RNN_FC_dim = 256

        # training parameters
        self.k = 101  # number of target category
        self.epochs = 120  # training epochs
        self.batch_size = 30
        self.learning_rate = 1e-4
        self.log_interval = 10  # interval for displaying training info

        # Select which frame to begin & end in videos
        self.begin_frame = 1
        self.end_frame = 29
        self.skip_frame = 1

        self.cnn_encoder = ResCNNEncoder(
            fc_hidden1=self.CNN_fc_hidden1,
            fc_hidden2=self.CNN_fc_hidden2, drop_p=self.dropout_p,
            CNN_embed_dim=self.CNN_embed_dim
        ).to(self.device)

        self.rnn_decoder = DecoderRNN(
            CNN_embed_dim=self.CNN_embed_dim,
            h_RNN_layers=self.RNN_hidden_layers,
            h_RNN=self.RNN_hidden_nodes,
            h_FC_dim=self.RNN_FC_dim, drop_p=self.dropout_p,
            num_classes=self.k
        ).to(self.device)

    def train(
        self, log_interval, model, device, train_loader,
        optimizer, epoch
    ):
        # set model as training mode
        cnn_encoder, rnn_decoder = model
        cnn_encoder.train()
        rnn_decoder.train()

        losses = []
        scores = []
        N_count = 0   # counting total trained sample in one epoch

        for batch_idx, (X, y) in enumerate(train_loader):
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1, )

            N_count += X.size(0)

            optimizer.zero_grad()
            # output has dim = (batch, number of classes)
            output = rnn_decoder(cnn_encoder(X))

            loss = F.cross_entropy(output, y)
            losses.append(loss.item())

            # to compute accuracy
            y_pred = torch.max(output, 1)[1]  # y_pred != output
            step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
            scores.append(step_score)         # computed on CPU

            loss.backward()
            optimizer.step()

            # show information
            if (batch_idx + 1) % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%'.format(
                    epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(), 100 * step_score))

        return losses, scores

    def validation(self, model, device, optimizer, test_loader):
        # set model as testing mode
        cnn_encoder, rnn_decoder = model
        cnn_encoder.eval()
        rnn_decoder.eval()

        test_loss = 0
        all_y = []
        all_y_pred = []
        with torch.no_grad():
            for X, y in test_loader:
                # distribute data to device
                X, y = X.to(device), y.to(device).view(-1, )

                output = rnn_decoder(cnn_encoder(X))

                loss = F.cross_entropy(output, y, reduction='sum')
                test_loss += loss.item()                 # sum up batch loss
                y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

                # collect all y and y_pred in all batches
                all_y.extend(y)
                all_y_pred.extend(y_pred)

        test_loss /= len(test_loader.dataset)

        # compute accuracy
        all_y = torch.stack(all_y, dim=0)
        all_y_pred = torch.stack(all_y_pred, dim=0)
        test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

        # show information
        print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, 100* test_score))

        # save Pytorch models of best record
        torch.save(cnn_encoder.state_dict(), os.path.join(save_model_path, 'cnn_encoder_epoch{}.pth'.format(epoch + 1)))  # save spatial_encoder
        torch.save(rnn_decoder.state_dict(), os.path.join(save_model_path, 'rnn_decoder_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
        torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'optimizer_epoch{}.pth'.format(epoch + 1)))      # save optimizer
        print("Epoch {} model saved!".format(epoch + 1))

        return test_loss, test_score