import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from amfbo import PROJECT_ROOT

warnings.filterwarnings('ignore')
import json
import os


class Config:
    input_dim = 128
    feature_size = 32
    num_classes = 5

    tcn_channels = [64, 128, 64]
    tcn_output_size = 64
    lstm_hidden_size = 64
    lstm_layers = 2
    dropout_rate = 0.2

    batch_size = 64
    learning_rate = 0.001
    num_epochs = 500
    patience = 8

    model_dir = str(PROJECT_ROOT / "models")
    result_dir = str(PROJECT_ROOT / "result")
    log_dir = str(PROJECT_ROOT / "logs")

    def __init__(self):
        self.create_dirs()

    def create_dirs(self):
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    @classmethod
    def load(cls, path):
        config = cls()
        with open(path, 'r') as f:
            config.__dict__.update(json.load(f))
        return config


class TimeSeriesDataset(Dataset):
    def __init__(self, data_list, label_list, scaler=None, fit_scaler=True):
        self.data_list = data_list
        self.label_list = label_list
        self.scaler = scaler or StandardScaler()

        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(label_list)

        if fit_scaler:
            self._fit_scaler()
        self.normalized_data = self._normalize_data()

    def _fit_scaler(self):
        """拟合标准化器"""
        all_data = np.vstack([data.numpy() for data in self.data_list if len(data) > 0])
        self.scaler.fit(all_data)

    def _normalize_data(self):
        """标准化数据"""
        normalized_data = []
        for data in self.data_list:
            if len(data) > 0:
                normalized = self.scaler.transform(data.numpy())
                normalized_data.append(torch.FloatTensor(normalized))
            else:
                normalized_data.append(torch.FloatTensor(np.zeros((1, 128))))
        return normalized_data

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.normalized_data[idx], self.encoded_labels[idx]

    def get_original_data(self, idx):
        return self.data_list[idx], self.label_list[idx]


def collate_fn(batch):
    """处理可变长度序列的批处理函数"""
    data_list, labels = zip(*batch)
    lengths = [len(seq) for seq in data_list]

    non_empty_data = [data for data in data_list if len(data) > 0]

    if len(non_empty_data) > 0:
        padded_data = nn.utils.rnn.pad_sequence(non_empty_data, batch_first=True)

        final_data = []
        data_idx = 0
        for length in lengths:
            if length == 0:
                final_data.append(torch.zeros(1, 128))
            else:
                final_data.append(padded_data[data_idx:data_idx + 1])
                data_idx += 1

        padded_data = torch.cat(final_data, dim=0)
    else:
        padded_data = torch.zeros(len(data_list), 1, 128)

    return padded_data, torch.tensor(labels), torch.tensor(lengths)


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size,
                      padding=padding, dilation=dilation))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size,
                      padding=padding, dilation=dilation))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)

        if out.shape != res.shape:
            min_length = min(out.size(2), res.size(2))
            out = out[:, :, :min_length]
            res = res[:, :, :min_length]

        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, dilation, dropout))

        self.network = nn.Sequential(*layers)
        self.output_projection = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        x = x.transpose(1, 2)
        features = self.network(x)
        features = features.transpose(1, 2)
        return self.output_projection(features)


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=32, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.feature_projection = nn.Linear(2 * hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.output_size = output_size

    def forward(self, x, lengths):
        batch_size = x.size(0)
        zero_length_indices = (lengths == 0)

        if zero_length_indices.any():
            final_output = torch.zeros(batch_size, self.output_size, device=x.device)

            non_zero_indices = ~zero_length_indices
            if non_zero_indices.any():
                non_zero_x = x[non_zero_indices]
                non_zero_lengths = lengths[non_zero_indices]

                packed_input = nn.utils.rnn.pack_padded_sequence(
                    non_zero_x, non_zero_lengths.cpu(), batch_first=True, enforce_sorted=False)
                packed_output, _ = self.lstm(packed_input)
                output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

                lstm_features = []
                for i, length in enumerate(non_zero_lengths):
                    lstm_features.append(output[i, length - 1, :])

                non_zero_features = torch.stack(lstm_features, dim=0)
                projected_features = self.feature_projection(self.dropout(non_zero_features))
                final_output[non_zero_indices] = projected_features

            return final_output
        else:
            packed_input = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_output, _ = self.lstm(packed_input)
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

            lstm_features = []
            for i, length in enumerate(lengths):
                lstm_features.append(output[i, length - 1, :])

            features = torch.stack(lstm_features, dim=0)
            return self.feature_projection(self.dropout(features))


class TCNLSTMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.tcn = TCN(config.input_dim, config.tcn_output_size,
                       config.tcn_channels, dropout=config.dropout_rate)

        self.lstm = BidirectionalLSTM(config.tcn_output_size, config.lstm_hidden_size,
                                      config.lstm_layers, output_size=config.feature_size,
                                      dropout=config.dropout_rate)

        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.feature_size, config.num_classes)
        )

        self.feature_extractor = nn.Sequential(
            nn.Linear(config.feature_size, config.feature_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )

    def forward(self, x, lengths):
        batch_size = x.size(0)
        zero_length_indices = (lengths == 0)

        if zero_length_indices.any():
            final_features = torch.zeros(batch_size, self.config.feature_size, device=x.device)
            final_class_output = torch.zeros(batch_size, self.config.num_classes, device=x.device)

            non_zero_indices = ~zero_length_indices
            if non_zero_indices.any():
                non_zero_x = x[non_zero_indices]
                non_zero_lengths = lengths[non_zero_indices]

                tcn_features = self.tcn(non_zero_x)
                lstm_features = self.lstm(tcn_features, non_zero_lengths)
                class_output = self.classifier(lstm_features)
                feature_output = self.feature_extractor(lstm_features)

                final_features[non_zero_indices] = feature_output
                final_class_output[non_zero_indices] = class_output

            return final_class_output, final_features
        else:
            tcn_features = self.tcn(x)
            lstm_features = self.lstm(tcn_features, lengths)
            class_output = self.classifier(lstm_features)
            feature_output = self.feature_extractor(lstm_features)
            return class_output, feature_output


class ModelTrainer:
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=config.patience // 2, factor=0.5, verbose=True)

        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0
        self.epochs_no_improve = 0

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss, correct, total = 0, 0, 0

        for batch_data, batch_labels, batch_lengths in tqdm(train_loader, desc="Training"):
            batch_data = batch_data.to(self.device)
            batch_labels = batch_labels.to(self.device)
            batch_lengths = batch_lengths.to(self.device)

            self.optimizer.zero_grad()
            class_output, _ = self.model(batch_data, batch_lengths)
            loss = self.criterion(class_output, batch_labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(class_output.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch_data, batch_labels, batch_lengths in val_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                batch_lengths = batch_lengths.to(self.device)

                class_output, _ = self.model(batch_data, batch_lengths)
                loss = self.criterion(class_output, batch_labels)

                total_loss += loss.item()
                _, predicted = torch.max(class_output.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy, all_preds, all_labels

    def train(self, train_loader):

        for epoch in range(self.config.num_epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            if train_acc > self.best_val_acc:
                self.best_val_acc = train_acc
                self.epochs_no_improve = 0
                self.save_checkpoint(epoch, True)
            else:
                self.epochs_no_improve += 1
                self.save_checkpoint(epoch, False)

            if self.epochs_no_improve >= self.config.patience:
                break

    def save_checkpoint(self, epoch, is_best):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'best_val_acc': self.best_val_acc
        }

        torch.save(checkpoint, os.path.join(self.config.model_dir, 'checkpoint.pth'))
        if is_best:
            torch.save(checkpoint, os.path.join(self.config.model_dir, 'best_model.pth'))


class ModelEvaluator:
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device

    def evaluate_model(self, test_loader):
        self.model.eval()
        all_preds, all_labels, all_features = [], [], []
        total_loss, correct, total = 0, 0, 0

        with torch.no_grad():
            for batch_data, batch_labels, batch_lengths in tqdm(test_loader, desc="Testing"):
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                batch_lengths = batch_lengths.to(self.device)

                class_output, features = self.model(batch_data, batch_lengths)
                loss = nn.CrossEntropyLoss()(class_output, batch_labels)

                total_loss += loss.item()
                _, predicted = torch.max(class_output.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                all_features.extend(features.cpu().numpy())

        accuracy = 100 * correct / total
        avg_loss = total_loss / len(test_loader)




class FeatureExtractor:
    def __init__(self, model_path, config_path, device='cpu'):
        self.config = Config.load(config_path)
        checkpoint = torch.load(model_path, map_location=device)

        self.model = TCNLSTMModel(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.model.to(device)

        self.label_encoder = LabelEncoder()
        self.device = device

    def load_label_encoder(self, classes):
        self.label_encoder.classes_ = np.array(classes)

    def extract(self, input_sequence):
        with torch.no_grad():
            sequence_length = len(input_sequence)

            if sequence_length == 0:
                return np.zeros(self.config.feature_size, dtype=np.float32)

            if hasattr(self, 'scaler'):
                input_sequence = self.scaler.transform(input_sequence.numpy())
                input_sequence = torch.FloatTensor(input_sequence)

            input_tensor = input_sequence.unsqueeze(0).to(self.device)
            length = torch.tensor([sequence_length]).to(self.device)

            _, features = self.model(input_tensor, length)
            return features.squeeze(0).cpu().numpy()

    def process_batch(self, data_list):
        results = []
        for data in data_list:
            features = self.get_32d_features(data)
            results.append(features)
        return results


def main_training_function(data_list, label_list):
    config = Config()
    config.num_classes = len(np.unique(label_list))
    config.save(os.path.join(config.model_dir, 'config.json'))

    dataset = TimeSeriesDataset(data_list, label_list)
    train_loader = DataLoader(dataset, batch_size=config.batch_size,
                              shuffle=True, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TCNLSTMModel(config)
    trainer = ModelTrainer(model, config, device)

    trainer.train(train_loader)

    checkpoint = torch.load(os.path.join(config.model_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    feature_extractor = FeatureExtractor(
        os.path.join(config.model_dir, 'best_model.pth'),
        os.path.join(config.model_dir, 'config.json'),
        device
    )
    feature_extractor.load_label_encoder(dataset.label_encoder.classes_)

    return feature_extractor


def generate_sample_data(num_samples=1000, max_length=100):

    data_list, label_list = [], []
    classes = ['class_A', 'class_B', 'class_C', 'class_D', 'class_E']

    for i in range(num_samples):
        length = np.random.randint(0, max_length + 1)

        if length == 0:
            data = torch.randn(0, 128)
        else:
            data = torch.randn(length, 128)

        label = np.random.choice(classes)

        data_list.append(data)
        label_list.append(label)

    return data_list, label_list


if __name__ == "__main__":

    data_list, label_list = generate_sample_data(1000, 100)

    feature_extractor = main_training_function(data_list, label_list)

    test_samples = [
        torch.randn(50, 128),
        torch.randn(0, 128),
        torch.randn(75, 128),
    ]
    features = feature_extractor.process_batch(test_samples)
