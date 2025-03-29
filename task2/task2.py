import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import re
import os
import sys
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class Config:
    batch_size = 128
    embed_dim = 100
    hidden_dim = 128
    filter_num = 128
    filter_size = [2, 3, 4]
    dropout = 0.5
    learning_rate = 0.001
    epochs = 20
    frequences = 3
    max_length = 50
    glove_path = 'glove.6B.100d.txt'
    seed = 42
    labels = ['Negative', 'Somewhat Negative', 'Neutral', 'Somewhat Positive', 'Positive']

torch.manual_seed(Config.seed)
np.random.seed(Config.seed)

class TextProcessor:
    def __init__(self, frequences=3):
        self.min_freq = frequences
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2word = {0: '<pad>', 1: '<unk>'}
        self.vocab_size = 2

    def build_vocab(self, texts):
        counter = defaultdict(int)
        for text in texts:
            tokens = self._tokenize(text)
            for token in tokens:
                counter[token] += 1
        idx = 2
        for token, count in counter.items():
            if count >= self.min_freq:
                self.word2idx[token] = idx
                self.idx2word[idx] = token
                idx += 1
        self.vocab_size = len(self.word2idx)
        print(f"Vocabulary size: {self.vocab_size}")

    def _tokenize(self, text):
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        return text.split()

    def numericalize(self, text):
        tokens = self._tokenize(text)
        return [self.word2idx.get(token, 1) for token in tokens]


class MovieDataset(Dataset):
    def __init__(self, texts, labels, processor):
        self.texts = texts
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        numericalized = self.processor.numericalize(text)
        if len(numericalized) == 0:
            numericalized = [1]
        return torch.LongTensor(numericalized), torch.tensor(label)

def collate_fn(batch):
    batch = [(x, y) for x, y in batch if len(x) > 0]
    inputs, labels = zip(*batch)
    lengths = torch.tensor([len(x) for x in inputs])
    if torch.any(lengths <= 0):
        print(f"发现无效长度序列，数量：{torch.sum(lengths <= 0)}")
        lengths = torch.clamp_min(lengths, 1)

    sorted_indices = torch.argsort(lengths, descending=True)
    inputs = [inputs[i] for i in sorted_indices]
    labels = [labels[i] for i in sorted_indices]
    lengths = lengths[sorted_indices]

    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    return (padded_inputs.to(device),torch.LongTensor(labels).to(device),lengths.to(device))

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, filter_num, filter_sizes, num_classes, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, filter_num, (fs, embed_dim)) for fs in filter_sizes])
        self.fc = nn.Linear(filter_num * len(filter_sizes), num_classes)
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(filter_num * len(filter_sizes))

    def forward(self, x):
        # x shape: [batch_size, seq_len]
        embedded = self.embedding(x).unsqueeze(1)  # [batch, 1, seq_len, embed_dim]
        conved = [torch.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [torch.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.bn(torch.cat(pooled, dim=1))
        return self.fc(self.dropout(cat))

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,n_layers=2, bidirectional=False, dropout=0.5, embedding_matrix=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(embedding_matrix)
        self.rnn = nn.RNN(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        if torch.any(lengths <= 0):
            lengths = torch.clamp_min(lengths, 1)
        embedded = self.dropout(self.embedding(x))
        packed_embedded = pack_padded_sequence(embedded,lengths.cpu(),batch_first=True,enforce_sorted=False)
        packed_output, hidden = self.rnn(packed_embedded)
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2], hidden[-1]), dim=1))
        else:
            hidden = self.dropout(hidden[-1])
        return self.fc(hidden)

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            bidirectional=True, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        if torch.any(lengths <= 0):
            invalid_indices = torch.where(lengths <= 0)[0]
            print(f"发现无效长度样本，索引：{invalid_indices}")
            lengths[lengths <= 0] = 1
        embedded = self.dropout(self.embedding(x))
        valid_lengths = torch.clamp_min(lengths.cpu(), 1)
        packed_embedded = pack_padded_sequence(
            embedded,
            valid_lengths,
            batch_first=True,
            enforce_sorted=False
        )
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        hidden = self.dropout(torch.cat((hidden[-2], hidden[-1]), dim=1))
        return self.fc(hidden)

def load_glove(glove_path, word2idx, embed_dim=100):
    if not os.path.exists(glove_path):
        return None
    word2vec = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            word2vec[word] = vector

    weight_matrix = np.random.normal(scale=0.6, size=(len(word2idx), embed_dim))
    matched = 0
    for word, idx in word2idx.items():
        if word in word2vec:
            weight_matrix[idx] = word2vec[word]
            matched += 1
        elif word == '<pad>':
            weight_matrix[idx] = np.zeros(embed_dim)
    print(f"Matched {matched}/{len(word2idx)} ({matched / len(word2idx):.2%}) words")
    return torch.tensor(weight_matrix, dtype=torch.float32)

def train_model(model, train_loader, val_loader, model_name):
    optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    best_acc = 0.0
    history = {'train_loss': [], 'val_acc': []}
    early_stop_counter = 0
    for epoch in range(Config.epochs):
        model.train()
        total_loss = 0.0
        for inputs, labels, lengths in train_loader:
            optimizer.zero_grad()
            if isinstance(model, (BiLSTM, RNN)):
                outputs = model(inputs, lengths)
            else:
                outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
        val_acc, _, _ = evaluate(model, val_loader)
        avg_loss = total_loss / len(train_loader.dataset)
        history['train_loss'].append(avg_loss)
        history['val_acc'].append(val_acc)
        scheduler.step(val_acc)
        print(f"Epoch {epoch + 1:02}/{Config.epochs}")
        print(f"Train Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2%}")
        if val_acc > best_acc:
            best_acc = val_acc
            early_stop_counter = 0
            torch.save(model.state_dict(), f"best_{model_name}.pth")
        else:
            early_stop_counter += 1
            if early_stop_counter >= 3:
                print("Early stopping triggered!")
                break
    return history

def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels, lengths in data_loader:
            if isinstance(model, (BiLSTM, RNN)):
                outputs = model(inputs, lengths)
            else:
                outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return accuracy, all_preds, all_labels

def plot_training(history, model_name):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.show()

def main():
    use_cnn = input("是否使用cnn模型(y/n)").lower() == 'y'
    use_rnn= input("是否使用rnn模型(y/n)").lower() == 'y'
    use_lstm = input("是否使用lstm(y/n)").lower() == 'y'
    try:
        data = pd.read_csv('../train.tsv', sep='\t')
    except FileNotFoundError:
        print("Error: train.tsv not found")
        sys.exit(1)
    processor = TextProcessor(frequences=Config.frequences)
    processor.build_vocab(data['Phrase'])
    X = data['Phrase'].values
    y = data['Sentiment'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=Config.sed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, random_state=Config.sed)
    train_dataset = MovieDataset(X_train, y_train, processor)
    val_dataset = MovieDataset(X_val, y_val, processor)
    test_dataset = MovieDataset(X_test, y_test, processor)
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size,shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size,collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size,collate_fn=collate_fn)
    glove_matrix = load_glove(Config.glove_path, processor.word2idx)
    cnn = TextCNN(
        vocab_size=processor.vocab_size,
        embed_dim=Config.embed_dim,
        filter_num=Config.filter_num,
        filter_sizes=Config.filter_size,
        num_classes=len(Config.labels),
        dropout=Config.dropout
    ).to(device)
    rnn = RNN(
        vocab_size=processor.vocab_size,
        embedding_dim=Config.embed_dim,
        hidden_dim=128,
        output_dim=len(Config.labels),
        n_layers=2,
        bidirectional=False,
        dropout=0.5,
        embedding_matrix=glove_matrix
    ).to(device)
    lstm = BiLSTM(
        vocab_size=processor.vocab_size,
        embed_dim=Config.embed_dim,
        hidden_dim=Config.hidden_dim,
        num_classes=len(Config.labels),
        num_layers=2,
        dropout=Config.dropout
    ).to(device)
    if glove_matrix is not None:
        cnn.embedding.weight.data.copy_(glove_matrix)
        lstm.embedding.weight.data.copy_(glove_matrix)
        rnn.embedding.weight.data.copy_(glove_matrix)
        rnn.embedding.weight.requires_grad = False
        cnn.embedding.weight.requires_grad = False
        lstm.embedding.weight.requires_grad = False
    if use_cnn:
        print("\nTraining TextCNN:")
        cnn_history = train_model(cnn, train_loader, val_loader, "textcnn")
        plot_training(cnn_history, "textcnn")
        cnn.load_state_dict(torch.load("best_textcnn.pth"))
        cnn_acc, cnn_preds, cnn_labels = evaluate(cnn, test_loader)
        print(f"cnn准确率: {cnn_acc:.2%}")
    if use_rnn:
        print("\nTraining RNN:")
        rnn_history = train_model(rnn, train_loader, val_loader, "rnn")
        plot_training(rnn_history, "rnn")
        rnn.load_state_dict(torch.load("best_rnn.pth"))
        rnn_acc, rnn_preds, rnn_labels = evaluate(rnn, test_loader)
        print(f"rnn准确率: {rnn_acc:.2%}")
    if use_lstm:
        print("\nTraining BiLSTM:")
        lstm_history = train_model(lstm, train_loader, val_loader, "bilstm")
        plot_training(lstm_history, "bilstm")
        lstm.load_state_dict(torch.load("best_bilstm.pth"))
        lstm_acc, lstm_preds, lstm_labels = evaluate(lstm, test_loader)
        print(f"lstm准确率: {lstm_acc:.2%}")

if __name__ == "__main__":
    main()