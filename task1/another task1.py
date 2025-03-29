import zipfile
import pandas as pd
import numpy as np
import re
from collections import Counter

class TextClassifier:
    def __init__(self, zip_path, vocab_size=3000):
        self.zip_path = zip_path
        self.vocab_size = vocab_size
        self.vocab = []
        self.label_map = {0: "negative", 1: "somewhat negative",
                        2: "neutral", 3: "somewhat positive", 4: "positive"}

    def _load_data(self):
        with zipfile.ZipFile(self.zip_path) as outer_z:
            with outer_z.open('train.tsv.zip') as inner_zip_file:
                with zipfile.ZipFile(inner_zip_file) as inner_z:
                    with inner_z.open('../train.tsv') as f:
                        df = pd.read_csv(f, sep='\t', encoding='utf-8')
        return df

    def _load_data_test(self):
        with zipfile.ZipFile(self.zip_path) as outer_z:
            with outer_z.open('test.tsv.zip') as inner_zip_file:
                with zipfile.ZipFile(inner_zip_file) as inner_z:
                    with inner_z.open('test.tsv') as f:
                        df = pd.read_csv(f, sep='\t', encoding='utf-8')
        return df.drop_duplicates(subset=['SentenceId', 'Phrase'], keep='first')

    def _clean_text(self, text):
        text = str(text).lower()
        return re.sub(r"\s+", ' ', text).split()  # 仅处理多余空格

    def build_features(self, ngram_range=(1, 1)):
        df = self._load_data()
        ngrams = []
        max_len = max(len(self._clean_text(p)) for p in df['Phrase'])
        valid_ngram = (ngram_range[0], min(ngram_range[1], max_len))
        for text in df['Phrase']:
            tokens = self._clean_text(text)
            for n in range(valid_ngram[0], valid_ngram[1] + 1):
                ngrams += [' '.join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        self.vocab = [item[0] for item in Counter(ngrams).most_common(self.vocab_size)]
        X = np.zeros((len(df), len(self.vocab)))
        for i, text in enumerate(df['Phrase']):
            tokens = self._clean_text(text)
            for n in range(valid_ngram[0], valid_ngram[1] + 1):
                for j in range(len(tokens) - n + 1):
                    ngram = ' '.join(tokens[j:j + n])
                    if ngram in self.vocab:
                        X[i, self.vocab.index(ngram)] += 1

        X = np.hstack([X, np.ones((X.shape[0], 1))])
        return X / (X.sum(axis=1, keepdims=True) + 1e-8), df['Sentiment'].values.astype(np.int32)

    def build_features_test(self, ngram_range=(1, 1)):
        df = self._load_data_test()
        X = np.zeros((len(df), len(self.vocab)))
        for i, text in enumerate(df['Phrase']):
            tokens = self._clean_text(text)
            for n in range(ngram_range[0], ngram_range[1] + 1):
                for j in range(len(tokens) - n + 1):
                    ngram = ' '.join(tokens[j:j + n])
                    if ngram in self.vocab:
                        X[i, self.vocab.index(ngram)] += 1
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        return X / (X.sum(axis=1, keepdims=True) + 1e-8)

    def train_test_split(self, X, y, test_size=0.2):
        np.random.seed(42)
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        n_test = int(len(y) * test_size)
        return X[indices[:-n_test]], X[indices[-n_test:]], y[indices[:-n_test]], y[indices[-n_test:]]

class SoftmaxRegression:
    def __init__(self, n_features, n_classes=5, lr=0.1, reg=0.01):
        self.W = np.random.randn(n_features, n_classes) * 0.01
        self.b = np.zeros(n_classes)
        self.lr = lr
        self.reg = reg

    def softmax(self, scores):
        exps = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exps / exps.sum(axis=1, keepdims=True)

    def compute_loss(self, X, y):
        scores = X.dot(self.W) + self.b
        probs = self.softmax(scores)
        correct_logprobs = -np.log(probs[np.arange(len(y)), y] + 1e-8)
        data_loss = np.sum(correct_logprobs) / len(y)
        reg_loss = 0.5 * self.reg * np.sum(self.W ** 2)
        return data_loss + reg_loss

    def compute_gradients(self, X, y):
        scores = X.dot(self.W) + self.b
        probs = self.softmax(scores)

        dscores = probs.copy()
        dscores[np.arange(len(y)), y] -= 1
        dscores /= len(y)

        dW = X.T.dot(dscores) + self.reg * self.W
        db = np.sum(dscores, axis=0)
        return dW, db

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=64):
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_epoch=0
        the_formol_train=0
        the_formol_val=0

        for epoch in range(epochs):
            indices = np.random.permutation(len(X_train))
            for i in range(0, len(X_train), batch_size):
                batch_indices = indices[i:i + batch_size]
                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]

                dW, db = self.compute_gradients(X_batch, y_batch)
                self.W -= self.lr * dW
                self.b -= self.lr * db

            train_loss = self.compute_loss(X_train, y_train)
            val_loss = self.compute_loss(X_val, y_val)

            if (train_loss-the_formol_train)**2+(val_loss-the_formol_val)**2<1e-7:
                break

            the_formol_train=train_loss
            the_formol_val=val_loss
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_W = self.W.copy()
                best_b = self.b.copy()

            print(f"Epoch {epoch + 1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        self.W = best_W
        self.b=best_b
        return train_losses, val_losses

    def predict(self, X):
        scores = X.dot(self.W) + self.b
        return np.argmax(scores, axis=1)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)

if __name__ == "__main__":
    processor = TextClassifier("sentiment-analysis-on-movie-reviews.zip")
    X, y = processor.build_features(ngram_range=(1, 1))
    print(f"成功加载数据！特征矩阵形状: {X.shape}, 标签数: {len(y)}")
    the_final_model=None
    best_model=None
    acc=0
    acc_1=0
    best_ngram=None

    for ngram in [(1, 1), (1, 2)]:
        print(f"\n正在处理 {ngram}-gram 特征")
        X, y = processor.build_features(ngram_range=ngram)
        X_train, X_t, y_train, y_t = processor.train_test_split(X, y, test_size=0.2)
        X_val, X_test, y_val, y_test = processor.train_test_split(X_t, y_t, test_size=0.5)

        for lr in [0.1, 0.05, 0.01]:
            model = SoftmaxRegression(n_features=X_train.shape[1], lr=lr)
            print(f"\n学习率: {lr}")
            model.train(X_train, y_train, X_val, y_val)
            val_acc = model.accuracy(X_val, y_val)
            print(f"验证集准确率: {val_acc:.2%}")
            val_acc = model.accuracy(X_test, y_test)
            print(f"测试集准确率: {val_acc:.2%}")
            if val_acc > acc:
                the_final_model = model
                acc = val_acc

        if acc>acc_1:
            best_model = the_final_model
            best_ngram=ngram
            acc_1 = acc

    print("最终评估模型：")
    X, y = processor.build_features(ngram_range=best_ngram)
    X_train, X_t, y_train, y_t = processor.train_test_split(X, y, test_size=0.2)
    X_val, X_test, y_val, y_test = processor.train_test_split(X_t, y_t, test_size=0.5)
    print(f"测试机准确率：{best_model.accuracy(X_test, y_test):.2%}")
    print(f"test.tsv答案如下:\n{best_model.predict(processor.build_features_test())}")



