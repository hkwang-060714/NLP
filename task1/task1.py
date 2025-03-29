import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

the_data = 'train.tsv'
data = pd.read_csv(the_data, sep='\t')
print("the shape", data.shape)
print(data.head())
x_all = data['Phrase']
y_all = data['Sentiment']


def data_split(x, y, size):
    np.random.seed(42)
    indices = np.random.permutation(len(y))
    n_test = int(len(y) * size)

    x = x.values if isinstance(x, pd.Series) else x
    y = y.values if isinstance(y, pd.Series) else y

    return (x[indices[:-n_test]], x[indices[-n_test:]],
            y[indices[:-n_test]], y[indices[-n_test:]])


x_train, x_test, y_train, y_test = data_split(x_all, y_all, 0.2)
x_train, x_val, y_train, y_val = data_split(x_train, y_train, 0.5)


class BagOfWords:
    def __init__(self, max_features=5000):
        self.vocab = None
        self.max_features = max_features

    def build_vocabulary(self, tokenized_documents):
        word_counts = defaultdict(int)
        for doc in tokenized_documents:
            for word in doc:
                word_counts[word] += 1
        sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
        self.vocab = {w: i for i, (w, _) in enumerate(sorted_words[:self.max_features - 1])}
        self.vocab['<UNK>'] = len(self.vocab)
        return self.vocab

    def vectorize(self, documents):
        data, rows, cols = [], [], []
        for i, doc in enumerate(documents):
            if isinstance(doc, str):
                tokens = doc.split()
            else:
                tokens = doc
            for word in tokens:
                idx = self.vocab.get(word, self.vocab['<UNK>'])
                if idx < self.max_features:
                    rows.append(i)
                    cols.append(idx)
                    data.append(1)
        return csr_matrix((data, (rows, cols)),
                          shape=(len(documents), self.max_features))


class NGramVectorizer:
    def __init__(self, n=2, max_features=10000):
        self.n = n
        self.max_features = max_features
        self.vocab = {}

    def _generate_ngrams(self, text):
        tokens = text.split()
        return [tuple(tokens[i:i + self.n]) for i in range(len(tokens) - self.n + 1)]

    def fit(self, texts):
        ngram_counts = defaultdict(int)
        for text in texts:
            for ngram in self._generate_ngrams(text):
                ngram_counts[ngram] += 1
        sorted_ngrams = sorted(ngram_counts.items(), key=lambda x: -x[1])
        self.vocab = {ngram: i for i, (ngram, _) in enumerate(sorted_ngrams[:self.max_features])}

    def transform(self, texts):
        data, rows, cols = [], [], []
        for i, text in enumerate(texts):
            for ngram in self._generate_ngrams(text):
                if ngram in self.vocab:
                    rows.append(i)
                    cols.append(self.vocab[ngram])
                    data.append(1)
        return csr_matrix((data, (rows, cols)),
                          shape=(len(texts), self.max_features))


class Softmax:
    def __init__(self, n_features, n_classes=5, learning_rate=0.05,
                 reg=0.01, epochs=50, batch_size=128):
        self.W = np.random.randn(n_features, n_classes) * 0.01
        self.b = np.zeros(n_classes)
        self.lr = learning_rate
        self.reg = reg
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_history = []
        self.val_acc_history = []

    def guiyi(self, z):
        max_z = np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z - max_z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def compute_loss(self, X, y):
        scores = X.dot(self.W) + self.b
        probs = self.guiyi(scores)
        return -np.mean(np.log(probs[np.arange(X.shape[0]), y] + 1e-8))

    def compute_gradients(self, X, y):
        m = X.shape[0]
        scores = X.dot(self.W) + self.b
        probs = self.guiyi(scores)

        probs[np.arange(m), y] -= 1
        dW = (X.T.dot(probs) / m) + self.reg * self.W
        db = np.mean(probs, axis=0)
        return dW, db

    def train(self, X, y, X_val, y_val):
        best_acc = 0
        X_batch=0
        y_batch=0
        for epoch in range(self.epochs):
            indices = np.random.permutation(X.shape[0])
            for i in range(0, X.shape[0], self.batch_size):
                batch_idx = indices[i:i + self.batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]

                dW, db = self.compute_gradients(X_batch, y_batch)
                self.W -= self.lr * dW
                self.b -= self.lr * db

            val_acc = self.accuracy(X_val, y_val)
            loss = self.compute_loss(X_batch, y_batch)
            self.loss_history.append(loss)
            self.val_acc_history.append(val_acc)
            if val_acc > best_acc:
                best_acc = val_acc
                best_W = self.W.copy()
                best_b = self.b.copy()
            print(f"Epoch {epoch + 1} Val Acc: {val_acc:.2%}")

        self.W = best_W
        self.b = best_b

    def predict(self, X):
        return np.argmax(X.dot(self.W) + self.b, axis=1)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)

if __name__ == "__main__":
    lr=float(input("请输入你所需要的学习率"))
    use_bow = input("是否使用词袋模型(y/n)").lower() == 'y'
    use_ngram = input("是否使用n-gram(y/n)").lower() == 'y'

    if use_ngram:
        try:
            the_ngram = int(input("请输入ngram特征（整数，如2或3）: "))
        except ValueError:
            print("输入无效，默认使用2-gram")
            the_ngram = 2

    feature_list = []

    if use_bow:
        bow_vectorizer = BagOfWords(max_features=5000)
        bow_vectorizer.build_vocabulary([doc.split() for doc in x_train])

        X_train_bow = bow_vectorizer.vectorize(x_train)
        X_val_bow = bow_vectorizer.vectorize(x_val)
        X_test_bow = bow_vectorizer.vectorize(x_test)
        feature_list.extend([X_train_bow, X_val_bow, X_test_bow])

    if use_ngram:
        ngram_vectorizer = NGramVectorizer(n=the_ngram, max_features=10000)
        ngram_vectorizer.fit(x_train)

        X_train_ngram = ngram_vectorizer.transform(x_train)
        X_val_ngram = ngram_vectorizer.transform(x_val)
        X_test_ngram = ngram_vectorizer.transform(x_test)
        feature_list.extend([X_train_ngram, X_val_ngram, X_test_ngram])

    X_train, X_val, X_test = feature_list[0], feature_list[1], feature_list[2]

    total_features = X_train.shape[1]
    model = Softmax(n_features=total_features,learning_rate=lr,epochs=100)

    model.train(X_train, y_train, X_val, y_val)
    print(f"测试集准确率: {model.accuracy(X_test, y_test):.2%}")

    plt.plot(model.loss_history, label='Training Loss')
    plt.plot(model.val_acc_history, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

