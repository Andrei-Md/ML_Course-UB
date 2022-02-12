from sklearn.tree import DecisionTreeClassifier
from sklearn import base
import numpy as np
import tqdm


class AdaBoost:
    '''
    based on SAMME algorithm - Stagewise Additive Modeling
    Zhu, Ji & Rosset, Saharon & Zou, Hui & Hastie, Trevor. (2006). Multi-class AdaBoost. Statistics and its interface. 2. 10.4310/SII.2009.v2.n3.a8.
    '''

    def __init__(self, base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimator_errors = None
        self.models = None
        self.alpha_m = None
        self.labels = {}
        self.labels2 = {}
        self.classes_no = 0

    def fit(self, X, y):
        self.models = []
        self.estimator_errors = []
        self.alpha_m = []
        n_samples = len(X)  # No of samples
        weight = np.ones(n_samples, dtype=np.float64)  # sample weights - init to 1
        weight /= weight.sum()

        k = len(np.unique(y))  # no of classes

        # create labels from classes type
        self.classes_no = len(np.unique(y))
        for i, cls in enumerate(np.unique(y)):
            self.labels[cls] = i
            self.labels2[i] = cls

        for estimator_no in tqdm.tqdm(range(self.n_estimators)):
            model = base.clone(self.base_estimator).fit(X, y, sample_weight=weight).predict
            predictions = model(X)
            predictions_truth = predictions != y

            # calculate the weak estimator error
            err_m = np.average(predictions_truth, weights=weight, axis=0)

            #compute the wright of the currect classifier
            alpha_m = (np.log((1 - err_m) / err_m) + np.log(k - 1))
            # update the weights, if the prediction was wrong then increase it
            weight = weight * np.exp(alpha_m * predictions_truth)
            #normalize
            # weight /= np.sum(weight)

            # save data
            self.estimator_errors.append(err_m)
            self.models.append(model)
            self.alpha_m.append(alpha_m)

    def predict(self, X):
        '''
        output is calculated by weighted voted
        :param X: predict set
        :return: predicted labels for each entry in the predict set
        '''
        y_pred = []
        for alpha_m, model in tqdm.tqdm(zip(self.alpha_m, self.models)):
            y_new = []
            for predict in model(X):
                pred = np.full(self.classes_no, fill_value=-1 / (self.classes_no - 1), dtype=np.float64)
                pred[self.labels[predict]] = 1
                y_new.append(pred)
            y_pred.append([y * alpha_m for y in y_new])
        y_pred = np.sum(y_pred, axis=0)
        labels_no = [np.argmax(y) for y in y_pred]
        return [self.labels2[i] for i in labels_no]

def flatten_vec(vectors):
    vecs = []
    for vec in vectors:
        vecs.append(np.ndarray.flatten(vec))
    return np.asarray(vecs)

def main():
    from sklearn.metrics import confusion_matrix as CM
    from sklearn.datasets import fetch_openml
    from sklearn import metrics
    from tensorflow import keras
    keras.__version__
    from sklearn.model_selection import train_test_split

    # MNIST
    mnist = fetch_openml('mnist_784', version=1)
    mnist.keys()
    X, y = mnist["data"], mnist["target"]
    X_train, X_test, y_train, y_test = train_test_split(X[:5000], y[:5000], test_size=0.2)  # 80% training and 20% test
    y_test = y_test.to_numpy().astype(np.float)
    y_train = y_train.to_numpy().astype(np.float)

    # MNIST fashion
    # fashion_mnist = keras.datasets.fashion_mnist
    # (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    # X_train = flatten_vec(X_train)
    # X_test = flatten_vec(X_test)

    clf = AdaBoost(n_estimators=500)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("F1 score:", metrics.f1_score(y_test, y_pred, average='micro'))

    print("Performance:", 100 * sum(y_pred == y_test) / len(y_test))
    print("Confusion Matrix:\n", CM(y_test, y_pred))


if __name__ == "__main__":
    main()
