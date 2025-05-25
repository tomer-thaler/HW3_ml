#################################
# Your name: Tomer Thaler
#################################


import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
import matplotlib.pyplot as plt

"""
Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels



def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements SGD for hinge loss.
    """
    n_samples, n_features=data.shape
    w=np.zeros(n_features)

    for t in range(1,T + 1):
        eta_t=eta_0/t
        i=np.random.randint(n_samples)
        x_i=data[i]
        y_i=labels[i]

        margin=y_i * np.dot(w,x_i)

        if margin<1:
            w=(1-eta_t)*w + eta_t*C*y_i*x_i
        else:
            w=(1-eta_t)*w

    return w


#################################

# Place for additional code

def my_accuracy_score(y_true, y_pred):
    """
    Computes the fraction of correctly predicted labels.
    Args:
        y_true: true labels (numpy array of +1/-1)
        y_pred: predicted labels (numpy array of +1/-1)
    Returns:
        Accuracy as a float between 0 and 1.
    """
    correct = np.sum(y_true == y_pred)
    return correct / len(y_true)

def my_linear_predict(X, w):
    """
    Computes the vector of dot products w^T x_i for each row x_i in X.
    Args:
        X: shape (n_samples, n_features) — data matrix
        w: shape (n_features,) — weight vector
    Returns:
        predictions: shape (n_samples,) — real-valued margins
    """
    n_samples = X.shape[0]
    predictions = np.zeros(n_samples)
    for i in range(n_samples):
        predictions[i] = np.dot(w, X[i])
    return predictions

def cross_validate_eta0_plot(train_data, train_labels, validation_data, validation_labels, T=1000, C=1, num_runs=10):
    """
    Runs cross-validation to select eta_0 using average validation accuracy over 10 runs.
    Shows and saves a plot of validation accuracy as a function of eta_0 (log scale).

    Args:
        train_data: training features
        train_labels: training labels
        validation_data: validation features
        validation_labels: validation labels
        T: number of SGD iterations (default 1000)
        C: regularization constant (default 1)
        num_runs: number of runs to average per eta_0 (default 10)
    """


    eta0_values = [10 ** i for i in range(-5, 6)]  # 10^-5 to 10^5
    avg_accuracies = []

    for eta_0 in eta0_values:
        accs = []
        for _ in range(num_runs):
            w = SGD_hinge(train_data, train_labels, C, eta_0, T)
            preds = np.sign(my_linear_predict(validation_data,w))

            if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
                continue  # skip unstable runs

            acc = my_accuracy_score(validation_labels, preds)
            accs.append(acc)

        mean_acc = np.mean(accs) if accs else 0.0
        avg_accuracies.append(mean_acc)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.semilogx(eta0_values, avg_accuracies, marker='o')
    plt.xlabel("eta_0 (log scale)")
    plt.ylabel("Average validation accuracy")
    plt.title("Validation Accuracy vs eta_0 (T=1000, C=1)")
    plt.grid(True)
    plt.savefig("eta0_validation_plot.png")
    plt.show()


def cross_validate_C_plot(
    train_data,
    train_labels,
    validation_data,
    validation_labels,
    eta0_best=10,
    T=1000,
    num_runs=10
):
    """
    Cross-validate to find the best C, keeping eta_0 fixed (best from part (a)).
    For each C on a log scale, run SGD_hinge `num_runs` times, average the
    validation accuracy, and plot/save the curve.

    Args
    ----
    train_data, train_labels         : training set
    validation_data, validation_labels : validation set
    eta0_best (float)                : the learning-rate chosen in part (a)
    T (int)                          : SGD iterations per run (default 1000)
    num_runs (int)                   : runs to average per C (default 10)
    """

    C_values = [10 ** i for i in range(-5, 6)]
    avg_accuracies = []

    for C in C_values:
        accs = []
        for _ in range(num_runs):
            w = SGD_hinge(train_data, train_labels, C, eta0_best, T)

            # If you wrote custom predictor / accuracy functions, use them:
            margins = my_linear_predict(validation_data, w)
            preds = np.sign(margins)
            if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
                continue  # skip unstable run
            acc = my_accuracy_score(validation_labels, preds)
            accs.append(acc)

        mean_acc = np.mean(accs) if accs else 0.0
        avg_accuracies.append(mean_acc)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.semilogx(C_values, avg_accuracies, marker='o')
    plt.xlabel("C (log scale)")
    plt.ylabel("Average validation accuracy")
    plt.title(f"Validation Accuracy vs C (T={T}, eta_0={eta0_best})")
    plt.grid(True)
    plt.savefig("C_validation_plot.png")
    plt.show()

def main():
    train_data, train_labels, validation_data, validation_labels, _, _ = helper()
    #part (a)
    cross_validate_eta0_plot(train_data, train_labels, validation_data, validation_labels)
    #part (b)
    cross_validate_C_plot(train_data, train_labels, validation_data, validation_labels)

if __name__ == "__main__":
    print("hello\n")
    main()

#################################
