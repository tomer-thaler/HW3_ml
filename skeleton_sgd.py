#################################
# Your name: Tomer Thaler
#################################


import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing

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
    from sklearn.metrics import accuracy_score
    import matplotlib.pyplot as plt

    eta0_values = [10 ** i for i in range(-5, 6)]  # 10^-5 to 10^5
    avg_accuracies = []

    for eta_0 in eta0_values:
        accs = []
        for _ in range(num_runs):
            w = SGD_hinge(train_data, train_labels, C, eta_0, T)
            preds = np.sign(validation_data @ w)

            if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
                continue  # skip unstable runs

            acc = accuracy_score(validation_labels, preds)
            accs.append(acc)

        mean_acc = np.mean(accs) if accs else float('nan')
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

def main():
    # Load the data
    train_data, train_labels, validation_data, validation_labels, _, _ = helper()

    # Run part (a): cross-validate eta_0 and save the plot
    cross_validate_eta0_plot(train_data, train_labels, validation_data, validation_labels)

if __name__ == "__main__":
    print("hello\n")
    main()

#################################
