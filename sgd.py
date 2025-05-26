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
    SGD with hinge loss.
    """
    n_samples, n_features=data.shape
    w=np.zeros(n_features)

    for t in range(1,T+1):
        eta_t=eta_0/t
        i=np.random.randint(n_samples)
        x_i=data[i]
        y_i=labels[i]

        margin=y_i*np.dot(w,x_i)

        if margin<1:
            w=(1-eta_t)*w + eta_t*C*y_i*x_i
        else:
            w=(1-eta_t)*w

    return w


#################################
# Place for additional code

def my_accuracy_score(y_true, y_pred):
    """
    Accuracy score.
    """
    correct=np.sum(y_true==y_pred)  # count how many are correct
    return correct/len(y_true)  # divide by total

def my_linear_predict(X, w):
    """
    Predict margins.
    """
    n_samples=X.shape[0]
    predictions=np.zeros(n_samples)
    for i in range(n_samples):
        predictions[i]=np.dot(w,X[i])  # compute w^T x_i
    return predictions

def cross_validate_eta0_plot(train_data, train_labels, validation_data, validation_labels, T=1000, C=1, num_runs=10):
    """
    Findes best eta_0.
    """
    eta0_values=[10**i for i in range(-5,6)]
    avg_accuracies=[]

    for eta_0 in eta0_values:
        accs=[]
        for _ in range(num_runs):
            w=SGD_hinge(train_data, train_labels, C, eta_0, T)
            preds=np.sign(my_linear_predict(validation_data,w))

            if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
                continue  # skip bad run

            acc=my_accuracy_score(validation_labels, preds)
            accs.append(acc)

        mean_acc=np.mean(accs) if accs else 0.0
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

def cross_validate_C_plot(train_data, train_labels, validation_data, validation_labels, eta0_best=10, T=1000, num_runs=10):
    """
    Find best C.
    """
    C_values=[10**i for i in range(-5,6)]
    avg_accuracies=[]

    for C in C_values:
        accs=[]
        for _ in range(num_runs):
            w=SGD_hinge(train_data, train_labels, C, eta0_best, T)
            preds=np.sign(my_linear_predict(validation_data, w))
            if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
                continue  # skip bad run
            acc=my_accuracy_score(validation_labels, preds)
            accs.append(acc)

        mean_acc=np.mean(accs) if accs else np.nan
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

def visualize_weight(train_data, train_labels, C, eta_0, T=20000):
    """
    Shows chosen weight vector.
    """
    w=SGD_hinge(train_data, train_labels, C, eta_0, T)
    plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
    plt.savefig("best_weight_visualization.png")
    plt.show()
    return w

def test_accuracy(test_data, test_labels, w):
    """
    Test accuracy of best classifier on the test set.
    """
    preds=np.sign(my_linear_predict(test_data, w))
    acc=my_accuracy_score(test_labels, preds)
    print(f"Test set accuracy: {acc}")
    return acc


def main():
    #load the data using your helper (thanks)
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels=helper()
    #part (a)
    cross_validate_eta0_plot(train_data, train_labels, validation_data, validation_labels)
    #part (b)
    cross_validate_C_plot(train_data, train_labels, validation_data, validation_labels)
    #Part (c)
    best_eta_0=10
    best_C=1e-5
    w_final=visualize_weight(train_data, train_labels, C=best_C, eta_0=best_eta_0)
    #Part (d)
    test_accuracy(test_data, test_labels, w_final)

if __name__ == "__main__":
    print("hello\n")
    main()

#################################
