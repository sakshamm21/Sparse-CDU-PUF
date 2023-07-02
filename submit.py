import numpy as np

# You are not allowed to use any ML libraries e.g. sklearn, scipy, keras, tensorflow etc

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py
# DO NOT INCLUDE OTHER PACKAGES LIKE SKLEARN, SCIPY, KERAS,TENSORFLOW ETC IN YOUR CODE
# THE USE OF ANY MACHINE LEARNING LIBRARIES WILL RESULT IN A STRAIGHT ZERO

# DO NOT CHANGE THE NAME OF THE METHOD my_fit BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length


def HT(v, k):
    t = np.zeros_like(v)
    if k < 1:
        return t
    else:
        ind = np.argsort(abs(v))[-k:]
        t[ind] = v[ind]
        return t


def my_fit(X_trn, y_trn):
    

    MODEL_SIZE = 2048
    MODEL_SPARSITY = 512

    model = np.random.randn(MODEL_SIZE)   # model = np.random.random(MODEL_SIZE)
    learning_rate = 0.001  
    max_iter = 10000
    momentum = 0.95  #0.5
    last_grad = 0

    for i in range(max_iter):
        y_ = X_trn @ model
        delta = (y_trn - y_)
        error = np.average(delta ** 2)
        print(f"iterm={i}, error={error}")
        dw = np.average(-2 * learning_rate * delta * X_trn.T, axis=1)
        grad = dw + last_grad * momentum
        last_grad = grad
        model -= grad
        model = HT(model, MODEL_SPARSITY)
        

    return model  # Return the trained model

