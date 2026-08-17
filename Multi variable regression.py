import numpy as np

def separator():
    print("----" * 50)

def model(w, x, b):
    prediction = np.dot(w, x) + b
    return prediction

def total_cost(w,x,b,y):
    m = x.shape[0]
    cost = 0

    for i in range(m):
        f_wb_i = (model(w, x[i], b) - y[i]) ** 2
        cost = cost + f_wb_i

    cost = cost / (2 * m)
    return cost

def gradient_decent(w, x, b, y, alpha, iterations):
    m = x.shape[0]

    for iteration in range(iterations):
        dj_dw = np.zeros(len(w))
        dj_db = 0

        for i in range(m):
            error = model(w, x[i], b) - y[i]
            dj_dw = dj_dw + error * x[i]
            dj_db = dj_db + error

        dj_dw = dj_dw / m
        dj_db = dj_db / m

        w = w - alpha * dj_dw
        b = b - alpha * dj_db
    return w, b
