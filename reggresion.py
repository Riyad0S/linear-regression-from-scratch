import numpy as np

def separator():
    print("----" * 50)

def model(x, w, b):
    return x * w + b

def compute_cost(x, w, b, y):
    m = x.shape[0]
    cost = 0
    f_wb = np.zeros(m)

    for i in range(m):
        f_wb[i] = model(x[i], w, b)
        cost = cost + (f_wb[i] - y[i]) ** 2

    cost = cost / (2 * m)
    return cost

def gradient_descent(x, w, b, y, alpha, iterations):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(iterations):
        dj_dw = 0
        dj_db = 0

        for j in range(m):
            error = model(x[j], w, b) -y[j]

            dj_dw = dj_dw + error * x[j]
            dj_db = dj_db + error

        dj_dw = dj_dw / m
        dj_db = dj_db / m

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

    return w, b

initial_w = 0
initial_b = 0
alpha = 0.000001
iterations = 1000
x_train = np.array([2104, 5, 45])
y_train = np.array([460, 232, 178])

w_final, b_final = gradient_descent(x_train, initial_w, initial_b, y_train, alpha, iterations)

initial_cost = compute_cost(x_train, initial_w, initial_b, y_train)
final_cost = compute_cost(x_train, w_final, b_final, y_train)

print(f"Initial cost({initial_cost})")
print(f"Final cost({final_cost})")
separator()
print(f"Initial w({initial_w})")
print(f"Final w({w_final})")
separator()
print(f"Initial b({initial_b})")
print(f"Final b({b_final})")
separator()
prediction = model(3000, w_final, b_final)
print(prediction)


