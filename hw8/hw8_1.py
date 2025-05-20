import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# given data
x = np.array([4.0, 4.2, 4.5, 4.7, 5.1, 5.5, 5.9, 6.3])
y = np.array([102.6, 113.2, 130.1, 142.1, 167.5, 195.1, 224.9, 256.8])

# (a) y = ax^2 + bx + c
coeffs_quad = np.polyfit(x, y, 2)  # 次方為2
poly_quad = np.poly1d(coeffs_quad)
y_quad_pred = poly_quad(x)
error_quad = np.sum((y - y_quad_pred) ** 2)

# (b) y = b * e^(a * x)
def exp_func(x, a, b):
    return b * np.exp(a * x)

params_exp, _ = curve_fit(exp_func, x, y, p0=(0.1, 1))
y_exp_pred = exp_func(x, *params_exp)
error_exp = np.sum((y - y_exp_pred) ** 2)

# (c) y = b * x^n
def power_func(x, b, n):
    return b * x ** n

params_pow, _ = curve_fit(power_func, x, y, p0=(1, 2))
y_pow_pred = power_func(x, *params_pow)
error_pow = np.sum((y - y_pow_pred) ** 2)

# print
print(" (a) a x^2 + bx + c approximation")
print(f" a = {coeffs_quad[0]:.4f}, b = {coeffs_quad[1]:.4f}, c = {coeffs_quad[2]:.4f}")
print(f" (SSE): {error_quad:.5f}")

print("\n (b) y = b * e^(a * x)")
print(f" a = {params_exp[0]:.4f}, b = {params_exp[1]:.4f}")
print(f" (SSE): {error_exp:.5f}")

print("\n (c) y = b * x^n")
print(f" b = {params_pow[0]:.4f}, n = {params_pow[1]:.4f}")
print(f" (SSE): {error_pow:.5f}")

# plot
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='black', label='Data')
plt.plot(x, y_quad_pred, label='(a) ax^2+bx+c', linestyle='--')
plt.plot(x, y_exp_pred, label='(b) be^ax', linestyle='-.')
plt.plot(x, y_pow_pred, label='(c) bx^n', linestyle=':')
plt.legend()
plt.title("Least Squares Approximations")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
