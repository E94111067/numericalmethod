import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
from scipy.integrate import quad

# 定義 f(x)
def f(x):
    return 0.5 * np.cos(x) + 0.25 * np.sin(2 * x)

# 建立二次多項式基底函數
phi = [lambda x: 1, lambda x: x, lambda x: x**2]

# 計算 inner product <f, phi_i> 及 <phi_i, phi_j> 組成的矩陣
A = np.zeros((3, 3))
b = np.zeros(3)

for i in range(3):
    for j in range(3):
        A[i, j], _ = quad(lambda x: phi[i](x) * phi[j](x), -1, 1)
    b[i], _ = quad(lambda x: f(x) * phi[i](x), -1, 1)

# 解聯立方程 A c = b 得到多項式係數
c = np.linalg.solve(A, b)

# 顯示結果
print(f"Least squares polynomial coefficients (from degree 0 to 2): {c}")

# 繪圖比較原函數與逼近多項式
x_vals = np.linspace(-1, 1, 400)
f_vals = f(x_vals)
p_vals = c[0] + c[1]*x_vals + c[2]*x_vals**2

plt.plot(x_vals, f_vals, label='f(x)', linewidth=2)
plt.plot(x_vals, p_vals, label='Least squares approximation (deg 2)', linestyle='--')
plt.legend()
plt.grid(True)
plt.title("Least Squares Polynomial Approximation (Degree 2)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
