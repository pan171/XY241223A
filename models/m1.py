import numpy as np


def model_1(w, delta_p, tau_y, r_w, v_m):
    """定义Lietard-Griffiths模型方程"""
    return (
        (delta_p / tau_y) * w**3
        + 6 * r_w * (delta_p / tau_y) * w**2
        - (9 / np.pi) * v_m
    )


def df_model_1(w, delta_p, tau_y, r_w):
    """计算Lietard-Griffiths模型方程的导数"""
    return 3 * (delta_p / tau_y) * w**2 + 12 * r_w * (delta_p / tau_y) * w


def newton_method_Lietard(w0, delta_p, tau_y, r_w, V_m, tol=1e-6, max_iter=100):
    """使用牛顿法求解 w"""
    w = w0
    for _ in range(max_iter):
        f_w = model_1(w, delta_p, tau_y, r_w, V_m)
        df_w = df_model_1(w, delta_p, tau_y, r_w)

        if abs(df_w) < 1e-8:
            print("导数接近零，牛顿法可能失败。")
            return None

        w_new = w - f_w / df_w

        if abs(w_new - w) < tol:
            return w_new

        w = w_new

    print("牛顿法未能在最大迭代次数内收敛")
    return None


# 示例输入
delta_p = 1.0  # 例如 1 Pa
tau_y = 0.5  # 例如 0.5 Pa
r_w = 0.1  # 例如 0.1 m
V_m = 0.02  # 例如 0.02 m³
w0 = 1.0  # 初始猜测值

# 计算 w 的根
w_root = newton_method_Lietard(w0, delta_p, tau_y, r_w, V_m)
print("求解出的 w =", w_root)
