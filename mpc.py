import casadi as ca
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mlp import MLP
import torch


# 系统参数
M = 1.0
m = 0.1
l = 0.5
g = 9.8
I = (1/3) * m * l**2

def pendulum_dynamics(x, u):
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    F = u[0]

    sin_theta = ca.sin(x3)
    cos_theta = ca.cos(x3)
    total_mass = M + m * sin_theta ** 2

    x1_dot = x2
    x2_dot = (F - m * l * x4**2 * sin_theta - m * g * sin_theta * cos_theta) / total_mass
    theta1_dot = x4
    theta2_dot = (m * l * x4**2 * sin_theta * cos_theta + (M+m) * g * sin_theta - F * cos_theta) / (l * total_mass)

    return ca.vertcat(x1_dot, x2_dot, theta1_dot, theta2_dot)

def mpc_control(N=20, dt=0.02):
    F_max = 10

    # 定义符号变量
    X0 = ca.MX.sym('X0', 4)  # 初始状态
    U = ca.MX.sym('U', N)    # 控制输入序列
    X = [X0]

    cost = 0

    # boundary of inputs
    lbx = [-F_max] * N
    ubx = [F_max] * N

    for k in range(N):
        x_k = X[-1]
        u_k = U[k]

        # Dynamics forward
        x_next = x_k + dt * pendulum_dynamics(x_k, ca.vertcat(u_k))
        X.append(x_next)

        # Cost function
        cost += 10*(x_k[0]**2) + 100*(x_k[2]**2) + 0.1*(u_k**2)

    X = ca.horzcat(*X)

    opt_vars = U
    nlp = {
        'x': opt_vars,
        'p': X0,
        'f': cost,
        'g': []
    }

    opts = {
        'ipopt.print_level': 0,
        'print_time': 0,
        'ipopt.tol': 1e-4
    }

    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
    return solver, lbx, ubx

def simulate_single(initial_state, solver, lbx, ubx, T=100, dt=0.02):
    """
    给定初始状态，使用MPC仿真一条轨迹。
    返回 states 和 inputs。
    """
    times = np.arange(0, T, dt)
    states = [initial_state]
    inputs = []

    u_init = np.zeros(20)  # 初始控制猜测

    for t in times:
        x0 = np.array(states[-1])

        # 解决MPC问题
        solver_args = {'x0': u_init, 'p': x0, 'lbx': lbx, 'ubx': ubx}
        sol = solver(**solver_args)

        u_opt = np.array(sol['x']).flatten()
        F_opt = u_opt[0]

        # 记录控制输入
        inputs.append(F_opt)

        # 更新状态
        dx = np.array(pendulum_dynamics(x0, np.array([F_opt]))).flatten()
        x_new = x0 + dx * dt
        states.append(x_new)

    return np.array(states), np.array(inputs)

def predict_inputs(model, states):
    model.eval()
    X = states[:-1]
    X_tensor = torch.tensor(X, dtype=torch.float32).to(next(model.parameters()).device)
    with torch.no_grad():
        predictions = model(X_tensor)
    predictions = predictions.cpu().numpy().flatten()
    return np.clip(predictions, -10, 10)

def compare_inputs(mpc_inputs, model_inputs, dt):
    """
    绘图对比 MPC 控制输入与模型预测输入。
    """
    times = np.arange(0, len(mpc_inputs) * dt, dt)
    
    plt.figure()
    plt.plot(times, mpc_inputs, label='MPC Input', linewidth=2)
    plt.plot(times, model_inputs, label='Model Prediction', linestyle='--')
    plt.title("MPC vs Model Predicted Input")
    plt.xlabel("Time (s)")
    plt.ylabel("Control Input (Force)")
    plt.legend()
    plt.grid(True)
    plt.savefig("input_comparison.png")

    # 差值
    input_diff = np.abs(np.array(mpc_inputs) - np.array(model_inputs))
    plt.figure()
    plt.plot(times, input_diff, label='Absolute Input Error', color='red')
    plt.title("Prediction Error")
    plt.xlabel("Time (s)")
    plt.ylabel("Error (|F_MPC - F_model|)")
    plt.grid(True)
    plt.savefig("input_error.png")

def simulate_single_mlp(model, initial_state, solver, lbx, ubx, T=100, dt=0.02):
    """
    给定初始状态，使用MPC仿真一条轨迹。
    返回 states 和 inputs。
    """
    times = np.arange(0, T, dt)
    states = [initial_state]
    inputs = []
    init = predict_inputs(model, initial_state)
    u_init = np.linspace(init[0], 0, 20)  # 初始控制猜测
    x0 = np.array(states[-1])

    for t in times:
        
        solver_args = {'x0': u_init, 'p': x0, 'lbx': lbx, 'ubx': ubx}
        sol = solver(**solver_args)

        u_opt = np.array(sol['x']).flatten()
        F_opt = u_opt[0]

        # 记录控制输入
        inputs.append(F_opt)

        # 更新状态
        dx = np.array(pendulum_dynamics(x0, np.array([F_opt]))).flatten()
        x_new = x0 + dx * dt
        states.append(x_new)
        x0 = np.array(states[-1])

        # 解决MPC问题
        init = predict_inputs(model, x0)
        u_init = np.linspace(init[0], inputs[-1], 20)

    return np.array(states), np.array(inputs)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    model = MLP(4, 64, 3).to(device)
    model.load_state_dict(torch.load('best_mlp_controller_kfold.pth', map_location=device))
    model.eval()

    DT = 0.02
    solver, lbx, ubx = mpc_control(N=20, dt = DT)
    x_init = np.array([0.5, 0, -0.5, 0])
    states, actions = simulate_single(x_init, solver, lbx, ubx, dt = DT)
    states2, actions2 = simulate_single_mlp(model, x_init, solver, lbx, ubx, dt = DT)

    times = np.arange(0, len(actions)*DT, DT)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(times, states[:-1, 2], label='theta')
    plt.title("Pendulum Angle")
    plt.subplot(2, 1, 2)
    plt.plot(times, actions, label='Force')
    plt.title("Control Input (Force)")
    plt.savefig('mpc_result.png')
    times = np.arange(0, len(actions)*DT, DT)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(times, states2[:-1, 2], label='theta')
    plt.title("Pendulum Angle")
    plt.subplot(2, 1, 2)
    plt.plot(times, actions2, label='Force')
    plt.title("Control Input (Force)")
    plt.savefig('mpcmlp_result.png')

    #compare_inputs(actions, predict_inputs(model, states), DT)
