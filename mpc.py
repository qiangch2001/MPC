import casadi as ca
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyDOE import lhs

# 系统参数
M = 1.0
m = 0.2
l = 0.5
g = 9.81
I = (1/3) * m * l**2

def pendulum_dynamics(x, u):
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    F = u[0]

    sin_theta = ca.sin(x3)
    cos_theta = ca.cos(x3)
    total_mass = M + m

    x1_dot = x2
    x2_dot = (F + m * l * x4**2 * sin_theta - m * g * sin_theta * cos_theta) / total_mass
    theta1_dot = x4
    theta2_dot = (m * l * x4**2 * sin_theta * cos_theta - m * g * sin_theta - F * cos_theta) / (l * (4/3 - m * cos_theta**2 / total_mass))

    return ca.vertcat(x1_dot, x2_dot, theta1_dot, theta2_dot)

def mpc_control(N=20, dt=0.2):
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

def simulate_single(initial_state, solver, lbx, ubx, T=100, dt=0.2):
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

def sample_initial_state_lhs(n_samples=300):
    ranges = {
        'x': (-0.5, 0.5),
        'x_dot': (-1.0, 1.0),
        'theta': (np.pi - 0.3, np.pi + 0.3),
        'theta_dot': (-2.0, 2.0)
    }

    samples = lhs(4, samples=n_samples)
    initial_states = []
    for i in range(n_samples):
        x = ranges['x'][0] + (ranges['x'][1] - ranges['x'][0]) * samples[i, 0]
        x_dot = ranges['x_dot'][0] + (ranges['x_dot'][1] - ranges['x_dot'][0]) * samples[i, 1]
        theta = ranges['theta'][0] + (ranges['theta'][1] - ranges['theta'][0]) * samples[i, 2]
        theta_dot = ranges['theta_dot'][0] + (ranges['theta_dot'][1] - ranges['theta_dot'][0]) * samples[i, 3]
        initial_states.append(np.array([x, x_dot, theta, theta_dot]))

    print("work")
    return np.array(initial_states)

def collect_dataset(num_trajectories=300):
    solver, lbx, ubx = mpc_control()
    initial_states = sample_initial_state_lhs(num_trajectories)
    all_states = []
    all_actions = []

    i = 0
    for x_init in initial_states:
        i += 1
        print(i)
        states, actions = simulate_single(x_init, solver, lbx, ubx, T=2, dt=0.2)
        all_states.append(states[:-1])  # 去掉最后一个状态
        all_actions.append(actions)

    all_states = np.vstack(all_states)
    all_actions = np.hstack(all_actions).reshape(-1, 1)

    print(f"采集到 {all_states.shape[0]} 个样本")
    np.save('states.npy', all_states)
    np.save('actions.npy', all_actions)
    print("数据集保存完成！")

if __name__ == '__main__':
    solver, lbx, ubx = mpc_control()
    x_init = np.array([0, 0, np.pi - 0.1, 0])
    states, actions = simulate_single(x_init, solver, lbx, ubx)

    times = np.arange(0, len(actions)*0.02, 0.02)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(times, states[:-1, 2], label='theta')
    plt.title("Pendulum Angle")
    plt.subplot(2, 1, 2)
    plt.plot(times, actions, label='Force')
    plt.title("Control Input (Force)")
    plt.savefig('mpc_result.png')

    collect_dataset()
