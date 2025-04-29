import torch
from mlp import MLP
import matplotlib.pyplot as plt
import numpy as np
from mpc import mpc_control, pendulum_dynamics, simulate_single

def simulate_single_mlp(initial_state, model_path='best_mlp_controller.pth', T=10, dt=0.2):
    """
    使用训练好的MLP模型进行仿真。
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    model = MLP().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    times = np.arange(0, T, dt)
    states = [initial_state]
    inputs = []

    for t in times:
        x0 = np.array(states[-1])
        x0_tensor = torch.tensor(x0, dtype=torch.float32, device=device).unsqueeze(0)  # shape (1,4)

        with torch.no_grad():
            F_pred = model(x0_tensor).item()  # 预测力

        # 记录控制输入
        inputs.append(F_pred)

        # 更新状态
        dx = np.array(pendulum_dynamics(x0, np.array([F_pred]))).flatten()
        x_new = x0 + dx * dt
        states.append(x_new)

    return np.array(states), np.array(inputs)

if __name__ == '__main__':
    # 单次仿真测试
    solver, lbx, ubx = mpc_control()
    x_init = np.array([0, 0, np.pi - 0.1, 0])
    states, actions = simulate_single(x_init, solver, lbx, ubx)

    # 画图
    times = np.arange(0, len(actions)*0.02, 0.02)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(times, states[:-1, 2], label='theta')
    plt.title("Pendulum Angle")
    plt.subplot(2, 1, 2)
    plt.plot(times, actions, label='Force')
    plt.title("Control Input (Force)")
    plt.savefig('mpc_result.png')

    # 使用MLP控制器进行仿真
    x_init = np.array([0, 0, np.pi - 0.1, 0])
    states_mlp, actions_mlp = simulate_single_mlp(x_init)

    # 画图
    times = np.arange(0, len(actions_mlp)*0.02, 0.02)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(times, states_mlp[:-1, 2], label='theta (MLP)')
    plt.title("Pendulum Angle (MLP Controller)")
    plt.subplot(2, 1, 2)
    plt.plot(times, actions_mlp, label='Force (MLP)')
    plt.title("Control Input (Force) (MLP Controller)")
    plt.savefig('mlp_result.png')
