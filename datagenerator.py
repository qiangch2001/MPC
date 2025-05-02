from mpc import mpc_control, simulate_single
import numpy as np
from pyDOE import lhs

def sample_initial_state_lhs(n_samples=30):
    ranges = {
        'x': (-0.5, 0.5),
        'x_dot': (-1.0, 1.0),
        'theta': (-1, 1),
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

def collect_dataset(dt, N, num_trajectories=500):
    solver, lbx, ubx = mpc_control()
    initial_states = sample_initial_state_lhs(num_trajectories)
    all_states = []
    all_actions = []

    i = 0
    for x_init in initial_states:
        i += 1
        print(i)
        states, actions = simulate_single(x_init, solver, lbx, ubx, T=2 * N * dt, dt=dt)
        all_states.append(states[:-1])  # 去掉最后一个状态
        all_actions.append(actions)

    all_states = np.vstack(all_states)
    all_actions = np.hstack(all_actions).reshape(-1, 1)

    print(f"采集到 {all_states.shape[0]} 个样本")
    np.save('states.npy', all_states)
    np.save('actions.npy', all_actions)
    print("数据集保存完成！")

if __name__ == '__main__':
    DT = 0.05
    collect_dataset(dt = DT, N = 20)
