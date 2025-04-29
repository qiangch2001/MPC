import numpy as np
import torch
import casadi as ca
import matplotlib.pyplot as plt
from mpc import pendulum_dynamics, mpc_control, simulate_single
from mlp import MLP

class LearningAugmentedMPC:
    def __init__(self, mpc_horizon=20, dt=0.02):
        # Initialize MPC controller
        self.mpc_solver, self.lbx, self.ubx = mpc_control(N=mpc_horizon, dt=dt)
        self.mpc_horizon = mpc_horizon
        self.dt = dt
        
        # Load trained MLP model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mlp = MLP().to(self.device)
        self.mlp.load_state_dict(torch.load('best_mlp_controller.pth'))
        self.mlp.eval()
        
    def mlp_predict(self, state):
        """Predict control using the trained MLP"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            return self.mlp(state_tensor).cpu().numpy()[0]
    
    def solve_mpc(self, current_state, warm_start=None):
        """Solve MPC with optional warm start from MLP"""
        if warm_start is None:
            # Default initialization if no warm start provided
            u_init = np.zeros(self.mpc_horizon)
        else:
            # Use MLP prediction as initial guess for first control input
            # and create a decaying sequence for the horizon
            u_init = np.full(self.mpc_horizon, warm_start)
            decay = np.linspace(1.0, 0.5, self.mpc_horizon)
            u_init = u_init * decay
        
        # Solve MPC
        solver_args = {
            'x0': u_init,
            'p': current_state,
            'lbx': self.lbx,
            'ubx': self.ubx
        }
        sol = self.mpc_solver(**solver_args)
        
        return np.array(sol['x']).flatten()
    
    def simulate(self, initial_state, T=100, use_mlp_warmstart=True):
        """Simulate the system with learning-augmented MPC"""
        times = np.arange(0, T, self.dt)
        states = [initial_state]
        inputs = []
        mlp_predictions = []
        
        for t in times:
            current_state = states[-1]
            
            # Get MLP prediction (for warm start or comparison)
            mlp_pred = self.mlp_predict(current_state)
            mlp_predictions.append(mlp_pred)
            
            # Solve MPC with optional warm start
            if use_mlp_warmstart:
                u_opt = self.solve_mpc(current_state, warm_start=mlp_pred)
            else:
                u_opt = self.solve_mpc(current_state)
            
            # Apply first control input
            F_opt = u_opt[0]
            inputs.append(F_opt)
            
            # Simulate forward
            dx = np.array(pendulum_dynamics(current_state, np.array([F_opt]))).flatten()
            x_new = current_state + dx * self.dt
            states.append(x_new)
        
        return {
            'times': times,
            'states': np.array(states[:-1]),  # remove last state
            'inputs': np.array(inputs),
            'mlp_predictions': np.array(mlp_predictions)
        }

def plot_results(results, title):
    """Plot simulation results"""
    plt.figure(figsize=(12, 8))
    
    # Plot pendulum angle
    plt.subplot(3, 1, 1)
    plt.plot(results['times'], results['states'][:, 2], label='θ (rad)')
    plt.axhline(y=np.pi, color='r', linestyle='--', label='Desired (π)')
    plt.ylabel('Angle (rad)')
    plt.title(f'{title} - Pendulum Angle')
    plt.legend()
    
    # Plot control inputs
    plt.subplot(3, 1, 2)
    plt.plot(results['times'], results['inputs'], label='MPC Control')
    plt.plot(results['times'], results['mlp_predictions'], '--', label='MLP Prediction')
    plt.ylabel('Force (N)')
    plt.title('Control Inputs')
    plt.legend()
    
    # Plot cart position
    plt.subplot(3, 1, 3)
    plt.plot(results['times'], results['states'][:, 0], label='x (m)')
    plt.ylabel('Position (m)')
    plt.title('Cart Position')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}_results.png')
    plt.close()

if __name__ == '__main__':
    # Initialize the learning-augmented MPC controller
    la_mpc = LearningAugmentedMPC()
    
    # Initial state: [x, x_dot, theta, theta_dot]
    initial_state = np.array([0.1, 0, np.pi - 0.2, 0])  # Slightly perturbed from upright
    
    # Simulate with MLP warm start
    aug_results = la_mpc.simulate(initial_state, use_mlp_warmstart=True)
    plot_results(aug_results, "Learning-Augmented MPC")
    
    # Simulate with standard MPC (no warm start) for comparison
    standard_results = la_mpc.simulate(initial_state, use_mlp_warmstart=False)
    plot_results(standard_results, "Standard MPC")
    
    # Compare computation times
    import time
    
    # Time learning-augmented MPC
    start = time.time()
    _ = la_mpc.simulate(initial_state, use_mlp_warmstart=True)
    aug_time = time.time() - start
    
    # Time standard MPC
    start = time.time()
    _ = la_mpc.simulate(initial_state, use_mlp_warmstart=False)
    std_time = time.time() - start
    
    print(f"\nPerformance Comparison:")
    print(f"Learning-Augmented MPC time: {aug_time:.4f} sec")
    print(f"Standard MPC time: {std_time:.4f} sec")
    print(f"Speedup: {std_time/aug_time:.2f}x")