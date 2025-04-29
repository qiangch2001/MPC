import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import ParameterGrid
from collections import defaultdict
import matplotlib.pyplot as plt
import time

class BasicDataset:
    """兼容原始states.npy和actions.npy的数据加载器"""
    def __init__(self, states_path='states.npy', actions_path='actions.npy'):
        self.states = np.load(states_path)
        self.actions = np.load(actions_path)
        
        # 数据标准化
        self.state_mean = self.states.mean(axis=0)
        self.state_std = self.states.std(axis=0)
        self.action_mean = self.actions.mean()
        self.action_std = self.actions.std()
        
    def get_torch_dataset(self, normalize=True):
        """返回PyTorch数据集"""
        states = self.states.copy()
        actions = self.actions.copy()
        
        if normalize:
            states = (states - self.state_mean) / (self.state_std + 1e-8)
            actions = (actions - self.action_mean) / (self.action_std + 1e-8)
        
        return TensorDataset(
            torch.FloatTensor(states),
            torch.FloatTensor(actions)
        )

class HyperparameterTuner:
    def __init__(self, dataset):
        self.dataset = dataset
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = []
        self.best_model_info = None
        
    def get_default_hyperparameters(self):
        """定义超参数搜索空间"""
        return {
            'hidden_size': [64, 128, 256],
            'num_layers': [2, 3, 4],
            'learning_rate': [1e-3, 3e-4, 1e-4],
            'batch_size': [64, 128, 256],
            'dropout_rate': [0.0, 0.1, 0.2],
            'activation': ['ReLU', 'LeakyReLU', 'ELU']
        }
    
    class TunableMLP(nn.Module):
        """可配置的MLP模型"""
        def __init__(self, input_size=4, output_size=1, 
                     hidden_size=128, num_layers=3, 
                     dropout_rate=0.0, activation='ReLU'):
            super().__init__()
            
            # 激活函数选择
            activations = {
                'ReLU': nn.ReLU(),
                'LeakyReLU': nn.LeakyReLU(0.1),
                'ELU': nn.ELU()
            }
            act_fn = activations[activation]
            
            layers = []
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(act_fn)
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(act_fn)
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))
            
            layers.append(nn.Linear(hidden_size, output_size))
            self.model = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.model(x)
    
    def train_model(self, params, num_epochs=100, val_ratio=0.2):
        """训练和验证模型"""
        torch.manual_seed(42)
        
        # 创建训练和验证集
        full_dataset = self.dataset.get_torch_dataset()
        val_size = int(val_ratio * len(full_dataset))
        train_size = len(full_dataset) - val_size
        train_set, val_set = torch.utils.data.random_split(
            full_dataset, [train_size, val_size])
        
        train_loader = DataLoader(
            train_set, 
            batch_size=params['batch_size'], 
            shuffle=True
        )
        val_loader = DataLoader(
            val_set, 
            batch_size=params['batch_size'], 
            shuffle=False
        )
        
        # 初始化模型
        model = self.TunableMLP(
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout_rate=params['dropout_rate'],
            activation=params['activation']
        ).to(self.device)
        
        # 定义优化器和损失函数
        optimizer = optim.Adam(
            model.parameters(), 
            lr=params['learning_rate']
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        criterion = nn.MSELoss()
        
        # 训练循环
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # 验证阶段
            val_loss = 0.0
            model.eval()
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item()
            
            # 记录指标
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            
            scheduler.step(avg_val_loss)
            
            # 打印进度
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | "
                      f"Train Loss: {avg_train_loss:.4f} | "
                      f"Val Loss: {avg_val_loss:.4f}")
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict()
        
        return {
            'model': model,
            'best_model_state': best_model_state,
            'history': history,
            'best_val_loss': best_val_loss
        }
    
    def run_grid_search(self, param_grid=None, num_epochs=50):
        """执行网格搜索"""
        if param_grid is None:
            param_grid = self.get_default_hyperparameters()
        
        grid = ParameterGrid(param_grid)
        print(f"Starting grid search with {len(grid)} combinations...")
        
        for i, params in enumerate(grid):
            print(f"\nTraining model {i+1}/{len(grid)} with params:")
            print(params)
            
            start_time = time.time()
            result = self.train_model(params, num_epochs)
            
            # 记录结果
            run_result = {
                'params': params,
                'train_history': result['history'],
                'best_val_loss': result['best_val_loss'],
                'training_time': time.time() - start_time,
                'model_state': result['best_model_state']
            }
            self.results.append(run_result)
            
            # 更新最佳模型
            if self.best_model_info is None or \
               result['best_val_loss'] < self.best_model_info['best_val_loss']:
                self.best_model_info = run_result
                torch.save(result['model'].state_dict(), 'best_model_weights.pth')
            
            print(f"Completed in {run_result['training_time']:.1f}s | "
                  f"Best Val Loss: {result['best_val_loss']:.4f}")
        
        print("\nGrid search completed!")
        return self.results
    
    def analyze_results(self):
        """分析并可视化结果"""
        if not self.results:
            print("No results to analyze. Run grid search first.")
            return
        
        # 找出最佳配置
        best_run = self.best_model_info
        print("\n=== Best Hyperparameters ===")
        for k, v in best_run['params'].items():
            print(f"{k:>15}: {v}")
        
        print("\n=== Best Validation Loss ===")
        print(f"{best_run['best_val_loss']:.6f}")
        
        # 可视化训练曲线
        plt.figure(figsize=(12, 6))
        for result in self.results:
            if result == best_run:
                plt.plot(result['train_history']['val_loss'], 
                        'r-', lw=2, label='Best Config')
            else:
                plt.plot(result['train_history']['val_loss'], 
                        'gray', alpha=0.2)
        
        plt.title('Validation Loss Across Configurations')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('val_loss_curves.png')
        plt.close()
        
        # 保存完整结果
        import pandas as pd
        results_df = pd.DataFrame([{
            **r['params'],
            'val_loss': r['best_val_loss'],
            'training_time': r['training_time']
        } for r in self.results])
        results_df.to_csv('hyperparameter_results.csv', index=False)
        
        return best_run

# 使用示例
if __name__ == '__main__':
    # 1. 加载原始数据集
    print("Loading dataset...")
    dataset = BasicDataset(states_path='states.npy', actions_path='actions.npy')
    
    # 2. 初始化调优器
    tuner = HyperparameterTuner(dataset)
    
    # 3. 定义自定义搜索空间 (或使用默认值)
    custom_grid = {
        'hidden_size': [128, 256],
        'num_layers': [2, 3],
        'learning_rate': [1e-3, 3e-4],
        'batch_size': [128, 256],
        'dropout_rate': [0.0, 0.1],
        'activation': ['ReLU', 'LeakyReLU']
    }
    
    # 4. 运行网格搜索 (减少epochs数量以加快测试)
    print("\nStarting grid search...")
    results = tuner.run_grid_search(custom_grid, num_epochs=30)
    
    # 5. 分析结果
    print("\nAnalyzing results...")
    best_run = tuner.analyze_results()
    
    # 6. 保存最佳模型
    torch.save({
        'model_params': best_run['params'],
        'state_dict': best_run['model_state'],
        'val_loss': best_run['best_val_loss'],
        'normalization': {
            'state_mean': dataset.state_mean,
            'state_std': dataset.state_std,
            'action_mean': dataset.action_mean,
            'action_std': dataset.action_std
        }
    }, 'best_model_full.pth')
    
    print("\nBest model saved to 'best_model_full.pth'")