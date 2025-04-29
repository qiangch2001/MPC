import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 128),   # 输入状态是4维
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)    # 输出是一个控制力（scalar）
        )
    
    def forward(self, x):
        return self.model(x)
    
if __name__ == '__main__':
    # 加载数据
    states = np.load('states.npy')  # (样本数, 4)
    actions = np.load('actions.npy')  # (样本数, 1)

    # 转为Tensor
    states_tensor = torch.tensor(states, dtype=torch.float32)
    actions_tensor = torch.tensor(actions, dtype=torch.float32)

    # 数据集
    dataset = TensorDataset(states_tensor, actions_tensor)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)



    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    device = torch.device(device)
    model = MLP().to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # 早停参数
    n_epochs = 500
    patience = 10  # 容忍多少个epoch没有提升
    best_loss = float('inf')
    trigger_times = 0

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for batch_states, batch_actions in train_loader:
            batch_states, batch_actions = batch_states.to(device), batch_actions.to(device)

            optimizer.zero_grad()
            preds = model(batch_states)
            loss = loss_fn(preds, batch_actions)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_states.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.6f}")

        # Early Stopping 检查
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            trigger_times = 0  # 有提升，重置
            # 也可以在这里保存最佳模型
            torch.save(model.state_dict(), 'best_mlp_controller.pth')
        else:
            trigger_times += 1
            print(f'No improvement. Trigger times: {trigger_times}')
            if trigger_times >= patience:
                print('Early stopping triggered!')
                break

    print("训练完成，最佳模型已保存为 best_mlp_controller.pth！")
