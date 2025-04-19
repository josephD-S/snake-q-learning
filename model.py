import torch
import torch.nn as nn
import torch.optim as optim
import os 
import numpy as np

class Linear_QNet(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.layers = nn.Sequential(
                                    nn.Linear(2052, hidden_size),
                                    nn.Linear(hidden_size, hidden_size//2),
                                    nn.Linear(hidden_size//2, hidden_size//4),
                                    nn.Linear(hidden_size//4, output_size)
        )

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float)
        return self.layers(x)

    def save(self, file_name="snake_model.pth"):
        model_folder_path = "./model"
        os.makedirs(model_folder_path, exist_ok=True)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma 
        self.model = model
        self.optim = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)

        if len(state.shape) == 1:
            # reshape to (1, x)
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state, 0)
            done = (done, )

        # 1: predicted Q values with current state
        prediction = self.model(state)

        target = prediction.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action).item()] = Q_new

        # 2: Q_new = r + y * max(next_predicted Q value)
        # prediction.clone()
        # preds[argmax(action)] = Q_new
        self.optim.zero_grad()
        loss = self.criterion(target, prediction)
        loss.backward()

        self.optim.step()

        return torch.mean(target[-1]).detach().numpy()



