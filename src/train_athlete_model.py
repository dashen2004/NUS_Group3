import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Step 1: Read the CSV file and prepare data
data = pd.read_csv('athlete_data.csv')

# Convert columns to the appropriate data types manually if needed
data = data.astype({
    'Feature1': 'float',
    'Feature2': 'float',
    'Feature3': 'float',
    'Feature4': 'float',
    'Feature5': 'float',
    'Feature6': 'float',
    'Feature7': 'float',
    'Feature8': 'float',
    'Label': 'int'
})

# Separate features and labels
X = data[['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Feature7', 'Feature8']].values
y = data['Label'].values

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Print shapes to ensure everything is correct
print(f'X_train_tensor shape: {X_train_tensor.shape}')
print(f'y_train_tensor shape: {y_train_tensor.shape}')
print(f'X_test_tensor shape: {X_test_tensor.shape}')
print(f'y_test_tensor shape: {y_test_tensor.shape}')

# Print some data samples to ensure correctness
print(f'First 5 samples of X_train_tensor: {X_train_tensor[:5]}')
print(f'First 5 samples of y_train_tensor: {y_train_tensor[:5]}')

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)


# Define the neural network model with dropout
class AthleteNetwork(nn.Module):
    def __init__(self):
        super(AthleteNetwork, self).__init__()
        self.layer1 = nn.Linear(8, 16)   # 第一隐藏层，32个神经元
        self.dropout1 = nn.Dropout(0.5) # Dropout层，防止过拟合
        self.layer2 = nn.Linear(16, 8)  # 第二隐藏层，16个神经元
        self.dropout2 = nn.Dropout(0.5) # Dropout层
        self.layer3 = nn.Linear(8, 4)   # 第三隐藏层，8个神经元
        self.layer4 = nn.Linear(4, 1)    # 输出层

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.dropout1(x)
        x = F.relu(self.layer2(x))
        x = self.dropout2(x)
        x = F.relu(self.layer3(x))
        x = torch.sigmoid(self.layer4(x))
        return x



# Instantiate the model
model = AthleteNetwork()

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training loop with early stopping
epochs = 50
best_loss = float('inf')
patience = 5
trigger_times = 0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        running_loss += loss.item()

    scheduler.step()

    avg_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}')

    # Early stopping
    if avg_loss < best_loss:
        best_loss = avg_loss
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping!")
            break

# Evaluation on test data
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total}%')




# 假设 new_data 是一个包含新样本特征的 NumPy 数组，形状为 (n_samples, 8)
new_data = [[22, 9.764615385, 0.012591026, 1.230769231, 1, 9.69, 1169.777336, 1]]

# 标准化新数据
new_data = scaler.transform(new_data)

# 转换为 PyTorch tensor
new_data_tensor = torch.tensor(new_data, dtype=torch.float32)

# 预测新数据的标签
model.eval()
with torch.no_grad():
    new_outputs = model(new_data_tensor)
    new_predictions = (new_outputs > 0.5).float()

print(f'Predicted labels: {new_predictions.numpy()}')
