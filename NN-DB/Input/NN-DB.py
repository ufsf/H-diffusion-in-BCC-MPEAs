import logging
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set the directory containing the .npy files
soap_folder_path_1 = '../../input-250-1/'
soap_data_list = []
# Loop through the sorted file list and load each .npy file
for i in range(1, 251):
    soap_file_path = soap_folder_path_1 + 'dsoap_' + str(i) + '.npy'
    soap_data = np.load(soap_file_path)
    soap_data_list.append(soap_data)

soap_folder_path_2 = '../../input-250-2/'
# Loop through the sorted file list and load each .npy file
for i in range(1, 251):
    soap_file_path = soap_folder_path_2 + 'dsoap_' + str(i) + '.npy'
    soap_data = np.load(soap_file_path)
    soap_data_list.append(soap_data)

SOAP_all = np.vstack(soap_data_list)

logging.info("Loaded SOAP train data from all files.")
logging.info(f"SOAP_all shape: {SOAP_all.shape}")


SE_1 = np.load('../../output-250/Barrier-1.npy')
SE_2 = np.load('../../output-250/Barrier-2.npy')
SE_all = np.concatenate((SE_1, SE_2))
logging.info("Loaded SE train data from all files.")
logging.info(f"SE_all shape: {SE_all.shape}")

# Define configurable parameters
input_size = 3696  # Example input size
hidden_layers = [32, 32, 32, 32]  # List of hidden layer sizes
output_size = 1  # Single output for regression
batch_size = 128
learning_rate = 0.001
num_epochs = 1500
patience = 100

# Convert numpy arrays to torch tensors
features = torch.tensor(SOAP_all, dtype=torch.float32)
targets = torch.tensor(SE_all, dtype=torch.float32).view(-1, 1)

# Create a TensorDataset and DataLoader
dataset = TensorDataset(features, targets)
train_size = int(0.80 * len(dataset))
valid_size = int(0.10 * len(dataset))
test_size = len(dataset) - train_size - valid_size
train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class SOAPNet(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super(SOAPNet, self).__init__()
        layers = []
        in_features = input_size
        for hidden_units in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_units))
            layers.append(nn.ReLU())
            in_features = hidden_units
        layers.append(nn.Linear(in_features, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

model = SOAPNet(input_size, hidden_layers, output_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

def train_model(num_epochs, patience, model, train_loader, valid_loader):
    min_val_loss = float('inf')
    early_stop_counter = 0
    train_losses = []
    valid_losses = []

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            output = torch.exp(output)  # Ensure output is positive
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for data, target in valid_loader:
                output = model(data)
                output = torch.exp(output)  # Ensure output is positive
                valid_loss += criterion(output, target).item()
        avg_val_loss = valid_loss / len(valid_loader)
        valid_losses.append(avg_val_loss)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        logging.info(f'Epoch {epoch+1}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}, Time: {epoch_duration:.2f} seconds')

        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            logging.info(f'Saved new best model with validation loss: {min_val_loss}')
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            logging.info('Early stopping triggered.')
            break

    return train_losses, valid_losses

# Example call to train_model and subsequent plotting
train_losses, valid_losses = train_model(num_epochs, patience, model, train_loader, valid_loader)

plt.figure(figsize=(10, 6))
plt.plot(train_losses[1:], label='Training Loss')  # Adjust to use only from the second element if needed
plt.plot(valid_losses[1:], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.savefig("training.png")

model.load_state_dict(torch.load('best_model.pth'))
model.eval()  # Set the model to evaluation mode

test_loss = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        output = torch.exp(output)  # Ensure output is positive
        test_loss += criterion(output, target).item()
avg_test_loss = test_loss / len(test_loader)
print(f'Test Loss: {avg_test_loss}')

# Collect true values and predictions
true_values = []
predicted_values = []
with torch.no_grad():
    for data, targets in test_loader:
        outputs = model(data)
        outputs = torch.exp(outputs)  # Ensure outputs are positive
        predicted_values.extend(outputs.numpy())
        true_values.extend(targets.numpy())

# Convert lists to numpy arrays for easier manipulation
y_true = np.array(true_values).flatten()
y_pred = np.array(predicted_values).flatten()

##### Visualizing True vs Predicted Values #####
plt.figure(figsize=(10, 6))
plt.scatter(y_true, y_pred, alpha=0.5)
plt.title('True vs Predicted Values')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--')  # Diagonal line
plt.savefig("comp.png")  

##### Visualizing Residuals #####
residuals = y_true - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.title('Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.hlines(y=0, xmin=min(y_pred), xmax=max(y_pred), colors='red', linestyles='--')  # Zero line
plt.savefig("Residuals.png")  

##### Calculating Performance Metrics #####
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R-squared: {r2}")
