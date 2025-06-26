import torch
import torch.nn as nn
import torch.optim as optim

# Define the first MLP model
model_1 = nn.Sequential(
    nn.Linear(input_size, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU()
    # Add more layers if needed
)

# Define the second MLP model
model_2 = nn.Sequential(
    nn.Linear(output_of_model_1_size, 32),
    nn.ReLU(),
    nn.Linear(32, 10),  # Assuming 10 classes for classification
    nn.Softmax(dim=1)  # Softmax for multi-class classification
    # Add more layers if needed
)

# Combined model: Chain the models
class CombinedModel(nn.Module):
    def __init__(self, model_1, model_2):
        super(CombinedModel, self).__init__()
        self.model_1 = model_1
        self.model_2 = model_2

    def forward(self, x):
        x = self.model_1(x)
        x = self.model_2(x)
        return x

combined_model = CombinedModel(model_1, model_2)

# Define the loss function
criterion = nn.CrossEntropyLoss()  # Assuming a classification task

# Define optimizer
optimizer = optim.Adam(combined_model.parameters(), lr=0.001)

# Assuming you have PyTorch DataLoader (train_loader)
for epoch in range(10):  # Training for 10 epochs
    for inputs, targets in train_loader:
        # Forward pass
        outputs_of_model_1 = model_1(inputs)
        outputs = combined_model(outputs_of_model_1)

        # Calculate loss
        loss = criterion(outputs, targets)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Save weights of the first MLP model (model_1)
torch.save(model_1.state_dict(), 'model_1_weights.pth')

# Save weights of the second MLP model (model_2)
torch.save(model_2.state_dict(), 'model_2_weights.pth')