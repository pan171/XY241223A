import torch
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, rho, device='cpu'):
        super(CustomModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size).to(device)
        self.layer2 = nn.Linear(hidden_size, output_size).to(device)
        self.lambda1 = torch.zeros(hidden_size, requires_grad=False).to(device)
        self.rho = rho  # penalty factor
        self.mu = 1.1
        self.gamma = 0.77
        self.prev_constraint_violation = 0.0
        # Add device attribute for use in methods
        self.device = device
        self._reset_parameters()

    def _reset_parameters(self):
        # He initialization for layers with ReLU activation
        nn.init.kaiming_uniform_(self.layer1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.layer2.weight, mode='fan_in', nonlinearity='relu')
        self.layer1 = self.layer1.to(self.device)
        self.layer2 = self.layer2.to(self.device)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.sigmoid(x)
        x = self.layer2(x)
        return x

    def enforce_constraints(self):
        with torch.no_grad():
            current_constraint_violation = 0.0  # Initialize current violation sum
            weights_l1 = self.layer1.weight[:, :3]  # the first 3 features' weights
            weights_l2 = self.layer2.weight  # hidden -> output layer's weights

            for i in range(3):
                for j in range(weights_l1.shape[0]):  # for each neuron in hidden
                    product = weights_l1[j, i] * weights_l2[:, j]
                    constraint_violation = torch.max(-product, torch.zeros_like(product))
                    # In this case, we want the product to be positive, so we negate it to find violations
                    current_constraint_violation += constraint_violation.sum().item()
                    # Correction should aim to make product positive
                    # Since we want product > 0, we adjust by adding the absolute value of violation times a correction factor
                    correction = (self.rho * constraint_violation * weights_l2[:, j]).sum()
                    weights_l1[j, i] += correction  # Note the change here; we're adding to make the product positive
                    
                    # Update lambda1 for accounting purposes, though it's not directly used in correction here
                    self.lambda1[j] += self.rho * constraint_violation.sum()

            # Update rho based on the comparison of current and previous constraint violations
            if current_constraint_violation >= self.gamma * self.prev_constraint_violation:
                self.rho *= self.mu
            self.prev_constraint_violation = current_constraint_violation

    def customed_enforce_constraints(self, fm=3, fn=3):
        with torch.no_grad():
            current_constraint_violation = 0.0  # Initialize current violation sum
            weights_l1 = self.layer1.weight[:, :fm]  # the first fm features' weights
            weights_l2 = self.layer2.weight  # hidden -> output layer's weights

            for i in range(fm):  # Loop through the first fm features
                for j in range(weights_l1.shape[0]):  # for each neuron in hidden layer
                    product = weights_l1[j, i] * weights_l2[:, j]
                    if i < fn:  # For the first fn features, we want the product to be positive
                        constraint_violation = torch.max(-product, torch.zeros_like(product))
                        # Correction should aim to make product positive
                        correction = (self.rho * constraint_violation * weights_l2[:, j]).sum()
                    else:  # For the rest features, we want the product to be negative
                        constraint_violation = torch.max(product, torch.zeros_like(product))
                        # Correction should aim to make product negative
                        correction = -(self.rho * constraint_violation * weights_l2[:, j]).sum()

                    current_constraint_violation += constraint_violation.sum().item()

                    weights_l1[j, i] += correction  # Apply the correction
                    # Update lambda1 for accounting purposes
                    self.lambda1[j] += self.rho * constraint_violation.sum()

            # Update rho based on the comparison of current and previous constraint violations
            if current_constraint_violation >= self.gamma * self.prev_constraint_violation:
                self.rho *= self.mu
            self.prev_constraint_violation = current_constraint_violation