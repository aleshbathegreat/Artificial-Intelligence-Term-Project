import torch
import torch.nn as nn
import numpy as np
import random

X = torch.tensor(features.values, dtype=torch.float32)
y = torch.tensor(merged['mean_arousal'].values, dtype=torch.float32).view(-1, 1)  # assuming regression

# Instantiate model
model = Emotion_Classifier(input_size=X.shape[1], hidden_size=128, num_classes=1)  # Use 1 if doing regression

# Loss function
criterion = nn.MSELoss()

# Flatten model parameters into a single vector
def get_weights(model):
    return torch.cat([p.view(-1) for p in model.parameters()])

# Load a flat vector of weights into model
def set_weights(model, flat_weights):
    idx = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_weights[idx:idx+numel].view_as(p))
        idx += numel

# Evaluate loss
def evaluate(model, X, y):
    with torch.no_grad():
        output = model(X)
        loss = criterion(output, y)
    return loss.item()

# Simulated Annealing function
def simulated_annealing(model, X, y, num_iterations=1000, initial_temp=1.0, cooling_rate=0.003):
    current_weights = get_weights(model).clone()
    current_loss = evaluate(model, X, y)
    best_weights = current_weights.clone()
    best_loss = current_loss

    temp = initial_temp

    for step in range(num_iterations):
        # Perturb weights
        new_weights = current_weights + 0.1 * torch.randn_like(current_weights)
        set_weights(model, new_weights)
        new_loss = evaluate(model, X, y)

        delta = new_loss - current_loss

        # Accept new weights if better, or probabilistically if worse
        if delta < 0 or random.random() < torch.exp(-delta / temp):
            current_weights = new_weights
            current_loss = new_loss

            if new_loss < best_loss:
                best_loss = new_loss
                best_weights = new_weights.clone()
        else:
            set_weights(model, current_weights)  # revert to old

        # Cool down
        temp *= (1 - cooling_rate)

        if step % 100 == 0:
            print(f"Step {step}: Current Loss = {current_loss:.4f}, Best Loss = {best_loss:.4f}, Temp = {temp:.4f}")

    # Set model to best found
    set_weights(model, best_weights)
    return model
