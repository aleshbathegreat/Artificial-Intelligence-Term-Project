import torch
import numpy as np

# Utility function
def get_flat_params(model):
    return torch.cat([param.view(-1) for param in model.parameters()])

def set_flat_params(model, flat_params):
    pointer = 0
    for param in model.parameters():
        num_params = param.numel()
        param.data.copy_(flat_params[pointer:pointer + num_params].view_as(param))
        pointer += num_params

# Fitness evaluation using cross-entropy loss
def evaluate(model, X, y, criterion):
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        loss = criterion(outputs, y)
    return loss.item()

# Evolutionary Algorithm for optimizing neural network weights
def evolutionary_algorithm(model, X_train, y_train, criterion,
                           population_size=30, generations=50,
                           mutation_rate=0.1, elite_fraction=0.2):

    param_size = get_flat_params(model).shape[0]
    population = [torch.randn(param_size) for _ in range(population_size)]

    elite_count = max(1, int(elite_fraction * population_size))

    for generation in range(generations):
        fitness_scores = []
        for individual in population:
            set_flat_params(model, individual)
            fitness = evaluate(model, X_train, y_train, criterion)
            fitness_scores.append((fitness, individual))

        # Sort by fitness (lower is better)
        fitness_scores.sort(key=lambda x: x[0])
        best_individuals = [ind for _, ind in fitness_scores[:elite_count]]

        new_population = best_individuals.copy()
        while len(new_population) < population_size:
            parent1, parent2 = np.random.choice(best_individuals, 2, replace=False)
            alpha = torch.rand(1).item()
            child = alpha * parent1 + (1 - alpha) * parent2

            # Mutation
            if np.random.rand() < mutation_rate:
                child += torch.randn(param_size) * 0.05

            new_population.append(child)

        population = new_population

        if generation % 10 == 0:
            print(f"Generation {generation}, Best Loss: {fitness_scores[0][0]:.4f}")

    # Set model to best individual
    best_params = fitness_scores[0][1]
    set_flat_params(model, best_params)

    return model
