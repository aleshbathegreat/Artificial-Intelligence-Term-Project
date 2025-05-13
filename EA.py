import random
import copy

# Hyperparameters for EA
population_size = 20
mutation_rate = 0.1
num_generations = 30

input_size = features.shape[1]
hidden_size = 128
num_classes = 2  # predicting arousal and valence
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Evaluation metric: Mean Squared Error
loss_fn = nn.MSELoss()

# Create a model and return flattened weights
def model_to_vector(model):
    return torch.cat([param.data.view(-1) for param in model.parameters()])

# Apply flattened weights to a model
def vector_to_model(model, vec):
    pointer = 0
    for param in model.parameters():
        num_param = param.numel()
        param.data = vec[pointer:pointer+num_param].view(param.size())
        pointer += num_param
    return model

# Initialize population of random models
def init_population():
    population = []
    for _ in range(population_size):
        model = Emotion_Classifier(input_size, hidden_size, num_classes).to(device)
        weights = model_to_vector(model)
        population.append(weights)
    return population

# Fitness function: negative loss (since we want to minimize it)
def fitness_function(weights, x_tensor, y_tensor):
    model = Emotion_Classifier(input_size, hidden_size, num_classes).to(device)
    model = vector_to_model(model, weights)
    model.eval()
    with torch.no_grad():
        preds = model(x_tensor)
        loss = loss_fn(preds, y_tensor)
    return -loss.item()

# Mutation: add small noise to weights
def mutate(weights):
    new_weights = weights.clone()
    for i in range(len(new_weights)):
        if random.random() < mutation_rate:
            new_weights[i] += torch.randn(1).to(device) * 0.1
    return new_weights

# Crossover: average weights of two parents
def crossover(w1, w2):
    return (w1 + w2) / 2.0

# Evolutionary optimization loop
x_tensor = torch.tensor(features, dtype=torch.float32).to(device)
y_tensor = torch.tensor(labels, dtype=torch.float32).to(device)

population = init_population()

for gen in range(num_generations):
    fitness_scores = [fitness_function(w, x_tensor, y_tensor) for w in population]
    sorted_pop = [w for _, w in sorted(zip(fitness_scores, population), key=lambda x: x[0], reverse=True)]

    print(f"Generation {gen}, Best Fitness: {max(fitness_scores):.4f}")

    # Selection: top 20%
    top_k = int(population_size * 0.2)
    new_population = sorted_pop[:top_k]

    # Reproduction: fill rest with crossover + mutation
    while len(new_population) < population_size:
        parent1, parent2 = random.sample(new_population[:top_k], 2)
        child = mutate(crossover(parent1, parent2))
        new_population.append(child)

    population = new_population

# Get best weights and evaluate final model
best_weights = population[0]
best_model = Emotion_Classifier(input_size, hidden_size, num_classes).to(device)
best_model = vector_to_model(best_model, best_weights)

# Save or use best_model for prediction
