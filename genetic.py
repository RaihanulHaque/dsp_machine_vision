import random
def f(x):
    # return x**3 - 60*x**2 + 900*x + 100
    return x**2

# GA Parameters
POP_SIZE = 100
GENES = 5  
GENERATIONS = 300
MUTATION_RATE = 0.01

# Generate random individual (binary string of 5 bits)
def generate_individual():
    return [random.randint(0, 1) for _ in range(GENES)]

# Decode binary to integer
def decode(individual):
    return int("".join(map(str, individual)), 2)

# Selection - Tournament selection
def select(population, fitnesses):
    i, j = random.sample(range(len(population)), 2)
    return population[i] if fitnesses[i] > fitnesses[j] else population[j]

# Crossover - Single point
def crossover(p1, p2):
    point = random.randint(1, GENES - 1)
    return p1[:point] + p2[point:], p2[:point] + p1[point:]

# Mutation - Flip bit
def mutate(individual):
    return [bit if random.random() > MUTATION_RATE else 1 - bit for bit in individual]

# Main loop
population = [generate_individual() for _ in range(POP_SIZE)]

for generation in range(GENERATIONS):
    decoded = [decode(ind) for ind in population]
    fitnesses = [f(x) for x in decoded]

    next_gen = []
    for _ in range(POP_SIZE // 2):
        parent1 = select(population, fitnesses)
        parent2 = select(population, fitnesses)
        child1, child2 = crossover(parent1, parent2)
        next_gen.append(mutate(child1))
        next_gen.append(mutate(child2))

    population = next_gen

# Final Result
best_individual = max(population, key=lambda ind: f(decode(ind))) 
best_x = decode(best_individual)
best_y = f(best_x)

print(f"Best solution: x = {best_x}, f(x) = {best_y}")