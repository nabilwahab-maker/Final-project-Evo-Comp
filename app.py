import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# ==================================================
# LOAD DATA
# ==================================================
def load_data(file):
    if file is not None:
        df = pd.read_csv(file)
        if df.iloc[:, 0].dtype == object:
            df = df.iloc[:, 1:]
        df = df.apply(pd.to_numeric, errors='coerce')
        return df.dropna().values

    return np.array([
        [10, 20, 5, 15],
        [8, 12, 20, 10],
        [15, 5, 10, 25]
    ])

# ==================================================
# FITNESS FUNCTION (MAKESPAN)
# ==================================================
def fitness_function(sequence, data):
    n_machines, n_jobs = data.shape
    finish = np.zeros((n_machines, n_jobs))

    for m in range(n_machines):
        for j in range(n_jobs):
            job = sequence[j]
            p = data[m, job]

            if m == 0 and j == 0:
                finish[m, j] = p
            elif m == 0:
                finish[m, j] = finish[m, j-1] + p
            elif j == 0:
                finish[m, j] = finish[m-1, j] + p
            else:
                finish[m, j] = max(
                    finish[m-1, j],
                    finish[m, j-1]
                ) + p

    return finish[-1, -1]  # MAKESPAN = FITNESS

# ==================================================
# GENETIC ALGORITHM
# ==================================================
def run_ga(data, pop_size, mutation_rate, generations):
    n_jobs = data.shape[1]

    population = [
        random.sample(range(n_jobs), n_jobs)
        for _ in range(pop_size)
    ]

    best_fitness_history = []

    for _ in range(generations):
        population.sort(key=lambda s: fitness_function(s, data))

        best_fitness = fitness_function(population[0], data)
        best_fitness_history.append(best_fitness)

        new_population = population[:2]  # elitism

        while len(new_population) < pop_size:
            parent1, parent2 = random.sample(population[:10], 2)

            cut = random.randint(1, n_jobs - 1)
            child = parent1[:cut] + [
                j for j in parent2 if j not in parent1[:cut]
            ]

            if random.random() < mutation_rate:
                a, b = random.sample(range(n_jobs), 2)
                child[a], child[b] = child[b], child[a]

            new_population.append(child)

        population = new_population

    best_sequence = population[0]
    final_fitness = fitness_function(best_sequence, data)

    return best_fitness_history, best_sequence, final_fitness

# ==================================================
# STREAMLIT UI
# ==================================================
st.set_page_config(page_title="GA Scheduling Optimizer", layout="wide")
st.title("🧬 Genetic Algorithm for Job Scheduling")

uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type="csv")
pop_size = st.sidebar.slider("Population Size", 10, 100, 30)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1)
generations = st.sidebar.slider("Generations", 10, 500, 50)

if st.button("Start GA Optimization"):
    data = load_data(uploaded_file)

    history, best_seq, final_fitness = run_ga(
        data, pop_size, mutation_rate, generations
    )

    # ======================
    # FINAL RESULTS
    # ======================
    col1, col2, col3 = st.columns(3)
    col1.metric("Final Fitness Value", final_fitness)
    col2.metric("Optimized Makespan", final_fitness)
    col3.write(f"**Best Job Sequence:** {best_seq}")

    # ======================
    # CONVERGENCE PLOT
    # ======================
    st.subheader("Fitness Convergence")
    fig, ax = plt.subplots()
    ax.plot(history)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (Makespan)")
    ax.set_title("GA Fitness Convergence")
    st.pyplot(fig)
