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
        data = df.dropna().values

        # Ensure flow shop orientation (machines x jobs)
        if data.shape[0] < data.shape[1]:
            data = data.T

        return data

    # DEFAULT: 10 MACHINES x 5 JOBS (minutes)
    return np.array([
        [10, 20, 5, 15, 12],
        [8, 12, 20, 10, 18],
        [15, 5, 10, 25, 14],
        [12, 18, 8, 20, 10],
        [20, 10, 15, 5, 18],
        [14, 16, 12, 18, 9],
        [9, 14, 20, 10, 16],
        [18, 9, 14, 12, 20],
        [10, 15, 18, 14, 8],
        [16, 12, 10, 20, 14]
    ])

# ==================================================
# METRICS (MAKESPAN + IDLE TIME)
# ==================================================
def calculate_metrics(sequence, data):
    n_machines, n_jobs = data.shape
    finish = np.zeros((n_machines, n_jobs))

    for m in range(n_machines):
        for j in range(n_jobs):
            job = sequence[j]
            p = data[m, job]

            if m == 0 and j == 0:
                start = 0
            elif m == 0:
                start = finish[m, j-1]
            elif j == 0:
                start = finish[m-1, j]
            else:
                start = max(finish[m-1, j], finish[m, j-1])

            finish[m, j] = start + p

    makespan = finish[-1, -1]

    total_processing = np.sum(data)
    idle_time = (makespan * n_machines) - total_processing

    utilization = total_processing / (makespan * n_machines)

    return makespan, idle_time, utilization

# ==================================================
# FITNESS FUNCTION (GP-STYLE)
# ==================================================
def fitness_function(sequence, data, w_m, w_i):
    makespan, idle, _ = calculate_metrics(sequence, data)

    fitness = (w_m * makespan) + (w_i * idle)
    return fitness

# ==================================================
# GENETIC ALGORITHM
# ==================================================
def run_ga(data, pop_size, mutation_rate, generations, w_m, w_i):
    n_jobs = data.shape[1]

    population = [
        random.sample(range(n_jobs), n_jobs)
        for _ in range(pop_size)
    ]

    history = []

    for _ in range(generations):
        population.sort(
            key=lambda s: fitness_function(s, data, w_m, w_i)
        )

        best_fitness = fitness_function(
            population[0], data, w_m, w_i
        )
        history.append(best_fitness)

        new_population = population[:2]

        while len(new_population) < pop_size:
            p1, p2 = random.sample(population[:10], 2)

            cut = random.randint(1, n_jobs - 1)
            child = p1[:cut] + [j for j in p2 if j not in p1[:cut]]

            if random.random() < mutation_rate:
                a, b = random.sample(range(n_jobs), 2)
                child[a], child[b] = child[b], child[a]

            new_population.append(child)

        population = new_population

    best_seq = population[0]
    final_fitness = fitness_function(best_seq, data, w_m, w_i)

    return history, best_seq, final_fitness

# ==================================================
# STREAMLIT UI
# ==================================================
st.set_page_config(page_title="GA Scheduling (Idle Time)", layout="wide")

st.title("🧬 Genetic Algorithm for Job Scheduling (Idle Time Based)")

st.sidebar.header("GA Parameters")
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type="csv")
pop_size = st.sidebar.slider("Population Size", 10, 100, 20)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1)
generations = st.sidebar.slider("Generations", 10, 500, 100)

st.sidebar.header("Fitness Weights")
w_m = st.sidebar.slider("Weight – Makespan", 0.0, 1.0, 0.7)
w_i = st.sidebar.slider("Weight – Idle Time", 0.0, 1.0, 0.3)

if st.button("Start Optimization"):
    data = load_data(uploaded_file)

    history, best_seq, fitness = run_ga(
        data, pop_size, mutation_rate, generations, w_m, w_i
    )

    makespan, idle, util = calculate_metrics(best_seq, data)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Optimized Makespan", f"{makespan:.0f}")
    c2.metric("Total Idle Time", f"{idle:.0f}")
    c3.metric("Machine Utilization", f"{util*100:.2f}%")
    c4.metric("Total Fitness Value", f"{fitness:.2f}")

    st.write(f"**Best Job Sequence:** {best_seq}")

    st.subheader("📈 Fitness Convergence")
    fig, ax = plt.subplots()
    ax.plot(history)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness Value")
    st.pyplot(fig)
