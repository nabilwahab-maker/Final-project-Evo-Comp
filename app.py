import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# ==================================================
# LOAD DATA (AUTO-FIX ORIENTATION)
# ==================================================
def load_data(file):
    if file is not None:
        df = pd.read_csv(file)

        if df.iloc[:, 0].dtype == object:
            df = df.iloc[:, 1:]

        df = df.apply(pd.to_numeric, errors='coerce')
        data = df.dropna().values

        if data.shape[0] < data.shape[1]:
            data = data.T

        return data

    # DEFAULT: 10 MACHINES x 5 JOBS
    return np.array([
        [10, 8, 15, 12, 14],
        [20, 12, 5, 18, 10],
        [5, 20, 10, 7, 9],
        [15, 10, 25, 14, 11],
        [12, 18, 14, 9, 13],
        [9, 11, 8, 6, 7],
        [14, 16, 12, 10, 15],
        [11, 9, 13, 8, 10],
        [10, 14, 9, 11, 12],
        [13, 10, 15, 14, 9]
    ])

# ==================================================
# METRICS
# ==================================================
def calculate_metrics(sequence, data):
    n_machines, n_jobs = data.shape
    start = np.zeros((n_machines, n_jobs))
    finish = np.zeros((n_machines, n_jobs))

    for m in range(n_machines):
        for j in range(n_jobs):
            job = sequence[j]
            p = data[m, job]

            if m == 0 and j == 0:
                start[m, j] = 0
            elif m == 0:
                start[m, j] = finish[m, j-1]
            elif j == 0:
                start[m, j] = finish[m-1, j]
            else:
                start[m, j] = max(finish[m-1, j], finish[m, j-1])

            finish[m, j] = start[m, j] + p

    makespan = finish[-1, -1]

    total_processing = np.sum(data)
    total_available = makespan * n_machines
    idle_time = total_available - total_processing

    utilization = total_processing / total_available

    return makespan, idle_time, utilization

# ==================================================
# NORMALIZED FITNESS FUNCTION
# ==================================================
def fitness_function(sequence, data, w_m, w_i, w_u):
    makespan, idle, util = calculate_metrics(sequence, data)

    # 🔧 NORMALIZATION
    norm_makespan = makespan / np.max(data)
    norm_idle = idle / (makespan * data.shape[0])

    fitness = (
        w_m * norm_makespan +
        w_i * norm_idle -
        w_u * util
    )

    return fitness

# ==================================================
# GENETIC ALGORITHM
# ==================================================
def run_ga(data, pop_size, mutation_rate, generations,
           w_m, w_i, w_u):

    n_jobs = data.shape[1]

    population = [
        random.sample(range(n_jobs), n_jobs)
        for _ in range(pop_size)
    ]

    history = []

    for _ in range(generations):
        population.sort(
            key=lambda s: fitness_function(s, data, w_m, w_i, w_u)
        )

        best_fit = fitness_function(population[0], data, w_m, w_i, w_u)
        history.append(best_fit)

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
    best_fit = fitness_function(best_seq, data, w_m, w_i, w_u)

    return history, best_seq, best_fit

# ==================================================
# STREAMLIT UI
# ==================================================
st.set_page_config(page_title="Multi-Objective GA Scheduling", layout="wide")
st.title("🧬 Multi-Objective Genetic Algorithm for Job Scheduling")

uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type="csv")
pop_size = st.sidebar.slider("Population Size", 10, 100, 20)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1)
generations = st.sidebar.slider("Generations", 10, 500, 100)

st.sidebar.header("Objective Weights (Σ ≤ 1)")
w_m = st.sidebar.slider("Makespan", 0.0, 1.0, 0.4)
w_i = st.sidebar.slider("Idle Time", 0.0, 1.0, 0.4)
w_u = st.sidebar.slider("Utilization", 0.0, 1.0, 0.2)

if w_m + w_i + w_u > 1:
    st.error("Sum of weights must be ≤ 1")
    st.stop()

if st.button("Start Optimization"):
    data = load_data(uploaded_file)

    history, best_seq, best_fit = run_ga(
        data, pop_size, mutation_rate, generations,
        w_m, w_i, w_u
    )

    makespan, idle, util = calculate_metrics(best_seq, data)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Makespan (min)", f"{makespan:.2f}")
    c2.metric("Total Idle Time (min)", f"{idle:.2f}")
    c3.metric("Machine Utilization", f"{util*100:.2f}%")
    c4.metric("Normalized Fitness", f"{best_fit:.4f}")

    st.write("**Best Job Sequence:**", best_seq)

    # Convergence
    st.subheader("📈 Fitness Convergence")
    fig, ax = plt.subplots()
    ax.plot(history)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Normalized Fitness")
    st.pyplot(fig)

    # Gantt Chart (KEKAL)
    st.subheader("📅 Optimized Gantt Chart")
    n_machines, n_jobs = data.shape
    finish = np.zeros((n_machines, n_jobs))
    fig, ax = plt.subplots()

    for m in range(n_machines):
        for j in range(n_jobs):
            job = best_seq[j]
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
            ax.barh(m, p, left=start)
            ax.text(start + p/2, m, f"J{job+1}",
                    ha='center', va='center', color='white')

    ax.set_yticks(range(n_machines))
    ax.set_yticklabels([f"Machine {i+1}" for i in range(n_machines)])
    ax.set_xlabel("Time (minutes)")
    st.pyplot(fig)
