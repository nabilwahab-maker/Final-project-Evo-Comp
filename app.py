import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# ==================================================
# LOAD DATA (Processing Time in MINUTES)
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
# MULTI-OBJECTIVE METRICS (TIME IN MINUTES)
# ==================================================
def calculate_metrics(sequence, data):
    n_machines, n_jobs = data.shape
    start = np.zeros((n_machines, n_jobs))
    finish = np.zeros((n_machines, n_jobs))

    for m in range(n_machines):
        for j in range(n_jobs):
            job = sequence[j]
            p = data[m, job]  # minutes

            if m == 0 and j == 0:
                start[m, j] = 0
            elif m == 0:
                start[m, j] = finish[m, j-1]
            elif j == 0:
                start[m, j] = finish[m-1, j]
            else:
                start[m, j] = max(
                    finish[m-1, j],
                    finish[m, j-1]
                )

            finish[m, j] = start[m, j] + p

    makespan = finish[-1, -1]                 # minutes
    waiting_time = np.sum(start)               # minutes

    total_processing = np.sum(data)             # minutes
    utilization = total_processing / (makespan * n_machines)

    return makespan, waiting_time, utilization

# ==================================================
# FITNESS FUNCTION (UNITLESS SCORE)
# ==================================================
def fitness_function(sequence, data, w_m, w_w, w_u):
    makespan, waiting, util = calculate_metrics(sequence, data)

    fitness = (
        w_m * makespan +
        w_w * waiting -
        w_u * util * 100
    )
    return fitness  # unitless

# ==================================================
# GENETIC ALGORITHM
# ==================================================
def run_ga(data, pop_size, mutation_rate, generations,
           w_m, w_w, w_u):

    n_jobs = data.shape[1]

    population = [
        random.sample(range(n_jobs), n_jobs)
        for _ in range(pop_size)
    ]

    history = []

    for _ in range(generations):
        population.sort(
            key=lambda s: fitness_function(s, data, w_m, w_w, w_u)
        )

        history.append(
            fitness_function(population[0], data, w_m, w_w, w_u)
        )

        new_population = population[:2]

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

    best_seq = population[0]
    final_fitness = fitness_function(best_seq, data, w_m, w_w, w_u)

    return history, best_seq, final_fitness

# ==================================================
# STREAMLIT UI
# ==================================================
st.set_page_config(page_title="Multi-Objective GA Scheduling", layout="wide")

st.title("🧬 Multi-Objective GA for Job Scheduling")
st.write(
    "Processing times, makespan, and waiting time are measured in **minutes**. "
    "The final fitness value is an **aggregated unitless score** used for optimization."
)

# Sidebar
st.sidebar.header("Algorithm Parameters")
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type="csv")
pop_size = st.sidebar.slider("Population Size", 10, 100, 20)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1)
generations = st.sidebar.slider("Generations", 10, 500, 100)

st.sidebar.header("Multi-Objective Weights (Σ ≤ 1)")
w_m = st.sidebar.slider("Weight – Makespan", 0.0, 1.0, 0.5)
w_w = st.sidebar.slider("Weight – Waiting Time", 0.0, 1.0, 0.3)
w_u = st.sidebar.slider("Weight – Machine Utilization", 0.0, 1.0, 0.2)

if w_m + w_w + w_u > 1.0:
    st.sidebar.error("Sum of weights must be ≤ 1")
    st.stop()

# ==================================================
# RUN
# ==================================================
if st.button("Start GA Optimization"):
    data = load_data(uploaded_file)

    history, best_seq, final_fitness = run_ga(
        data, pop_size, mutation_rate, generations,
        w_m, w_w, w_u
    )

    makespan, waiting, util = calculate_metrics(best_seq, data)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Makespan (minutes)", f"{makespan:.2f}")
    c2.metric("Total Waiting Time (minutes)", f"{waiting:.2f}")
    c3.metric("Machine Utilization (%)", f"{util*100:.2f}")
    c4.metric("Final Fitness Value (unitless)", f"{final_fitness:.2f}")

    st.write(f"**Best Job Sequence:** {best_seq}")

    st.subheader("📈 Aggregated Fitness Convergence")
    fig, ax = plt.subplots()
    ax.plot(history)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Aggregated Fitness (unitless)")
    ax.set_title("Multi-Objective GA Convergence")
    st.pyplot(fig)
