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

        # Buang kolum pertama jika bukan numeric (contoh Job ID)
        if df.iloc[:, 0].dtype == object:
            df = df.iloc[:, 1:]

        df = df.apply(pd.to_numeric, errors='coerce')
        return df.dropna().values

    # Default dataset
    return np.array([
        [10, 20, 5, 15],
        [8, 12, 20, 10],
        [15, 5, 10, 25]
    ])

# ==================================================
# FITNESS FUNCTION (FLOW SHOP MAKESPAN)
# ==================================================
def calculate_makespan(sequence, data):
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

    return finish[-1, -1]

# ==================================================
# GENETIC ALGORITHM (GA)
# ==================================================
def run_ga(data, pop_size, mutation_rate, generations):
    n_jobs = data.shape[1]

    # Initial population (permutation of jobs)
    population = [
        random.sample(range(n_jobs), n_jobs)
        for _ in range(pop_size)
    ]

    history = []

    for _ in range(generations):
        population.sort(key=lambda s: calculate_makespan(s, data))
        history.append(calculate_makespan(population[0], data))

        # Elitism (ambil 2 terbaik)
        new_population = population[:2]

        while len(new_population) < pop_size:
            parent1, parent2 = random.sample(population[:10], 2)

            cut = random.randint(1, n_jobs - 1)
            child = parent1[:cut] + [
                j for j in parent2 if j not in parent1[:cut]
            ]

            # Mutation (swap)
            if random.random() < mutation_rate:
                a, b = random.sample(range(n_jobs), 2)
                child[a], child[b] = child[b], child[a]

            new_population.append(child)

        population = new_population

    best_sequence = population[0]
    best_makespan = calculate_makespan(best_sequence, data)

    return history, best_sequence, best_makespan, data

# ==================================================
# STREAMLIT UI (SAMA MACAM ES)
# ==================================================
st.set_page_config(page_title="GA Scheduling Optimizer", layout="wide")

st.title("🧬 Genetic Algorithm (GA) for Job Scheduling")
st.write(
    "Bahagian ini menggunakan **Genetic Algorithm** "
    "untuk mengoptimumkan **Makespan** dalam Flow Shop Scheduling."
)

# Sidebar Parameters
st.sidebar.header("Algorithmic Parameters")
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type="csv")
pop_size = st.sidebar.slider("Population Size", 10, 100, 20)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1)
gen_val = st.sidebar.slider("Generations", 10, 500, 100)

# ==================================================
# RUN BUTTON
# ==================================================
if st.button("Start GA Optimization"):
    data = load_data(uploaded_file)

    hist, best_seq, best_m, raw_data = run_ga(
        data, pop_size, mutation_rate, gen_val
    )

    # ======================
    # METRICS
    # ======================
    col1, col2 = st.columns(2)
    col1.metric("Optimized Makespan", f"{best_m} mins")
    col2.write(f"**Best Sequence:** {best_seq}")

    # ======================
    # CONVERGENCE PLOT
    # ======================
    st.subheader("📈 Convergence Analysis")
    fig, ax = plt.subplots()
    ax.plot(hist)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Makespan")
    ax.set_title("GA Fitness Convergence")
    st.pyplot(fig)

    # ======================
    # GANTT CHART (MATPLOTLIB)
    # ======================
    st.subheader("📅 Optimized Gantt Chart")

    finish = np.zeros((raw_data.shape[0], raw_data.shape[1]))
    fig, ax = plt.subplots()

    for m in range(raw_data.shape[0]):
        for j in range(raw_data.shape[1]):
            job = best_seq[j]
            p = raw_data[m, job]

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
            ax.text(start + p / 2, m, f"J{job+1}",
                    ha='center', va='center', color='white')

    ax.set_yticks(range(raw_data.shape[0]))
    ax.set_yticklabels([f"Machine {i+1}" for i in range(raw_data.shape[0])])
    ax.set_xlabel("Time")
    ax.set_title("GA Optimized Schedule")
    st.pyplot(fig)

