import streamlit as st
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

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
            p_time = data[m, job]

            if m == 0 and j == 0:
                finish[m, j] = p_time
            elif m == 0:
                finish[m, j] = finish[m, j-1] + p_time
            elif j == 0:
                finish[m, j] = finish[m-1, j] + p_time
            else:
                finish[m, j] = max(finish[m-1, j],
                                   finish[m, j-1]) + p_time

    return finish[-1, -1]

# ==================================================
# GENETIC ALGORITHM
# ==================================================
def run_ga(data, pop_size, generations, mutation_rate):
    n_jobs = data.shape[1]

    # Initial population (job permutations)
    population = [
        random.sample(range(n_jobs), n_jobs)
        for _ in range(pop_size)
    ]

    history = []

    for _ in range(generations):
        population.sort(key=lambda s: calculate_makespan(s, data))
        best_fitness = calculate_makespan(population[0], data)
        history.append(best_fitness)

        # Elitism
        new_population = population[:2]

        # Generate offspring
        while len(new_population) < pop_size:
            p1, p2 = random.sample(population[:10], 2)
            cut = random.randint(1, n_jobs - 1)

            child = p1[:cut] + [j for j in p2 if j not in p1[:cut]]

            # Mutation (swap)
            if random.random() < mutation_rate:
                a, b = random.sample(range(n_jobs), 2)
                child[a], child[b] = child[b], child[a]

            new_population.append(child)

        population = new_population

    best_solution = population[0]
    best_makespan = calculate_makespan(best_solution, data)

    return history, best_solution, best_makespan, data

# ==================================================
# STREAMLIT UI
# ==================================================
st.set_page_config(page_title="GA Scheduling Optimizer", layout="wide")

st.title("🧬 Genetic Algorithm (GA) for Job Scheduling")
st.write(
    "This system applies **Genetic Algorithm** with selection, crossover, "
    "and mutation to minimise **Makespan** in Flow Shop Scheduling."
)

# Sidebar Parameters
st.sidebar.header("Algorithmic Parameters")
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type="csv")
pop_size = st.sidebar.slider("Population Size", 10, 100, 30)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1)
generations = st.sidebar.slider("Generations", 10, 500, 100)

# ==================================================
# RUN BUTTON
# ==================================================
if st.button("Start GA Optimization"):
    data = load_data(uploaded_file)

    history, best_seq, best_makespan, raw_data = run_ga(
        data, pop_size, generations, mutation_rate
    )

    # ======================
    # METRICS
    # ======================
    col1, col2 = st.columns(2)
    col1.metric("Optimized Makespan", f"{best_makespan} mins")
    col2.write(f"**Best Job Sequence:** {best_seq}")

    # ======================
    # CONVERGENCE PLOT
    # ======================
    st.subheader("📈 Convergence Analysis")
    fig, ax = plt.subplots()
    ax.plot(history)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Makespan")
    ax.set_title("GA Fitness Convergence")
    st.pyplot(fig)

    # ======================
    # GANTT CHART (PLOTLY)
    # ======================
    st.subheader("📅 Optimized Gantt Chart")

    gantt_data = []
    finish_times = np.zeros((raw_data.shape[0], raw_data.shape[1]))

    for m in range(raw_data.shape[0]):
        for j in range(raw_data.shape[1]):
            job = best_seq[j]
            p_time = raw_data[m, job]

            if m == 0 and j == 0:
                start = 0
            elif m == 0:
                start = finish_times[m, j-1]
            elif j == 0:
                start = finish_times[m-1, j]
            else:
                start = max(finish_times[m-1, j],
                            finish_times[m, j-1])

            end = start + p_time
            finish_times[m, j] = end

            gantt_data.append(dict(
                Task=f"Machine {m+1}",
                Start=start,
                Finish=end,
                Resource=f"Job {job+1}"
            ))

    df_plot = pd.DataFrame(gantt_data)
    df_plot['Start'] = pd.to_datetime(df_plot['Start'], unit='m', origin='2026-01-01')
    df_plot['Finish'] = pd.to_datetime(df_plot['Finish'], unit='m', origin='2026-01-01')

    fig_gantt = ff.create_gantt(
        df_plot,
        index_col='Resource',
        show_colorbar=True,
        group_tasks=True,
        title="GA Optimized Schedule"
    )

    st.plotly_chart(fig_gantt, use_container_width=True)
