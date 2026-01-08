import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Job Scheduling using Genetic Algorithm",
    layout="wide"
)

# ==========================
# TITLE
# ==========================
st.title("🧬 Job Scheduling Optimization Using Genetic Algorithm")
st.caption(
    "A multi-objective optimisation approach focusing on makespan, waiting time, "
    "and machine idle time using evolutionary computation."
)

st.divider()

# ==========================
# DATASET (DUMMY – DECLARED)
# ==========================
def generate_jobs(n):
    """
    Dummy dataset generator.
    Processing time is randomly generated.
    This dataset is synthetically created and declared as dummy data.
    """
    return [random.randint(1, 20) for _ in range(n)]

# ==========================
# FITNESS EVALUATION
# ==========================
def evaluate(schedule, jobs, num_machines):
    machine_time = [0] * num_machines
    waiting_time = 0

    for job in schedule:
        idx = job
        machine = machine_time.index(min(machine_time))
        waiting_time += machine_time[machine]
        machine_time[machine] += jobs[idx]

    makespan = max(machine_time)
    idle_time = sum(makespan - t for t in machine_time)

    return makespan, waiting_time, idle_time

# ==========================
# GENETIC OPERATORS
# ==========================
def crossover(p1, p2):
    cut = random.randint(1, len(p1) - 2)
    child = p1[:cut] + [j for j in p2 if j not in p1[:cut]]
    return child

def mutate(individual, rate):
    if random.random() < rate:
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]
    return individual

# ==========================
# GENETIC ALGORITHM
# ==========================
def genetic_algorithm(jobs, num_machines, pop_size, generations, mutation_rate):
    population = [random.sample(range(len(jobs)), len(jobs)) for _ in range(pop_size)]
    history = []

    for _ in range(generations):
        scored = []
        for ind in population:
            m, w, i = evaluate(ind, jobs, num_machines)
            fitness = m + 0.5 * w + 0.2 * i  # multi-objective (weighted)
            scored.append((fitness, ind))

        scored.sort(key=lambda x: x[0])
        population = [ind for _, ind in scored[:pop_size // 2]]
        history.append(scored[0][0])

        new_pop = population.copy()
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(population, 2)
            child = crossover(p1, p2)
            child = mutate(child, mutation_rate)
            new_pop.append(child)

        population = new_pop

    return scored[0][1], history

# ==========================
# GANTT CHART
# ==========================
def plot_gantt(schedule, jobs, num_machines):
    machine_time = [0] * num_machines
    fig, ax = plt.subplots()

    for job in schedule:
        machine = machine_time.index(min(machine_time))
        start = machine_time[machine]
        duration = jobs[job]
        ax.barh(f"Machine {machine+1}", duration, left=start)
        machine_time[machine] += duration

    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")
    ax.set_title("Gantt Chart of Job Scheduling")
    return fig

# ==========================
# SIDEBAR
# ==========================
st.sidebar.header("⚙️ Algorithm Configuration")

num_jobs = st.sidebar.slider("Number of Jobs", 5, 15, 8)
num_machines = st.sidebar.slider("Number of Machines", 1, 5, 3)
pop_size = st.sidebar.slider("Population Size", 10, 100, 30)
generations = st.sidebar.slider("Generations", 20, 200, 60)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.0, 0.5, 0.1)

jobs = generate_jobs(num_jobs)

# ==========================
# MAIN CONTENT
# ==========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Job Dataset (Dummy Data)")
    st.write(
        "The processing time for each job is generated using dummy data. "
        "This is done to simulate a real scheduling environment."
    )
    st.table({
        "Job ID": [f"J{i+1}" for i in range(num_jobs)],
        "Processing Time": jobs
    })

with col2:
    st.subheader("🎯 Optimisation Objectives")
    st.markdown("""
    - **Minimise Makespan** – finish all jobs as early as possible  
    - **Minimise Waiting Time** – reduce job waiting duration  
    - **Minimise Idle Time** – avoid machine underutilisation  

    These objectives often conflict, making the problem multi-objective.
    """)

st.divider()

# ==========================
# RUN BUTTON
# ==========================
if st.button("🚀 Run Genetic Algorithm", use_container_width=True):
    best, history = genetic_algorithm(
        jobs, num_machines, pop_size, generations, mutation_rate
    )

    makespan, waiting, idle = evaluate(best, jobs, num_machines)

    st.subheader("✅ Optimisation Results")

    c1, c2, c3 = st.columns(3)
    c1.metric("Makespan", makespan)
    c2.metric("Waiting Time", waiting)
    c3.metric("Idle Time", idle)

    st.write(
        "The values above represent the best solution found after the evolutionary process. "
        "Different trade-offs can be observed depending on parameter settings."
    )

    st.divider()

    st.subheader("📈 Fitness Convergence")
    st.write(
        "The convergence curve shows how the fitness value improves over generations."
    )
    fig, ax = plt.subplots()
    ax.plot(history)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness Value")
    st.pyplot(fig)

    st.divider()

    st.subheader("🗓️ Final Schedule (Gantt Chart)")
    st.pyplot(plot_gantt(best, jobs, num_machines))

