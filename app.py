import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pso import optimize_student, DAYS

st.set_page_config(page_title="Student Study Load Optimization", layout="wide")

st.title(" Student Study Load Optimization using PSO")
st.markdown("""
Optimizes weekly study schedules for a **selected student** using PSO.  
Balances daily workload and minimizes overload. Classes may be **split** or moved to empty days to ensure no day exceeds the ideal load.
""")

# -------------------------
# Load dataset
# -------------------------
data = pd.read_csv("dataset.csv")
students = data["StudentName"].unique()

# Sidebar: PSO parameters
st.sidebar.header("⚙️ PSO Parameters")
iterations = st.sidebar.slider("Iterations", 10, 150, 80)
particles = st.sidebar.slider("Number of Particles", 5, 30, 15)

# -------------------------
# Select student
# -------------------------
selected_student = st.selectbox("Select Student", students)
student_data = data[data["StudentName"] == selected_student]

# -------------------------
# Run PSO for selected student
# -------------------------
optimized_data, fitness_curve, ideal_daily_hours, moves_summary = optimize_student(
    student_data, iterations=iterations, particles=particles
)

# -------------------------
# Before/After study load
# -------------------------
before_load = student_data.groupby("Day")["Duration"].sum().reindex(DAYS, fill_value=0)
after_load = optimized_data.groupby("OptimizedDay")["Duration"].sum().reindex(DAYS, fill_value=0)

# -------------------------
# Bar chart: Before vs After
# -------------------------
st.subheader(f"📊 Daily Study Load: {selected_student}")
fig, ax = plt.subplots(figsize=(8,4))
x = np.arange(len(DAYS))
width = 0.35

ax.bar(x - width/2, before_load, width, label="Before PSO", color="orange")
ax.bar(x + width/2, after_load, width, label="After PSO", color="green")
ax.axhline(y=ideal_daily_hours, color="blue", linestyle="--", label="Ideal Daily Hours")
ax.set_xticks(x)
ax.set_xticklabels(DAYS)
ax.set_ylabel("Study Hours")
ax.set_title("Daily Study Load: Before vs After PSO")
ax.legend()
ax.grid(axis='y')

for i in range(len(DAYS)):
    ax.text(i - width/2, before_load[i]+0.05, f"{before_load[i]:.1f}", ha='center', va='bottom')
    ax.text(i + width/2, after_load[i]+0.05, f"{after_load[i]:.1f}", ha='center', va='bottom')

st.pyplot(fig)

# -------------------------
# Fitness convergence
# -------------------------
st.subheader("📈 PSO Fitness Convergence")
fig2, ax2 = plt.subplots(figsize=(8,3))
ax2.plot(fitness_curve, marker='o', color="#00796b")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Best Fitness")
ax2.set_title(f"Fitness Convergence: {selected_student}")
ax2.grid(True)
st.pyplot(fig2)

# -------------------------
# Moves & splits summary
# -------------------------
st.subheader("📌 Moves & Splits Summary")
if moves_summary:
    moves_df = pd.DataFrame(moves_summary)
    moves_df = moves_df.sort_values(by=["FromDay", "Course"])
    st.dataframe(moves_df)
else:
    st.info("No moves or splits were needed. The schedule is already within ideal load.")

# -------------------------
# Optimization Explanation
# -------------------------
st.subheader("📝 Optimization Explanation")
for day in DAYS:
    b_load = before_load[day]
    a_load = after_load[day]
    if b_load == 0 and a_load == 0:
        st.markdown(f"- **{day}:** No classes before or after PSO.")
    elif b_load == 0 and a_load > 0:
        st.markdown(f"- **{day}:** Empty day before PSO, received {a_load}h from moved/split classes.")
    elif a_load < b_load:
        st.markdown(f"- **{day}:** Reduced load from {b_load}h to {a_load}h by moving/splitting classes to other days.")
    elif a_load == b_load:
        st.markdown(f"- **{day}:** Load remains {a_load}h, no changes needed.")
    else:
        st.markdown(f"- **{day}:** Optimized to {a_load}h (was {b_load}h).")

# -------------------------
# Optional: Download optimized schedule
# -------------------------
st.download_button(
    "💾 Download Optimized Schedule",
    optimized_data.to_csv(index=False),
    "optimized_schedule.csv",
    "text/csv"
)
