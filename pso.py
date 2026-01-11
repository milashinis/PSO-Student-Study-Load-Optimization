import numpy as np
import pandas as pd

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
NUM_DAYS = len(DAYS)

# -------------------------
# Fitness Function (per student)
# -------------------------
def fitness(position, durations, ideal_daily_hours, w1=0.6, w2=0.4):
    daily_load = np.zeros(NUM_DAYS)
    for i, d in enumerate(position):
        daily_load[int(d)] += durations[i]

    # Balance = variance of daily load
    balance = np.var(daily_load) / (ideal_daily_hours + 1e-6)
    # Overload = sum of hours exceeding ideal
    overload = np.sum(np.maximum(daily_load - ideal_daily_hours, 0))

    return -(w1 * balance + w2 * overload)


# -------------------------
# PSO + Post-Optimization Redistribution
# -------------------------
def optimize_student(student_df, iterations=80, particles=20):
    durations = student_df["Duration"].values
    num_records = len(durations)
    ideal_daily_hours = durations.sum() / NUM_DAYS

    # PSO parameters
    w, c1, c2 = 0.7, 1.8, 1.8

    # Initialize particles and velocities
    particles_pos = np.random.randint(0, NUM_DAYS, (particles, num_records))
    velocities = np.random.uniform(-1, 1, (particles, num_records))

    pbest = particles_pos.copy()
    pbest_scores = np.array([fitness(p, durations, ideal_daily_hours) for p in particles_pos])
    gbest = pbest[np.argmax(pbest_scores)]
    fitness_curve = []

    for _ in range(iterations):
        for i in range(particles):
            r1, r2 = np.random.rand(), np.random.rand()
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (pbest[i] - particles_pos[i])
                + c2 * r2 * (gbest - particles_pos[i])
            )
            particles_pos[i] = np.clip(np.round(particles_pos[i] + velocities[i]), 0, NUM_DAYS - 1)
            score = fitness(particles_pos[i], durations, ideal_daily_hours)
            if score > pbest_scores[i]:
                pbest[i] = particles_pos[i].copy()
                pbest_scores[i] = score
        gbest = pbest[np.argmax(pbest_scores)]
        fitness_curve.append(np.max(pbest_scores))

    # -------------------------
    # Apply PSO result
    # -------------------------
    optimized = student_df.copy().reset_index(drop=True)
    optimized["OptimizedDay"] = [DAYS[int(i)] for i in gbest]

    moves_summary = []

    # -------------------------
    # Post-optimization balancing (split/move to respect ideal load)
    # -------------------------
    def update_day_load(df):
        return df.groupby("OptimizedDay")["Duration"].sum().reindex(DAYS, fill_value=0).to_dict()

    day_load = update_day_load(optimized)

    for day in DAYS:
        while day_load[day] > ideal_daily_hours:
            courses_in_day = optimized[optimized["OptimizedDay"] == day]
            if courses_in_day.empty:
                break

            # Pick the longest course
            course_idx = courses_in_day["Duration"].idxmax()
            course = optimized.loc[course_idx]

            excess = day_load[day] - ideal_daily_hours
            move_duration = min(course["Duration"], excess)

            # Find day with minimum load
            min_day = min(day_load, key=day_load.get)
            if min_day == day:
                break  # No other day available

            # Split or move
            if move_duration < course["Duration"]:
                # Reduce original course
                optimized.at[course_idx, "Duration"] -= move_duration
                # Create new split course
                new_row = course.copy()
                new_row["Duration"] = move_duration
                new_row["OptimizedDay"] = min_day
                optimized = pd.concat([optimized, pd.DataFrame([new_row])], ignore_index=True)

                moves_summary.append({
                    "Course": course["Course"],
                    "FromDay": day,
                    "ToDay": min_day,
                    "OriginalDuration": course["Duration"],
                    "MovedDuration": move_duration
                })
            else:
                # Move entire course
                optimized.at[course_idx, "OptimizedDay"] = min_day
                moves_summary.append({
                    "Course": course["Course"],
                    "FromDay": day,
                    "ToDay": min_day,
                    "OriginalDuration": course["Duration"],
                    "MovedDuration": course["Duration"]
                })

            # Update day_load after each move/split
            day_load = update_day_load(optimized)

    return optimized, fitness_curve, ideal_daily_hours, moves_summary
