# Student Study Load Optimization using Particle Swarm Optimization and Streamlit

## Overview

This project presents a study schedule optimization system for university students using Particle Swarm Optimization (PSO). The system aims to create balanced study schedules by distributing study hours efficiently throughout the week while minimizing daily overload. An interactive Streamlit dashboard is used to visualize schedules, optimization performance, and study load distribution.

## Objectives

* Optimize weekly study schedules for students.
* Balance study hours across different days.
* Minimize excessive study workload on any single day.
* Improve time management and study efficiency.
* Provide an interactive interface for schedule analysis.

## Methodology

### Particle Swarm Optimization (PSO)

Particle Swarm Optimization is a population-based optimization algorithm inspired by the social behavior of birds and fish schools. In this project, each particle represents a potential study schedule.

The PSO algorithm iteratively updates particles based on:

* Personal Best (pBest)
* Global Best (gBest)
* Velocity and Position Updates

### Fitness Function

The optimization considers two objectives:

1. **Balance Daily Study Hours**

   * Minimize variance in study hours across days.

2. **Minimize Daily Overload**

   * Reduce study hours exceeding the ideal daily workload.

Weighted Fitness Function:

Fitness = (0.7 × Balance Score) + (0.3 × Overload Score)

## Dataset

The dataset contains student study information with the following attributes:

| Column      | Description            |
| ----------- | ---------------------- |
| StudentName | Name of student        |
| Course      | Subject/Course         |
| Day         | Study day              |
| TimeSlot    | Scheduled time         |
| Duration    | Study duration (hours) |

## Features

* Particle Swarm Optimization implementation
* Multi-objective study schedule optimization
* Interactive Streamlit dashboard
* Timetable visualization
* Daily study load analysis
* Fitness convergence graph
* Performance metrics and statistics

## Technologies Used

* Python
* Streamlit
* NumPy
* Pandas
* Matplotlib

## Project Structure

```text
├── app.py
├── pso.py
├── data.csv
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
```

2. Navigate to the project folder:

```bash
cd student-study-load-optimization
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit application:

```bash
streamlit run app.py
```

## Results

The PSO algorithm successfully generates optimized study schedules by balancing daily study hours and reducing overload. The Streamlit dashboard provides visual insights through:

* Optimized study timetable
* Daily study load comparison
* Fitness convergence analysis
* Optimization performance metrics

## Future Improvements

* Personalized study preferences
* Multi-student optimization
* Adaptive PSO parameter tuning
* Integration with academic calendars
* Mobile-friendly interface


