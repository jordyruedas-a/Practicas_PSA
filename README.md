# PRACTICAS_PSA-MAIN: Multi-Module Adaptive Systems & AI Engineering Portfolio

**Author:** Jordan Emanuel Ruedas Vázquez  
**Last Updated:** June 2026  
**Language:** Python 3.9+  
**Core Technologies:** NumPy, Matplotlib, scikit-fuzzy, Tkinter, Browser Extensions (Manifest V2), Google Gemini API.


## 📋 Executive Summary

This repository represents a consolidated, enterprise-grade portfolio of five distinct software engineering projects. It demonstrates a progression from foundational algorithm implementation to complex adaptive systems and AI-integrated automation.

The projects are structured as independent modules (`Practica1` through `Practica4` and `PIA`). The underlying theme is **Data Science, Adaptive Control, and User Interface Automation**.


## 🧩 Project Architecture

The repository is organized into the following structure:

```
PRACTICAS_PSA-MAIN/
├── PIA/                         # Final Capstone: Adaptive PSO System
│   ├── controlador_difuso.py    # Fuzzy logic controller implementation
│   ├── funciones_prueba.py      # Benchmark functions for optimization
│   ├── pso_adaptativo.py        # Core PSO algorithm with adaptive logic
│   ├── main.py                  # Orchestrator for experiments and demos
│   ├── requirements.txt         # Python dependencies
│   └── *.png                    # Visualization results
├── Practica1/                   # Multi-Agent System Simulation
│   └── semaforo.py              # Traffic simulation with adaptive control
├── Practica2/                   # Graph Theory & Data Structures
│   ├── ejemplo_practica2.py     # Mandelbrot fractal generator
│   ├── practica2.py             # Social network analysis from adjacency matrix
│   ├── matriz_amigos.txt        # Input adjacency matrix data
│   └── resultados_red.txt       # Output analysis results
├── Practica3/                   # Unsupervised Machine Learning
│   └── practica3.py             # K-Means clustering from scratch
└── Practica4/                   # Multi-Agent Auction System
    └── practica4.py             # Agent-based economic simulation
```


## 🚀 Module Breakdown & Technical Deep Dive

### 1. Practica1: Adaptive Traffic Control (Multi-Agent System)
- **Objective:** Simulate an intelligent traffic intersection using an adaptive controller.
- **Core Logic:** The system models an intersection with 4 movements (NS, SN, EW, WE). It uses a `Controlador` class that dynamically adjusts green light times based on real-time queue lengths (demand). It implements a conflict matrix to prevent unsafe overlapping green phases.
- **Architecture:** 
  - `Auto`, `Carril` (Queue), `Semaforo` (State machine), `Interseccion` (Matrix of conflicts), `Controlador` (Proportional allocation logic), `Simulacion` (Tick-based loop).
- **UI:** Built with `tkinter` for a visual representation of the intersection and queues.
- **📺 Video:** [Practica 1 - Semaforo](https://www.youtube.com/watch?v=z3s-yv4QfFU)

### 2. Practica2: Graph Theory & Computational Geometry
- **Objective:** Perform graph analysis on a social network and generate a Mandelbrot fractal.
- **Sub-module A (Social Network):** 
  - **Input:** Reads a symmetric adjacency matrix from `matriz_amigos.txt` (representing a 10-vertex graph).
  - **Analysis:** Calculates graph metrics: Number of vertices (`n`), number of edges (`m`), and degree centrality for each vertex using list comprehensions.
  - **Output:** Writes the structured results to `resultados_red.txt` for auditing.
- **Sub-module B (Mandelbrot Set):** 
  - **Core Logic:** Implements the complex plane iteration `z = z² + c` using NumPy vectorization for performance. It maps pixel coordinates to a defined complex region and applies a color palette (`matplotlib.cm.hot`) to represent the number of iterations before divergence.
- **📺 Video:** [Practica 2 - Graph & Fractal](https://www.youtube.com/watch?v=vJDP41fNUNs)

### 3. Practica3: Unsupervised Machine Learning (K-Means Clustering)
- **Objective:** Segment synthetic customer data (Age vs. Income) using a custom implementation of the K-Means algorithm.
- **Technical Implementation:**
  - **Data Generation:** Uses `sklearn.datasets.make_blobs` to create 4 distinct clusters, adding synthetic noise to Age and Income to simulate real-world variance.
  - **Algorithm:** Implements K-Means++ for deterministic centroid initialization, Euclidean distance for cluster assignment, and a convergence check based on centroid movement tolerance.
  - **Optimization:** Includes a **StandardScaler** preprocessing step to normalize Age and Income data, ensuring the Euclidean distance is not biased by the large scale of Income.
  - **Visualization:** Provides the "Elbow Method" (`metodo_codo`) to determine the optimal number of clusters (`k`) and visualizes the evolution of centroids across iterations.
- **📺 Video:** [Practica 3 - K-Means](https://www.youtube.com/watch?v=DdLJGgUARwk)

### 4. Practica4: Multi-Agent Auction System
- **Objective:** Simulate a decentralized auction where autonomous agents bid on items.
- **Core Entities:**
  - `AgenteComprador`: Has a budget, strategy (conservative, aggressive, random), and an internal state to decide bid amounts.
  - `AgenteSubastador`: Coordinates the auction flow, announces items, validates bids, and handles the conflict-free adjudication.
  - `Mensaje`: A standardized protocol class (`remitente`, `destinatario`, `tipo`, `contenido`) for inter-agent communication via queues.
  - `SistemaSubastas`: The high-level orchestrator that spawns agent threads and manages the simulation lifecycle.
- **Concurrency:** Uses Python's `threading` module with `queue.Queue` to handle asynchronous message passing between the auctioneer and bidders.
- **📺 Video:** [Practica 4 - Auction](https://www.youtube.com/watch?v=idKU-oxFuA8)

### 5. PIA (Proyecto Integrador): Adaptive Swarm Intelligence
- **Objective:** Implement a Particle Swarm Optimization (PSO) algorithm enhanced with a Fuzzy Logic controller to dynamically tune its hyperparameters (`w`, `c1`, `c2`).
- **System Architecture:**
  - **Core Algorithm (`pso_adaptativo.py`):**
    - `Particula` class storing position, velocity, and personal best.
    - `PSOAdaptativo` class handling the standard PSO loop.
    - **Adaptive Feature:** Calculates two metrics at each iteration: **Diversity** (how spread out the particles are) and **Improvement** (rate of change of the global best fitness). These metrics are fed into the fuzzy controller.
  - **Fuzzy Controller (`controlador_difuso.py`):**
    - Uses `scikit-fuzzy` to define membership functions (e.g., `baja`, `media`, `alta` for Diversity).
    - **Rule Engine:** Contains 9 expert rules (e.g., *IF Diversity is low AND Improvement is stagnant THEN w is high and c2 is high*). This allows the algorithm to automatically switch between **Exploration** (finding new regions) and **Exploitation** (fine-tuning the best solution).
  - **Benchmarking (`main.py` & `funciones_prueba.py`):**
    - Evaluates the system against 4 benchmark functions: **Esfera** (Sphere), **Rosenbrock**, **Rastrigin**, and **Ackley**.
    - Compares the adaptive PSO against a traditional static PSO and generates statistical comparisons (Boxplots) and convergence graphs.
- **📺 Video:** [PIA - PSO Final Project](https://www.youtube.com/watch?v=TvV5Wk1bsWg)



## ⚙️ Environment Setup & Dependencies

To run this project locally, set up a Python virtual environment and install the dependencies:

# 1. Create a virtual environment
python -m venv venv

# 2. Activate the environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r PIA/requirements.txt


**`requirements.txt` includes:**
- `numpy>=1.21.0` (Numerical operations)
- `matplotlib>=3.4.0` (Data visualization)
- `scikit-fuzzy>=0.4.2` (Fuzzy logic engine for PIA)



## 📊 Key Technical Variables Explained

| Variable (Project) | Description | Data Type |
| :--- | :--- | :--- |
| `matriz_conflictos` (P1) | Adjacency matrix defining which traffic movements cannot be green simultaneously. | `List[List[int]]` |
| `matriz_amigos` (P2) | Adjacency matrix of a 10-node social network graph. | `List[List[int]]` |
| `X_norm` (P3) | The feature matrix (Age, Income) after `StandardScaler` normalization. | `numpy.ndarray` |
| `k` (P3) | The number of clusters selected for the K-Means algorithm. | `int` |
| `mensaje.tipo` (P4) | Message type for agent communication (e.g., `PUJA`, `ACTUALIZACION_PUJA`). | `str` |
| `diversidad_val` (PIA) | Normalized metric (0-1) representing particle dispersion in the swarm. | `float` |
| `mejora_val` (PIA) | Normalized metric (0-1) representing the rate of improvement of the global best. | `float` |


## 🧪 How to Run Each Module

Each module is designed to be executable independently. Navigate to the module's directory and run the main script.

**Example:**
# Run the PSO Adaptive System
cd PIA
python main.py

# Run the Traffic Simulation
cd ../Practica1
python semaforo.py

# Run the Graph Analysis
cd ../Practica2
python practica2.py



