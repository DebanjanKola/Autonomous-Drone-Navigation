# Autonomous Drone Delivery System

> Autonomous drone navigation trained using **Approximate Q-Learning with Linear Function Approximation** inside a custom Gym-compatible simulation environment.

---

##  Problem Statement

Autonomous drone delivery requires sequential decision-making under real-world constraints:

- **Obstacles & Boundaries** — The drone must navigate safely through a structured grid map.
- **Stochastic Wind** — External forces perturb motion and require corrective action.
- **Dynamic Obstacles** — Moving birds introduce unpredictable collision risk.
- **Battery Limits** — Every step drains power; efficiency directly affects mission success.

Classical tabular Q-learning is infeasible here due to the size of the state space. Deep networks, while powerful, add unnecessary complexity and opacity. Linear function approximation offers a **scalable, interpretable, and efficient** alternative.

---

##  Reinforcement Learning Formulation

### MDP Components

| Component     | Description |
|---------------|-------------|
| **State** `s` | Drone's position on the grid |
| **Action** `a` | Discrete directional movement primitives |
| **Reward** `R` | Shaped reward encouraging safe & efficient delivery |
| **Discount** `γ` | `0.99` — long-horizon planning |
| **Feature** `f(s,a)` | 4-dimensional hand-engineered state-action features |

### Reward Structure

| Event             | Reward        | Purpose              |
|-------------------|---------------|----------------------|
| Successful Delivery | `+100`      | Terminal objective   |
| Per-Step Cost      | `−1`         | Path efficiency      |
| Collision / Failure | `−100` to `−300` | Safety enforcement |

### Algorithm: Approximate Q-Learning

The action-value function is approximated linearly:

```
Q̂(s, a; w) = wᵀ · f(s, a)
```

The TD error drives weight updates via stochastic gradient descent:

```
δₜ = Rₜ₊₁ + γ · max_a' Q̂(sₜ₊₁, a') − Q̂(sₜ, aₜ)
w  ← w + α · δₜ · f(sₜ, aₜ)
```

### Feature Representation

Four hand-engineered features encode:
1. Progress toward the delivery goal
2. Proximity to obstacles or boundaries
3. Movement / time cost
4. Terminal success indicator

---

## ⚙️ Experimental Setup

| Hyperparameter        | Value              |
|-----------------------|--------------------|
| Training Episodes     | ~500               |
| Exploration Schedule  | ε: `0.99` → `0.01` (decaying) |
| Discount Factor (γ)   | `0.99`             |
| Evaluation Policy     | ε ≈ `0.01` (near-greedy) |
| Feature Dimensions    | 4                  |

---

##  System Architecture

```
Grid-Based Map Environment (Gym-Compatible)
            │
            ▼
   State Observation (Grid Position)
            │
            ▼
  Feature Extraction  →  f(s, a) ∈ ℝ⁴
            │
            ▼
  Linear Q-Function   →  Q̂(s, a; w) = wᵀf(s, a)
            │
            ▼
   ε-Greedy Action Selection
            │
            ▼
  Environment Step  →  Reward + Next State
            │
            ▼
   TD Error Computation & Weight Update
            │
            ▼
       Repeat Until Convergence
```

---

