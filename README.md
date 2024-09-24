# Neural Assembly Simulation Project

## Introduction

This repository contains code for simulating operations in the assembly model of brain computation, based on the concepts introduced in the following papers:

1. **"Brain Computation by Assemblies of Neurons"** by Papadimitriou et al., *Proceedings of the National Academy of Sciences*, 2020.
2. **"The Architecture of a Biologically Plausible Language Organ"** by Mitropolsky et al., 2023.

The project models how assemblies of neurons can perform computational tasks, including language processing and acquisition, illustrating the computational power of neural circuits in processing complex cognitive functions.

---

## Table of Contents

- [Background](#background)
  - [Neural Assemblies and the Assembly Calculus](#neural-assemblies-and-the-assembly-calculus)
  - [The NEMO Model](#the-nemo-model)
- [Project Overview](#project-overview)
  - [Relation to the Papers](#relation-to-the-papers)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Running the Parser Simulation](#running-the-parser-simulation)
  - [Running Learning Experiments](#running-learning-experiments)
  - [Running Overlap Simulations](#running-overlap-simulations)
  - [Running Other Simulations](#running-other-simulations)
  - [Running Tests](#running-tests)
- [Project Modules](#project-modules)
  - [Brain Simulation (`brain.py`)](#brain-simulation-brainpy)
  - [Brain Utilities (`brain_util.py`)](#brain-utilities-brain_utilpy)
  - [Learning Experiments (`learner.py`)](#learning-experiments-learnerpy)
  - [Overlap Simulations (`overlap_sim.py`)](#overlap-simulations-overlap_simpy)
  - [Parser Simulation (`parser.py`)](#parser-simulation-parserpy)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

---

## Background

### Neural Assemblies and the Assembly Calculus

Neural assemblies are groups of neurons that fire together, representing specific concepts or cognitive functions. The **Assembly Calculus** proposed by Papadimitriou et al. provides a computational framework modeling how these assemblies perform complex computations through operations like projection, association, and merge.

**Key concepts:**

- **Projection**: Creating a new assembly in a downstream area that represents a copy of an existing assembly.
- **Association**: Increasing the overlap between two assemblies to represent an association between their respective concepts.
- **Merge**: Combining two assemblies to form a new assembly representing the combination of their respective concepts.

### The NEMO Model

The **NEMO (NEuronal MOdels)** model, introduced by Mitropolsky et al., enhances the Assembly Calculus by incorporating additional biological realism:

- **Distinct Neuron Types**: Modeling excitatory and inhibitory neurons.
- **Plasticity Mechanisms**: Implementing Hebbian plasticity and synaptic weight adjustments.
- **Brain Areas and Connectivity**: Simulating multiple brain areas with specific connectivity patterns.

---

## Project Overview

This project implements simulations demonstrating how neural assemblies can perform computational tasks, focusing on language processing and Turing machine concepts.

### Relation to the Papers

#### "Brain Computation by Assemblies of Neurons"

- **Implementing Assembly Operations**: The project uses projection, association, and merge operations as described in the paper.
- **Modeling Syntax Processing**: Demonstrates how merge operations represent syntactic structures in language.
- **Simulating Neural Computation**: Reflects the theoretical underpinnings of neural computation through assemblies.

#### "The Architecture of a Biologically Plausible Language Organ"

- **Implementing the NEMO Model**: Incorporates the NEMO model to enhance biological realism.
- **Modeling Language Acquisition**: Simulates how assemblies learn representations of nouns and verbs from grounded input.
- **Exploring Computational Power**: Investigates whether enhancing the neural model yields new computational capabilities.

---

## Project Structure

- **`brain.py`**: Core implementation of the brain model, including neuron assemblies and synaptic connections.
- **`brain_util.py`**: Utility functions for simulations, including saving/loading models and computing overlaps.
- **`learner.py`**: Implements learning experiments, including word acquisition and syntax learning.
- **`overlap_sim.py`**: Simulations examining overlap preservation in assemblies and the effect of associations.
- **`parser.py`**: Code for parsing sentences using the brain simulation framework.
- **`simulations.py`**: Runs various simulations to test neural assemblies' behavior.
- **`turing_sim.py`**: Explores how neural assemblies can simulate Turing machine concepts.
- **`tests.py`** & **`simulations_test.py`**: Unit tests for the brain model and simulations.
- **`README.md`**: Project introduction and usage guide (this file).
- **`DOCUMENTATION.md`**: Detailed documentation of modules and classes.

---

## Getting Started

### Prerequisites

- **Python 3.x**
- **Required Libraries**:
  - `numpy`
  - `matplotlib`
  - `unittest`
  - `pickle`

Install dependencies using pip:

```bash
pip install numpy matplotlib
```

### Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/neural-assembly-simulation.git
   cd neural-assembly-simulation
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Running the Parser Simulation

To run the parser simulation:

```bash
python parser.py
```

Modify the `sentence` variable in the `parse()` function to test different inputs.

### Running Learning Experiments

To run learning experiments for word acquisition and syntax learning:

```bash
python learner.py
```

Modify parameters within `learner.py` to test different scenarios, such as varying lexicon sizes or plasticity parameters.

### Running Overlap Simulations

To run overlap simulations:

```bash
python overlap_sim.py
```

This executes simulations examining overlap preservation in assemblies.

### Running Other Simulations

Run `simulations.py` to execute various tests:

```bash
python simulations.py
```

Run `turing_sim.py` to explore Turing machine concepts:

```bash
python turing_sim.py
```

### Running Tests

To run the unit tests:

```bash
python simulations_test.py
python tests.py
```

---

## Project Modules

### Brain Simulation (`brain.py`)

Provides classes for simulating brain areas and their interactions:

- **`Area`**: Represents a brain area containing neurons.
- **`Brain`**: Manages multiple areas and stimuli, handling neural activity and synaptic plasticity.

**Usage Example**:

```python
from brain import Brain

# Initialize the brain
brain_model = Brain(p=0.05)

# Add areas
brain_model.add_area('VisualCortex', n=1000, k=100, beta=0.05)

# Simulate projections
brain_model.project(
    areas_by_stim={'visual_stimulus': ['VisualCortex']},
    dst_areas_by_src_area={'VisualCortex': ['AuditoryCortex']}
)
```

### Brain Utilities (`brain_util.py`)

Utility functions for simulations:

- **Saving/Loading Models**: `sim_save`, `sim_load`
- **Computing Overlaps**: `overlap`, `get_overlaps`

**Usage Example**:

```python
import brain_util as bu

# Save the brain model
bu.sim_save('brain_model.pkl', brain_model)

# Compute overlap between assemblies
overlap_count = bu.overlap(assembly_a, assembly_b)
```

### Learning Experiments (`learner.py`)

Implements learning experiments:

- **Word Acquisition**: Simulates learning of nouns and verbs from grounded input.
- **Syntax Learning**: Demonstrates how syntax and word order can be learned.

**Usage Example**:

```python
from learner import LearnBrain

# Initialize the brain
brain = LearnBrain(p=0.05, LEX_k=100)

# Train with simple sentences
brain.train_simple(30)

# Test word retrieval
retrieved_word = brain.testIndexedWord(PHON_INDICES["RUN"])
print(f"Retrieved word: {retrieved_word}")
```

### Overlap Simulations (`overlap_sim.py`)

Simulations examining overlap preservation:

**Usage Example**:

```python
import overlap_sim

assembly_overlap, proj_overlap = overlap_sim.overlap_sim()
print(f"Assembly Overlap: {assembly_overlap}")
```

### Parser Simulation (`parser.py`)

Contains code for parsing sentences using the brain simulation framework:

- **Languages Supported**: English and Russian
- **Grammar Rules**: Simulates parsing using predefined grammar rules.

**Usage Example**:

```python
from parser import parse

# Parse an English sentence
parse("cats chase mice", language="English")

# Parse a Russian sentence
parse("kot vidit sobaku", language="Russian")
```

---

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**: Click the "Fork" button at the top right of the repository page.

2. **Clone Your Fork**:

   ```bash
   git clone https://github.com/yourusername/neural-assembly-simulation.git
   ```

3. **Create a New Branch**:

   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make Changes**: Add your features or fix bugs.

5. **Commit Changes**:

   ```bash
   git commit -am "Add your message here"
   ```

6. **Push to Your Fork**:

   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**: Go to the original repository and click "New Pull Request".

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## References

- Papadimitriou, C. H., Vempala, S. S., Mitropolsky, D., Collins, M., & Maass, W. (2020). **Brain Computation by Assemblies of Neurons**. *Proceedings of the National Academy of Sciences*, 117(25), 14464–14472. [Link](https://www.pnas.org/doi/full/10.1073/pnas.2001893117)

- Mitropolsky, D., & Papadimitriou, C. H. (2023). **The Architecture of a Biologically Plausible Language Organ**. [arXiv:2306.15364](https://arxiv.org/abs/2306.15364)

---

This README provides an introduction to the neural assembly simulation project, including background information, usage instructions, and an overview of the modules. For detailed documentation of classes and methods, please refer to the [DOCUMENTATION.md](DOCUMENTATION.md) file.