Protein Length Periodicity Analysis

This repository holds my work on analyzing periodic patterns in eukaryotic protein lengths. Everything here is from a small study where I cleaned the data, ran exploratory checks, and reproduced the periodicity signals reported in previous work.

Project Structure

data/
• diverse_eukaryotic_enzymes.csv – raw dataset
• diverse_eukaryotic_enzymes_preprocessed.csv – cleaned version

scripts/
• Python scripts I wrote to run the analysis (protein-analysis-python.py, protein-analysis-notebook.py, sad-python-implementation.py)

results/
• Final figures generated from the analysis (Figure_1, Figure_2, etc.)

Overview

The main goal was to see whether protein length distributions across a set of eukaryotic species show recurring periodic signals. I used Python to process the datasets, compute frequency patterns, and regenerate the figures.

All code used to run the analysis is in the scripts folder, the datasets are in data, and generated plots are in results.
