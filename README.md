# Toxicity Progression Analysis

This repository contains tools and scripts for analyzing toxicity progression in online conversations. It leverages natural language processing (NLP) techniques and machine learning models to detect and study the evolution of toxic behavior in textual data. The motivation towards this project stems from meeting the requirements of the graduate-level course STATS 507: Data Science & Analytics using Python at the University of Michigan.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

---

## Features

- **Toxicity Detection:** Automatically detect toxic language in text using pretrained models.
- **Progression Analysis:** Monitor and visualize how toxicity evolves over time in conversations.
- **Customizable Models:** Extend or replace models to suit specific datasets or use cases.
- **Data Preprocessing:** Comprehensive tools for cleaning and preparing text data.
- **Visualization:** Generate insightful visualizations of toxicity trends.

---

## Requirements

- Python 3.8 or later
- Required libraries (listed in `requirements.txt`)

## Project Structure

toxicity-progression-analysis/
├── data/                   # Input datasets
├── ouput/                  # Processed logs and visualizations
├── src/                    # Source code for analysis and utilities
│   ├── preprocess.py       # Preprocessing scripts
│   ├── analyze_toxicity.py # Main toxicity analysis script
│   ├── visualize_trends.py # Visualization utilities
│   └── models/             # Machine learning models
├── tests/                  # Model Performance
├── notebooks               # Codebase on Jupyter
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

## Acknowledgements

Taking this opportunity to be grateful towards Professor Xian Zhang for the invaluable guidance and for making this project both challenging and rewarding. A big thank you to my classmates for keeping me sharp with their insightful questions and occasional curveballs. Thanks to the open-source community for providing resources, guidance and technical advice that made this project possible.
