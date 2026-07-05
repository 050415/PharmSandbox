# Drug-Drug Interaction Prediction and Alternative Recommendation System

This project focuses on the prediction of Drug-Drug Interactions (DDIs) and recommends safer alternative drug combinations. It addresses the critical need for automated DDI detection due to the complexity and sheer volume of biomedical data and proposes a solution integrating transformer-based and traditional ML models with ensemble learning.

---

## Table of Contents

- [Introduction](#introduction)
- [Objectives](#objectives)
- [Methodology](#methodology)
- [Datasets Used](#datasets-used)
- [Results](#results)
- [Conclusion & Future Work](#conclusion--future-work)

---

## Introduction

Detecting drug-drug interactions (DDIs) is crucial in healthcare to avoid harmful side effects from concurrent medications. Manual analysis is impractical due to the vast biomedical literature. This project automates the process and further enhances healthcare decision-making by recommending safe alternatives.

---

## Objectives

- Develop and fine-tune DDI detection models using BioBERT.
- Utilize Word2Vec embeddings for contextual understanding.
- Combine model outputs via ensemble learning for robustness.
- Recommend alternative drug combinations if harmful interactions are found.

---

## Methodology

This project presents a **multi-pipeline framework** for Drug-Drug Interaction (DDI) prediction and alternative drug recommendation. It aims to fulfill two main objectives:

1. **Detect if two drugs interact negatively**, and  
2. **Recommend an alternative, non-interacting drug pair** if a negative interaction is detected.

The system is structured into two primary pipelines:

---

### Pipeline 1: DDI Detection

#### Model 1: DDI Detection Using BioBERT

- **Dataset**: Utilizes the **DDICorpus dataset**, which includes:
  - 784 DrugBank documents with curated drug interaction texts
  - 233 MedLine abstracts selected using the query “drug-drug interactions”
- **Annotations**: Contains manually annotated pharmacological substances and interaction types.
- **Approach**: BioBERT (a domain-specific variant of BERT pretrained on biomedical literature) is fine-tuned to classify the nature of interaction between drug entities based on the sentence context.
- **Objective**: Determine the presence and type of DDI from sentence-level biomedical text.

#### Model 2: DDI Detection Using Contextual Embeddings (Word2Vec)

- **Data Source**: Abstracts from **PubMed**.
- **Approach**:
  - Train a **Word2Vec** model on PubMed abstracts to capture biomedical context.
  - Represent drug pairs as vector pairs using the trained embeddings.
  - A classifier predicts whether the pair results in a negative interaction.
- **Class Imbalance Handling**:  
  Applies **SMOTE (Synthetic Minority Over-sampling Technique)** to generate synthetic examples for the minority class and mitigate bias toward the majority class.

#### Ensemble: Model Fusion with Meta-Classifier

- Uses a **stacking ensemble method** to combine predictions from Model 1 and Model 2.
- A **meta-classifier** takes the individual predictions as input and produces the final decision.
- **Advantage**: 
  - Model 1 captures textual semantic nuances, while Model 2 leverages contextual similarity in biomedical language.
  - The ensemble mitigates individual model weaknesses and improves overall robustness in DDI prediction.

---

### Pipeline 2: Alternative Drug Recommendation

This pipeline activates when a harmful DDI is detected.

- For each drug in the interacting pair, the system identifies the **Top-N most pharmacologically similar drugs**.
- Each candidate pair is **re-evaluated using the DDI detection model**.
- The **most similar pair with no predicted interaction** is selected as the alternative.

> 🔁 This iterative process ensures that the recommended drug pair is both **pharmacologically relevant** and **safe**.

---

### Validation Framework for Pipeline 2

To verify the functional similarity of alternative drug candidates, a **Patent Validation Framework** is implemented.

#### Framework Architecture

A multi-stage pipeline performs the following:

1. **Data Collection**  
   - Retrieve compound names and convert to **PubChem CIDs**.
   - Fetch related **patent identifiers** using PubChem and Google Patents.

2. **Descriptor Generation**  
   - Use **Gemini 1.5 Pro (Google's multimodal AI model)** to extract **functional descriptors** from patent text (titles, abstracts, descriptions).

3. **Similarity Analysis**  
   - Compute **similarity scores** between drug candidates based on functional descriptors.

4. **Results Management**  
   - Store and organize outputs in structured formats (e.g., CSV) for further use and reporting.

#### Key Components

- **PubChem Integration**
  - Convert drug names to **Compound Identifiers (CIDs)**
  - Retrieve associated **patent IDs**

- **Patent Extraction**
  - Scrape Google Patents while adhering to rate limits
  - Extract text content (title, abstract, description)

- **AI-Powered Analysis**
  - Generate interpretable descriptors using **Gemini**
  - Score similarity between drugs using descriptor overlap and semantic analysis

- **Reporting**
  - Output results as structured records
  - Generate **CSV reports** for external validation or visualization



---

## Datasets Used

1. **DDI Corpus** – Annotated drug-drug interactions with sentence-level context.
2. **PubMed Abstracts** – Biomedical abstracts used to train Word2Vec.
3. **DrugBank XML** – Drug descriptions, names, and product details.

---

## Results

- BioBERT performs well on sentence-level DDI extraction.
- Word2Vec models struggle with class imbalance; balancing improves minority class recall.
- Ensemble model outperforms individual models slightly.
- Recommendation system filters out risky drug pairs and suggests alternatives effectively.


The results of the models implemented in this project are presented below.

### Results of Our Models

| **Method**               | **Precision** | **Recall** | **F1-Score** |
|--------------------------|---------------|------------|--------------|
| **BioBERT Model**        | 0.86          | 0.88       | 0.87         |
| Random Forest            | 0.52          | 0.57       | 0.49         |
| Decision Tree            | 0.52          | 0.57       | 0.46         |
| XGB-SMOTE                | 0.59          | 0.64       | 0.58         |
| SVM-SMOTE                | 0.60          | 0.64       | 0.61         |
| **Ensemble-Based Models**|               |            |              |
| Logistic Regression      | 0.90          | 0.86       | 0.88         |
| Random Forest            | 0.89          | 0.87       | 0.88         |
| Gradient Boosting        | 0.90          | 0.87       | 0.88         |
| XGBoost                  | 0.90          | 0.88       | 0.89         |
| Neural Network           | 0.89          | 0.88       | 0.89         |


---

## Conclusion & Future Work

This project highlights the effectiveness of combining transformer-based and traditional ML models for DDI prediction. Key future directions include:

- Improving similarity metrics for drug recommendations.
- Building a framework to validate recommended alternatives.
- Extending to multi-drug interactions and integrating with EHR systems.

---

## Contributors
- Abhayjit Singh Gulati
- Soumya Sangam Jha
- Achyut Agarwal
- **Guide:** Prof. Ananthanarayana V. S., NITK Surathkal

---
