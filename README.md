# ML for Healthcare – ECG Beat Classification

**Author:** Mehjabeen Shaik  
**Course / Capstone Project**  
**Domain:** Machine Learning for Healthcare  



##  Project Overview

This project focuses on **automatic ECG heartbeat classification** using **interpretable machine learning techniques**, with an emphasis on **clinical transparency, real-time performance, and robustness under limited and imbalanced data conditions**.

Two classical and well-cited ECG classification approaches from the literature are implemented, analyzed, and compared against a K-Nearest Neighbors (KNN) baseline. The goal is to understand the **trade-offs between interpretability and predictive performance**, particularly for rare arrhythmic beats that are clinically critical but under-represented in real-world datasets.



##  Problem Motivation

Electrocardiogram (ECG) analysis plays a crucial role in diagnosing cardiac arrhythmias. While modern deep learning approaches can achieve high accuracy, they often behave as **black-box models**, limiting their adoption in clinical settings where **explainability and trust** are essential.

Key challenges addressed in this project include:
- Severe **class imbalance**, where life-threatening arrhythmias occur far less frequently than normal beats
- The need for **real-time inference** on portable or resource-constrained medical devices
- **Inter-patient variability** in ECG morphology
- Limited availability of expert-labeled medical data
- The requirement for **transparent decision logic** in healthcare systems



##  Project Report

 The **complete academic project report**, including full methodology, literature survey, experiments, and conclusions, is available here:

 **ML_for_Healthcare_Report.pdf**

This README serves as a **technical guide and summary** of the work, while the PDF contains the full formal documentation.



##  Methods Implemented

###  Method A: Decision Tree Induction (DTI)
- Based on: *Kaur & Singla (2016)*
- Uses **wavelet-energy features** and **RR-interval information**
- Trains a **C4.5-style decision tree**
- Produces **explicit if–then rules** for classification
- Designed for **real-time inference (< 5 ms per beat)**

**Strengths**
- High interpretability
- Transparent decision logic
- Suitable for clinical environments

**Limitations**
- Reduced sensitivity to rare arrhythmic classes
- Performance constrained by shallow tree depth



###  Method B: Wavelet Energy + Linear Discriminant Analysis (LDA)
- Based on: *de Chazal et al. (2004)*
- Applies **3-level Daubechies-4 wavelet decomposition**
- Extracts **four compact energy features**
- Uses **Linear Discriminant Analysis** for classification

**Strengths**
- Extremely lightweight and fast
- Minimal feature set
- Clear linear decision boundaries

**Limitations**
- Linear separability limits rare-class detection
- Less expressive for complex ECG morphologies



###  Method C: K-Nearest Neighbors (Baseline)
- Applied to normalized raw ECG waveforms
- Used as a **performance benchmark**
- Achieves the highest overall accuracy

**Trade-off**
- High accuracy but **low interpretability**
- Not ideal for clinical deployment without explanation mechanisms



##  Dataset

- **Dataset:** ECG5000  
- **Source:** UCR Time Series Archive  
- **Samples:** 140 time-points per heartbeat  
- **Classes:** 5 ANSI/AAMI heartbeat categories  
- **Characteristics:** Highly imbalanced with rare arrhythmias  



##  Experimental Results (Summary)

| Method | Accuracy |
|------|---------|
| Decision Tree (DTI) | ~82% |
| Wavelet + LDA | ~85% |
| KNN Baseline | ~92% |

**Key Observation:**  
Higher accuracy models tend to sacrifice interpretability, while transparent models struggle with rare arrhythmic beats.



##  Real-Time Performance

All implemented models were evaluated for **real-time feasibility**:
- Decision Tree: ~4–5 ms per beat
- LDA: ~2 ms per beat
- Suitable for streaming ECG monitoring scenarios



##  Challenges & Limitations

- Severe **class imbalance** limits sensitivity to rare arrhythmias
- Feature compression may lose subtle waveform characteristics
- Linear models struggle with complex ECG patterns
- Generalization across datasets (ECG5000 vs MIT-BIH) remains challenging



##  Future Work

- Incorporate richer temporal and morphological features
- Apply cost-sensitive or class-balanced learning
- Explore ensemble methods (Random Forests, boosted trees)
- Combine interpretable filters with deep learning backends
- Investigate patient-specific adaptation strategies



##  Learning Outcomes

Through this project, I gained hands-on experience in:
- Interpretable machine learning for healthcare
- Signal processing using wavelet transforms
- Handling imbalanced medical datasets
- Evaluating trade-offs between accuracy and explainability
- Designing models for real-time medical applications



##  References

- Kaur, B., & Singla, S. (2016). *ECG Analysis with Signal Classification Using Decision Tree Induction (DTI)*  
- de Chazal, P., O’Dwyer, M., & Reilly, R. (2004). *IEEE Transactions on Biomedical Engineering*  
- ECG5000 Dataset – UCR Time Series Archive  



