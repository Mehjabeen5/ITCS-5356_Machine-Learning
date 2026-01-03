
# PROJECT REPORT

**Project Title:** ML for Healthcare  
**Name:** Mehjabeen Shaik (801455091)

---

## Method A
**Authors:** B. Kaur and S. Singla  
**Title:** “ECG Analysis with Signal Classification Using Decision Tree Induction (DTI)”  
**Conference:** 2016 Annual Conference of the Computer Society of India, in Advances in Intelligent Systems and Computing  
**Year:** 2016  
**Link:** https://dl.acm.org/doi/10.1145/2979779.2979874

---

## Method B
**Authors:** Pádraig de Chazal, Michael O’Dwyer, and Ray B. Reilly  
**Title:** “Automatic classification of heartbeats using ECG morphology and heartbeat interval features”  
**Journal:** IEEE Transactions on Biomedical Engineering  
**Year:** 2004  
**Link:** https://doi.org/10.1109/TBME.2004.827359

---

# 1) Introduction

## 1.1) Problem Statement

### (Method A)
Accurate ECG beat classification is essential for timely arrhythmia detection,  
yet many high‐performing models—like SVMs or neural nets—act as “black boxes” that clinicians cannot inspect.  
Moreover, rare arrhythmic beats are under‐represented, and real‐time monitoring on portable devices demands  
lightweight, low‐latency solutions. This work addresses the challenge of creating an interpretable, rule‐based ECG classifier  
that uses only a small set of compact waveform features to deliver fast, transparent beat labeling without sacrificing accuracy.

### (Method B)
Accurate, real‐time classification of individual heartbeats is crucial for monitoring and diagnosing cardiac arrhythmias,  
yet conventional approaches rely on labor‐intensive manual annotation or complex feature sets that hinder deployment  
in resource‐constrained settings. Many existing automated algorithms require extensive signal preprocessing,  
large numbers of hand‐crafted descriptors, or computationally heavy models—making them difficult to generalize, interpret,  
and run on portable devices. This paper addresses the challenge of designing a lightweight ECG beat classifier  
that uses only a small set of wavelet‐based morphology descriptors and simple heartbeat‐interval features  
to achieve high sensitivity and specificity across five clinically recommended classes, while maintaining transparency  
and computational efficiency.

---

## Performance comparision

- **Method A to Method C**
  - Method B (DTI): 0.8233  
  - Method C (KNN): 0.9213  

- **Method B to Method C**
  - Method B (LDA): 0.8473  
  - Method C (KNN): 0.9213  

B vs. C: LDA loses fine morphology detail → underfits rare classes.  
A vs. C: DTI gains interpretability but similarly underfits rare beats.  
B vs. A: SVM/RBF (if tested) or deeper trees could boost rare‐beat detection.

---

## 1.2) Motivation and Challenges

### Method A
- Interpretability: Clinicians require clear, rule‐based decision logic rather than opaque “black‐box” models.  
- Real-Time Efficiency: Continuous heartbeat monitoring demands sub‐5 ms inference on lightweight hardware.  
- Data Scarcity: Annotated arrhythmia examples are limited and imbalanced.

### Method B
- Real-time Arrhythmia Detection  
- Inter-patient Variability  
- Limited Labeled Data  
- Class Imbalance  
- Interpretability & Trust  
- Computational Footprint  

---

## 8) Conclusion

### Method A
This project implemented and evaluated a decision‐tree–based ECG beat classifier on the ECG5000 dataset, achieving 82.3% accuracy  
with fully transparent if–then rules. The model proved fast (<5 ms/beat) and interpretable, though it underperformed on rare arrhythmias.

### Method B
This project reimplemented and evaluated a classic wavelet‐energy + LDA beat classifier on the ECG5000 dataset, demonstrating that  
four compact features can yield ~84.7% accuracy and real‐time performance (<2 ms/beat) on low‐power hardware.

---

## 9) My Contributions

### Method A
- Coded a Python pipeline to extract DWT‐energy features and train a DecisionTreeClassifier.  
- Added and compared a 5‐NN baseline.

### Method B
- Implemented wavelet‐energy + LDA classification.  
- Evaluated accuracy and per‐class metrics.

---

## 10) References
See original report for full reference list.
