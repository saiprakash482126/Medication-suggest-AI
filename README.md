# 💊 Medication Suggest AI — AI-Powered Clinical Decision Support System

### 🔥 Intelligent Medication Recommendations | FastAPI | OpenAI | RAG | Docker | Kubernetes | AWS EKS

[![Python](https://img.shields.io/badge/Python-3.10+-blue)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)]()
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue)]()
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Orchestrated-326CE5)]()
[![AWS](https://img.shields.io/badge/AWS-EKS-orange)]()
[![Status](https://img.shields.io/badge/Status-Active-success)]()

---

<img width="515" height="700" alt="Screenshot 2026-05-13 165833" src="https://github.com/user-attachments/assets/05bcad06-0c66-4b21-9a06-a4e8636308f4" />


## 🚀 Overview

**Medication Suggest AI** is an AI-powered healthcare assistant designed to support doctors and healthcare professionals in generating intelligent medication recommendations based on patient symptoms, diagnoses, and clinical information.

The platform combines **Large Language Models (LLMs)**, **Retrieval-Augmented Generation (RAG)**, and medical reference datasets to deliver accurate, contextual, and explainable medication suggestions.

The solution is built with a cloud-native architecture and can be deployed using **Docker**, **Kubernetes**, and **AWS EKS** for enterprise scalability.

---

## 🎯 Problem Statement

Healthcare professionals often spend valuable time:

* Searching for appropriate medications
* Reviewing treatment guidelines
* Cross-referencing symptoms and diagnoses
* Verifying medication recommendations
* Managing increasing patient consultation workloads

Manual decision support processes can lead to delays and inconsistencies in treatment recommendations.

---

## 💡 Solution

Medication Suggest AI automates the medication recommendation workflow by:

1. Accepting patient symptoms and diagnosis details
2. Processing clinical information using AI
3. Retrieving relevant medical knowledge
4. Generating contextual medication recommendations
5. Providing structured treatment insights
6. Supporting healthcare professionals in clinical decision-making

---

## ✨ Key Features

### 🩺 Symptom-Based Medication Recommendation

Generates medication suggestions based on patient symptoms and clinical conditions.

### 🤖 AI-Powered Clinical Reasoning

Uses advanced LLMs to analyze medical context and generate intelligent recommendations.

### 📚 Retrieval-Augmented Generation (RAG)

Enhances recommendation accuracy using medical reference datasets.

### ⚡ Real-Time API Responses

Fast and scalable API endpoints powered by FastAPI.

### 🐳 Dockerized Deployment

Containerized architecture for portability and simplified deployment.

### ☸️ Kubernetes Ready

Supports enterprise-scale deployments using Kubernetes and AWS EKS.

### 🔄 CI/CD Automation

Automated build and deployment pipelines using GitHub Actions.

---

## 🏗️ System Architecture

```text
Patient Symptoms
        │
        ▼
 FastAPI Backend
        │
        ▼
 AI Processing Layer
        │
        ▼
 Medical Knowledge Base
        │
        ▼
 RAG Retrieval Engine
        │
        ▼
 OpenAI / LLM Service
        │
        ▼
 Medication Recommendation
        │
        ▼
 API Response
```

---

## 🛠️ Technology Stack

### Backend

* Python
* FastAPI
* Uvicorn

### Artificial Intelligence

* OpenAI GPT Models
* Prompt Engineering
* Retrieval-Augmented Generation (RAG)

### Data Processing

* Pandas
* NumPy

### DevOps & Cloud

* Docker
* Kubernetes
* AWS EKS
* GitHub Actions

### API Development

* REST APIs
* JSON-based communication

---

## 📂 Project Structure

```text
Medication-suggest-AI/
│
├── app/
├── routes/
├── services/
├── models/
├── data/
├── prompts/
│
├── Dockerfile
├── requirements.txt
├── main.py
│
├── .github/
│   └── workflows/
│
├── .k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── namespace.yaml
│
└── README.md
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/saiprakash482126/Medication-suggest-AI.git
```

### Navigate to Project

```bash
cd Medication-suggest-AI
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows

```bash
venv\Scripts\activate
```

Linux / Mac

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Application

```bash
uvicorn main:app --reload
```

Access API:

```text
http://localhost:8000
```

Swagger Documentation:

```text
http://localhost:8000/docs
```

---

## 🐳 Docker Deployment

Build Docker Image:

```bash
docker build -t medication-suggest-ai .
```

Run Container:

```bash
docker run -p 8000:8000 medication-suggest-ai
```

---

## ☸️ Kubernetes Deployment

Deploy Application:

```bash
kubectl apply -f .k8s/
```

Verify Resources:

```bash
kubectl get pods
kubectl get svc
kubectl get deployments
```

---

## 🚀 CI/CD Pipeline

GitHub Actions workflow automates:

* Source Code Checkout
* Docker Image Build
* Image Push
* Kubernetes Deployment
* Deployment Validation
* Rollout Monitoring

---

## 📈 Business Impact

✅ Faster clinical decision support

✅ Improved recommendation consistency

✅ Reduced manual medication lookup

✅ Enhanced healthcare productivity

✅ Scalable cloud-native architecture

✅ Production-ready deployment workflow

---

## 🔮 Future Enhancements

* Drug Interaction Detection
* Dosage Recommendation Engine
* EHR Integration
* Voice-Based Patient Consultation
* Multi-Language Support
* Medical Report Generation
* Clinical Risk Analysis

---

## 👨‍💻 Author

**Sai Prakash**

Data Engineer | AI Engineer | Cloud & DevOps Enthusiast

GitHub: https://github.com/saiprakash482126

Repository:
https://github.com/saiprakash482126/Medication-suggest-AI
