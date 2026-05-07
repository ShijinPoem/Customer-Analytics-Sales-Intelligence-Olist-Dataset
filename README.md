# Customer Analytics & Churn Prediction Project

## Project Overview

This project analyzes the Brazilian E-Commerce Public Dataset by Olist to uncover customer purchasing behavior, sales performance, delivery trends, and customer retention patterns.

The project also includes a machine learning pipeline for customer churn prediction and a business-oriented dashboard for data visualization and decision-making support.

The goal is to simulate a real-world end-to-end business data science workflow, including:
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Machine learning modeling
- Business dashboard development
- Insight generation and storytelling

---

# Business Problem

Customer retention is one of the most important challenges in e-commerce businesses.

This project aims to:
- Understand customer purchasing behavior
- Identify potential churn patterns
- Analyze operational performance
- Support business decision-making using data

---

# Dataset

Dataset: Brazilian E-Commerce Public Dataset by Olist

The dataset contains information about:
- Customers
- Orders
- Payments
- Products
- Sellers
- Reviews
- Delivery information

Main tables used:
- `olist_customers_dataset.csv`
- `olist_orders_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_order_payments_dataset.csv`

---

# Project Structure

```text
customer-analytics-project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_overview.ipynb
│   ├── 02_data_processing.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_modeling.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── modeling.py
│   └── utils.py
│
├── dashboard/
│   └── dashboard_screenshot.png
│
├── requirements.txt
├── README.md
