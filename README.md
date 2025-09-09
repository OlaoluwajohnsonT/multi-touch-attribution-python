# Multi-Touch Attribution in Python

## Project Brief
This project implements **multi-touch attribution (MTA)** models in Python to evaluate marketing channel contributions to conversions.  
It combines rule-based, probabilistic, and predictive approaches to guide **data-driven budget allocation**.

## Aim
- Quantify the true contribution of each channel and campaign.  
- Support budget reallocation decisions for improved ROI.  
- Provide an interpretable framework for marketers and analysts.

## Models Implemented
- **Rule-Based:** First-Touch, Last-Touch, Linear, Position-Based, Time-Decay  
- **Probabilistic:** Markov Chain (removal effect)  
- **Game-Theoretic:** Shapley Value Attribution  
- **Predictive:** Logistic Regression (conversion likelihood modeling)  

## Key Insights
- **Referral & Display Ads** consistently emerge as strong assist drivers.  
- **Direct & Email** are over-credited in simple rule-based models.  
- **Shapley attribution** is the most balanced and recommended as the primary model.  

## 📂 Repository Structure
- `data/` → sample datasets  
- `notebooks/` → step-by-step analysis in Jupyter  
- `src/` → reusable Python scripts  
- `outputs/` → charts & tables  

## Getting Started
1. Clone the repo:
   ```bash
   git clone https://github.com/<your-username>/multi-touch-attribution-python.git
   cd multi-touch-attribution-python
