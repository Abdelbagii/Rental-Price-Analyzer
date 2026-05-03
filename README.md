# Rental Price Analyzer

A machine learning web application that predicts estimated monthly rental prices in the USA using property and location features.

## Project Overview

USA Rental Price Analyzer is a regression-based machine learning project built with Python and Streamlit. The system allows users to enter property details such as region, state, property type, square feet, bedrooms, bathrooms, pet policy, furnishing status, laundry options, parking options, and location coordinates.

The model predicts the estimated monthly rent in USD and classifies it into a rent category:

- Low Rent
- Medium Rent
- High Rent

This project is designed as a practical real estate price prediction system for portfolio and GitHub presentation.

## Features

- Predict monthly rental price in USD
- Uses USA-based housing rental data
- Clean Streamlit web interface
- Supports property details such as bedrooms, bathrooms, square feet, and location
- Includes lifestyle and housing options such as pets, smoking, parking, laundry, and furnishing
- Displays rent category based on predicted price
- Uses a Random Forest regression model

## Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- Random Forest Regressor
- Joblib

## Dataset

The project uses a USA housing rental dataset with property listing information.

Main dataset file:

```text
housing_train.csv
