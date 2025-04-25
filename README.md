# Route Optimization Using Machine Learning

This project demonstrates a route optimization model for freight transport from Rome to Hamburg. 
Three test routes were randomly selected from the dataset and used to evaluate
the model performance based on cost and time efficiency using Machine Learning techniques.

## Technologies Used

- **Python**: Programming language used for implementation.
- **TensorFlow**: Deep learning framework for building and training the model.
- **NumPy** and **Pandas**: For data handling and preprocessing.
- **Google Colab**: For running and testing the project in the cloud.

## Input Data

The input dataset includes numerical values representing both cost-related and time-related factors, divided into two categories:

### Category 1 – Cost Factors

- Fuel cost
- Tolls
- Tunnel/bridge fees
- Pollution taxes
- Fines (speeding, etc.)
- Parking
- Driver’s wage
- Driver’s meal cost
- Driver’s accommodation

### Category 2 – Time Factors

- Route length (km)
- Border delays
- Traffic jams
- Roadworks
- Seasonal weather conditions
- Stops for rest, refueling, meals, toll booths
- Road safety

## Output Data

The model predicts two outputs:

1. **Total cost** (in EUR)
2. **Total travel time** (in hours)

## Machine Learning Approach

- **Type of Learning**: Supervised Learning
- **Model**: Neural Network implemented using TensorFlow/Keras
- **Loss Function**: Mean Squared Error (MSE) for both cost and time predictions
- **Optimizer**: Adam

## Training & Evaluation

- **Training Samples**: Manually generated synthetic data (due to absence of real datasets)
- **Training Duration**: Varies by epoch and batch size, tested 100 or 200 epochs
- **Validation**: Performed using a held-out test set from synthetic examples
- **Evaluation Metric**: Mean Absolute Error (MAE), R² Score

## Predefined Routes

The model was tested on 30 specific routes. Test size=0.1, 90 % Learning and 10% Testing.
The chosen routes are random every time.
Here are some of the results:

Chosen route number: 4
Actual value: Total Cost = 2500.00 EUR, Total Travel Time = 54.00 h
Prediction   : Total Cost = 2498.99 EUR, Total Travel Time = 53.43 h
--------------------------------------------------

Chosen route number: 23
Actual value: Total Cost = 2600.00 EUR, Total Travel Time = 56.50 h
Prediction   : Total Cost = 2621.35 EUR, Total Travel Time = 56.61 h
--------------------------------------------------

Chosen route number: 6
Actual value: Total Cost = 2390.00 EUR, Total Travel Time = 50.00 h
Prediction   : Total Cost = 2371.61 EUR, Total Travel Time = 49.86 h

## Notes

This is a simplified demo project intended for portfolio purposes. 