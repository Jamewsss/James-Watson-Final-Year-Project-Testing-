import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

#laod data for training
data = pd.read_csv("localisationdata.csv")
data = data[:7]
data = data[['4', '5', '6', '7', '8', '9']].dropna()

# Define the neighbors of each node using string representations
neighbors = {
    '4': ['5', '6'],
    '5': ['4', '7'],
    '6': ['4', '5', '7', '8'],
    '7': ['8', '5'],
    '8': ['9', '7'],
    '9': ['8','7', '5', '4'],
}


models = {}

#train lienar regression models
for node, neighbor_keys in neighbors.items():
    if all(n in data.columns for n in neighbor_keys):
        X = data[neighbor_keys].values  
        y = data[node].values  
        if len(y) > 0 and X.shape[0] > 0:
            clf = LinearRegression()
            clf.fit(X, y)
            models[node] = clf

#function to predict the voltage of null nodes
def predict_voltage(node, voltage_evidence):
    if node not in neighbors or models[node] is None:
        return None
    
    #prepare voltage evidence as input features
    if all(n in voltage_evidence for n in neighbors[node]):
        neighbor_voltages = [voltage_evidence[n] for n in neighbors[node]]
        return models[node].predict([neighbor_voltages])[0]
    return None

#initialize voltage evidence for known nodes
voltage_evidence = {'4': 0.86, '5': 0.73, '7': 0.778, '8': 0.805}
voltage_evidence = {'4': 0.52, '5': 0.37, '7': 0.68, '8': 0.69}
#voltage_evidence = {'4': 0.531, '5': 0.575, '7': 0.748, '8': 0.7043}
#voltage_evidence = {'4': 0.7706, '5': 0.6143, '7': 0.4443, '8': 0.346}
#voltage_evidence = {'4': 0.7286, '5': 0.4843, '7': 0.5567, '8': 0.622}
#voltage_evidence = {'4': 0.54, '5': 0.37, '7': 0.68, '8': 0.71}

#testing
voltage_evidence = {'4': 0.7387, '5': 0.5027, '7': 0.5657, '8': 0.632} #passed
voltage_evidence = {'4': 0.7517, '5': 0.529, '7': 0.5833, '8': 0.6469} #passed
voltage_evidence = {'4': 0.543, '5': 0.587, '7': 0.752, '8': 0.7103} #passed
voltage_evidence = {'4': 0.5713, '5': 0.6073, '7': 0.7623, '8': 0.723} #passed
voltage_evidence = {'4': 0.7917, '5': 0.7023, '7': 0.5993, '8': 0.4033} #passed
voltage_evidence = {'4': 0.8047, '5': 0.717, '7': 0.616, '8': 0.4337} #passed
voltage_evidence = {'4': 0.7493, '5': 0.73, '7': 0.758, '8': 0.6643} #passed
voltage_evidence = {'4': 0.761, '5': 0.74, '7': 0.7663, '8': 0.6757} #passed


#iteratively update predictions
max_iterations = 10
tolerance = 1e-3
for _ in range(max_iterations):
    old_voltage_6 = voltage_evidence.get('6', None)
    old_voltage_9 = voltage_evidence.get('9', None)

    predicted_voltage_6 = predict_voltage('6', voltage_evidence)
    predicted_voltage_9 = predict_voltage('9', voltage_evidence)

    if predicted_voltage_6 is not None:
        voltage_evidence['6'] = predicted_voltage_6
    if predicted_voltage_9 is not None:
        voltage_evidence['9'] = predicted_voltage_9

    # Check for convergence
    if all(v is not None for v in [old_voltage_6, old_voltage_9, predicted_voltage_6, predicted_voltage_9]):
        if abs(old_voltage_6 - predicted_voltage_6) < tolerance and abs(old_voltage_9 - predicted_voltage_9) < tolerance:
            break

# Print the predicted voltages
print(f"Predicted voltage for node 6: {voltage_evidence.get('6')}")
print(f"Predicted voltage for node 9: {voltage_evidence.get('9')}")

neighbors = {
    '4': ['5', '6'],
    '5': ['4', '7'],
    '6': ['4'],
    '7': ['8', '5'],
    '8': ['9', '7'],
    '9': ['8','6'],
}


#identify the two lowest valued nodes
def find_lowest_voltage_pair(voltage_evidence):
    connected_pairs = []
    for node, connected_nodes in neighbors.items():
        for connected in connected_nodes:
            if node in voltage_evidence and connected in voltage_evidence:
                connected_pairs.append((node, connected, voltage_evidence[node], voltage_evidence[connected]))

   
    connected_pairs.sort(key=lambda x: x[2] + x[3])
    if connected_pairs:
        return connected_pairs[0]
    return None

lowest_pair = find_lowest_voltage_pair(voltage_evidence)
if lowest_pair:
    print(f"The two lowest voltage nodes that are directly connected are {lowest_pair[0]} and {lowest_pair[1]} with voltages {lowest_pair[2]} and {lowest_pair[3]}.")
else:
    print("No connected node pairs found or voltages are missing.")










