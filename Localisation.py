from collections import deque
import random
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# this script is for calculating the peak fault voltage from the data measured at each bus


df = pd.read_csv('locationtesting.csv')
#set section of data included in data frame
df = df[11000:12000]

segment_size = 1  

    
def segment_data(data, window_size):
        return [data[i:i+window_size] for i in range(0, len(data) - window_size + 1, window_size)]

#data segmentation
segments = segment_data(df, segment_size)

for j in range(6):
    j = j+44



    def extract_features(df):
        features = {
            'feature1': (df[f'A{j}'].mean()),
            'feature2': (df[f'B{j}'].mean()),
            'feature3': (df[f'C{j}'].mean())
        
        }
        return features
    


    feature_list = [extract_features(segment) for segment in segments]
    features_df = pd.DataFrame(feature_list)


    

    value = (np.mean(features_df['feature1'])+ np.mean(features_df['feature2'])+ np.mean(features_df['feature3']))/3

    def find_peaks(data, num_peaks=50):
        #find the indices of the top 50 peaks
        top_peaks_indices = np.argpartition(data, -num_peaks)[-num_peaks:]
        top_peaks_indices = top_peaks_indices[np.argsort(-data[top_peaks_indices])]  

        
        peak_values = data[top_peaks_indices]

        return top_peaks_indices, peak_values
    peaks, _ = find_peaks((features_df['feature1']))
    #extract peak values
    peak_values = (features_df['feature1'])[peaks]
    
    # Find the minimum 
    min_peak_value1 = np.min(peak_values)

    peaks, _ = find_peaks((features_df['feature2']))
   
    peak_values = (features_df['feature2'])[peaks]
  
    min_peak_value2 = np.min(peak_values)

    peaks, _ = find_peaks((features_df['feature3']))
    
    peak_values = (features_df['feature3'])[peaks]
    
    min_peak_value3 = np.min(peak_values)

    value = (min_peak_value1+min_peak_value2+min_peak_value3)/3

    print(j)
    print(value)
    
    

