import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

df = pd.read_csv("bus9_faultdatanew2.csv")
def process_dataC(df, i):
    def segment_data(data, window_size):
            return [data[i:i+window_size] for i in range(0, len(data) - window_size + 1, window_size)]

    segment_size = 5
    segments = segment_data(df, segment_size)

    def extract_features(segment):
            return {
                'feature1': segment[f'A{i}'].mean(),
                'feature2': segment[f'B{i}'].mean(),
                'feature3': segment[f'C{i}'].mean(),
            }

    feature_list = [extract_features(segment) for segment in segments]
    features_df = pd.DataFrame(feature_list)
    features_df = features_df[2200:2400]
    #features_df =features_df[2000:2200]
    new_X = features_df
        
    
    
    from joblib import load
    scaler = load('scaler_phaseC.joblib')
    X_scaled = scaler.transform(new_X)
    pca = load('pca_phaseC.joblib')
    X_pca = pca.transform(X_scaled)

    X_original = X_pca
    ######################################################
    

    
    import numpy as np
    from sklearn.covariance import EllipticEnvelope
    import matplotlib.pyplot as plt

    
    ee = EllipticEnvelope(contamination=0.15)
    
    ee.fit(X_pca)
    
    y_pred = ee.predict(X_pca)
    
    inliers = X_pca[y_pred == 1]
    outliers = X_pca[y_pred == -1]

    X_original = inliers

    ######################################################


    pca = PCA(n_components=2)
    pca.fit(X_pca)

    # First principal component
    first_pc = pca.components_[0]

    # Calculate the angle of the major axis
    angle_of_major_axis = np.arctan2(first_pc[1], first_pc[0])
    angle_in_degrees = np.degrees(angle_of_major_axis)
    print('angle')
    print(angle_in_degrees)
    def adjust_major_axis_width(X, desired_width):
        
        
        desired_width = 2.6
        #Phase C
        
        
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        # Calculate the current width along the major axis
        current_width = np.max(X_pca[:, 1]) - np.min(X_pca[:, 1])
        print(current_width)
        scale_factor = desired_width / current_width
        #scale_factor=1
        print("SCALE FACTOR" + str(scale_factor))
        # Scale the major axis
        X_pca[:, 1] *= scale_factor

        minor_axis_mean = np.mean(X_pca[:, 1])
        print(minor_axis_mean)
        
        
        X_adjusted = pca.inverse_transform(X_pca)

        
        sum_xy = X[:, 0] + X[:, 1]
        max_index = np.argmax(sum_xy)
        min_index = np.argmin(sum_xy)
        print("Original max point based on x+y:", X[max_index])
        print("Original min point based on x+y:", X[min_index])
        
        original_spread_x = X[max_index, 0] - X[min_index, 0]
        original_spread_y = X[max_index, 1] - X[min_index, 1]

        
        sum_xy_adjusted = X_adjusted[:, 0] + X_adjusted[:, 1]
        new_max_index = np.argmax(sum_xy_adjusted)
        new_min_index = np.argmin(sum_xy_adjusted)
        print("New max point based on x+y:", X_adjusted[new_max_index])
        print("New min point based on x+y:", X_adjusted[new_min_index])
        
        new_spread_x = X_adjusted[new_max_index, 0] - X_adjusted[new_min_index, 0]
        new_spread_y = X_adjusted[new_max_index, 1] - X_adjusted[new_min_index, 1]

        print(original_spread_x)
        print(original_spread_y)
        print(new_spread_x)
        print(new_spread_y)
        
        correction_scale_x = abs(original_spread_x / new_spread_x)
        correction_scale_y = abs(original_spread_y / new_spread_y)
        print(correction_scale_x)
        print(correction_scale_y)


        shift = max(X_adjusted[:,0]) - max(X[:,0])
        # Apply the correction scaling
        #X_adjusted[:, 0] *= correction_scale_x 
        #X_adjusted[:,0] = X_adjusted[:,0]+shift
        #X_adjusted[:, 1] *= correction_scale_y

        pca = PCA(n_components=2)
        pca.fit(X_adjusted)

        #alignmnet
        first_pc = pca.components_[0]
        angle_of_major_axis2 = np.arctan2(first_pc[1], first_pc[0])
        angle_in_degrees2 = np.degrees(angle_of_major_axis2)
        print("angle 2")
        print(angle_in_degrees2)


        alignment_angle = -angle_of_major_axis + angle_of_major_axis2
        alignment_matrix = np.array([[np.cos(alignment_angle), -np.sin(alignment_angle)],
                             [np.sin(alignment_angle), np.cos(alignment_angle)]])
        aligned_data = np.dot(X_adjusted, alignment_matrix)

        # Plotting aligned data
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(aligned_data[:, 0], aligned_data[:, 1], alpha=0.2)
        plt.title('Aligned Data')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.axis('equal')
        plt.grid(True)
        plt.show()
        """

        return aligned_data



    # Desired new width for the major axis, e.g., increase by 50%
    new_width = 1*(np.max(X_original[:, 0]) - np.min(X_original[:, 0]))
    #new_width = 0.92
    # Adjust the width of the major axis
    X_adjusted = adjust_major_axis_width(X_original, new_width)
    #X_adjusted = X_adjusted/1.8

    # Visualization
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(X_original[:, 0], X_original[:, 1], alpha=0.5, label='Original')
    plt.title('Original Data')
    plt.axis('equal')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(X_adjusted[:, 0], X_adjusted[:, 1], alpha=0.5, color='red', label='Width Adjusted')
    plt.title('Major Axis Width Adjusted')
    plt.axis('equal')
    plt.legend()

    plt.show()
    
    #X_adjusted = X_original
    X_adjusted = pd.DataFrame(X_adjusted, columns=['x', 'y'])
    return X_adjusted