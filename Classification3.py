from sklearn_rvm import EMRVC
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from joblib import load
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from data2_rvmc import process_dataC
from data2_rvmb import process_dataB
from data2_rvma import process_dataA

import time

plt.rcParams['axes.labelsize'] = 18  
plt.rcParams['xtick.labelsize'] = 16  
plt.rcParams['ytick.labelsize'] = 16  
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['legend.fontsize'] = 18

#this file runs each RVM sequentially and cycles through the data in the test file

j = 0
df = pd.read_csv("testdata2new.csv")

for j in range(10):

    j= j+4

    rvm = load('rvm_model3_phaseA.joblib')
    #process data before classification
    X_pca = process_dataA(df, j)
    start = time.time()
    new_predictions = rvm.predict(X_pca)
    #class probabilities
    probabilities = rvm.predict_proba(X_pca)
    end = time.time()

    print(end-start)
    average_probabilities = np.mean(probabilities, axis=0)
    print("Average probability for each class:", average_probabilities)
    probabilities_class_0 = []
    probabilities_class_1 = []

        
    for pred, prob in zip(new_predictions, probabilities):
            if pred == 0:
                probabilities_class_0.append(prob[0])
            else:
                probabilities_class_1.append(prob[1])

    probabilities_class_0 = np.array(probabilities_class_0)
    probabilities_class_1 = np.array(probabilities_class_1)
    average_probability_class_0 = np.mean(probabilities_class_0)
    average_probability_class_1 = np.mean(probabilities_class_1)

    print("Average Probability for Class 0:", average_probability_class_0)
    print("Average Probability for Class 1:", average_probability_class_1)

    #count points in each class
    one_count = 0
    zero_count = 0
    for i in range(len(new_predictions)):
        if new_predictions[i] == 1:
            one_count = one_count+1
        else:
            zero_count = zero_count+1
    print(one_count)
    print(zero_count)

    if one_count> zero_count:
        print("Phase faulty: YES")
        phaseA = True
    else:
        print("Phase faulty: NO")
        phaseA = False
    print(X_pca)
    
    
    x_min, x_max = X_pca['x'].min() - 1, X_pca['x'].max() + 1
    y_min, y_max = X_pca['y'].min() - 1, X_pca['y'].max() + 1

    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))

    
    Z = rvm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    #plotting the decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4)  
    plt.scatter(X_pca['x'], X_pca['y'], c=new_predictions, cmap=plt.cm.coolwarm, s=20)
    plt.title(f'Fault Data Plotted On The RVM Decision Plane')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.xlim([x_min, x_max])  
    plt.ylim([y_min, y_max])  
    classes = np.unique(new_predictions)
    class_labels = ['No Fault', 'Fault'] 

    
    from matplotlib.colors import ListedColormap
    colors = ListedColormap(['blue', 'red'])  
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors(i), markersize=10, label=class_labels[i]) for i in range(len(classes))]

    
    plt.legend(handles=handles, title="Classes")
    plt.show()




    ################################################
    rvm = load('rvm_model3_phaseB.joblib')

    X_pca = process_dataB(df, j)

    new_predictions = rvm.predict(X_pca)

    #class probabilities
    probabilities = rvm.predict_proba(X_pca)
    average_probabilities = np.mean(probabilities, axis=0)
    print("Average probability for each class:", average_probabilities)
    probabilities_class_0 = []
    probabilities_class_1 = []

        
    for pred, prob in zip(new_predictions, probabilities):
            if pred == 0:
                probabilities_class_0.append(prob[0])
            else:
                probabilities_class_1.append(prob[1])

    probabilities_class_0 = np.array(probabilities_class_0)
    probabilities_class_1 = np.array(probabilities_class_1)
    average_probability_class_0 = np.mean(probabilities_class_0)
    average_probability_class_1 = np.mean(probabilities_class_1)

    print("Average Probability for Class 0:", average_probability_class_0)
    print("Average Probability for Class 1:", average_probability_class_1)
    

    # count the number of data points in each class
    print(new_predictions)
    one_count = 0
    zero_count = 0
    for i in range(len(new_predictions)):
        if new_predictions[i] == 1:
            one_count = one_count+1
        else:
            zero_count = zero_count+1
    print(one_count)
    print(zero_count)

    if one_count> zero_count:
        print("Phase faulty: YES")
        phaseB = True
    else:
        print("Phase faulty: NO")
        phaseB = False
    print(X_pca)
    
    x_min, x_max = X_pca['x'].min() - 1, X_pca['x'].max() + 1
    y_min, y_max = X_pca['y'].min() - 1, X_pca['y'].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))

    
    Z = rvm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    #plotting the decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4)  
    plt.scatter(X_pca['x'], X_pca['y'], c=new_predictions, cmap=plt.cm.coolwarm, s=20)
    plt.title(f'Fault Data Plotted On The RVM Decision Plane')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.xlim([x_min, x_max])  
    plt.ylim([y_min, y_max])  
    classes = np.unique(new_predictions)
    class_labels = ['No Fault', 'Fault']  

    
    from matplotlib.colors import ListedColormap
    colors = ListedColormap(['blue', 'red'])  
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors(i), markersize=10, label=class_labels[i]) for i in range(len(classes))]

    
    plt.legend(handles=handles, title="Classes")
    plt.show()


    #############################################################################


    rvm = load('rvm_model3_phaseC.joblib')

    X_pca = process_dataC(df, j)

    new_predictions = rvm.predict(X_pca)

    probabilities = rvm.predict_proba(X_pca)
    average_probabilities = np.mean(probabilities, axis=0)
    print("Average probability for each class:", average_probabilities)
    probabilities_class_0 = []
    probabilities_class_1 = []

 
    for pred, prob in zip(new_predictions, probabilities):
            if pred == 0:
                probabilities_class_0.append(prob[0])
            else:
                probabilities_class_1.append(prob[1])

    probabilities_class_0 = np.array(probabilities_class_0)
    probabilities_class_1 = np.array(probabilities_class_1)
    average_probability_class_0 = np.mean(probabilities_class_0)
    average_probability_class_1 = np.mean(probabilities_class_1)

    print("Average Probability for Class 0:", average_probability_class_0)
    print("Average Probability for Class 1:", average_probability_class_1)

    print(new_predictions)
    one_count = 0
    zero_count = 0
    for i in range(len(new_predictions)):
        if new_predictions[i] == 1:
            one_count = one_count+1
        else:
            zero_count = zero_count+1
    print(one_count)
    print(zero_count)

    if one_count> zero_count:
        print("Phase faulty: YES")
        phaseC = True
    else:
        print("Phase faulty: NO")
        phaseC = False
    print(X_pca)
    

    x_min, x_max = X_pca['x'].min() - 1, X_pca['x'].max() + 1
    y_min, y_max = X_pca['y'].min() - 1, X_pca['y'].max() + 1

    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))

    
    Z = rvm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4)  
    plt.scatter(X_pca['x'], X_pca['y'], c=new_predictions, cmap=plt.cm.coolwarm, s=20)
    plt.title(f'Fault Data Plotted On The RVM Decision Plane')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.xlim([x_min, x_max])  
    plt.ylim([y_min, y_max])  
    classes = np.unique(new_predictions)
    class_labels = ['No Fault', 'Fault']  
   
   #check ground involvement
    if (phaseA == True and phaseB == True) or (phaseB == True and phaseC ==True) or (phaseA == True and phaseC == True):

        def segment_data(data, window_size):
            return [data[i:i+window_size] for i in range(0, len(data) - window_size + 1, window_size)]

        segment_size = 5
        segments = segment_data(df, segment_size)

        def extract_features(segment):
            return {
                'feature1': segment[f'F{j}'].mean(),
                
            }

        
        feature_list = [extract_features(segment) for segment in segments]
        features_df = pd.DataFrame(feature_list)
       
        
        features_df = features_df[2200:2400]

        ground_involved = False
        for i in range(len(features_df)):
            if features_df.iloc[i]['feature1'] > 0:
                ground_involved = True

        
        if ground_involved:
            print('ground involved')
        else:
            print('ground not involved')