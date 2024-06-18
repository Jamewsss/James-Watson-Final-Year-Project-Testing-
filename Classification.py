
from sklearn_rvm import EMRVC
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

#This file trains the RVM models using the training data, and performs predictions on a section of the training data.

###############################################
from sklearn_rvm import EMRVC
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

#load training data from file
df = pd.read_csv("bus9_faultdatanew2.csv")


df.fillna(method='ffill', inplace=True)


df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(method='ffill', inplace=True) 

#function sgements the data to reduce data size but preserve information
def segment_data(data, window_size):
    return [data[i:i+window_size] for i in range(0, len(data) - window_size + 1, window_size)]


segment_size = 5


segments = segment_data(df, segment_size)


def extract_features(segment):
    features = {
        'feature1': segment['A10'].mean(),
        'feature2': segment['B10'].mean(),
        'feature3': segment['C10'].mean(),
        'faultlabel': segment['label'].iloc[0]
    }
    return features


feature_list = [extract_features(segment) for segment in segments]


features_df = pd.DataFrame(feature_list)


features_df['faultlabel'] = features_df['faultlabel'].astype(int)

X = features_df[['feature1', 'feature2', 'feature3']]
Y = features_df['faultlabel']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

#standardise the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mean = scaler.mean_
variance = scaler.var_

#Perform PCA
pca = PCA(n_components=2)  
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)




############################################

plt.figure(figsize=(8, 6))


# 'red' for 1, 'blue' for 0
colors = ['blue' if label == 0 else 'red' for label in y_train]
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=colors, label=['Class 0', 'Class 1'], s=8)


plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot of X_train with Red/Blue Labels')
plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Class 0'),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Class 1')])


plt.show()




###########################################
from joblib import dump
print("BEGIN")
dump(pca, 'pca_phaseC.joblib')
dump(scaler, 'scaler_phaseC.joblib')
print("END")
#fit the RVM to the processed training data
rvm = EMRVC(kernel='rbf', n_iter_posterior=100, gamma=5)
rvm.fit(X_train_pca, y_train)

#make predictions using the test data
y_pred = rvm.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")



dump(rvm, 'rvm_model3_phaseC.joblib')


const_feature3_value = np.mean(X_train_scaled[:, 2])  


x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))



Z = rvm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolor='k', s=20)
plt.title('RVM Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
