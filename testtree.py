import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import confusion_matrix
import seaborn as sns

clf = joblib.load('decision_tree_model2.joblib')

plt.rcParams['axes.labelsize'] = 12  
plt.rcParams['xtick.labelsize'] = 10  
plt.rcParams['ytick.labelsize'] = 10  
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 12

from sklearn.tree import plot_tree

#Plot the decision tree
def plot_decision_tree(model):
    plt.figure(figsize=(10,5))  
    plot_tree(model, filled=True, feature_names=None, class_names=None)
    plt.title("Decision Tree Visualization")
    plt.show()

#Plotting the loaded model
plot_decision_tree(clf)
# Load data
df = pd.read_csv("testdata2.csv")
print(len(df))
# set number of samples included in the data frame
#df = df[10500:11500]
#df = df[:4000]
df = df[:20000]
print(df.columns)

#plotting the sample data
plt.figure()
ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
plt.plot(df["D13"])
plt.title("Phasor Voltage Magnitude Time Series")
plt.xlabel("sample")
plt.ylabel("Phasor Voltage Magnitude (p.u)")
plt.show(block=True)



# extract voltage as feature
def extract_features(data):
    features = {
        'feature1': data['D13'].mean(),
        
    }
    return features

#extract features
features_list = [extract_features(df.loc[[i]]) for i in df.index]
features_df = pd.DataFrame(features_list)
print(features_df[:10])
print(len(features_df))

# label the data for accuracy check
def label_data(features_df):
    features_df['fault_label'] = 0  
    for i in range(len(features_df)):
        if features_df['feature1'][i] < 0.95 or features_df['feature1'][i] > 1.05:
            features_df['fault_label'][i] = 1
    return features_df

#features_df = label_data(features_df)
#more direct method of labelling data
features_df['fault_label'] = 0
features_df['fault_label'][11000:12000] = 1
features_df['fault_label'] = features_df['fault_label'].astype(int)
print(features_df.head())


# pass the data to the decision tree model
X_test = features_df[['feature1']]
Y_labels = features_df['fault_label']
Y_pred = clf.predict(X_test)

# plot histogram
plt.hist(Y_pred, bins=[-0.5, 0.5, 1.5], rwidth=0.8)
plt.xticks([0, 1])
plt.xlabel('Predicted Class')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted Classes')
plt.show(block=True)

print(np.unique(Y_labels))
print(np.unique(Y_pred))

cm = confusion_matrix(Y_labels, Y_pred)
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Fault', 'Fault'], yticklabels=['No Fault', 'Fault'])
#plt.xticks(fontsize=16)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show(block=True)

#Roughly selects the faulty section of data

below_threshold = features_df['feature1'] < 0.9


first_index = below_threshold.idxmax() if below_threshold.any() else None


last_index = below_threshold[::-1].idxmax() if below_threshold.any() else None

print("First index below threshold:", first_index)
print("Last index below threshold:", last_index)

