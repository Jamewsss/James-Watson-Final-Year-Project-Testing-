from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import joblib

plt.ion()


df = pd.read_csv("testdata2.csv")
#change the number of samples included in the data frame
df = df[:6000]
print(df.columns)

#plot the sample data
plt.figure()
ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
plt.plot(df["D11"])
plt.title("Phasor Voltage Magnitude Time Series")
plt.xlabel("samples")
plt.ylabel("Phasor Voltage Magnitude (p.u)")
plt.show()

#extract phasor voltage magnitude as feature
def extract_features(df):
    features = {
        'feature1': df['D11'].mean(),
    }
    return features


features_list = [extract_features(df.loc[[i]]) for i in df.index]
features_df = pd.DataFrame(features_list)

columns = features_df.columns
rows = features_df.shape[0]
print(columns)

# label the training data
def label_data(features_df):
    features_df['fault_label'] = 0  
    for i in range(len(features_df)):
        if features_df['feature1'][i] < 0.95 or features_df['feature1'][i] > 1.05:
            features_df['fault_label'][i] = 1
    return features_df

features_df = label_data(features_df)
features_df['fault_label'] = features_df['fault_label'].astype(int)
print(features_df.head())


for i in range(len(columns)):
    plt.figure(figsize=(10, 3))
    plt.plot(features_df.iloc[:, i])
    plt.title(f"feature {i + 1}")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.show()


X = features_df[['feature1']]
Y = features_df['fault_label']

if len(X) > 1:
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, Y_train)

    y_pred = clf.predict(X_test)
    print(y_pred)

    accuracy = accuracy_score(Y_test, y_pred)
    precision = precision_score(Y_test, y_pred)
    recall = recall_score(Y_test, y_pred)
    f1 = f1_score(Y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    joblib.dump(clf, 'decision_tree_model2.joblib')

    plt.figure(figsize=(10, 6))
    plot_tree(clf, feature_names=['feature1'], class_names=['No Fault', 'Fault'], filled=True)
    plt.show(block=True)

    cm = confusion_matrix(Y_test, y_pred)
    print(cm)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Fault', 'Fault'], yticklabels=['No Fault', 'Fault'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show(block=True)
else:
    print("Not enough data to split into training and testing sets.")

