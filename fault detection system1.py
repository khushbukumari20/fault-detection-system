import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder 
from imblearn.over_sampling import SMOTE 
from sklearn.model_selection import train_test_split 
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import f_classif 
from sklearn.ensemble import ExtraTreesClassifier 
from sklearn.naive_bayes import GaussianNB 
import joblib 
import os 
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score 
from sklearn.metrics import f1_score 
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report 
dataset=pd.read_csv("Dataset.csv") 
dataset.isnull().sum() 
# Create a count plot 
sns.set(style="darkgrid")  # Set the style of the plot 
plt.figure(figsize=(8, 6))  # Set the figure size 
# Replace 'dataset' with your actual DataFrame and 'Drug' with the column name 
ax = sns.countplot(x='attack', data=dataset, palette="Set3") 
plt.title("Count Plot")  # Add a title to the plot 
plt.xlabel("Categories")  # Add label to x-axis 
plt.ylabel("Count")  # Add label to y-axis 
# Annotate each bar with its count value 
for p in ax.patches: 
  ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), 
     ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), 
     textcoords='offset points') 
plt.show()  # Display the plot 
le= LabelEncoder() 
dataset['attack']=  le.fit_transform(dataset['attack']) 
dataset['protocol_type']=  le.fit_transform(dataset['protocol_type']) 
dataset['service']=  le.fit_transform(dataset['service']) 
dataset['flag']=  le.fit_transform(dataset['flag']) 
X=dataset.iloc[:,0:23] 
y=dataset.iloc[:,-1] 
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.20) 
labels=['Attack','NORMAL'] 
#defining global variables to store accuracy and other metrics 
precision = [] 
recall = [] 
fscore = [] 
accuracy = [] 
#function to calculate various metrics such as accuracy, precision etc 
def calculateMetrics(algorithm, predict, testY): 
  testY = testY.astype('int') 
  predict = predict.astype('int') 
  p = precision_score(testY, predict,average='macro') * 100 
  r = recall_score(testY, predict,average='macro') * 100 
  f = f1_score(testY, predict,average='macro') * 100 
  a = accuracy_score(testY,predict)*100  
  accuracy.append(a) 
  precision.append(p) 
  recall.append(r) 
  fscore.append(f) 
  print(algorithm+' Accuracy    : '+str(a)) 
  print(algorithm+' Precision   : '+str(p)) 
  print(algorithm+' Recall      : '+str(r)) 
  print(algorithm+' FSCORE      : '+str(f)) 
  report=classification_report(predict, testY,target_names=labels) 
  print('\n',algorithm+" classification report\n",report) 
  conf_matrix = confusion_matrix(testY, predict)  
  plt.figure(figsize =(5, 5))  
  ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, 
cmap="Blues" ,fmt ="g"); 
  ax.set_ylim([0,len(labels)]) 
  plt.title(algorithm+" Confusion matrix")  
  plt.ylabel('True class')  
  plt.xlabel('Predicted class')  
  plt.show() 
if os.path.exists('naive_bayes_model.pkl'): 
# Load the trained model from the file 
   clf = joblib.load('naive_bayes_model.pkl') 
   print("Model loaded successfully.") 
   predict = clf.predict(X_test) 
   calculateMetrics("Naive Bayes Classifier", predict, y_test) 
else: 
# Train the model (assuming X_train and y_train are defined) 
  clf = GaussianNB() 
  clf.fit(X_train, y_train) 
# Save the trained model to a file 
  joblib.dump(clf, 'naive_bayes_model.pkl') 
  print("Model saved successfully.") 
  predict = clf.predict(X_test) 
  calculateMetrics("Naive Bayes Classifier", predict, y_test) 
# Check if the model files exist 
if os.path.exists('extratrees_model.pkl'): 
# Load the trained model from the file 
   clf = joblib.load('extratrees_model.pkl') 
   print("Model loaded successfully.") 
   predict = clf.predict(X_test) 
   calculateMetrics("ExtraTreesClassifier", predict, y_test) 
else: 
# Train the model (assuming X_train and y_train are defined) 
   clf = ExtraTreesClassifier() 
   clf.fit(X_train, y_train) 
# Save the trained model to a file 
   joblib.dump(clf, 'extratrees_model.pkl')  
   print("Model saved successfuly.") 
   predict = clf.predict(X_test) 
   calculateMetrics("ExtraTreesClassifier", predict, y_test) 
#showing all algorithms performance values 
columns = ["Algorithm Name","Precison","Recall","FScore","Accuracy"] 
values = [] 
algorithm_names = ["Naive Bayes Classifier", "ExtraTreesClassifier"] 
for i in range(len(algorithm_names)): 
    values.append([algorithm_names[i],precision[i],recall[i],fscore[i],accuracy[i]]) 
temp = pd.DataFrame(values,columns=columns) 
temp 
dataset=pd.read_csv("test.csv") 
dataset['protocol_type']=  le.fit_transform(dataset['protocol_type']) 
dataset['service']=  le.fit_transform(dataset['service']) 
dataset['flag']=  le.fit_transform(dataset['flag']) 
# Make predictions on the selected test data 
predict = clf.predict(dataset) 
# Loop through each prediction and print the corresponding row 
for i, p in enumerate(predict): 
    if p == 0: 
       print(dataset.iloc[i])  # Print the row where prediction is failure 
       print("Row  {}"
"Attack".format(i)) 
    else: 
     print(dataset.iloc[i])  # Print the row where prediction is no failure 
       
     print("Row    {}"
  "NORMAL".format(i)) 
    dataset['Predicted']=predict