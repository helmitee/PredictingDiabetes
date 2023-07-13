import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss from sklearn.model_selection import train_test_split
from sklearn import tree


# function to filter out missing information
def drop_empty_info(df):
df = df[df['BMI'] != 0]
df = df[df['Age'] != 0]
df = df[df['BloodPressure'] != 0] df = df[df['Glucose'] != 0] return df


def bar_height(y): neg = 0
pos = 0 for i in y:
if i == 1: pos += 1
else:
neg +=1
return neg, pos

# converting the data into a dataframe and dropping unnecessary columns
diabetesData = pd.read_csv("diabetes.csv")
diabetesData = diabetesData.drop(columns=['Pregnancies', 'SkinThickness','Insulin', 'DiabetesPedigreeFunction']) 
diabetesData = drop_empty_info(diabetesData)

# initializing lists to help with histograms
val_errors = []
test_errors = []
train_errors = []
loss = []


def logistic_regression(diabetesData):
  X = diabetesData[['BMI', 'Age', 'Glucose', 'BloodPressure']].to_numpy() y = diabetesData['Outcome'].to_numpy()
  
  # separating the data into training data 70% and testing and validation data overall 30%
  X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, train_size=0.7)
  X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, train_size=0.5)
  
  # using logistic regression method to examine the data
  clf = LogisticRegression().fit(X_train, y_train)

  # computing predicted labels with training, validation and testing sets
  y_pred_val = clf.predict(X_val)
  y_pred_test = clf.predict(X_test)
  y_pred_train = clf.predict(X_train)

  # calculationg the accuracy of the predicted labels
  test_accuracy = accuracy_score(y_test, y_pred_test)
  val_accuracy = accuracy_score(y_val, y_pred_val)
  train_accuracy = accuracy_score(y_train, y_pred_train)

  # calculating and saving the prediction errors
  train_errors.append(1 - train_accuracy)
  val_errors.append(1 - val_accuracy)
  test_loss = log_loss(y_test, y_pred_test)
  test_errors.append(test_loss)

  print("Logistic Regression:")
  print("Training accuracy of the logistic regression is {:.0f} %".format(train_accuracy * 100))
  print("Validation accuracy of the logistic regression is {:.0f} %".format(val_accuracy * 100))
  print("Training error of the logistic regression is {:.0f} %".format((1 - train_accuracy) * 100))
  print("Validation error of the logistic regression is {:.0f} %".format((1 - val_accuracy) * 100))

  # visualizing the logistic regression
  ax1 = plt.subplot()
  ax1.set_xlabel("feature(BMI, Age, Glucose, BloodPressure)")
  ax1.set_yticks([0, 1])
  ax1.set_ylabel('label(Outcome)')
  ax1.set_title("Occurrence of diabetes (binarized)")
  ax1.scatter(X_train[:, 0], y_train, color='skyblue', s=30, label='training datapoints')
  ax1.scatter(X_val[:, 0], y_pred_val, color='r', s=2, label='predicted label') 
  ax1.legend() 
  plt.show()

  # computing a confusion matrix to examine the accuracy
  ax2 = plt.subplot()
  conf_mat = confusion_matrix(y_val, y_pred_val) 
  sns.heatmap(conf_mat, annot=True, fmt='g', ax=ax2) 
  ax2.set_xlabel('Predicted labels', fontsize=15) 
  ax2.set_ylabel('True labels', fontsize=15)
  ax2.set_title('Confusion Matrix (Logistic Regression)', fontsize=15) 
  plt.show()
  
  # logistic regression's loss function
  loss_val = log_loss(y_val, y_pred_val)
  loss.append(loss_val)


def decision_tree(diabetesData):
  binarized_BMI = np.where(diabetesData['BMI'] >= 25, 1, 0)
  binarized_Age = np.where(diabetesData['Age'] > 40, 1, 0)
  binarized_Glucose = np.where(diabetesData['Glucose'] >= 140, 1, 0) binarized_BloodPressure = np.where(diabetesData['BloodPressure'] >= 80, 1,0)
  
  diabetesData.insert(5, 'BinarizedBMI', binarized_BMI)
  diabetesData.insert(6, 'BinarizedAge', binarized_Age)
  diabetesData.insert(7, 'BinarizedGlucose', binarized_Glucose)
  diabetesData.insert(8, 'BinarizedBloodPressure', binarized_BloodPressure)

  X = diabetesData[['BinarizedBMI', 'BinarizedAge', 'BinarizedGlucose','BinarizedBloodPressure']]
  y = diabetesData['Outcome']
  X_train, X_test_val, y_train, y_test_val = train_test_split(X, y,train_size=0.7)
  X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val,train_size=0.5)

  clf = tree.DecisionTreeClassifier().fit(X_train, y_train)

  # computing predicted labels with training, validation and testing sets
  y_pred_val = clf.predict(X_val)
  y_pred_test = clf.predict(X_test)
  y_pred_train = clf.predict(X_train)

  # calculationg the accuracy of the predicted labels
  test_accuracy = accuracy_score(y_test, y_pred_test)
  val_accuracy = accuracy_score(y_val, y_pred_val)
  train_accuracy = accuracy_score(y_train, y_pred_train)

  # calculating and saving the prediction errors
  train_errors.append(1 - train_accuracy)
  val_errors.append(1 - val_accuracy)
  test_loss = log_loss(y_test, y_pred_test)
  test_errors.append(test_loss)

  print("\nDecision Tree:")
  print("Training accuracy of the decision tree is {:.0f} %".format(train_accuracy * 100))
  print("Validation accuracy of the decision tree is {:.0f} %".format(val_accuracy * 100))
  print("Training error of the decision tree is {:.0f} %".format((1 - train_accuracy) * 100))
  print("Validation error of the decision tree is {:.0f} %".format((1 - val_accuracy) * 100))

  # Visualizing the decision tree model
  feature_names = ["BMI", "Age", "Glucose", "BloodPressure"]
  class_names = ["No diabetes", "Diabetes"]

  plt.subplots(figsize=(17, 8))
  tree.plot_tree(clf, feature_names=feature_names, filled=True, rounded=True, class_names=class_names)
  plt.show()

  # computing a confusion matrix to examine the accuracy
  conf_mat = confusion_matrix(y_val, y_pred_val)

  ax = plt.subplot()
  sns.heatmap(conf_mat, annot=True, fmt='g', ax=ax) 
  ax.set_xlabel('Predicted labels', fontsize=15) 
  ax.set_ylabel('True labels', fontsize=15) 
  ax.set_title('Confusion Matrix (Decision Tree)', fontsize=15) 
  plt.show()

  # decision tree's loss function
  loss_val = log_loss(y_val, y_pred_val)
  loss.append(loss_val)


def main(): logistic_regression(diabetesData) decision_tree(diabetesData)
  height = bar_height(diabetesData['Outcome'])

  # Comparing logistic regression's and decision tree's training and␣ ↪validation errors with histograms
  ax3 = plt.subplot()
  ax3.bar(np.arange(2) + 0.2, val_errors, 0.4, color='cornflowerblue',label='validation error')
  ax3.bar(np.arange(2) - 0.2, train_errors, 0.4, color='violet',label='training error')
  ax3.set_ylim(0, 0.35)
  ax3.set_xticks(np.arange(2))
  ax3.set_xticklabels(['Logistic regression', 'Decision tree'])
  ax3.set_title('Comparison of training and validation errors')
  ax3.legend()
  plt.show()

  # Comparing logistic regression's and decision tree's calculated logistic loss functions with histograms
  ax1 = plt.subplot()
  ax1.bar(np.arange(2), loss, 0.5, color='palegreen', label='loss function')
  ax1.set_ylim(0, 11)
  ax1.set_xticks(np.arange(2))
  ax1.set_xticklabels(['Logistic regression', 'Decision tree'])
  ax1.set_title('Comparison of the loss functions')
  ax1.legend()
  plt.show()

  # Comparing test errors
  ax5 = plt.subplot()
  ax5.bar(np.arange(2), test_errors, 0.5, color='salmon', label='Test errors')
  ax5.set_ylim(0, 11)
  ax5.set_xticks(np.arange(2))
  ax5.set_xticklabels(['Logistic regression', 'Decision tree'])
  ax5.set_title('Comparison of testing errors')
  ax5.legend()
  plt.show()

  # Printing the testing errors
  print("Testing error of the logistic regression is {:.2f}".format((test_errors[0])))

main()





