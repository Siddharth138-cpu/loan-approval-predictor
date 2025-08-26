import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display


from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.model_selection import cross_val_predict, validation_curve
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

from xgboost import XGBClassifier

import optuna

from scipy.stats import kurtosis, skew

from time import time

from warnings import simplefilter
simplefilter("ignore")

pd.set_option('display.max_columns', None)
data = pd.read_csv(r"D:\projects\sid\siddd.venv\loan_data.csv")

data.head()
data
print(f'The dataset has {data.shape[0]} rows and {data.shape[1]} columns.')
print(f'The dataset has {data.duplicated().sum()} duplicate values.')
data.dtypes
data.describe()
### Figures ###
bigfig = plt.figure(figsize=(12,6))

(top, middle, bottom) = bigfig.subfigures(3,1)
### Top figure ###
top.subplots_adjust(left=.1, right=.9, wspace=.4, hspace=.4)

fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(12,6))

ax1 = sns.histplot(data=data, x='person_age', bins=20, ax=ax1, color="red")
ax1.set_xlim(20, 100)
ax1.set_title('Age Distribution', size=15)

ax2 = sns.histplot(data=data, x='person_gender', ax=ax2,color="purple")
ax2.set_title('Gender Distribution', size=15)

plt.tight_layout()

### Middle figure ###
middle.subplots_adjust(left=.1, right=.9, wspace=.4, hspace=.4)

fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(12,6))

ax1 = sns.histplot(data=data, x='person_education', ax=ax1, color="orange")
ax1.set_title('Education Distribution', size=15)

ax2 = sns.histplot(data=data, x='person_income', ax=ax2,color="blue")
ax2.set_title('Income Distribution', size=15)
ax2.set_xlim(0, 400000)

plt.tight_layout()

### Bottom figure ###
bottom.subplots_adjust(left=.1, right=.9, wspace=.4, hspace=.4)

fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(12,6))

ax1 = sns.histplot(data=data, x='person_emp_exp',bins=20, ax=ax1,color="green")
ax1.set_title('Years of Employment Experience', size=15)
ax1.set_xlim(0, 50)

ax2 = sns.histplot(data=data, x='person_home_ownership', ax=ax2,color="Brown")
ax2.set_title('Home Ownership Status', size=15)
plt.tight_layout()
num_features_one = ['person_age', 'person_income', 'person_emp_exp']

for col in num_features_one:

    print(f"Skewness of {col}: {skew(data[col])}")
    print(f"Kurtosis of {col}: {kurtosis(data[col])}")
    print()
### Figures ###
bigfig = plt.figure(figsize=(12,6))

(top, bottom) = bigfig.subfigures(2,1)

### Top figure ###
top.subplots_adjust(left=.1, right=.9, wspace=.4, hspace=.4)

fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(12,6))

ax1 = sns.histplot(data=data, x='loan_amnt', bins=20, ax=ax1,kde=True,color="green")
ax1.set_title('Loan Amount Distribution', size=15)

ax2 = sns.histplot(data=data, x='credit_score', ax=ax2,kde=True,color="darkblue")
ax2.set_title('Purpose of the Loan', size=15)

plt.tight_layout()

### Bottom figure ###
bottom.subplots_adjust(left=.1, right=.9, wspace=.4, hspace=.4)

fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(12,6))

ax1 = sns.histplot(data=data, x='loan_int_rate', bins=10, ax=ax1, kde=True,color="red")
ax1.set_title('Loan interest rate', size=15)

ax2 = sns.histplot(data=data, x='loan_percent_income', bins=20, ax=ax2, kde=True,color="purple")
ax2.set_title('Loan as Percentage of the Income', size=15)

plt.tight_layout()
num_features_two = ['loan_amnt', 'loan_int_rate', 'loan_percent_income']

for col in num_features_two:

    print(f"Skewness of {col}: {skew(data[col])}")
    print(f"Kurtosis of {col}: {kurtosis(data[col])}")
    print()
##########################
loan_approval = {0:'Loan Was Rejected', 1:'Loan Was Approved'}
data['loan_approval'] = data['loan_status'].map(loan_approval)

val = data['loan_approval'].value_counts()
##########################

### Figures ###
bigfig = plt.figure(figsize=(12,6))

(top, bottom) = bigfig.subfigures(2,1)

### Top figure ###
top.subplots_adjust(left=.1, right=.9, wspace=.4, hspace=.4)

fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(12,6))

ax1 = sns.histplot(data=data, x='previous_loan_defaults_on_file', ax=ax1,color="Turquoise")
ax1.set_title('Previous Loan Defaults', size=15)

ax2 = sns.histplot(data=data, x='cb_person_cred_hist_length',bins=10,ax=ax2,color="navy")
ax2.set_title('Length of Credit History in Years', size=15)

plt.tight_layout()

### Bottom figure ###
bottom.subplots_adjust(left=.1, right=.9, wspace=.4, hspace=.4)

fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(12,6))

ax1 = sns.histplot(data=data, x='loan_approval', ax=ax1,color="purple")
ax1.set_title('Loan Approval Status (Target Variable)', size=15)

ax = val.plot(kind='pie', labels=['Rejected', 'Approved'], autopct="%1.1f%%", 
              shadow=True, colors=["maroon", "gold"], explode=(0.03, 0.03))

ax.set_title('Loan Approval Status (Target Variable)', size=15)
plt.show()
data.drop('loan_approval', axis=1, inplace=True)
#######################################
int_rate_default = data.groupby('previous_loan_defaults_on_file')['loan_int_rate'].mean().reset_index(name='avg_interest_rate')

int_rate_purpose = data.groupby('loan_intent')['loan_int_rate'].mean().reset_index(name='avg_interest_rate')
#######################################

### Figures ###
bigfig = plt.figure(figsize=(12,6))

(top, bottom) = bigfig.subfigures(2,1)

### Top figure ###
top.subplots_adjust(left=.1, right=.9, wspace=.4, hspace=.4)

fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(12,6))

ax1 = sns.scatterplot(data=data, x='loan_percent_income', y='loan_int_rate', ax=ax1,hue="loan_percent_income")
ax1.set_title('Loan interest Rate vs Loan Percent Income', size=15)

ax2 = sns.barplot(data=int_rate_default, x='previous_loan_defaults_on_file', y='avg_interest_rate', ax=ax2,hue="previous_loan_defaults_on_file")
ax2.set_title('Loan interest Rate vs Previous Default', size=15)

plt.tight_layout()

### Bottom figure ###
bottom.subplots_adjust(left=.1, right=.9, wspace=.4, hspace=.4)

fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(12,6))

ax1 = sns.barplot(data=int_rate_purpose, x='loan_intent', y='avg_interest_rate', ax=ax1,hue="loan_intent")
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
ax1.set_title('Loan interest Rate vs Loan Purpose', size=15)

ax2 = sns.scatterplot(data=data, x='loan_amnt', y='loan_int_rate', ax=ax2, hue="loan_amnt")
ax2.set_title('Loan interest Rate vs Loan Amount', size=15)

plt.tight_layout()
selected_features = ['loan_int_rate', 
                     'person_emp_exp', 
                     'loan_amnt', 
                     'loan_percent_income',
                     'cb_person_cred_hist_length',
                     'credit_score']

plt.figure(figsize=(12,8))
sns.heatmap(data[selected_features].corr(method='pearson'),annot=True,fmt='.2f',annot_kws={"fontsize":8}, cmap='coolwarm')
plt.title('Correlation heatmap',fontsize=30)

plt.tight_layout()
plt.show()
num_features = [col for col in data.columns if data[col].dtypes != 'O']

num_features.remove('loan_status')


def plot_boxplots(data):

    for i in range(3):

        fig, (ax1,ax2,ax3) = plt.subplots(ncols=3, figsize=(12,5))
        ax1 = sns.boxplot(data[num_features[i*3]], ax=ax1,palette="viridis")
        ax1.set_title('Boxplot of '+str(num_features[i*3]), fontsize=15)
        ax2 = sns.boxplot(data[num_features[i*3+1]], ax=ax2,palette="viridis")
        ax2.set_title('Boxplot of '+str(num_features[i*3+1]), fontsize=15)
        if i < 2:
            ax3 = sns.boxplot(data[num_features[i*3+2]], ax=ax3,palette="viridis")
            ax3.set_title('Boxplot of '+str(num_features[i*3+2]), fontsize=15)
    
        fig.suptitle(f"Boxplots of the Numerical Variables", fontsize=24)    
    
        plt.tight_layout()


plot_boxplots(data)
def outliers_percentage(data):

    outliers_perc = []

    for k,v in data.items():
        # Column must be of numeric type (not object)
        if data[k].dtype != 'O':
            q1 = v.quantile(0.25)
            q3 = v.quantile(0.75)
            irq = q3 - q1
            v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
            perc = np.shape(v_col)[0] * 100.0 / np.shape(data)[0]
            out_tuple = (k,int(perc))
            outliers_perc.append(out_tuple)
            print("Column %s outliers = %.2f%%" % (k,perc))

outliers_percentage(data[num_features])  
def remove_outliers_iqr(data, column):

    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter the data
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data


capped_data = data.copy()

for col in num_features:

    capped_data = remove_outliers_iqr(capped_data, col)
import matplotlib.pyplot as plt
import seaborn as sns

def plot_distributions(data, capped_data, num_features):
    for i in range(3):
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12, 5))

        sns.kdeplot(data[num_features[i * 3]], ax=ax1, color='steelblue', label='+ outliers')
        sns.kdeplot(capped_data[num_features[i * 3]], ax=ax1, color='orange', label='no outliers')
        ax1.set_title('Distribution of ' + str(num_features[i * 3]), fontsize=15)
        ax1.legend(fontsize=8, loc='upper right')

        sns.kdeplot(data[num_features[i * 3 + 1]], ax=ax2, color='steelblue', label='+ outliers')
        sns.kdeplot(capped_data[num_features[i * 3 + 1]], ax=ax2, color='orange', label='no outliers')
        ax2.set_title('Distribution of ' + str(num_features[i * 3 + 1]), fontsize=15)
        ax2.legend(fontsize=8, loc='upper right')

        if i * 3 + 2 < len(num_features):  # Ensure index doesn't go out of range
            sns.kdeplot(data[num_features[i * 3 + 2]], ax=ax3, color='steelblue', label='+ outliers')
            sns.kdeplot(capped_data[num_features[i * 3 + 2]], ax=ax3, color='orange', label='no outliers')
            ax3.set_title('Distribution of ' + str(num_features[i * 3 + 2]), fontsize=15)
            ax3.legend(fontsize=8, loc='upper right')

        plt.tight_layout()
        plt.show()
def plot_distributions(data, capped_data):

    for i in range(3):

        fig, (ax1,ax2,ax3) = plt.subplots(ncols=3, figsize=(12,5))

        ax1 = sns.kdeplot(data[num_features[i*3]], ax=ax1, color='steelblue', label='+ outliers')
        ax1 = sns.kdeplot(capped_data[num_features[i*3]], ax=ax1, color='orange', label='no outliers')
        ax1.set_title('Distribution of '+str(num_features[i*3]), fontsize=15)
        ax1.legend(fontsize=8, loc='upper right')

        ax2 = sns.kdeplot(data[num_features[i*3+1]], ax=ax2, color='steelblue', label='+ outliers')
        ax2 = sns.kdeplot(capped_data[num_features[i*3+1]], ax=ax2, color='orange', label='no outliers')
        ax2.set_title('Distribution of '+str(num_features[i*3+1]), fontsize=15)
        ax2.legend(fontsize=8, loc='upper right')

        if i < 2:
            ax3 = sns.kdeplot(data[num_features[i*3]], ax=ax3, color='steelblue', label='+ outliers')
            ax3 = sns.kdeplot(capped_data[num_features[i*3]], ax=ax3, color='orange', label='no outliers')
            ax3.set_title('Distribution of '+str(num_features[i*3]), fontsize=15)
            ax3.legend(fontsize=8, loc='upper right')
    
        plt.tight_layout()


plot_distributions(data, capped_data)    
print('##### Skewness and kurtosis after outliers capping ##### \n')

for col in num_features:

    print(f"Skewness of {col}: {skew(capped_data[col])}")
    print(f"Kurtosis of {col}: {kurtosis(capped_data[col])}")
    print()
print('##### Skewness and kurtosis before outliers capping ##### \n')

for col in num_features_one:

    print(f"Skewness of {col}: {skew(data[col])}")
    print(f"Kurtosis of {col}: {kurtosis(data[col])}")
    print()
### Label encoding ###
encoder = LabelEncoder()

for col in ['person_gender', 'previous_loan_defaults_on_file']:

    capped_data[col] = encoder.fit_transform(capped_data[col])

### One-hot encoding ###    
capped_data = pd.get_dummies(capped_data)

capped_data.head()
mm_scaler = MinMaxScaler()
std_scaler = StandardScaler()

STD_list = ['person_age', 
            'person_income',
            'person_emp_exp',
            'loan_amnt', 
            'loan_int_rate', 
            'loan_percent_income',
            'cb_person_cred_hist_length', 
            'credit_score']

capped_data[STD_list] = std_scaler.fit_transform(capped_data[STD_list])

capped_data.head()
X = capped_data.drop('loan_status', axis=1)
y = capped_data['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
# Random Forest Model
random_forest = RandomForestClassifier(random_state=1, max_depth=20)
random_forest.fit(X_train, y_train)

importances = pd.DataFrame({'feature':X_train.columns, 'importance':np.round(random_forest.feature_importances_,3)})
importances = importances.sort_values('importance', ascending=False)
importances.head(15)
plt.figure(figsize=(12,6))

sns.barplot(importances[importances['importance'] > 0.02], x='feature', y='importance', hue = 'feature')

plt.title('Feature Importances > 0.02', fontsize=25)
plt.xlabel('feature', fontsize=15)
plt.xticks(fontsize=8, rotation=45)
plt.ylabel('relative importance', fontsize=15)
    
plt.tight_layout()
plt.show()
feat_list = ['person_age', 'person_income', 'person_emp_exp',
             'loan_amnt', 'loan_int_rate', 'loan_percent_income',
             'cb_person_cred_hist_length', 'credit_score']

plt.figure(figsize=(12,8))
sns.heatmap(capped_data[feat_list].corr(method='pearson'), annot=True, fmt='.2f', annot_kws={"fontsize":8},cmap='coolwarm')
plt.title('Correlation heatmap', fontsize=30)

plt.tight_layout()
plt.show()
sns.pairplot(capped_data[feat_list], size=2, corner=True)
def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}

    # Fit the learner to the training data using slicing with 'sample_size'
    start = time() # Get start time
    learner = learner.fit(X_train[:sample_size],y_train[:sample_size])
    end = time() # Get end time
    
    # Calculate the training time
    results['train_time'] = end - start

    #  Get the predictions on the test set,
    #  then get predictions on the first 300 training samples
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time() # Get end time

    # Calculate the total prediction time
    results['pred_time'] = end - start
            
    # Compute accuracy score on the first 300 training samples
    results['accuracy_train'] = accuracy_score(y_train[:300],predictions_train)
     # Compute accuracy score on test set
    results['accuracy_test'] = accuracy_score(y_test,predictions_test)

    # Compute recall score on the first 300 training samples
    results['recall_train'] = recall_score(y_train[:300],predictions_train,average='macro')

    # Compute recall score on test set
    results['recall_test'] = recall_score(y_test,predictions_test,average='macro')
       
    # Success
    print("{} trained on {} samples.".format(learner.__class__.__name__,sample_size))
        
    # Return the results
    return results
# Initialize the three models
clf_A = GradientBoostingClassifier(random_state=42)
clf_B = AdaBoostClassifier(DecisionTreeClassifier(random_state=42))
clf_C = RandomForestClassifier(random_state=42)
clf_D = XGBClassifier(random_state=42)
clf_E = SVC(random_state=42)

# Calculate the number of samples for 1%, 10%, 25%, 50%, 75% and 100% of the training data
samples_1   = int(round(len(X_train) / 100))
samples_10  = int(round(len(X_train) / 10))
samples_25  = int(round(len(X_train) / 4))
samples_50  = int(round(len(X_train) / 2))
samples_75  = int(round(len(X_train) * 0.75))
samples_100 = len(X_train)

# Collect results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C, clf_D, clf_E]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i,samples in enumerate([samples_1, samples_10, samples_25, samples_50, samples_75, samples_100]):
        results[clf_name][i] = \
        train_predict(clf,samples,X_train, y_train, X_test, y_test)
# Printing out the values
for i in results.items():
    print(i[0])
    display(pd.DataFrame(i[1]).rename(columns={0:'1%', 1:'10%', 2:'25%', 3:'50%', 4:'75%', 5:'100%'}))  
def get_test_scores(model_name:str,preds,y_test_data):
    '''
    Generate a table of test scores.

    In:
        model_name (string): Your choice: how the model will be named in the output table
        preds: numpy array of test predictions
        y_test_data: numpy array of y_test data

    Out:
        table: a pandas df of precision, recall, f1, and accuracy scores for your model
    '''
    accuracy  = accuracy_score(y_test_data, preds)
    precision = precision_score(y_test_data, preds, average='macro')
    recall    = recall_score(y_test_data, preds, average='macro')
    f1        = f1_score(y_test_data, preds, average='macro')

    table = pd.DataFrame({'model': [model_name],'precision': [precision],'recall': [recall],
                          'F1': [f1],'accuracy': [accuracy]})

    return table
def plot_validation_curve(clf,X,y,CV,param_name,param_range,y_lim=[0.8, 0.95]):

    train_scores, test_scores = validation_curve(
                estimator = clf, 
                X = X_train, 
                y = y_train, 
                param_name = param_name, 
                param_range = param_range,
                cv = CV)

    train_mean = np.mean(train_scores,axis=1)
    train_std = np.std(train_scores,axis=1)
    test_mean = np.mean(test_scores,axis=1)
    test_std = np.std(test_scores,axis=1)

    plt.plot(param_range, train_mean, 
         color='blue', marker='o', 
         markersize=5, label='training accuracy')

    plt.fill_between(param_range, train_mean + train_std,
                 train_mean - train_std, alpha=0.15,
                 color='blue')

    plt.plot(param_range, test_mean, 
         color='green', linestyle='--', 
         marker='s', markersize=5, 
         label='validation accuracy')

    plt.fill_between(param_range, 
                 test_mean + test_std,
                 test_mean - test_std, 
                 alpha=0.15, color='green')

    plt.xlim([param_range[0], param_range[-1]])
    plt.ylim(y_lim)
    plt.grid()
    plt.legend(loc='lower right')
    plt.xlabel(f'{param_name}')
    plt.ylabel('Accuracy')
    plt.title(f'Validation Curve of {param_name}')

    plt.tight_layout()
    plt.gcf().patch.set_facecolor('lightsteelblue')
    plt.gca().set_facecolor('lemonchiffon')
    plt.show()
gb_class = GradientBoostingClassifier(random_state=42)

plot_validation_curve(gb_class, X_train, y_train, 4, 'n_estimators', [25,50,100,200,300,400,500], y_lim=[0.9,1.])
plot_validation_curve(gb_class, X_train, y_train, 4, 'max_features', [20,50,100,150,200], y_lim=[0.9,1.])
plot_validation_curve(gb_class, X_train, y_train, 4, 'learning_rate', [0.01,0.05,0.1,0.25,0.5,0.75,1.], y_lim=[0.88,1.])
plot_validation_curve(gb_class, X_train, y_train, 4, 'max_depth',[1,2,3,4,5,6,7,8], y_lim=[0.9,1.])
def gradboost_objective(trial, X, y, cv, scoring):
    """
      An objective function to tune hyperparameters of Gradient Boosting Classifier.
      Args:
        trial: an Optuna trial
        X: DataFrame object, features
        y: Series object, Labels
        cv: k folds to cross-validate
        scoring: String, evaluation metric
      Return:
        Mean test accuracy
      """

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 260, 340, step=10),
        "learning_rate": trial.suggest_float("learning_rate", 0.15, 0.25, step=0.025),
        "max_depth": trial.suggest_int("max_depth", 5, 5),
        "random_state": 42,
        }
    
    # Perform cross validation
    gb_class = GradientBoostingClassifier(**params)

    # Compute scores
    scores = cross_validate(gb_class, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    accuracy = scores["test_score"].mean()
    
    return accuracy


gradboost_study = optuna.create_study(direction = "maximize")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
func = lambda trial: gradboost_objective(trial, X_train, y_train, cv=kf, scoring="accuracy")

# Start optimizing with 20 trials
gradboost_study.optimize(func, n_trials=20)

print(f"The highest accuracy reached by this study: {(gradboost_study.best_value)*100:.1f}%.")

print("Best params:")
for key, value in gradboost_study.best_params.items():
    print(f"\t{key}: {value}")
params = gradboost_study.best_params

gradboost_model = GradientBoostingClassifier(**params)

gradboost_model.fit(X_train, y_train)
    
test_preds_gradboost = gradboost_model.predict(X_test)

gradboost_test_results = get_test_scores('GradientBoosting + Optuna', test_preds_gradboost, y_test)

gradboost_test_results
# Generate array of values for confusion matrix
cm = confusion_matrix(y_test, test_preds_gradboost, labels=gradboost_model.classes_)

ax = sns.heatmap(cm,annot=True)
ax.set_title('Confusion Matrix [GradientBoosting (test)]', fontsize=15)
ax.xaxis.set_ticklabels(['Rejected', 'Approved']) 
ax.yaxis.set_ticklabels(['Rejected', 'Approved']) 
ax.set_xlabel("Predicted")
ax.set_ylabel("Target")

plt.tight_layout()
rf_class = RandomForestClassifier(random_state=42)

plot_validation_curve(rf_class, X_train, y_train, 4, 'n_estimators', [25,50,100,200,300], y_lim=[0.9,1.05])
plot_validation_curve(rf_class, X_train, y_train, 4, 'max_features', [20,50,100,150,200], y_lim=[0.9,1.05])
plot_validation_curve(rf_class, X_train, y_train, 4, 'min_samples_split', [10,20,50,75,100,200,300], y_lim=[0.9,1.])
def rf_objective(trial, X, y, cv, scoring):
    """
      An objective function to tune hyperparameters of Gradient Boosting Classifier.
      Args:
        trial: an Optuna trial
        X: DataFrame object, features
        y: Series object, Labels
        cv: k folds to cross-validate
        scoring: String, evaluation metric
      Return:
        Mean test accuracy
      """

    params = {
        #"n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
        #"max_features": trial.suggest_int("max_features", 50, 200, step=50),
        "min_samples_split": trial.suggest_int("min_samples_split", 20, 80, step=5),
        "random_state": 42,
        }
    
    # Perform cross validation
    rf_class = RandomForestClassifier(**params)

    # Compute scores
    scores = cross_validate(rf_class, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    accuracy = scores["test_score"].mean()
    
    return accuracy


rf_study = optuna.create_study(direction = "maximize")

func = lambda trial: rf_objective(trial, X_train, y_train, cv=kf, scoring="accuracy")

# Start optimizing with 20 trials
rf_study.optimize(func, n_trials=20)

print(f"The highest accuracy reached by this study: {(rf_study.best_value)*100:.1f}%.")

print("Best params:")
for key, value in rf_study.best_params.items():
    print(f"\t{key}: {value}")
params = rf_study.best_params

rf_model = RandomForestClassifier(**params)

rf_model.fit(X_train, y_train)

test_preds_rf = rf_model.predict(X_test)

rf_test_results = get_test_scores('RandomForest + Optuna', test_preds_rf, y_test)

rf_test_results
# Generate array of values for confusion matrix
cm = confusion_matrix(y_test, test_preds_rf, labels=rf_model.classes_)

ax = sns.heatmap(cm,annot=True)
ax.set_title('Confusion Matrix [RandomForest (test)]', fontsize=15)
ax.xaxis.set_ticklabels(['Rejected', 'Approved']) 
ax.yaxis.set_ticklabels(['Rejected', 'Approved']) 
ax.set_xlabel("Predicted")
ax.set_ylabel("Target")

plt.tight_layout()
xgb = XGBClassifier(random_state=42)

plot_validation_curve(xgb, X_train, y_train, 4, 'max_depth', [1,2,3,4,5,6,8,10], y_lim=[0.9,1.])
plot_validation_curve(xgb, X_train, y_train, 4, 'lambda', [2,4,8,12,20,50,100,1000], y_lim=[0.9,1.])
def xgb_objective(trial, X, y, cv, scoring):
    """
      An objective function to tune hyperparameters of Gradient Boosting Classifier.
      Args:
        trial: an Optuna trial
        X: DataFrame object, features
        y: Series object, Labels
        cv: k folds to cross-validate
        scoring: String, evaluation metric
      Return:
        Mean test accuracy
      """

    params = {
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "lambda": trial.suggest_int("lambda", 20, 100, step=10),
        "random_state": 42,
        }
    
    # Perform cross validation
    xgb_class = XGBClassifier(**params)

    # Compute scores
    scores = cross_validate(xgb_class, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    accuracy = scores["test_score"].mean()
    
    return accuracy


xgb_study = optuna.create_study(direction = "maximize")

func = lambda trial: xgb_objective(trial, X_train, y_train, cv=kf, scoring="accuracy")

# Start optimizing with 20 trials
xgb_study.optimize(func, n_trials=20)

print(f"The highest accuracy reached by this study: {(xgb_study.best_value)*100:.1f}%.")

print("Best params:")
for key, value in xgb_study.best_params.items():
    print(f"\t{key}: {value}")
params = xgb_study.best_params

xgb_model = XGBClassifier(**params)

xgb_model.fit(X_train, y_train)

test_preds_xgb = xgb_model.predict(X_test)

xgb_test_results = get_test_scores('XGBoost + Optuna', test_preds_xgb, y_test)

xgb_test_results
# Generate array of values for confusion matrix
cm = confusion_matrix(y_test, test_preds_xgb, labels=xgb_model.classes_)

ax = sns.heatmap(cm,annot=True)
ax.set_title('Confusion Matrix [XGBoost (test)]', fontsize=15)
ax.xaxis.set_ticklabels(['Rejected', 'Approved']) 
ax.yaxis.set_ticklabels(['Rejected', 'Approved']) 
ax.set_xlabel("Predicted")
ax.set_ylabel("Target")

plt.tight_layout()
svc_model = SVC(random_state=42, probability=True)

svc_model.fit(X_train, y_train)

test_preds_svc = svc_model.predict(X_test)

svc_test_results = get_test_scores('SVC Classifier', test_preds_svc, y_test)

svc_test_results
voting_classifier = VotingClassifier(estimators=[
        ('XGB', xgb_model),
        ('SVC', svc_model),
        ('GradBoost', gradboost_model)
    ], voting='soft', verbose=False)

voting_classifier.fit(X_train, y_train)

test_preds_vote = voting_classifier.predict(X_test)

vote_test_results = get_test_scores('Voting Classifier', test_preds_vote, y_test)

vote_test_results
# Generate array of values for confusion matrix
cm = confusion_matrix(y_test, test_preds_vote, labels=voting_classifier.classes_)

ax = sns.heatmap(cm,annot=True)
ax.set_title('Confusion Matrix [Voting Class. (test)]', fontsize=15)
ax.xaxis.set_ticklabels(['Rejected', 'Approved']) 
ax.yaxis.set_ticklabels(['Rejected', 'Approved']) 
ax.set_xlabel("Predicted")
ax.set_ylabel("Target")

plt.tight_layout()
stacking_clf = StackingClassifier(
    estimators=[
        ('XGB', xgb_model),
        ('SVC', svc_model),
        ('GradBoost', gradboost_model)
    ],
    final_estimator=rf_model,
    cv=kf
)

stacking_clf.fit(X_train, y_train)

test_preds_stack = stacking_clf.predict(X_test)

stack_test_results = get_test_scores('Stacking Classifier', test_preds_stack, y_test)

stack_test_results
# Generate array of values for confusion matrix
cm = confusion_matrix(y_test, test_preds_stack, labels=stacking_clf.classes_)

ax = sns.heatmap(cm,annot=True)
ax.set_title('Confusion Matrix [Stacking Class. (test)]', fontsize=15)
ax.xaxis.set_ticklabels(['Rejected', 'Approved']) 
ax.yaxis.set_ticklabels(['Rejected', 'Approved']) 
ax.set_xlabel("Predicted")
ax.set_ylabel("Target")

plt.tight_layout()
estimators = [
        ('XGB', xgb_model),
        ('SVC', svc_model),
        ('GradBoost', gradboost_model),
        ('RF', rf_model)
    ]


def ensemble_objective(trial, X, y, cv, scoring):
    
    params = {
        'xgboost_weight': trial.suggest_float('xgboost_weight', 0.0, 1.0),
        'svc_weight': trial.suggest_float('svc_weight', 0.0, 1.0),
        'gradboost_weight': trial.suggest_float('gradboost_weight', 0.0, 1.0),
        'rf_weight': trial.suggest_float('rf_weight', 0.0, 1.0)
    }
    
    scores = []
        
    voting_class = VotingClassifier(
        estimators = estimators,
        weights = [params['xgboost_weight'], params['svc_weight'], params['gradboost_weight'], params['rf_weight']]
    )

    # Compute scores
    scores = cross_validate(voting_class, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    accuracy = scores["test_score"].mean()
        
    return accuracy
ensemble_study = optuna.create_study(direction="maximize")

func = lambda trial: ensemble_objective(trial, X_train, y_train, cv=kf, scoring="accuracy")

# Start optimizing with 20 trials
ensemble_study.optimize(func, n_trials=20)

print(f"The highest accuracy reached by this study: {(ensemble_study.best_value)*100:.1f}%.")

print("Best params:")
for key, value in ensemble_study.best_params.items():
    print(f"\t{key}: {value}")

val_weigths = list(params.values())

val_weigths
clf_votesoft_final = VotingClassifier(
    estimators = [('XGB', xgb_model),
                ('SVC', svc_model),
                ('GradBoost', gradboost_model),
                ('RF', rf_model)],
    weights = val_weigths)

clf_votesoft_final.fit(X_train, y_train)
test_preds_votesoft = clf_votesoft_final.predict(X_test)

votesoft_results = get_test_scores('EnsembleVoting + Optuna', test_preds_votesoft, y_test)

votesoft_results
# Generate array of values for confusion matrix
cm = confusion_matrix(y_test, test_preds_votesoft, labels=clf_votesoft_final.classes_)

ax = sns.heatmap(cm,annot=True)
ax.set_title('Confusion Matrix [Stacking Class. (test)]', fontsize=15)
ax.xaxis.set_ticklabels(['Rejected', 'Approved']) 
ax.yaxis.set_ticklabels(['Rejected', 'Approved']) 
ax.set_xlabel("Predicted")
ax.set_ylabel("Target")

plt.tight_layout()
final_test_results = (pd.concat([gradboost_test_results, 
                                 rf_test_results, 
                                 xgb_test_results, 
                                 vote_test_results,
                                 stack_test_results,
                                 votesoft_results], axis=0).sort_values('accuracy', ascending=False))

final_test_results