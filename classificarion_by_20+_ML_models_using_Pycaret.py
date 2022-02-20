import pandas as pd
import sys
import os

path = sys.argv[1]

data = pd.read_csv(path)
data.drop('Peptide Sequence', axis=1, inplace=True)
data.head()

from pycaret.classification import *

s = setup(data=data, target='Target', silent=True)
cm = compare_models()
one = pull(cm)
one = one.iloc[:, 0:2]
one

setup(data=data, target='Target', normalize = True, normalize_method = 'zscore', silent=True)
cm1 = compare_models()
two = pull(cm1)
two = two.iloc[:, 1]
two

setup(data=data, target='Target', normalize = True, normalize_method = 'minmax', silent=True)
cm2 = compare_models()
three = pull(cm2)
three = three.iloc[:, 1]
three

setup(data=data, target='Target', normalize = True, normalize_method = 'maxabs', silent=True)
cm3 = compare_models()
four = pull(cm3)
four = four.iloc[:, 1]
four


setup(data=data, target='Target', normalize = True, normalize_method = 'robust', silent=True)
cm4 = compare_models()
five = pull(cm4)
five = five.iloc[:, 1]
five

df1 = pd.concat([one, two, three, four, five], axis=1)
df1.columns = ['Model', 'Accuracy without Normalization', 'Accuracy z-score', 'Accuracy minmax', 'Accuracy Maxabs', 'Accuracy Robust']
df1.head()
df1.to_csv('output-101903706-Normalization.csv', index=False)


setup(data=data, target='Target', feature_selection = True, feature_selection_method = 'classic', feature_selection_threshold = 0.2, silent=True)
cm1 = compare_models()
two2 = pull(cm1)
two2 = two2.iloc[:, 1]
setup(data=data, target='Target', feature_selection = True, feature_selection_method = 'classic', feature_selection_threshold = 0.5, silent=True)
cm2 = compare_models()
three2 = pull(cm2)
three2 = three2.iloc[:, 1]
setup(data=data, target='Target', feature_selection = True, feature_selection_method = 'boruta', feature_selection_threshold = 0.2, silent=True)
cm3 = compare_models()
four2 = pull(cm3)
four2 = four2.iloc[:, 1]
setup(data=data, target='Target', feature_selection = True, feature_selection_method = 'boruta', feature_selection_threshold = 0.5, silent=True)
cm4 = compare_models()
five2 = pull(cm4)
five2 = five2.iloc[:, 1]

df2 = pd.concat([one, two2, three2, four2, five2], axis=1)
df2.columns = ['Model', 'Accuracy without Feature Selection', 'Accuracy Classic 0.2', 'Accuracy Classic 0.5', 'Accuracy Boruta 0.2', 'Accuracy Boruta 0.5']
df2.head()
df2.to_csv('output-101903706-FeatureSelection.csv', index=False)


setup(data=data, target='Target', remove_outliers = True, outliers_threshold = 0.02, silent=True)
cm2 = compare_models()
two3 = pull(cm2)
two3 = two3.iloc[:, 1]
setup(data=data, target='Target', remove_outliers = True, outliers_threshold = 0.04, silent=True)
cm3 = compare_models()
three3 = pull(cm3)
three3 = three3.iloc[:, 1]
setup(data=data, target='Target', remove_outliers = True, outliers_threshold = 0.06, silent=True)
cm4 = compare_models()
four3 = pull(cm4)
four3 = four3.iloc[:, 1]
setup(data=data, target='Target', remove_outliers = True, outliers_threshold = 0.08, silent=True)
cm5 = compare_models()
five3 = pull(cm5)
five3 = five3.iloc[:, 1]

df3 = pd.concat([one, two3, three3, four3, five3], axis=1)
df3.columns = ['Model', 'Accuracy without Outlier Removal', 'Accuracy Threshold 0.02', 'Accuracy Threshold 0.04', 'Accuracy Threshold 0.06', 'Accuracy Threshold 0.08']
df3.head()
df3.to_csv('output-101903706-OutlierRemoval.csv', index=False)


setup(data=data, target='Target', pca = True, pca_method = 'linear', silent=True)
cm2 = compare_models()
two4 = pull(cm2)
two4 = two4.iloc[:, 1]
setup(data=data, target='Target', pca = True, pca_method = 'kernel', silent=True)
cm3 = compare_models()
three4 = pull(cm3)
three4 = three4.iloc[:, 1]
setup(data=data, target='Target', pca = True, pca_method = 'incremental', silent=True)
cm4 = compare_models()
four4 = pull(cm4)
four4 = four4.iloc[:, 1]

df4 = pd.concat([one, two4, three4, four4], axis=1)
df4.columns = ['Model', 'Accuracy without PCA', 'Accuracy linear', 'Accuracy kernel', 'Accuracyincremental']
df4.head()
df4.to_csv('output-101903706-PCA.csv', index=False)

setup(data=data, target='Target', silent=True)
rfModel = create_model('rf')
plot_model(rfModel, plot='confusion_matrix', save=True)
os.rename('Confusion Matrix.png', 'output_101903706_confusion_matrix.png')
plot_model(rfModel, plot='learning', save=True)
os.rename('Learning Curve.png', 'output_101903706_learning.png')
plot_model(rfModel, plot='auc', save=True)
os.rename('AUC.png', 'output_101903706_auc.png')
plot_model(rfModel, plot='boundary', save=True)
os.rename('Decision Boundary.png', 'output_101903706_boundary.png')
plot_model(rfModel, plot='feature', save=True)
os.rename('Feature Importance.png', 'output_101903706_featur.png')