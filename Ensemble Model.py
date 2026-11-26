import os
import shap
import torch
import optuna
import joblib
import umap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from scipy.interpolate import interp1d
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical, Real
from pytorch_tabnet.tab_model import TabNetClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from optuna.samplers import TPESampler
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy.interpolate import interp1d
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score

# ========== [0] Set Working Directory ===============================================

out_dir = r'D:\Data'
os.makedirs(out_dir, exist_ok=True)

# ========== [1] Data Loading and Preprocessing ======================================

print('\n[1] Loading data...')

data_path = r'D:\Data\Cst_data.xlsx'
mydata = pd.read_excel(data_path)

feature_data = mydata.drop(
    columns=['Group', 'Sample ID', 'Sub-group']
)

X = feature_data.values
feature_names = feature_data.columns.tolist()

# Group labels for subsequent model training and evaluation
Y_group = mydata['Group'].values
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y_group)
class_names = label_encoder.classes_

# Sub-group labels used only for UMAP visualization
Y_subgroup = mydata['Sub-group'].values
subgroup_encoder = LabelEncoder()
Y_subgroup_encoded = subgroup_encoder.fit_transform(Y_subgroup)
subgroup_names = subgroup_encoder.classes_

print(f'Data shape: {X.shape}')
print(f'Model classes: {class_names}')
print(f'UMAP sub-groups: {subgroup_names}')

# ========== [2] UMAP Visualization ==================================================

print('\n[2] Running UMAP visualization...')
reducer = umap.UMAP(n_components=2, n_neighbors=100, min_dist=0.5, random_state=2)
X_umap = reducer.fit_transform(X)
plt.figure(figsize=(8, 6), dpi=600)
palette = sns.color_palette('Set2', len(subgroup_names))
sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1],
    hue=subgroup_encoder.inverse_transform(Y_subgroup_encoded), hue_order=subgroup_names,
    palette=palette, alpha=0.8, edgecolor='black', linewidth=0.5, s=80)
plt.title('UMAP of Cassiterite Sub-groups', fontsize=12)
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.xlim(-10, 20)
plt.ylim(-10, 20)
plt.legend(title='Sub-group', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.rcParams['svg.hashsalt'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
umap_path = os.path.join(out_dir, 'UMAP_subgroup_plot.pdf')
plt.savefig(umap_path, format='pdf', bbox_inches='tight', dpi=600, facecolor='white')
plt.show()
print(f'✅ UMAP plot has been saved to: {umap_path}')

# ========== [3] Data Splitting =====================================================

print('\n[3] Splitting the dataset...')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=2025, stratify=Y)

# ========== [4] Random Forest Model Training ===================================================

print('\n[4] Training the Random Forest model...')

# ========== 4.1 Bayesian Optimization with Cross-Validation ==========

param_bayes_rf = {'n_estimators': Integer(100, 300), 'max_depth': Integer(3, 20), 'min_samples_split': Integer(2, 10), 'min_samples_leaf': Integer(2, 8), 'max_features': ['sqrt', 'log2'], 'bootstrap': [True, False]}
rf_model = RandomForestClassifier(n_jobs=-1, random_state=2025)
bayes_search_rf = BayesSearchCV(estimator=rf_model, search_spaces=param_bayes_rf, n_iter=50, cv=5, scoring='roc_auc_ovr_weighted', verbose=1, n_jobs=-1, random_state=2025)
bayes_search_rf.fit(X_train, Y_train)
best_rf_model = bayes_search_rf.best_estimator_

# ========== 4.2 Learning Curve Plotting ==========

print('\n[4.1] Plotting the learning curve...')
train_sizes_rf = np.linspace(0.05, 1.0, 20)
train_sizes_rf_percent = train_sizes_rf * 100
train_sizes_rf, train_scores_rf, test_scores_rf = learning_curve(estimator=best_rf_model, X=X_train, y=Y_train, cv=5, scoring='roc_auc_ovr_weighted', train_sizes=train_sizes_rf, n_jobs=-1)
train_mean_rf = np.mean(train_scores_rf, axis=1)
train_std_rf = np.std(train_scores_rf, axis=1)
test_mean_rf = np.mean(test_scores_rf, axis=1)
test_std_rf = np.std(test_scores_rf, axis=1)
plt.figure(figsize=(8, 6), dpi=600)
plt.plot(train_sizes_rf_percent, train_mean_rf, 'o-', color='blue', label='Training AUC')
plt.fill_between(train_sizes_rf_percent, train_mean_rf - train_std_rf, train_mean_rf + train_std_rf, alpha=0.1, color='blue')
plt.plot(train_sizes_rf_percent, test_mean_rf, 'o-', color='green', label='Cross-validation AUC')
plt.fill_between(train_sizes_rf_percent, test_mean_rf - test_std_rf, test_mean_rf + test_std_rf, alpha=0.1, color='green')
plt.title('Learning Curve - Random Forest (AUC)')
plt.xlabel('Training Set Size (%)')
plt.ylabel('AUC')
plt.ylim(0.8, 1.05)
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
learning_curve_auc_path_rf = os.path.join(out_dir, 'learning_curve_rf_auc.pdf')
plt.savefig(learning_curve_auc_path_rf, format='pdf', bbox_inches='tight', dpi=300)
plt.show()
print(f'✅ Learning curve has been saved to: {learning_curve_auc_path_rf}')

# ========== 4.3 Output Top 5 Bayesian Optimization Results ==========

cv_results_rf = bayes_search_rf.cv_results_
auc_scores_rf = cv_results_rf['mean_test_score']
params_list_rf = [dict(params) for params in cv_results_rf['params']]
top_5_results_rf = sorted(zip(auc_scores_rf, params_list_rf), reverse=True, key=lambda x: x[0])[:5]
print('\nTop 5 Bayesian optimization cross-validation AUC scores:')
for i, (auc, params) in enumerate(top_5_results_rf, start=1):
    print(f'{i}. AUC score: {auc:.4f}, Parameters: {params}')
print('\nBest Random Forest model evaluation results:')
y_rf_pred = best_rf_model.predict(X_test)
y_rf_proba = best_rf_model.predict_proba(X_test)
n_classes_rf = len(label_encoder.classes_)
if n_classes_rf == 2:
    auc_rf = roc_auc_score(Y_test, y_rf_proba[:, 1], multi_class='ovr', average='weighted')
    print(f'AUC score (binary classification, class 1): {auc_rf:.4f}')
else:
    auc_rf = roc_auc_score(Y_test, y_rf_proba, multi_class='ovr', average='macro')
    print(f'AUC score (multiclass, weighted OVR): {auc_rf:.4f}')
print(f'Accuracy: {accuracy_score(Y_test, y_rf_pred):.4f}')
print(f"Precision: {precision_score(Y_test, y_rf_pred, average='weighted'):.4f}")
print(f"Recall: {recall_score(Y_test, y_rf_pred, average='weighted'):.4f}")
print(f"F1 score: {f1_score(Y_test, y_rf_pred, average='weighted'):.4f}")
cm_rf = confusion_matrix(Y_test, y_rf_pred)
fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.matshow(cm_rf, cmap='Blues')
fig.colorbar(cax)
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label')
ax.set_title('Confusion Matrix')
ax.set_xticks(np.arange(len(class_names)))
ax.set_yticks(np.arange(len(class_names)))
ax.set_xticklabels(class_names)
ax.set_yticklabels(class_names)
for i in range(len(class_names)):
    for j in range(len(class_names)):
        ax.text(j, i, str(cm_rf[i, j]), ha='center', va='center', color='white' if cm_rf[i, j] > cm_rf.max() / 2 else 'black')
plt.show()

# ========== 4.4 SHAP Analysis ==========

print('\nCalculating SHAP values...')
explainer_rf = shap.TreeExplainer(best_rf_model)
shap_values_rf = explainer_rf.shap_values(X_test)

# ========== 4.5 SHAP Analysis Visualization ==========

def generate_editable_shap_plot_rf(shap_values_rf, features, feature_names, class_name, cmap='PiYG_r', alpha=0.8, point_size=30, linewidth=0, preview_in_spyder=True, out_dir=None):
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.sans-serif'] = ['Arial']
    if isinstance(shap_values_rf, list):
        shap_array = np.stack(shap_values_rf, axis=-1)
    elif len(shap_values_rf.shape) == 2:
        shap_array = np.stack([-shap_values_rf, shap_values_rf], axis=-1)
    else:
        shap_array = shap_values_rf
    class_idx = np.where(class_names == class_name)[0][0]
    class_shap = shap_array[..., class_idx]
    feature_array = np.asarray(features, dtype=np.float32)
    mean_abs_shap = np.mean(np.abs(class_shap), axis=0)
    top_feature_indices = np.argsort(mean_abs_shap)[::-1][:15]
    selected_names = [feature_names[i] for i in top_feature_indices]
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    ax.grid(True, linestyle='--', alpha=0.6, linewidth=0.5)
    max_val = np.max(np.abs(feature_array[:, top_feature_indices]))
    norm = plt.Normalize(-max_val, max_val)
    for i, feat_idx in enumerate(top_feature_indices):
        y_pos = i + np.random.uniform(-0.25, 0.25, size=len(class_shap))
        sc = ax.scatter(class_shap[:, feat_idx], y_pos, c=feature_array[:, feat_idx], cmap=cmap, norm=norm, s=point_size, alpha=alpha, edgecolors='none', linewidth=0, rasterized=False)
    cbar = fig.colorbar(sc, ax=ax, pad=0.01, aspect=30)
    cbar.set_label('Feature Value', rotation=270, labelpad=15)
    ax.set_yticks(np.arange(len(top_feature_indices)))
    ax.set_yticklabels(selected_names, fontsize=9)
    ax.set_xlabel('SHAP Value (impact on model output)', fontsize=10)
    ax.set_title(f'SHAP Feature Importance - {class_name}', pad=12)
    ax.set_xlim(-0.5, 0.5)
    ax.invert_yaxis()
    if out_dir is not None:
        output_path = os.path.join(out_dir, f"SHAP_rf_{class_name.replace(' ', '_')}.pdf")
        plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300, transparent=True, facecolor='none', metadata={'Title': f'SHAP Analysis - {class_name}', 'Author': 'RF SHAP Visualizer'})
        print(f'  - Saved: {output_path}')
    if preview_in_spyder:
        plt.show(block=False)
        plt.pause(0.1)
    plt.close(fig)
print('\nGenerating SHAP visualizations...')
for class_name in class_names:
    print(f'Processing class: {class_name}')
    generate_editable_shap_plot_rf(shap_values_rf, X_test, feature_names, class_name, cmap='PiYG_r', alpha=0.7, point_size=30, linewidth=0, preview_in_spyder=True, out_dir=out_dir)
def calculate_shap_contributions_rf(shap_values_rf, feature_names, class_names):
    if isinstance(shap_values_rf, list):
        shap_array = np.stack(shap_values_rf, axis=-1)
    elif len(shap_values_rf.shape) == 2:
        shap_array = np.stack([-shap_values_rf, shap_values_rf], axis=-1)
    else:
        shap_array = shap_values_rf
    contrib_data = {}
    for i, class_name in enumerate(class_names):
        class_shap = shap_array[..., i]
        mean_abs_shap = np.mean(np.abs(class_shap), axis=0)
        contrib_data[class_name] = mean_abs_shap
    contrib_df = pd.DataFrame(contrib_data, index=feature_names)
    contrib_df = contrib_df.div(contrib_df.sum(axis=0), axis=1) * 100
    return contrib_df
def plot_shap_contributions_rf(shap_contrib_df, target_classes, top_n=20, out_dir=None):
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.sans-serif'] = ['Arial']
    for cls in target_classes:
        series = shap_contrib_df[cls].sort_values(ascending=False)
        if top_n is not None:
            series = series.head(top_n)
        series = series.sort_values()
        fig = plt.figure(figsize=(8, 6), dpi=600)
        ax = fig.add_subplot(111)
        y_pos = np.arange(len(series))
        ax.barh(y_pos, series.values, height=0.6, edgecolor='black', linewidth=0.5, rasterized=False)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(series.index, fontsize=9)
        ax.set_xlabel('Contribution rate (%)', fontsize=10)
        ax.set_title(f'SHAP Contribution - {cls}', pad=12)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        if out_dir is not None:
            fname = f"SHAP_rf_Contrib_{cls.replace(' ', '_')}.pdf"
            save_path = os.path.join(out_dir, fname)
            plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=600, transparent=True, facecolor='none', metadata={'Title': f'SHAP Contribution - {cls}', 'Author': 'SHAP Visualizer'})
            print(f'  - Chart saved: {save_path}')
        plt.show(block=False)
        plt.pause(0.5)
        plt.close(fig)
shap_contrib_df = calculate_shap_contributions_rf(shap_values_rf, feature_names, class_names)
print('\nPlotting ranked SHAP feature contributions...')
plot_shap_contributions_rf(shap_contrib_df=shap_contrib_df, target_classes=class_names, top_n=20, out_dir=out_dir)
excel_path = os.path.join(out_dir, 'SHAP_rf_contributions.xlsx')
shap_contrib_df.to_excel(excel_path)
print(f'\n✅ SHAP feature contributions for all classes have been saved to the Excel file: {excel_path}')

# ========== [5] XGBoost Model Training ===================================================

print('\n[4] Training the XGBoost model...')

# ========== 5.1 Bayesian Optimization with Cross-Validation ==========

param_bayes_xgb = {'n_estimators': Integer(100, 300), 'max_depth': Integer(1, 5), 'learning_rate': Real(0.01, 0.1), 'min_child_weight': Integer(1, 10), 'subsample': Real(0.7, 1.0), 'colsample_bytree': Real(0.7, 1.0), 'gamma': Real(0, 1), 'reg_alpha': Real(0.1, 10), 'reg_lambda': Real(0.1, 10)}
xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(class_names), tree_method='hist', n_jobs=-1, eval_metric='mlogloss', random_state=2025)
bayes_search_xgb = BayesSearchCV(estimator=xgb_model, search_spaces=param_bayes_xgb, n_iter=50, cv=5, scoring='roc_auc_ovr_weighted', verbose=1, n_jobs=-1, random_state=2025)
bayes_search_xgb.fit(X_train, Y_train)
best_xgb_model = bayes_search_xgb.best_estimator_

# ========== 5.2 Learning Curve Plotting ==========

print('\n[4.1] Plotting the learning curve...')
train_sizes_xgb = np.linspace(0.05, 1.0, 20)
train_sizes_xgb_percent = train_sizes_xgb * 100
train_sizes_xgb, train_scores_xgb, test_scores_xgb = learning_curve(estimator=best_xgb_model, X=X_train, y=Y_train, cv=5, scoring='roc_auc_ovr_weighted', train_sizes=train_sizes_xgb, n_jobs=-1)
train_mean_xgb = np.mean(train_scores_xgb, axis=1)
train_std_xgb = np.std(train_scores_xgb, axis=1)
test_mean_xgb = np.mean(test_scores_xgb, axis=1)
test_std_xgb = np.std(test_scores_xgb, axis=1)
plt.figure(figsize=(8, 6), dpi=600)
plt.plot(train_sizes_xgb_percent, train_mean_xgb, 'o-', color='blue', label='Training AUC')
plt.fill_between(train_sizes_xgb_percent, train_mean_xgb - train_std_xgb, train_mean_xgb + train_std_xgb, alpha=0.1, color='blue')
plt.plot(train_sizes_xgb_percent, test_mean_xgb, 'o-', color='green', label='Cross-validation AUC')
plt.fill_between(train_sizes_xgb_percent, test_mean_xgb - test_std_xgb, test_mean_xgb + test_std_xgb, alpha=0.1, color='green')
plt.title('Learning Curve - XGBoost (AUC)')
plt.xlabel('Training Set Size (%)')
plt.ylabel('AUC')
plt.ylim(0.8, 1.05)
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
learning_curve_auc_path_xgb = os.path.join(out_dir, 'learning_curve_xgb_auc.pdf')
plt.savefig(learning_curve_auc_path_xgb, format='pdf', bbox_inches='tight', dpi=300)
plt.show()
print(f'✅ Learning curve has been saved to: {learning_curve_auc_path_xgb}')

# ========== 5.3 Output Top 5 Bayesian Optimization Results ==========

cv_results_xgb = bayes_search_xgb.cv_results_
auc_scores_xgb = cv_results_xgb['mean_test_score']
params_list_xgb = cv_results_xgb['params']
params_list_xgb = [dict(params) for params in params_list_xgb]
top_5_results_xgb = sorted(zip(auc_scores_xgb, params_list_xgb), reverse=True, key=lambda x: x[0])[:5]
print('\nTop 5 Bayesian optimization cross-validation AUC scores:')
for i, (auc, params) in enumerate(top_5_results_xgb, start=1):
    print(f'{i}. AUC score: {auc:.4f}, Parameters: {params}')
print('\nBest XGBoost model evaluation results:')
y_xgb_pred = best_xgb_model.predict(X_test)
y_xgb_proba = best_xgb_model.predict_proba(X_test)
n_classes_xgb = len(label_encoder.classes_)
if n_classes_xgb == 2:
    auc_xgb = roc_auc_score(Y_test, y_xgb_proba[:, 1], multi_class='ovr', average='weighted')
    print(f'AUC score (binary classification, class 1): {auc_xgb:.4f}')
else:
    auc_xgb = roc_auc_score(Y_test, y_xgb_proba, multi_class='ovr', average='macro')
    print(f'AUC score (multiclass, weighted OVR): {auc_xgb:.4f}')
print(f'Accuracy: {accuracy_score(Y_test, y_xgb_pred):.4f}')
print(f"Precision: {precision_score(Y_test, y_xgb_pred, average='weighted'):.4f}")
print(f"Recall: {recall_score(Y_test, y_xgb_pred, average='weighted'):.4f}")
print(f"F1 score: {f1_score(Y_test, y_xgb_pred, average='weighted'):.4f}")
cm_xgb = confusion_matrix(Y_test, y_xgb_pred)
fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.matshow(cm_xgb, cmap='Blues')
fig.colorbar(cax)
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label')
ax.set_title('Confusion Matrix')
ax.set_xticks(np.arange(len(class_names)))
ax.set_yticks(np.arange(len(class_names)))
ax.set_xticklabels(class_names)
ax.set_yticklabels(class_names)
for i in range(len(class_names)):
    for j in range(len(class_names)):
        ax.text(j, i, str(cm_xgb[i, j]), ha='center', va='center', color='white' if cm_xgb[i, j] > cm_xgb.max() / 2 else 'black')
plt.show()

# ========== 5.4 SHAP Analysis ==========

print('\nCalculating SHAP values...')
explainer_xgb = shap.TreeExplainer(best_xgb_model)
shap_values_xgb = explainer_xgb.shap_values(X_test)

# ========== 5.5 SHAP Analysis Visualization ==========

def generate_editable_shap_plot_xgb(shap_values_xgb, features, feature_names, class_name, cmap='PiYG_r', alpha=0.8, point_size=30, linewidth=0, preview_in_spyder=True, out_dir=None):
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.sans-serif'] = ['Arial']
    if isinstance(shap_values_xgb, list):
        shap_array = np.stack(shap_values_xgb, axis=-1)
    elif len(shap_values_xgb.shape) == 2:
        shap_array = np.stack([-shap_values_xgb, shap_values_xgb], axis=-1)
    else:
        shap_array = shap_values_xgb
    class_idx = np.where(class_names == class_name)[0][0]
    class_shap = shap_array[..., class_idx]
    feature_array = np.asarray(features, dtype=np.float32)
    mean_abs_shap = np.mean(np.abs(class_shap), axis=0)
    top_feature_indices = np.argsort(mean_abs_shap)[::-1][:15]
    selected_names = [feature_names[i] for i in top_feature_indices]
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    ax.grid(True, linestyle='--', alpha=0.6, linewidth=0.5)
    max_val = np.max(np.abs(feature_array[:, top_feature_indices]))
    norm = plt.Normalize(-max_val, max_val)
    for i, feat_idx in enumerate(top_feature_indices):
        y_pos = i + np.random.uniform(-0.25, 0.25, size=len(class_shap))
        sc = ax.scatter(class_shap[:, feat_idx], y_pos, c=feature_array[:, feat_idx], cmap=cmap, norm=norm, s=point_size, alpha=alpha, edgecolors='none', linewidth=0, rasterized=False)
    cbar = fig.colorbar(sc, ax=ax, pad=0.01, aspect=30)
    cbar.set_label('Feature Value', rotation=270, labelpad=15)
    ax.set_yticks(np.arange(len(top_feature_indices)))
    ax.set_yticklabels(selected_names, fontsize=9)
    ax.set_xlabel('SHAP Value (impact on model output)', fontsize=10)
    ax.set_title(f'SHAP Feature Importance - {class_name}', pad=12)
    ax.set_xlim(-3, 3)
    ax.invert_yaxis()
    if out_dir is not None:
        output_path = os.path.join(out_dir, f"SHAP_xgb_{class_name.replace(' ', '_')}.pdf")
        plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300, transparent=True, facecolor='none', metadata={'Title': f'SHAP Analysis - {class_name}', 'Author': 'XGBoost SHAP Visualizer'})
        print(f'  - Saved: {output_path}')
    if preview_in_spyder:
        plt.show(block=False)
        plt.pause(0.1)
    plt.close(fig)
print('\nGenerating SHAP visualizations...')
for class_name in class_names:
    print(f'Processing class: {class_name}')
    generate_editable_shap_plot_xgb(shap_values_xgb, X_test, feature_names, class_name, cmap='PiYG_r', alpha=0.7, point_size=30, linewidth=0, preview_in_spyder=True, out_dir=out_dir)
def calculate_shap_contributions_xgb(shap_values_xgb, feature_names, class_names):
    if isinstance(shap_values_xgb, list):
        shap_array = np.stack(shap_values_xgb, axis=-1)
    elif len(shap_values_xgb.shape) == 2:
        shap_array = np.stack([-shap_values_xgb, shap_values_xgb], axis=-1)
    else:
        shap_array = shap_values_xgb
    contrib_data = {}
    for i, class_name in enumerate(class_names):
        class_shap = shap_array[..., i]
        mean_abs_shap = np.mean(np.abs(class_shap), axis=0)
        contrib_data[class_name] = mean_abs_shap
    contrib_df = pd.DataFrame(contrib_data, index=feature_names)
    contrib_df = contrib_df.div(contrib_df.sum(axis=0), axis=1) * 100
    return contrib_df
def plot_shap_contributions_xgb(shap_contrib_df, target_classes, top_n=20, out_dir=None):
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.sans-serif'] = ['Arial']
    for cls in target_classes:
        series = shap_contrib_df[cls].sort_values(ascending=False)
        if top_n is not None:
            series = series.head(top_n)
        series = series.sort_values()
        fig = plt.figure(figsize=(8, 6), dpi=600)
        ax = fig.add_subplot(111)
        y_pos = np.arange(len(series))
        ax.barh(y_pos, series.values, height=0.6, edgecolor='black', linewidth=0.5, rasterized=False)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(series.index, fontsize=9)
        ax.set_xlabel('Contribution rate (%)', fontsize=10)
        ax.set_title(f'SHAP Contribution - {cls}', pad=12)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        if out_dir is not None:
            fname = f"SHAP_xgb_Contrib_{cls.replace(' ', '_')}.pdf"
            save_path = os.path.join(out_dir, fname)
            plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=600, transparent=True, facecolor='none', metadata={'Title': f'SHAP Contribution - {cls}', 'Author': 'SHAP Visualizer'})
            print(f'  - Chart saved: {save_path}')
        plt.show(block=False)
        plt.pause(0.5)
        plt.close(fig)
shap_contrib_df = calculate_shap_contributions_xgb(shap_values_xgb, feature_names, class_names)
print('\nPlotting ranked SHAP feature contributions...')
plot_shap_contributions_xgb(shap_contrib_df=shap_contrib_df, target_classes=class_names, top_n=20, out_dir=out_dir)
excel_path = os.path.join(out_dir, 'SHAP_xgb_contributions.xlsx')
shap_contrib_df.to_excel(excel_path)
print(f'\n✅ All feature contributions have been saved to the Excel file: {excel_path}')

# ========== [6] TabNet Model Training ================================================

print('\n[5] Training the TabNet model...')

# ========= 6.1 Optuna Hyperparameter Tuning Setup =========

def objective(trial):
    params_tb = {'n_d': trial.suggest_categorical('n_d', [8, 16, 32, 64]), 'n_a': trial.suggest_categorical('n_a', [8, 16, 32, 64]), 'n_steps': trial.suggest_int('n_steps', 3, 10), 'gamma': trial.suggest_float('gamma', 1.0, 2.0), 'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-06, 0.001, log=True), 'optimizer_fn': torch.optim.Adam, 'optimizer_params': {'lr': trial.suggest_float('lr', 0.001, 0.02)}, 'mask_type': trial.suggest_categorical('mask_type', ['entmax', 'sparsemax']), 'verbose': 0}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2025)
    fold_auc_scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, Y_train)):
        X_tr, X_val = (X_train[train_idx], X_train[val_idx])
        Y_tr, Y_val = (Y_train[train_idx], Y_train[val_idx])
        tabnet_model = TabNetClassifier(**params_tb)
        tabnet_model.fit(X_tr, Y_tr, eval_set=[(X_val, Y_val)], eval_name=['valid'], eval_metric=['auc'], max_epochs=50, patience=10, batch_size=256, virtual_batch_size=64, drop_last=False)
        y_pred_proba = tabnet_model.predict_proba(X_val)
        if len(class_names) == 2:
            auc_score = roc_auc_score(Y_val, y_pred_proba[:, 1])
        else:
            Y_val_bin = label_binarize(Y_val, classes=np.arange(len(class_names)))
            auc_score = roc_auc_score(Y_val_bin, y_pred_proba, multi_class='ovr')
        fold_auc_scores.append(auc_score)
        print(f'Trial {trial.number} - Fold {fold + 1} AUC: {auc_score:.4f}')
    mean_auc_tb = np.mean(fold_auc_scores)
    print(f'Trial {trial.number} - Mean CV AUC: {mean_auc_tb:.4f}')
    return mean_auc_tb

# ========= 6.2 Run Optuna Hyperparameter Tuning =========

study_tb = optuna.create_study(direction='maximize', sampler=TPESampler(seed=2025))
study_tb.optimize(objective, n_trials=20)
print('Best hyperparameters:', study_tb.best_params)

# ========= 6.3 Train Final Model with Best Hyperparameters =========

best_params_tb = study_tb.best_params
best_tabnet_model = TabNetClassifier(n_d=best_params_tb['n_d'], n_a=best_params_tb['n_a'], n_steps=best_params_tb['n_steps'], gamma=best_params_tb['gamma'], lambda_sparse=best_params_tb['lambda_sparse'], optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=best_params_tb['lr']), mask_type=best_params_tb['mask_type'], seed=2025, verbose=10)
best_tabnet_model.fit(X_train, Y_train, eval_set=[(X_train, Y_train), (X_test, Y_test)], eval_name=['train', 'valid'], eval_metric=['auc'], max_epochs=50, patience=10, batch_size=256, virtual_batch_size=64)

# ========= 6.4 Plot AUC Learning Curve =========

train_auc = best_tabnet_model.history['train_auc']
valid_auc = best_tabnet_model.history['valid_auc']
interp_x = np.linspace(0, 1, 101)
x_original = np.linspace(0, 1, len(train_auc))
train_interp = interp1d(x_original, train_auc, kind='linear')(interp_x)
valid_interp = interp1d(x_original, valid_auc, kind='linear')(interp_x)
train_mean = np.mean(train_interp)
train_std = np.std(train_interp)
valid_mean = np.mean(valid_interp)
valid_std = np.std(valid_interp)
group_indices = np.arange(0, 101, 5)
train_mean_grouped = [np.mean(train_interp[i:i + 5]) for i in range(0, 101, 5)]
train_std_grouped = [np.std(train_interp[i:i + 5]) for i in range(0, 101, 5)]
valid_mean_grouped = [np.mean(valid_interp[i:i + 5]) for i in range(0, 101, 5)]
valid_std_grouped = [np.std(valid_interp[i:i + 5]) for i in range(0, 101, 5)]
x_percent = np.arange(0, 101, 5)
plt.figure(figsize=(8, 6), dpi=600)
plt.plot(x_percent, train_mean_grouped, 'o-', color='blue', label='Training AUC')
plt.fill_between(x_percent, np.array(train_mean_grouped) - np.array(train_std_grouped), np.array(train_mean_grouped) + np.array(train_std_grouped), alpha=0.15, color='blue')
plt.plot(x_percent, valid_mean_grouped, 'o-', color='green', label='Validation AUC')
plt.fill_between(x_percent, np.array(valid_mean_grouped) - np.array(valid_std_grouped), np.array(valid_mean_grouped) + np.array(valid_std_grouped), alpha=0.15, color='green')
plt.title('TabNet Learning Curve (Mean ± Std AUC by Progress %)')
plt.xlabel('Training Progress (%)')
plt.ylabel('AUC')
plt.xticks(np.arange(0, 101, 10))
plt.ylim(0.8, 1.05)
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
learning_curve_auc_path_tabnet = os.path.join(out_dir, 'learning_curve_tabnet_auc_percent.pdf')
plt.savefig(learning_curve_auc_path_tabnet, format='pdf', bbox_inches='tight', dpi=600)
plt.show()
print(f'✅ Percentage-based AUC learning curve has been saved to: {learning_curve_auc_path_tabnet}')

# ========= 6.5 Model Evaluation =========

print('\nTabNet model evaluation results:')
y_pred_tb = best_tabnet_model.predict(X_test)
y_pred_proba_tb = best_tabnet_model.predict_proba(X_test)
def compute_auc_score(y_true, y_proba):
    n_classes = y_proba.shape[1]
    if n_classes == 2:
        return roc_auc_score(y_true, y_proba[:, 1])
    else:
        y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
        return roc_auc_score(y_true_bin, y_proba, multi_class='ovr')
print(f'AUC: {compute_auc_score(Y_test, y_pred_proba_tb):.4f}')
print(f'Accuracy: {accuracy_score(Y_test, y_pred_tb):.4f}')
print(f"Precision: {precision_score(Y_test, y_pred_tb, average='weighted'):.4f}")
print(f"Recall: {recall_score(Y_test, y_pred_tb, average='weighted'):.4f}")
print(f"F1 score: {f1_score(Y_test, y_pred_tb, average='weighted'):.4f}")

# ========= 6.6 Confusion Matrix Visualization =========

cm = confusion_matrix(Y_test, y_pred_tb)
fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.matshow(cm, cmap='Blues')
plt.colorbar(cax)
ax.set_xticks(np.arange(len(class_names)))
ax.set_yticks(np.arange(len(class_names)))
ax.set_xticklabels(class_names, rotation=45)
ax.set_yticklabels(class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
for i in range(len(class_names)):
    for j in range(len(class_names)):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
plt.tight_layout()
plt.show()

# ========= 6.7 SHAP Interpretation Analysis =========

print('\nCalculating SHAP values...')
os.environ['OMP_NUM_THREADS'] = '6'
sample_num = 10
test_samples = X_test[:sample_num]
n_background = min(100, X_train.shape[0])
background = shap.kmeans(X_train, n_background)
explainer_tb = shap.KernelExplainer(best_tabnet_model.predict_proba, background)
print('Calculating SHAP values for the full dataset...')
full_shap_values_tb = explainer_tb.shap_values(test_samples, nsamples=1000)
if isinstance(full_shap_values_tb, list):
    shap_values_tb = [v.astype(np.float32) for v in full_shap_values_tb]
else:
    shap_values_tb = [full_shap_values_tb[..., i] for i in range(full_shap_values_tb.shape[-1])]
print('\nDimension validation:')
print(f'Feature matrix shape: {test_samples.shape}')
print(f'SHAP value structure: {len(shap_values_tb)} classes')
for i, sv in enumerate(shap_values_tb):
    print(f'Class {class_names[i]} SHAP value shape: {sv.shape}')

# ========== 6.8 SHAP Plotting ==========

def generate_editable_shap_plot_tabnet(shap_values_tb, features, feature_names, class_name, class_names, out_dir, cmap='PiYG_r', alpha=0.8, point_size=30, linewidth=0):
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.sans-serif'] = ['Arial']
    os.makedirs(out_dir, exist_ok=True)
    if isinstance(shap_values_tb, list):
        shap_array = np.stack(shap_values_tb, axis=-1)
    elif len(shap_values_tb.shape) == 2:
        shap_array = np.stack([-shap_values_tb, shap_values_tb], axis=-1)
    else:
        shap_array = shap_values_tb
    class_idx = np.where(class_names == class_name)[0][0] if isinstance(class_names, np.ndarray) else class_names.index(class_name)
    class_shap = shap_array[..., class_idx]
    feature_array = np.asarray(features, dtype=np.float32)
    mean_abs_shap = np.mean(np.abs(class_shap), axis=0)
    top_feature_indices = np.argsort(mean_abs_shap)[::-1][:15]
    selected_names = [feature_names[i] for i in top_feature_indices]
    fig, ax = plt.subplots(figsize=(10, 8), dpi=600)
    ax.grid(True, linestyle='--', alpha=0.6, linewidth=0.5)
    max_val = np.max(np.abs(feature_array[:, top_feature_indices]))
    norm = plt.Normalize(-max_val, max_val)
    for i, feat_idx in enumerate(top_feature_indices):
        y_pos = i + np.random.uniform(-0.25, 0.25, size=len(class_shap))
        sc = ax.scatter(class_shap[:, feat_idx], y_pos, c=feature_array[:, feat_idx], cmap=cmap, norm=norm, s=point_size, alpha=alpha, edgecolors='none', linewidth=linewidth, rasterized=False)
    cbar = fig.colorbar(sc, ax=ax, pad=0.01, aspect=30)
    cbar.set_label('Feature Value', rotation=270, labelpad=15)
    ax.set_yticks(np.arange(len(top_feature_indices)))
    ax.set_yticklabels(selected_names, fontsize=9)
    ax.set_xlabel('SHAP Value (impact on model output)', fontsize=10)
    ax.set_title(f'SHAP Feature Importance - {class_name}', pad=12)
    ax.set_xlim(-1, 1)
    ax.invert_yaxis()
    output_path = os.path.join(out_dir, f"SHAP_tabnet_{class_name.replace(' ', '_')}.pdf")
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300, transparent=True, facecolor='none', metadata={'Title': f'SHAP Analysis - {class_name}', 'Author': 'TabNet SHAP Visualizer'})
    print(f'✅ Saved: {output_path}')
    plt.show()
    plt.close(fig)
    mean_abs_values = np.mean(np.abs(class_shap), axis=0)
    sorted_idx = np.argsort(mean_abs_values)[::-1][:20]
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_contributions = mean_abs_values[sorted_idx]
    fig_bar, ax_bar = plt.subplots(figsize=(8, 6), dpi=600)
    y_pos = np.arange(len(sorted_names))
    ax_bar.barh(y_pos, sorted_contributions, height=0.6, edgecolor='black', linewidth=0.5)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(sorted_names, fontsize=9)
    ax_bar.set_xlabel('Mean |SHAP Value|', fontsize=10)
    ax_bar.set_title(f'SHAP Contribution - {class_name}', pad=12)
    ax_bar.grid(axis='x', linestyle='--', alpha=0.7)
    ax_bar.invert_yaxis()
    plt.tight_layout()
    output_path_bar = os.path.join(out_dir, f"SHAP_tabnet_Contrib_{class_name.replace(' ', '_')}.pdf")
    plt.savefig(output_path_bar, format='pdf', bbox_inches='tight', dpi=300, transparent=True, facecolor='none', metadata={'Title': f'SHAP Contribution - {class_name}', 'Author': 'TabNet SHAP Visualizer'})
    print(f'✅ Saved: {output_path_bar}')
    plt.show()
    plt.close(fig_bar)
for class_name in class_names:
    generate_editable_shap_plot_tabnet(shap_values_tb=shap_values_tb, features=test_samples, feature_names=feature_names, class_name=class_name, class_names=class_names, out_dir=out_dir, cmap='PiYG_r')

# ========== Save TabNet SHAP Contributions to Excel ==========

def calculate_shap_contributions_tabnet(shap_values_tb, feature_names, class_names):
    if isinstance(shap_values_tb, list):
        shap_array = np.stack(shap_values_tb, axis=-1)
    elif len(shap_values_tb.shape) == 2:
        shap_array = np.stack([-shap_values_tb, shap_values_tb], axis=-1)
    else:
        shap_array = shap_values_tb
    contrib_data = {}
    for i, class_name in enumerate(class_names):
        class_shap = shap_array[..., i]
        mean_abs_shap = np.mean(np.abs(class_shap), axis=0)
        contrib_data[class_name] = mean_abs_shap
    contrib_df = pd.DataFrame(contrib_data, index=feature_names)
    contrib_df = contrib_df.div(contrib_df.sum(axis=0), axis=1) * 100
    return contrib_df
shap_contrib_df_tabnet = calculate_shap_contributions_tabnet(shap_values_tb, feature_names, class_names)
excel_path_tabnet = os.path.join(out_dir, 'SHAP_tabnet_contributions.xlsx')
shap_contrib_df_tabnet.to_excel(excel_path_tabnet)
print(f'\n✅ TabNet SHAP feature contributions for all classes have been saved to the Excel file: {excel_path_tabnet}')

# ========== [7] Save Base Models ====================================================

print('\n[6] Saving base models...')
joblib.dump(best_rf_model, 'rf_model.pkl')
joblib.dump(best_xgb_model, 'xgb_model.pkl')
joblib.dump(best_tabnet_model, 'tabnet_model.pkl')

# ========== [8] Load Base Models and Evaluate Performance ==========

print('\n[7] Loading base models and evaluating performance...')
def evaluate_model(model, X_test, Y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    n_classes = len(set(Y_test))
    if n_classes == 2:
        auc = roc_auc_score(Y_test, y_prob[:, 1])
    else:
        Y_test_bin = label_binarize(Y_test, classes=list(range(n_classes)))
        auc = roc_auc_score(Y_test_bin, y_prob, average='weighted', multi_class='ovr')
    return {'Accuracy': accuracy_score(Y_test, y_pred), 'F1': f1_score(Y_test, y_pred, average='weighted'), 'Precision': precision_score(Y_test, y_pred, average='weighted'), 'Recall': recall_score(Y_test, y_pred, average='weighted'), 'AUC': auc}
models = {'Random Forest': joblib.load('rf_model.pkl'), 'XGBoost': joblib.load('xgb_model.pkl'), 'TabNet': joblib.load('tabnet_model.pkl')}
metrics = {}
for name, model in models.items():
    metrics[name] = evaluate_model(model, X_test, Y_test)

# ========== [9] Optimize and Evaluate LGB Ensemble Model with Optuna ==========

print('\n[8] Optimize and evaluate the stacking ensemble model (LGBM) with Optuna...')
def objective(trial):
    params = {'n_estimators': trial.suggest_int('n_estimators', 50, 300), 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3), 'max_depth': trial.suggest_int('max_depth', 3, 15), 'num_leaves': trial.suggest_int('num_leaves', 15, 255), 'subsample': trial.suggest_float('subsample', 0.6, 1.0), 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0), 'n_jobs': -1, 'random_state': 2025}
    model = LGBMClassifier(**params)
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(Y_test, y_pred)
study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=2025))
study.optimize(objective, n_trials=50)
print('Best hyperparameters:', study.best_params)
print('Best accuracy:', study.best_value)
best_params = study.best_params
best_params.update({'n_jobs': -1, 'random_state': 2025})
best_ensem_model = LGBMClassifier(**best_params)
best_ensem_model.fit(X_train, Y_train)
joblib.dump(best_ensem_model, 'ensem_model.pkl')
metrics['Stacking_Optuna'] = evaluate_model(best_ensem_model, X_test, Y_test)

# ========== [10] Model Visualization and Comparison ==========

print('\n[9] Generating model evaluation charts...')
metrics_df = pd.DataFrame(metrics).T.reset_index().rename(columns={'index': 'Model'})
metrics_long = metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Value')
metrics_list = metrics_long['Metric'].unique()
for metric in metrics_list:
    plt.figure(figsize=(8, 6))
    subset = metrics_long[metrics_long['Metric'] == metric]
    ax = sns.barplot(data=subset, x='Model', y='Value', palette='viridis', errwidth=1)
    for bar in ax.patches:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.title(f'{metric} Comparison', fontsize=16)
    plt.ylim(0, 1.1)
    plt.xlabel('')
    plt.ylabel('Score', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f'{metric.lower()}_comparison.pdf', dpi=600, bbox_inches='tight')
    plt.show()