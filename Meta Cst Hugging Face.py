import os
import pandas as pd
import numpy as np
import joblib
import gradio as gr
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import torch
from pytorch_tabnet.tab_model import TabNetClassifier

# 全局配置
ELEMENT_FEATURES = ['Al', 'Sc', 'Ti', 'V', 'Fe', 'Ga', 'W', 'Sb', 'Zr', 'Hf', 'Nb', 'Ta', 'U']
DERIVED_FEATURES = ['Zr/Hf', 'Nb/Ta', 'Sb/W', 'Fe/Al', 'U/Hf', 'U/Zr']

class MetaCassiterite:
    def __init__(self):
        self.models = {
            "Random Forest": None,
            "XGBoost": None,
            "TabNet": None,
            "Ensemble": None
        }
        self.load_models()
        self.processed_data = None
        self.prediction_results = None
        self.sample_ids = None

    def resource_path(self, relative_path):
        """ 获取资源的绝对路径 """
        base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)

    def load_models(self):
        """加载所有预训练模型"""
        try:
            self.models["Random Forest"] = joblib.load(self.resource_path('best_rf_model.pkl'))
            self.models["XGBoost"] = joblib.load(self.resource_path('best_xgb_model.pkl'))
            self.models["TabNet"] = joblib.load(self.resource_path('best_tabnet_model.pkl'))
            self.models["Ensemble"] = joblib.load(self.resource_path('best_ensem_model.pkl'))
            return True
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            print("Hint: If you see sklearn/xgboost version warnings, they are usually non-fatal. "
                  "If prediction fails, please align runtime versions with the training environment.")
            return False

    def apply_log10_transform(self, df):
        """对元素特征进行Log10变换"""
        transformed_df = df.copy()
        for col in ELEMENT_FEATURES:
            if col in transformed_df.columns:
                mask = transformed_df[col].notna() & (transformed_df[col] > 0)
                transformed_df.loc[mask, col] = np.log10(transformed_df.loc[mask, col])
        return transformed_df

    def impute_missing_values(self, df):
        """多重插补缺失值"""
        cols_to_impute = [col for col in ELEMENT_FEATURES if col in df.columns]
        if not cols_to_impute:
            return df

        data_to_impute = df[cols_to_impute].copy()
        if not data_to_impute.isna().any().any():
            return df

        imputed_data_list = []
        for i in range(5):
            estimator = RandomForestRegressor(
                n_estimators=50, max_depth=7,
                n_jobs=-1, random_state=2025+i
            )
            imp = IterativeImputer(
                estimator=estimator, max_iter=15,
                random_state=2025+i, tol=0.01
            )
            imputed_data = imp.fit_transform(data_to_impute)
            imputed_data_list.append(imputed_data)

        imputed_avg = np.mean(imputed_data_list, axis=0)
        imputed_df = df.copy()
        imputed_df[cols_to_impute] = imputed_avg
        return imputed_df

    def create_derived_features(self, df):
        """创建派生特征"""
        df_with_derived = df.copy()

        if 'Zr' in df.columns and 'Hf' in df.columns:
            df_with_derived['Zr/Hf'] = df['Zr'] - df['Hf']
        if 'Nb' in df.columns and 'Ta' in df.columns:
            df_with_derived['Nb/Ta'] = df['Nb'] - df['Ta']
        if 'Sb' in df.columns and 'W' in df.columns:
            df_with_derived['Sb/W'] = df['Sb'] - df['W']
        if 'Fe' in df.columns and 'Al' in df.columns:
            df_with_derived['Fe/Al'] = df['Fe'] - df['Al']
        if 'U' in df.columns and 'Hf' in df.columns:
            df_with_derived['U/Hf'] = df['U'] - df['Hf']
        if 'U' in df.columns and 'Zr' in df.columns:
            df_with_derived['U/Zr'] = df['U'] - df['Zr']

        return df_with_derived

    def process_data(self, file_path):
        """处理上传的数据"""
        try:
            # 读取文件
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)

            # 识别样本ID
            possible_id_cols = ['ID', 'Sample_ID', 'SampleID', 'Sample', 'Sample ID', 'id']
            self.sample_id_column = next(
                (col for col in possible_id_cols if col in df.columns), None
            )
            self.sample_ids = df[self.sample_id_column].astype(str).tolist() if self.sample_id_column else [
                f"Sample_{i+1}" for i in range(len(df))]

            # 数据处理流程
            log_data = self.apply_log10_transform(df)
            imputed_data = self.impute_missing_values(log_data)
            processed_data = self.create_derived_features(imputed_data)

            # 准备特征数据
            feature_cols = [col for col in ELEMENT_FEATURES + DERIVED_FEATURES if col in processed_data.columns]
            X = processed_data[feature_cols].values
            self.processed_data = X
            self.feature_names = feature_cols

            return processed_data, "\n".join([
                f"Loading Successfully: {os.path.basename(file_path)}",
                f"Sample X {len(df)}",
                f"Features X {len(feature_cols)}"
            ])

        except Exception as e:
            return None, f"Failed to process: {str(e)}"

    def predict(self):
        """执行预测"""
        if self.processed_data is None:
            return None, "Please upload and process data"

        try:
            results = {"Sample_ID": self.sample_ids}
            class_map = {0: 'A', 1: 'B'}  # 根据你的实际类别修改

            # 随机森林预测
            rf_pred = self.models["Random Forest"].predict(self.processed_data)
            rf_proba = self.models["Random Forest"].predict_proba(self.processed_data)
            results["RF_Class"] = [class_map[p] for p in rf_pred]
            results["RF_Prob"] = np.max(rf_proba, axis=1)

            # XGBoost预测
            xgb_pred = self.models["XGBoost"].predict(self.processed_data)
            xgb_proba = self.models["XGBoost"].predict_proba(self.processed_data)
            results["XGB_Class"] = [class_map[p] for p in xgb_pred]
            results["XGB_Prob"] = np.max(xgb_proba, axis=1)

            # TabNet预测
            tabnet_pred = self.models["TabNet"].predict(self.processed_data)
            tabnet_proba = self.models["TabNet"].predict_proba(self.processed_data)
            results["TabNet_Class"] = [class_map[p] for p in tabnet_pred]
            results["TabNet_Prob"] = np.max(tabnet_proba, axis=1)

            # 集成预测
            stack_pred = self.models["Ensemble"].predict(self.processed_data)
            stack_proba = self.models["Ensemble"].predict_proba(self.processed_data)
            results["Ensemble_Class"] = [class_map[p] for p in stack_pred]
            results["Ensemble_Prob"] = np.max(stack_proba, axis=1)

            # 创建结果DataFrame
            self.prediction_results = pd.DataFrame(results)
            return self.prediction_results, "Prediction Finished"

        except Exception as e:
            return None, f"Error in prediciton: {str(e)}"
            
    def export_results(self, format_type="excel"):
        """Export Results (Excel or CSV)"""
        if self.prediction_results is None:
            return None, "No Results for Exporting，Please Finish Prediction"
            
        try:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            
            if format_type == "excel":
                output_file = f"Prediction_Results_{timestamp}.xlsx"
                self.prediction_results.to_excel(output_file, index=False)
            else:
                output_file = f"Prediction_Results_{timestamp}.csv"
                self.prediction_results.to_csv(output_file, index=False)
                
            return output_file, f"Exported successfully as {output_file}"
        except Exception as e:
            return None, f"Export Error: {str(e)}"

# 初始化系统
system = MetaCassiterite()

# Gradio界面
custom_css = """
    .title-text {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .main-button {
        width: 100%;
        padding: 15px;
        font-size: 16px;
        margin: 10px 0;
    }
    #三列模块
    .tab-nav button {
    font-size: 36px !important;
    font-weight: bold !important;
    flex: 1;
    padding: 12px !important;
    color: #9075ad !important; 
}
"""

with gr.Blocks(title="Meta Cassiterite 1.0", css=custom_css) as demo:
    gr.HTML("""
    <div class="title-text">
        💎 Meta Cassiterite 1.0 💎
    </div>
    """)

    with gr.Tabs() as tabs:
        with gr.Row():
            with gr.Column(scale=1):
                with gr.TabItem("📁 Upload & Process Data") as tab1:
                    file_input = gr.File(label="Upload file (CSV/Excel)")
                    process_btn = gr.Button("Process Data", elem_classes=["main-button"])
                    processed_data = gr.Dataframe(label="Processed Data", interactive=False)
                    process_output = gr.Textbox(label="Processing Status")

            with gr.Column(scale=1):
                with gr.TabItem("🔮 Prediction") as tab2:
                    predict_btn = gr.Button("Start Prediction", elem_classes=["main-button"])
                    prediction_results = gr.Dataframe(label="Predicted Results")
                    predict_output = gr.Textbox(label="Prediction Status")

            with gr.Column(scale=1):
                with gr.TabItem("💾 Export") as tab3:
                    with gr.Row():
                        excel_btn = gr.Button("Export Excel File", variant="primary", elem_classes=["main-button"])
                        csv_btn = gr.Button("Export CSV File", elem_classes=["main-button"])
                    export_output = gr.Textbox(label="Exporting Status")
                    export_download = gr.File(label="Download File")

    # 事件处理
    process_btn.click(
        fn=system.process_data,
        inputs=file_input,
        outputs=[processed_data, process_output]
    )

    predict_btn.click(
        fn=system.predict,
        outputs=[prediction_results, predict_output]
    )

    excel_btn.click(
        fn=lambda: system.export_results("excel"),
        outputs=[export_download, export_output]
    )
    
    csv_btn.click(
        fn=lambda: system.export_results("csv"),
        outputs=[export_download, export_output]
    )
    
    # Add the notes at the bottom
    gr.Markdown("""
        ## Notes:
        1. Please input element concentrations in order of Al, Sc, Ti, V, Fe, Ga, W, Sb, Zr, Hf, Nb, Ta, and U, and delete BDL and NA characters
        2. LOG10 transformation and imputation of element concentrations will be automatically processed
        3. Missing values will be imputed using Random Forest (average of 5 iterations)
        4. Prediction using multiple models (Random Forest, XGBoost, TabNet, Ensemble)
    """)

if __name__ == "__main__":
    # 在部署环境中显式指定监听地址/端口，避免容器启动后外部无法访问
    port = int(os.getenv("PORT", "7860"))
    # 关闭 SSR 实验特性，减少环境兼容性问题；如需开启可设置为 True
    demo.launch(server_name="0.0.0.0", server_port=port, ssr_mode=False, show_error=True)
