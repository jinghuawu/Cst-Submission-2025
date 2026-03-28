import os
import pandas as pd
import numpy as np
import joblib
import gradio as gr
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

ELEMENT_FEATURES = ['Al','Sc','Ti','V','Fe','Ga','W','Sb','Zr','Hf','Nb','Ta','U']
DERIVED_FEATURES = ['Zr/Hf','Nb/Ta','Sb/W','Fe/Al','U/Hf','U/Zr']

class MetaCassiterite:
    def __init__(self):
        self.models = {
            "Random Forest": None,
            "XGBoost": None,
            "TabNet": None,
            "Ensemble": None
        }
        self.processed_df = None
        self.processed_data = None
        self.prediction_results = None
        self.sample_ids = None
        self.sample_id_column = None
        self.feature_names = None
        self.load_models()

    def resource_path(self, relative_path: str) -> str:
        return os.path.join(os.path.abspath("."), relative_path)

    def load_models(self) -> bool:
        try:
            self.models["Random Forest"] = joblib.load(self.resource_path('best_rf_model.pkl'))
            self.models["XGBoost"] = joblib.load(self.resource_path('best_xgb_model.pkl'))
            self.models["TabNet"] = joblib.load(self.resource_path('best_tabnet_model.pkl'))
            self.models["Ensemble"] = joblib.load(self.resource_path('best_ensem_model.pkl'))
            return True
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return False

    def apply_log10_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        transformed_df = df.copy()
        for col in ELEMENT_FEATURES:
            if col in transformed_df.columns:
                transformed_df[col] = pd.to_numeric(transformed_df[col], errors='coerce')
                mask = transformed_df[col].notna() & (transformed_df[col] > 0)
                transformed_df.loc[mask, col] = np.log10(transformed_df.loc[mask, col])
        return transformed_df

    def impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_impute = [col for col in ELEMENT_FEATURES if col in df.columns]
        if not cols_to_impute:
            return df

        data_to_impute = df[cols_to_impute].copy()
        if not data_to_impute.isna().any().any():
            return df

        imputed_data_list = []
        for i in range(5):
            estimator = RandomForestRegressor(
                n_estimators=50,
                max_depth=7,
                n_jobs=-1,
                random_state=2025 + i
            )
            imp = IterativeImputer(
                estimator=estimator,
                max_iter=15,
                random_state=2025 + i,
                tol=0.01
            )
            imputed_data = imp.fit_transform(data_to_impute)
            imputed_data_list.append(imputed_data)

        imputed_avg = np.mean(imputed_data_list, axis=0)
        imputed_df = df.copy()
        imputed_df[cols_to_impute] = imputed_avg
        return imputed_df

    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def process_data(self, file_obj):
        if file_obj is None:
            return None, "Please upload a file."

        try:
            file_path = file_obj.name if hasattr(file_obj, 'name') else file_obj

            if str(file_path).endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)

            possible_id_cols = ['ID','Sample_ID','SampleID','Sample','Sample ID','id']
            self.sample_id_column = next((col for col in possible_id_cols if col in df.columns), None)
            self.sample_ids = (
                df[self.sample_id_column].astype(str).tolist()
                if self.sample_id_column else
                [f"Sample_{i+1}" for i in range(len(df))]
            )

            log_data = self.apply_log10_transform(df)
            imputed_data = self.impute_missing_values(log_data)
            processed_data = self.create_derived_features(imputed_data)

            feature_cols = [col for col in ELEMENT_FEATURES + DERIVED_FEATURES if col in processed_data.columns]
            self.processed_df = processed_data
            self.processed_data = processed_data[feature_cols].values
            self.feature_names = feature_cols

            msg = "\n".join([
                f"Loading Successfully: {os.path.basename(str(file_path))}",
                f"Sample X {len(df)}",
                f"Features X {len(feature_cols)}"
            ])
            return processed_data, msg

        except Exception as e:
            return None, f"Failed to process: {str(e)}"

    def predict(self):
        if self.processed_data is None:
            return None, "Please upload and process data"

        try:
            results = {"Sample_ID": self.sample_ids}
            class_map = {0: 'A', 1: 'B'}

            rf_pred = self.models["Random Forest"].predict(self.processed_data)
            rf_proba = self.models["Random Forest"].predict_proba(self.processed_data)
            results["RF_Class"] = [class_map[p] for p in rf_pred]
            results["RF_Prob"] = np.max(rf_proba, axis=1)

            xgb_pred = self.models["XGBoost"].predict(self.processed_data)
            xgb_proba = self.models["XGBoost"].predict_proba(self.processed_data)
            results["XGB_Class"] = [class_map[p] for p in xgb_pred]
            results["XGB_Prob"] = np.max(xgb_proba, axis=1)

            tabnet_pred = self.models["TabNet"].predict(self.processed_data)
            tabnet_proba = self.models["TabNet"].predict_proba(self.processed_data)
            results["TabNet_Class"] = [class_map[p] for p in tabnet_pred]
            results["TabNet_Prob"] = np.max(tabnet_proba, axis=1)

            stack_pred = self.models["Ensemble"].predict(self.processed_data)
            stack_proba = self.models["Ensemble"].predict_proba(self.processed_data)
            results["Ensemble_Class"] = [class_map[p] for p in stack_pred]
            results["Ensemble_Prob"] = np.max(stack_proba, axis=1)

            self.prediction_results = pd.DataFrame(results)
            return self.prediction_results, "Prediction Finished"

        except Exception as e:
            return None, f"Error in prediction: {str(e)}"

    def export(self, ftype):
        if self.prediction_results is None:
            return None
        path = f"Results.{ftype}"
        if ftype == "xlsx":
            self.prediction_results.to_excel(path, index=False)
        else:
            self.prediction_results.to_csv(path, index=False)
        return path


system = MetaCassiterite()

# =========================
bg_canvas_html = """
<div id="canvas-wrapper">
    <canvas id="hero-canvas"></canvas>
</div>
<script>
(function() {
    const forceDark = () => {
        document.documentElement.classList.add('dark');
        document.body.classList.add('dark');
    };
    forceDark();
    setInterval(forceDark, 1000);

    const canvas = document.getElementById('hero-canvas');
    const ctx = canvas.getContext('2d');
    let pts = [];
    const resize = () => { canvas.width = window.innerWidth; canvas.height = window.innerHeight; };
    window.addEventListener('resize', resize);
    resize();

    class Particle {
        constructor() {
            this.x = Math.random() * canvas.width;
            this.y = Math.random() * canvas.height;
            this.vx = (Math.random() - 0.5) * 1.5;
            this.vy = (Math.random() - 0.5) * 1.5;
        }
        update() {
            this.x += this.vx; this.y += this.vy;
            if(this.x<0||this.x>canvas.width) this.vx*=-1;
            if(this.y<0||this.y>canvas.height) this.vy*=-1;
        }
    }
    for(let i=0; i<100; i++) pts.push(new Particle());

    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "rgba(0, 242, 254, 0.6)";
        pts.forEach(p => {
            p.update();
            ctx.beginPath(); ctx.arc(p.x, p.y, 2, 0, Math.PI*2); ctx.fill();
        });
        ctx.strokeStyle = "rgba(0, 242, 254, 0.15)";
        ctx.lineWidth = 1;
        for(let i=0; i<pts.length; i++) {
            for(let j=i+1; j<pts.length; j++) {
                let d = Math.hypot(pts[i].x-pts[j].x, pts[i].y-pts[j].y);
                if(d < 180) {
                    ctx.beginPath(); ctx.moveTo(pts[i].x, pts[i].y);
                    ctx.lineTo(pts[j].x, pts[j].y); ctx.stroke();
                }
            }
        }
        requestAnimationFrame(animate);
    }
    animate();
})();
</script>
<style>
#canvas-wrapper {
    position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
    z-index: -1; background: radial-gradient(circle at center, #0a1128 0%, #020617 100%) !important;
}
</style>
"""

custom_css = """
.gradio-container { background: transparent !important; }
:root, .dark {
    --body-background-fill: transparent !important;
    --background-fill-primary: transparent !important;
}
.main-panel {
    background: rgba(16, 24, 48, 0.85) !important;
    border: 3px solid rgba(0, 242, 254, 0.6) !important;
    border-radius: 24px !important;
    backdrop-filter: blur(15px) saturate(180%);
    padding: 30px !important;
    box-shadow: 0 0 25px rgba(0, 242, 254, 0.15) !important;
    margin: 10px !important;
}
.glow-title {
    font-size: 4rem !important; font-weight: 900 !important; text-align: center;
    background: linear-gradient(90deg, #00f2fe, #4facfe, #7000ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 0 20px rgba(0, 242, 254, 0.6));
    margin: 40px 0 !important;
}
.tech-btn {
    background: linear-gradient(135deg, #00f2fe 0%, #4facfe 50%, #7000ff 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 800 !important;
    border-radius: 16px !important;
    box-shadow: 0 6px 20px rgba(79, 172, 254, 0.4) !important;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
}
.tech-btn:hover {
    transform: translateY(-3px) scale(1.03);
    box-shadow: 0 12px 30px rgba(112, 0, 255, 0.6) !important;
    filter: brightness(1.2);
}
.gr-markdown { color: #e0f2fe !important; }
.gr-markdown h2, .gr-markdown h3 { color: #00f2fe !important; }
table {
    border-collapse: collapse !important;
    width: 100% !important;
    margin: 20px 0 !important;
    background: rgba(10, 20, 40, 0.4) !important;
    border: 1px solid rgba(0, 242, 254, 0.3) !important;
}
th, td {
    border: 1px solid rgba(0, 242, 254, 0.2) !important;
    padding: 10px !important;
    text-align: center !important;
}
th { background: rgba(0, 242, 254, 0.1) !important; color: #00f2fe !important; }
footer { display: none !important; }
"""

with gr.Blocks(title="Cassiterite X | GeoAI", css=custom_css) as demo:
    gr.HTML(bg_canvas_html)
    gr.HTML("<div class='glow-title'> Cassiterite X </div>")

    with gr.Row():
        with gr.Column(elem_classes=["main-panel"], scale=2):
            gr.Markdown("### 📥 Data Upload")
            file_input = gr.File(label="Upload Dataset (XLSX / CSV)")
            process_btn = gr.Button("Data pre-process", elem_classes="tech-btn")
            processed_view = gr.Dataframe(interactive=False)
            process_log = gr.Textbox(label="Processing Log")

        with gr.Column(elem_classes=["main-panel"], scale=1):
            gr.Markdown("### 🧠 AI Prediction")
            predict_btn = gr.Button("Run prediction", elem_classes="tech-btn")
            prediction_view = gr.Dataframe(interactive=False)
            predict_log = gr.Textbox(label="Status Report")

        with gr.Column(elem_classes=["main-panel"], scale=1):
            gr.Markdown("### 📦 Data Export")
            export_xlsx = gr.Button("Export XLSX", elem_classes="tech-btn")
            export_csv = gr.Button("Export CSV", elem_classes="tech-btn")
            download_file = gr.File(label="Download Ready")

    with gr.Row(elem_classes=["main-panel"]):
        gr.Markdown(
            """
## 📝 Notes
1. **Input order**: Please input element concentrations in order of **Al, Sc, Ti, V, Fe, Ga, W, Sb, Zr, Hf, Nb, Ta, and U**. Ensure you delete BDL and NA characters.
2. **Auto-processing**: LOG10 transformation and imputation of element concentrations are handled automatically.
3. **Imputation logic**: Missing values are imputed using **Random Forest Iterative Imputer** (average of 5 iterations).
4. **Ensemble prediction**: Predictions are synthesized from multiple models: Random Forest, XGBoost, TabNet, and Ensemble.

 ### 📊 Table Example
    
| Sample_ID | Al | Sc | Ti | V | Fe | Ga | W | Sb | Zr | Hf | Nb | Ta | U |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| A-1 | 78.2 | 7454.1 | 16.1 | 9996.3 | 2270.0 | 31.6 | 893.2 | 277.0 | 171.8 | 700.6 | 73.7 | 3367.8 | 13.1 |
| A-2 | | | 20.7 | 9863.4 | 1241.9 | | | 250.5 | 156.1 | 707.6 | 53.6 | 1359.8 | 453.1 |
| A-3 | 21.2 | 3057.7 | | 1166.4 | 1834.7 | | 664.8 | 153.4 | | 375.7 | 44.5 | 2567.8 | 463.1 |
| B-1 | 46.4 | | 12.3 | 8179.5 | 1055.9 | 14.9 | 497.6 | 547.3 | 286.8 | 135.6 | 563.7 | | 253.1 |
            """
        )

    process_btn.click(system.process_data, inputs=file_input, outputs=[processed_view, process_log])
    predict_btn.click(system.predict, outputs=[prediction_view, predict_log])
    export_xlsx.click(lambda: system.export("xlsx"), outputs=download_file)
    export_csv.click(lambda: system.export("csv"), outputs=download_file)

if __name__ == "__main__":
    demo.launch()
