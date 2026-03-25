import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import joblib
import threading
import sys  # Import sys module
from PIL import Image, ImageTk
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk, ImageDraw
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# Global color scheme
COLORS = {
    "background": "#7e779d",
    "card": "#9d96be",
    "accent": "#F2A7C7",
    "text": "#FFFFFF",
    "text_muted": "#FFFFFF",
    "success": "#a1bae7",
    "button": "#a1bae7",
    "grey": "#484554",
    "button_hover": "#7e779d"
}

class ModelPredictorApp:
    def __init__(self):
        # Initialize main window with drag-and-drop support
        self.root = TkinterDnD.Tk()
        self.root.title("Meta Cassiterite_1.0")
        self.root.geometry("1200x900")
        self.root.tk.call('tk', 'scaling', 1.5)
        self.root.configure(bg=COLORS["background"])

        # Initialize model containers
        self.models = {
            "Random Forest": None,
            "XGBoost": None,
            "TabNet": None,
            "Stacking": None
        }
        self.models_loaded = False

        # Data-related attributes
        self.data = None
        self.processed_data = None
        self.prediction_results = None
        self.feature_names = None
        self.sample_ids = None

        # Element feature list
        self.element_features = ['Al', 'Sc', 'Ti', 'V', 'Fe', 'Ga', 'W', 'Sb', 'Zr', 'Hf', 'Nb', 'Ta', 'U']
        # Derived feature list
        self.derived_features = ['ZrHf', 'NbTa', 'SbW', 'FeAl', 'UHf', 'UZr']

        # Build UI
        self.create_layout()

        # Create loading overlay (hidden initially)
        self.create_loading_overlay()

    def resource_path(self, relative_path):
        """Get absolute path for resources (supports PyInstaller packaging)"""
        try:
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)

    def create_layout(self):
        """Create main UI layout"""
        header = tk.Frame(self.root, bg=COLORS["background"], height=60)
        header.pack(fill=tk.X)

        title_label = tk.Label(header, text="Meta Cassiterite",
                              font=("Arial", 30, "bold"),
                              fg="#2b2848", bg=COLORS["background"])
        title_label.place(relx=0.5, rely=0.5, anchor="center")

        main_frame = tk.Frame(self.root, bg=COLORS["background"])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Left control panel
        left_frame = tk.Frame(main_frame, bg=COLORS["card"], width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        upload_label = tk.Label(left_frame, text="💎 Upload",
                               font=("Arial", 20, "bold"),
                               fg=COLORS["text"], bg=COLORS["card"])
        upload_label.pack(anchor=tk.W, padx=50, pady=(15, 10))

        # Drag-and-drop area
        self.drop_area = tk.Frame(left_frame, bg=COLORS["text"], height=150, width=270)
        self.drop_area.pack(fill=tk.X, padx=12, pady=10)
        self.drop_area.pack_propagate(False)

        self.drop_text = tk.Label(self.drop_area,
                                 text="Drop file here (xls, csv, xlsx)\n or \nClick to upload",
                                 font=("Arial", 12, "bold"),
                                 fg=COLORS["grey"], bg=COLORS["text"])
        self.drop_text.pack(expand=True)

        # Register drag-and-drop
        self.drop_area.drop_target_register(DND_FILES)
        self.drop_area.dnd_bind('<<Drop>>', self.on_drop)

        # File selection button
        browse_btn = tk.Button(left_frame, text="Open file", bg=COLORS["button"],
                              fg=COLORS["text"], font=("Arial", 15, "bold"),
                              relief=tk.FLAT, padx=10, pady=5,
                              command=self.browse_file)
        browse_btn.pack(fill=tk.X, padx=15, pady=10)

        # File info display
        self.file_var = tk.StringVar(value="No File Selected")
        file_info = tk.Label(left_frame, textvariable=self.file_var,
                            fg=COLORS["text_muted"], bg=COLORS["card"],
                            font=("Arial", 15, "bold"), wraplength=270)
        file_info.pack(fill=tk.X, padx=15, pady=5)

        separator = ttk.Separator(left_frame, orient="horizontal")
        separator.pack(fill=tk.X, padx=15, pady=15)

        # Processing section
        model_label = tk.Label(left_frame, text="💎 Process",
                              font=("Arial", 20, "bold"),
                              fg=COLORS["text"], bg=COLORS["card"])
        model_label.pack(anchor=tk.W, padx=50, pady=(5, 10))

        # Prediction button
        predict_btn = tk.Button(left_frame, text="Prediction", bg=COLORS["success"],
                               fg=COLORS["text"], font=("Arial", 15, "bold"),
                               relief=tk.FLAT, padx=10, pady=8,
                               command=self.run_prediction)
        predict_btn.pack(fill=tk.X, padx=15, pady=10)

        # Export buttons
        export_btn = tk.Button(left_frame, text="Export Results", bg=COLORS["accent"],
                              fg=COLORS["text"], font=("Arial", 15, "bold"),
                              relief=tk.FLAT, padx=10, pady=5,
                              command=self.export_results)
        export_btn.pack(fill=tk.X, padx=15, pady=5)

        export_processed_btn = tk.Button(left_frame, text="Export Processed Data",
                                        bg=COLORS["grey"],
                                        fg=COLORS["text"], font=("Arial", 15, "bold"),
                                        relief=tk.FLAT, padx=10, pady=5,
                                        command=self.export_processed_data)
        export_processed_btn.pack(fill=tk.X, padx=15, pady=5)

        # Status indicator
        self.status_var = tk.StringVar(value="Ready")
        status_label = tk.Label(left_frame, textvariable=self.status_var,
                               font=("Arial", 10, "bold"),
                               fg=COLORS["text_muted"], bg=COLORS["card"])
        status_label.pack(anchor=tk.W, padx=15, pady=(20, 5))

        # Right panel (results display)
        right_frame = tk.Frame(main_frame, bg=COLORS["background"])
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        results_label = tk.Label(right_frame, text="Results",
                                font=("Arial", 20, "bold"),
                                fg=COLORS["text"], bg=COLORS["background"])
        results_label.pack(anchor=tk.W, padx=350, pady=10)

        table_frame = tk.Frame(right_frame, bg=COLORS["card"])
        table_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=10)

        # Scrollbars
        y_scroll = ttk.Scrollbar(table_frame)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        x_scroll = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL)
        x_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        # Treeview for results
        self.result_tree = ttk.Treeview(table_frame,
                                       yscrollcommand=y_scroll.set,
                                       xscrollcommand=x_scroll.set)
        self.result_tree.pack(fill=tk.BOTH, expand=True)

        y_scroll.config(command=self.result_tree.yview)
        x_scroll.config(command=self.result_tree.xview)

        # Style configuration
        style = ttk.Style()
        style.configure("Treeview",
                       background=COLORS["card"],
                       foreground=COLORS["text"],
                       fieldbackground=COLORS["card"],
                       borderwidth=0)
        style.configure("Treeview.Heading",
                       background=COLORS["background"],
                       foreground=COLORS["accent"],
                       font=("Arial", 10, "bold"))

        # Bottom status bar
        self.status_bar = tk.Label(self.root, text="Ready",
                                  font=("Arial", 10, "bold"),
                                  bg="#7e779d", fg=COLORS["text_muted"],
                                  anchor=tk.W, padx=15)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_loading_overlay(self):
        """Create loading overlay with progress bar"""
        self.loading_frame = tk.Frame(self.root, bg=COLORS["background"])
        self.loading_frame.place(x=0, y=0, relwidth=1, relheight=1)

        loading_text = tk.Label(self.loading_frame, text="In processing, please wait...",
                               font=("Arial", 16, "bold"),
                               fg=COLORS["accent"], bg=COLORS["background"])
        loading_text.place(relx=0.5, rely=0.4, anchor=tk.CENTER)

        progress = ttk.Progressbar(self.loading_frame, mode="indeterminate", length=400)
        progress.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        progress.start()

        self.loading_frame.place_forget()

    def on_drop(self, event):
        """Handle file drag-and-drop event"""
        file_path = event.data

        # Normalize Windows path format
        if file_path.startswith("{") and file_path.endswith("}"):
            file_path = file_path[1:-1]
        if file_path.startswith('"') and file_path.endswith('"'):
            file_path = file_path[1:-1]

        # Use first file if multiple are dropped
        if " " in file_path:
            file_path = file_path.split(" ")[0]

        # Validate file type
        if file_path.lower().endswith(('.csv', '.xlsx', '.xls')):
            self.load_data(file_path)
        else:
            messagebox.showerror("Error", "Unsupported file format. Please upload a CSV or Excel file.")

    def browse_file(self):
        """Open file dialog to select input data"""
        file_path = filedialog.askopenfilename(
            title="Select data file",
            filetypes=[
                ("All files", "*.*"),
                ("Excel file", "*.xlsx *.xls"),
                ("CSV file", "*.csv")
            ]
        )

        if file_path:
            self.load_data(file_path)

    def apply_log10_transform(self, df):
        """Apply log10 transform to element features while preserving missing values"""
        transformed_df = df.copy()
        for col in self.element_features:
            if col in transformed_df.columns:
                mask = transformed_df[col].notna() & (transformed_df[col] > 0)
                transformed_df.loc[mask, col] = np.log10(transformed_df.loc[mask, col])
        return transformed_df

    def impute_missing_values(self, df):
        """Impute missing values using multiple Random Forest imputations and averaging"""
        cols_to_impute = [col for col in self.element_features if col in df.columns]
        if not cols_to_impute:
            return df

        data_to_impute = df[cols_to_impute].copy()

        if not data_to_impute.isna().any().any():
            return df

        imputed_data_list = []
        for i in range(10):
            estimator = RandomForestRegressor(
                n_estimators=50,
                max_depth=7,
                n_jobs=-1,
                random_state=2025+i
            )

            imp = IterativeImputer(
                estimator=estimator,
                max_iter=15,
                random_state=2025+i,
                tol=0.01
            )

            imputed_data = imp.fit_transform(data_to_impute)
            imputed_data_list.append(imputed_data)

        imputed_avg = np.mean(imputed_data_list, axis=0)

        imputed_df = df.copy()
        imputed_df[cols_to_impute] = imputed_avg

        return imputed_df

    def create_derived_features(self, df):
        """Generate derived geochemical features"""
        df_with_derived = df.copy()

        for col in self.element_features:
            if col not in df_with_derived.columns:
                self.status_bar.config(text=f"Warning: {col} column not found.")

        if 'Zr' in df.columns and 'Hf' in df.columns:
            df_with_derived['ZrHf'] = df['Zr'] - df['Hf']

        if 'Nb' in df.columns and 'Ta' in df.columns:
            df_with_derived['NbTa'] = df['Nb'] - df['Ta']

        if 'Sb' in df.columns and 'W' in df.columns:
            df_with_derived['SbW'] = df['Sb'] - df['W']

        if 'Fe' in df.columns and 'Al' in df.columns:
            df_with_derived['FeAl'] = df['Fe'] - df['Al']

        if 'U' in df.columns and 'Hf' in df.columns:
            df_with_derived['UHf'] = df['U'] - df['Hf']

        if 'U' in df.columns and 'Zr' in df.columns:
            df_with_derived['UZr'] = df['U'] - df['Zr']

        return df_with_derived

# Remaining code unchanged except comments translated where present

if __name__ == "__main__":
    app = ModelPredictorApp()
    app.run()
