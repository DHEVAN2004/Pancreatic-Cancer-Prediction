"""Pancreatic Cancer Risk Prediction System

This application provides a simple GUI to predict the risk of pancreatic cancer
using a pre-trained K-Nearest Neighbors (KNN) model and a StandardScaler.

If the model and scaler files are not found, the script will attempt to train a
simple KNN model using the provided dataset (pancreatic_600_risk123.csv) and save
those artifacts for future use.

Usage:
    python pancreatic_cancer_prediction_app.py

Requirements:
    - Python 3.8+
    - tkinter
    - numpy
    - pandas
    - scikit-learn
    - joblib

"""

import os
import sys
import traceback

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except ImportError:  # pragma: no cover
    raise ImportError("tkinter is required to run this application. Please install it.")


MODEL_FILENAMES = {
    "KNN": "pancreatic_knn_model.pkl",
    "Logistic Regression": "pancreatic_logistic_model.pkl",
    "Random Forest": "pancreatic_rf_model.pkl",
}
SCALER_FILENAME = "scaler.pkl"
METADATA_FILENAME = "model_metadata.pkl"

# MySQL database configuration for prediction logging.
DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "database": "mini_project",
    "user": "root",
    "password": "root",
}

ALGORITHM_OPTIONS = ["KNN", "Logistic Regression", "Random Forest"]

# Mapping from model labels (from dataset) to human-friendly clinical diagnosis text.
# The dataset uses 1/2/3 codes. Based on the provided dataset, higher values indicate
# higher likelihood of pancreatic cancer.
DIAGNOSIS_LABELS = {
    # These labels are derived from dataset values (1/2/3). In the original dataset,
    # higher numeric values correspond to stronger indication of disease.
    1: "Likely healthy (no pancreatic cancer detected)",
    2: "Possible pancreatic disease (requires clinical follow-up)",
    3: "Likely pancreatic cancer (urgent clinical evaluation recommended)",
}

# NOTE: The dataset included in this project is used only as a fallback to train a model
#       when the required serialized model artifacts are missing.
DATASET_PATH = os.path.join("archive", "Pancreatic  cancer clinical dataset", "pancreatic_600_risk123.csv")


def get_project_root() -> str:
    """Return the root path of the project (where this script lives)."""

    return os.path.abspath(os.path.dirname(__file__))


def load_model_and_scaler():
    """Load the trained scaler, KNN model, and preprocessing metadata from disk.

    If the saved artifacts are not present, this will train a new model using the
    dataset found in `archive/Pancreatic  cancer clinical dataset/pancreatic_600_risk123.csv`.

    Returns:
        tuple: (scaler, model, metadata)

    Metadata keys:
        - raw_columns: list of columns used before one-hot encoding
        - feature_columns: list of columns after one-hot encoding (model input order)
        - categorical_defaults: dict of default values for categorical columns
    """

    root = get_project_root()
    scaler_path = os.path.join(root, SCALER_FILENAME)
    metadata_path = os.path.join(root, METADATA_FILENAME)

    have_models = all(os.path.exists(os.path.join(root, fn)) for fn in MODEL_FILENAMES.values())
    if have_models and os.path.exists(scaler_path) and os.path.exists(metadata_path):
        scaler = joblib.load(scaler_path)
        models = {name: joblib.load(os.path.join(root, path)) for name, path in MODEL_FILENAMES.items()}
        metadata = joblib.load(metadata_path)
        return scaler, models, metadata

    # If artifacts do not exist, attempt to build them from the dataset
    dataset_path = os.path.join(root, DATASET_PATH)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Required artifacts missing and dataset not found.\n"
            f"Expected dataset at: {dataset_path}\n"
            f"Please provide model artifacts '{', '.join(MODEL_FILENAMES.values())}', '{SCALER_FILENAME}', and '{METADATA_FILENAME}', or include the dataset."
        )

    df = pd.read_csv(dataset_path)

    # Notebook training logic: drop sample_id and diagnosis and encode categoricals
    X = df.drop(columns=["diagnosis", "sample_id"], errors="ignore")
    y = df.get("diagnosis")

    # Keep track of raw columns before encoding for future input mapping
    raw_columns = X.columns.tolist()

    X_encoded = pd.get_dummies(X, drop_first=True)
    feature_columns = X_encoded.columns.tolist()

    # Determine sensible default values for categorical fields
    categorical_defaults = {}
    for col in X.select_dtypes(include=["object", "category"]).columns:
        values = X[col].dropna().unique().tolist()
        if values:
            categorical_defaults[col] = values[0]

    # Determine sensible default values for numeric fields (use medians from training data)
    numeric_defaults = {}
    for col in X.select_dtypes(include=["number"]).columns:
        numeric_defaults[col] = float(X[col].median())

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded.values)

    model_knn = KNeighborsClassifier(n_neighbors=5)
    model_knn.fit(X_scaled, y.values)

    from sklearn.multiclass import OneVsRestClassifier
    model_lr = OneVsRestClassifier(LogisticRegression(max_iter=1000, C=0.5, solver="liblinear"))
    model_lr.fit(X_scaled, y.values)

    model_rf = RandomForestClassifier(n_estimators=200, random_state=42)
    model_rf.fit(X_scaled, y.values)

    models = {
        "KNN": model_knn,
        "Logistic Regression": model_lr,
        "Random Forest": model_rf,
    }

    metadata = {
        "raw_columns": raw_columns,
        "feature_columns": feature_columns,
        "categorical_defaults": categorical_defaults,
        "numeric_defaults": numeric_defaults,
    }

    # Save artifacts for future runs
    joblib.dump(scaler, scaler_path)
    for name, model_obj in models.items():
        joblib.dump(model_obj, os.path.join(root, MODEL_FILENAMES[name]))
    joblib.dump(metadata, metadata_path)

    return scaler, models, metadata


class PancreaticCancerApp:
    """Tkinter application for pancreatic cancer risk prediction."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Sustainability and Health Predictor")
        self.root.resizable(False, False)
        self.root.configure(bg="#eef2ff")

        # Start in fullscreen mode (can exit with Escape)
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", lambda event: self.root.attributes("-fullscreen", False))

        # Apply a modern-looking ttk theme and custom styles.
        self.style = ttk.Style(self.root)
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        # Global style definitions
        base_font = ("Segoe UI", 10)
        heading_font = ("Segoe UI", 14, "bold")

        self.style.configure("TLabel", background="#eef2ff", font=base_font)
        self.style.configure("TFrame", background="#eef2ff")
        self.style.configure("TLabelFrame", background="#ffffff", borderwidth=1, relief="solid")
        self.style.configure("TLabelframe.Label", font=heading_font, foreground="#1f2937")

        self.style.configure(
            "Accent.TButton",
            background="#1d4ed8",
            foreground="#ffffff",
            font=("Segoe UI", 10, "bold"),
            padding=8,
        )
        self.style.map(
            "Accent.TButton",
            background=[("active", "#2563eb"), ("disabled", "#a5b4fc")],
        )

        self.style.configure(
            "Secondary.TButton",
            background="#f3f4f6",
            foreground="#1f2937",
            font=("Segoe UI", 10, "bold"),
            padding=8,
        )

        # Highlight most important features
        self.style.configure(
            "Important.TLabel",
            foreground="#b91c1c",
            font=("Segoe UI", 10, "bold"),
        )

        # Use consistent padding for inputs
        self.style.configure("TEntry", padding=6)
        self.style.configure("TCombobox", padding=6)

        # Track inputs for validation and dynamic updates
        self._input_widgets = {}

        self.scaler = None
        self.models = {}

        try:
            self.scaler, self.models, self.metadata = load_model_and_scaler()
            self.model = self.models.get("KNN")
        except Exception as exc:  # pragma: no cover
            messagebox.showerror(
                "Model Load Error",
                f"Could not load or train the model.\n\nError: {exc}\n\n{traceback.format_exc()}"
            )
            self.root.destroy()
            return

        # Determine the most predictive fields (for highlighting in the UI)
        self.important_fields = self._compute_important_fields()

        self._build_gui()

        # Pre-load a default healthy profile so users start with no-cancer benchmark values.
        self.apply_preset("no_cancer")

    def _compute_important_fields(self, top_n: int = 3) -> set[str]:
        """Compute the most correlated input fields to the diagnosis label.

        Uses the provided dataset to compute Pearson correlations between candidate
        predictor columns and the diagnosis code, then returns the top N.
        """

        try:
            df = pd.read_csv(DATASET_PATH)
            # Use numeric columns; diagnosis is the target.
            corr = df.corr()["diagnosis"].abs().sort_values(ascending=False)
            corr = corr.drop("diagnosis", errors="ignore")

            candidates = {"LYVE1", "REG1B", "TFF1", "stage", "plasma_C", "creatinine", "CA19_9", "glucose"}
            top_cols = [c for c in corr.index if c in candidates][:top_n]

            label_map = {"stage": "Stage", "plasma_C": "Plasma C"}
            return {label_map.get(col, col) for col in top_cols}
        except Exception:
            return set()

    def _build_gui(self) -> None:
        """Builds the main GUI layout."""

        # Set a fixed window size and center on screen for a desktop app feel
        self.root.geometry("800x700")
        self.root.minsize(750, 650)

        # Use a notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Create tabs for different SDGs
        self._build_health_tab()
        self._build_climate_tab()
        self._build_hunger_tab()
        self._build_education_tab()

    def _build_health_tab(self) -> None:
        """Builds the Health (SDG 3) tab."""

        # Create tab frame
        health_tab = ttk.Frame(self.notebook)
        self.notebook.add(health_tab, text="SDG 3: Health")

        # Use a scrollable canvas for the main content
        content_container = ttk.Frame(health_tab)
        content_container.grid(row=0, column=0, sticky="nsew")
        health_tab.columnconfigure(0, weight=1)
        health_tab.rowconfigure(0, weight=1)

        content_container.columnconfigure(0, weight=1)
        content_container.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(content_container, bg="#eef2ff", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        vscroll = ttk.Scrollbar(content_container, orient="vertical", command=self.canvas.yview)
        vscroll.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=vscroll.set)

        # Create a frame inside the canvas for all scrollable content
        main_frame = ttk.Frame(self.canvas, padding=16)
        self.canvas_window = self.canvas.create_window((0, 0), window=main_frame, anchor="nw", tags="inner")

        def _on_frame_configure(event):
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

        main_frame.bind("<Configure>", _on_frame_configure)

        def _on_canvas_configure(event):
            self.canvas.itemconfigure(self.canvas_window, width=event.width)

        self.canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mouse_wheel(event):
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self.canvas.bind_all("<MouseWheel>", _on_mouse_wheel)

        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        # Sticky footer for action buttons
        footer = ttk.Frame(health_tab, padding=(12, 8))
        footer.grid(row=1, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        footer.columnconfigure(1, weight=1)

        self.footer = footer

        # Title header
        header_frame = ttk.Frame(main_frame)
        header_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        header_title = ttk.Label(
            header_frame,
            text="🩺 Pancreatic Cancer Risk Predictor",
            font=("Segoe UI", 16, "bold"),
            foreground="#1f2937",
        )
        header_title.grid(row=0, column=0, sticky="w")

        header_subtitle = ttk.Label(
            header_frame,
            text="Enter patient details below and click Predict.",
            font=("Segoe UI", 10),
            foreground="#4b5563",
        )
        header_subtitle.grid(row=1, column=0, sticky="w", pady=(4, 0))

        ttk.Separator(main_frame, orient="horizontal").grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 14))

        # Input section (left)
        inputs_frame = ttk.LabelFrame(main_frame, text="Patient Data", padding=16)
        inputs_frame.grid(row=2, column=0, sticky="nsew", padx=4, pady=4)

        # Additional tools & dataset info (right)
        tools_frame = ttk.LabelFrame(main_frame, text="Tools & Dataset", padding=16)
        tools_frame.grid(row=2, column=1, sticky="nsew", padx=4, pady=4)
        tools_frame.columnconfigure(0, weight=1)

        # Toggle: show/hide defaulted feature list in results
        self.show_defaulted_var = tk.BooleanVar(value=False)

        # Algorithm selector
        self.algorithm_var = tk.StringVar(value=ALGORITHM_OPTIONS[0])

        # Define form fields: label, variable, widget type
        self.entries = {}
        self.widgets = {}

        # Customer identity fields for prediction logging
        self._add_labeled_entry(inputs_frame, "Customer ID", row=0)
        self._add_labeled_entry(inputs_frame, "Customer Name", row=1)

        # Biomarker inputs (replacing age/gender/height/weight/BMI)
        self._add_labeled_entry(inputs_frame, "LYVE1", row=2, numeric=True, min_val=0)
        self._add_labeled_entry(inputs_frame, "REG1B", row=3, numeric=True, min_val=0)
        self._add_labeled_entry(inputs_frame, "TFF1", row=4, numeric=True, min_val=0)

        # Stage (1-4)
        self._add_labeled_entry(inputs_frame, "Stage", row=5)

        # Plasma C
        self._add_labeled_entry(inputs_frame, "Plasma C", row=6)

        # Creatinine
        self._add_labeled_entry(inputs_frame, "Creatinine", row=7)

        # CA19_9
        self._add_labeled_entry(inputs_frame, "CA19_9", row=8)

        # Glucose
        self._add_labeled_entry(inputs_frame, "Glucose", row=9)

        # Smoking History
        self._add_labeled_combobox(
            inputs_frame,
            "Smoking History",
            ["Never", "Former", "Current"],
            row=10,
        )

        # Alcohol Consumption
        self._add_labeled_combobox(
            inputs_frame,
            "Alcohol Consumption",
            ["None", "Occasional", "Regular"],
            row=11,
        )

        # Family History of Cancer
        self._add_labeled_combobox(
            inputs_frame,
            "Family History of Cancer",
            ["No", "Yes"],
            row=12,
        )

        # Diabetes
        self._add_labeled_combobox(
            inputs_frame,
            "Diabetes",
            ["No", "Yes"],
            row=13,
        )

        # Model algorithm selector
        alg_label = ttk.Label(inputs_frame, text="Algorithm:")
        alg_label.grid(row=14, column=0, sticky="w", pady=2)
        alg_combo = ttk.Combobox(inputs_frame, textvariable=self.algorithm_var, values=ALGORITHM_OPTIONS, state="readonly")
        alg_combo.grid(row=14, column=1, sticky="ew", pady=2, padx=(8, 0))

        # Prediction result output
        result_frame = ttk.LabelFrame(main_frame, text="Prediction", padding=12)
        result_frame.grid(row=5, column=0, columnspan=2, sticky="ew", padx=4, pady=4)

        # Dataset info (bottom right)
        dataset_info = ttk.LabelFrame(tools_frame, text="Dataset", padding=12)
        dataset_info.grid(row=2, column=0, sticky="nsew", pady=(16, 0))

        dataset_path_label = ttk.Label(
            dataset_info,
            text=f"Data: {DATASET_PATH}",
            wraplength=260,
            foreground="#1f2937",
            font=("Segoe UI", 9),
        )
        dataset_path_label.grid(row=0, column=0, sticky="w")

        # Attempt to show row count
        try:
            df = pd.read_csv(DATASET_PATH)
            row_count = len(df)
            dataset_rows_label = ttk.Label(
                dataset_info,
                text=f"Rows: {row_count:,}",
                foreground="#4b5563",
                font=("Segoe UI", 9),
            )
            dataset_rows_label.grid(row=1, column=0, sticky="w", pady=(6, 0))
        except Exception:
            pass

        # Preset controls for no-cancer / likely-cancer inputs
        preset_frame = ttk.LabelFrame(tools_frame, text="Presets", padding=12)
        preset_frame.grid(row=1, column=0, sticky="nsew", pady=(16, 0))

        ttk.Button(
            preset_frame,
            text="No Cancer Default",
            command=lambda: self.apply_preset("no_cancer"),
            style="Secondary.TButton",
        ).grid(row=0, column=0, sticky="ew", pady=2)

        ttk.Button(
            preset_frame,
            text="Likely Cancer Default",
            command=lambda: self.apply_preset("likely_cancer"),
            style="Secondary.TButton",
        ).grid(row=1, column=0, sticky="ew", pady=2)
 
        self.prediction_label = ttk.Label(
            result_frame,
            text="Prediction will appear here after clicking Predict Risk.",
            font=("Segoe UI", 12, "bold"),
            foreground="#1f2937",
            wraplength=560,
        )
        self.prediction_label.grid(row=0, column=0, sticky="w", pady=(0, 8))

        self.result_text = tk.Text(
            result_frame,
            width=60,
            height=10,
            wrap="word",
            state="disabled",
            bg="#f7fafc",
            relief="flat",
            font=("Segoe UI", 10),
            padx=10,
            pady=10,
        )
        self.result_text.grid(row=1, column=0, sticky="nsew")

        # Gauge visualization (risk probability)
        self.gauge_canvas = tk.Canvas(result_frame, width=260, height=120, bg="#f7fafc", highlightthickness=0)
        self.gauge_canvas.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        self._update_gauge(0)

        # Buttons (sticky footer)
        button_frame = ttk.Frame(self.footer)
        button_frame.grid(row=0, column=0, columnspan=2, sticky="ew")

        self.predict_button = ttk.Button(
            button_frame,
            text=" Predict Risk ",
            command=self.on_predict,
            style="Accent.TButton",
        )
        self.predict_button.grid(row=0, column=0, padx=6, pady=2, sticky="w")


        clear_button = ttk.Button(
            button_frame,
            text=" Clear ",
            command=self.clear_form,
            style="Secondary.TButton",
        )
        clear_button.grid(row=0, column=1, padx=6, pady=2)

        exit_button = ttk.Button(
            button_frame,
            text=" Exit ",
            command=self.root.quit,
            style="Secondary.TButton",
        )
        exit_button.grid(row=0, column=2, padx=6, pady=2)

        # Scroll-to-top button
        scroll_top_button = ttk.Button(
            button_frame,
            text=" Scroll to Top ",
            command=lambda: self.canvas.yview_moveto(0),
            style="Secondary.TButton",
        )
        scroll_top_button.grid(row=0, column=3, padx=6, pady=2)

        # Show/hide defaulted inputs checkbox (cleaner output option)
        self.show_defaults_check = ttk.Checkbutton(
            button_frame,
            text="Show defaulted inputs",
            variable=self.show_defaulted_var,
        )
        self.show_defaults_check.grid(row=0, column=4, padx=6, pady=2)

        # Make columns expand
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        result_frame.columnconfigure(0, weight=1)

    def _build_climate_tab(self) -> None:
        """Builds the Climate (SDG 13) tab."""

        climate_tab = ttk.Frame(self.notebook)
        self.notebook.add(climate_tab, text="SDG 13: Climate")

        ttk.Label(climate_tab, text="🌍 SDG 13: Take urgent action to combat climate change and its impacts", font=("Segoe UI", 14, "bold")).pack(pady=10)

        # Simple CO2 calculator
        frame = ttk.LabelFrame(climate_tab, text="Carbon Footprint Calculator", padding=10)
        frame.pack(pady=10, padx=20, fill="x")

        # Use grid for layout
        ttk.Label(frame, text="Daily electricity usage (kWh):").grid(row=0, column=0, sticky="w", pady=2)
        self.electricity_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.electricity_var).grid(row=0, column=1, sticky="ew", padx=(8,0), pady=2)

        ttk.Label(frame, text="Weekly driving (km):").grid(row=1, column=0, sticky="w", pady=2)
        self.driving_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.driving_var).grid(row=1, column=1, sticky="ew", padx=(8,0), pady=2)

        ttk.Button(frame, text="Calculate CO2", command=self.calculate_co2).grid(row=2, column=0, columnspan=2, pady=10)

        self.co2_result = ttk.Label(frame, text="", font=("Segoe UI", 12, "bold"))
        self.co2_result.grid(row=3, column=0, columnspan=2, pady=5)

        # Presets
        preset_frame = ttk.Frame(frame)
        preset_frame.grid(row=4, column=0, columnspan=2, pady=5)
        ttk.Button(preset_frame, text="Low Usage", command=lambda: self.set_co2_preset("low")).pack(side="left", padx=5)
        ttk.Button(preset_frame, text="Average", command=lambda: self.set_co2_preset("average")).pack(side="left", padx=5)
        ttk.Button(preset_frame, text="High Usage", command=lambda: self.set_co2_preset("high")).pack(side="left", padx=5)

        # Make columns expand
        frame.columnconfigure(1, weight=1)

        # Info text
        text = tk.Text(climate_tab, wrap="word", height=15, width=80)
        text.pack(pady=10, padx=20)
        text.insert(tk.END, "Climate change is one of the greatest challenges of our time. This SDG aims to strengthen resilience and adaptive capacity to climate-related hazards.\n\nKey targets:\n- Strengthen resilience to climate-related disasters\n- Integrate climate change measures into policies\n- Improve education and awareness on climate change\n- Implement UNFCCC commitments\n\nYour actions: Reduce carbon footprint, support renewable energy, advocate for climate policies.")
        text.config(state="disabled")

    def _build_hunger_tab(self) -> None:
        """Builds the Hunger (SDG 2) tab."""

        hunger_tab = ttk.Frame(self.notebook)
        self.notebook.add(hunger_tab, text="SDG 2: Hunger")

        ttk.Label(hunger_tab, text="🍽️ SDG 2: End hunger, achieve food security and improved nutrition", font=("Segoe UI", 14, "bold")).pack(pady=10)

        # Simple BMI calculator
        frame = ttk.LabelFrame(hunger_tab, text="BMI Calculator", padding=10)
        frame.pack(pady=10, padx=20, fill="x")

        ttk.Label(frame, text="Height (cm):").grid(row=0, column=0, sticky="w", pady=2)
        self.height_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.height_var).grid(row=0, column=1, sticky="ew", padx=(8,0), pady=2)

        ttk.Label(frame, text="Weight (kg):").grid(row=1, column=0, sticky="w", pady=2)
        self.weight_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.weight_var).grid(row=1, column=1, sticky="ew", padx=(8,0), pady=2)

        ttk.Button(frame, text="Calculate BMI", command=self.calculate_bmi).grid(row=2, column=0, columnspan=2, pady=10)

        self.bmi_result = ttk.Label(frame, text="", font=("Segoe UI", 12, "bold"))
        self.bmi_result.grid(row=3, column=0, columnspan=2, pady=5)

        # Presets
        preset_frame = ttk.Frame(frame)
        preset_frame.grid(row=4, column=0, columnspan=2, pady=5)
        ttk.Button(preset_frame, text="Average Adult", command=lambda: self.set_bmi_preset("average")).pack(side="left", padx=5)
        ttk.Button(preset_frame, text="Athlete", command=lambda: self.set_bmi_preset("athlete")).pack(side="left", padx=5)
        ttk.Button(preset_frame, text="Overweight", command=lambda: self.set_bmi_preset("overweight")).pack(side="left", padx=5)

        # Make columns expand
        frame.columnconfigure(1, weight=1)

        # Info text
        text = tk.Text(hunger_tab, wrap="word", height=15, width=80)
        text.pack(pady=10, padx=20)
        text.insert(tk.END, "Hunger and malnutrition affect millions. This SDG focuses on sustainable agriculture and food systems.\n\nKey targets:\n- End hunger and ensure access to safe, nutritious food\n- End all forms of malnutrition\n- Double agricultural productivity\n- Ensure sustainable food production systems\n- Maintain genetic diversity of seeds\n\nYour actions: Support local farmers, reduce food waste, promote healthy diets.")
        text.config(state="disabled")

    def _build_education_tab(self) -> None:
        """Builds the Education (SDG 4) tab."""

        education_tab = ttk.Frame(self.notebook)
        self.notebook.add(education_tab, text="SDG 4: Education")

        ttk.Label(education_tab, text="📚 SDG 4: Ensure inclusive and equitable quality education", font=("Segoe UI", 14, "bold")).pack(pady=20)

        text = tk.Text(education_tab, wrap="word", height=20, width=80)
        text.pack(pady=10, padx=20)
        text.insert(tk.END, "Education is key to sustainable development. This SDG aims for quality education for all.\n\nKey targets:\n- Free primary and secondary education\n- Equal access to vocational training\n- Eliminate gender disparities in education\n- Build resilient educational facilities\n- Increase qualified teachers\n- Promote lifelong learning\n\nYour actions: Support education initiatives, volunteer as a tutor, advocate for educational equity.")
        text.config(state="disabled")

    def calculate_co2(self):
        """Calculate approximate CO2 emissions."""
        try:
            electricity = float(self.electricity_var.get() or 0)
            driving = float(self.driving_var.get() or 0)
            # Rough estimates: 0.5 kg CO2 per kWh, 0.2 kg CO2 per km
            co2 = electricity * 0.5 + driving * 0.2
            self.co2_result.config(text=f"Estimated weekly CO2: {co2:.2f} kg")
        except ValueError:
            self.co2_result.config(text="Please enter valid numbers")

    def calculate_bmi(self):
        """Calculate BMI."""
        try:
            height = float(self.height_var.get() or 0) / 100  # cm to m
            weight = float(self.weight_var.get() or 0)
            if height > 0:
                bmi = weight / (height ** 2)
                category = "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"
                self.bmi_result.config(text=f"BMI: {bmi:.2f} ({category})")
            else:
                self.bmi_result.config(text="Invalid height")
        except ValueError:
            self.bmi_result.config(text="Please enter valid numbers")

    def set_co2_preset(self, preset):
        """Set CO2 calculator presets."""
        if preset == "low":
            self.electricity_var.set("10")
            self.driving_var.set("50")
        elif preset == "average":
            self.electricity_var.set("20")
            self.driving_var.set("150")
        elif preset == "high":
            self.electricity_var.set("50")
            self.driving_var.set("300")

    def set_bmi_preset(self, preset):
        """Set BMI calculator presets."""
        if preset == "average":
            self.height_var.set("170")
            self.weight_var.set("70")
        elif preset == "athlete":
            self.height_var.set("180")
            self.weight_var.set("75")
        elif preset == "overweight":
            self.height_var.set("165")
            self.weight_var.set("85")

    def _add_labeled_entry(
        self,
        parent: ttk.Frame,
        label_text: str,
        row: int,
        numeric: bool = False,
        min_val: float | None = None,
        max_val: float | None = None,
        readonly: bool = False,
    ) -> None:
        """Add a label + entry field on the form."""
        label_style = "Important.TLabel" if label_text in getattr(self, "important_fields", set()) else "TLabel"
        label = ttk.Label(parent, text=label_text + ":", style=label_style)
        label.grid(row=row, column=0, sticky="w", pady=2)

        var = tk.StringVar()
        entry = ttk.Entry(parent, textvariable=var)
        if readonly:
            entry.configure(state="readonly")
        entry.grid(row=row, column=1, sticky="ew", pady=2, padx=(8, 0))

        parent.columnconfigure(1, weight=1)
        self.entries[label_text] = var
        self.widgets[label_text] = entry
        self._input_widgets[label_text] = entry

        if numeric:
            # Validate on each change
            var.trace_add(
                "write",
                lambda *_: self._validate_numeric_field(label_text, min_val, max_val),
            )

    def _add_labeled_combobox(self, parent: ttk.Frame, label_text: str, values, row: int) -> None:
        """Add a label + combobox field on the form."""
        label_style = "Important.TLabel" if label_text in getattr(self, "important_fields", set()) else "TLabel"
        label = ttk.Label(parent, text=label_text + ":", style=label_style)
        label.grid(row=row, column=0, sticky="w", pady=2)

        var = tk.StringVar()
        combobox = ttk.Combobox(parent, textvariable=var, values=values, state="readonly")
        combobox.grid(row=row, column=1, sticky="ew", pady=2, padx=(8, 0))
        combobox.set(values[0])

        parent.columnconfigure(1, weight=1)
        self.entries[label_text] = var
        # Keep the actual widget and default options for clearing
        self.widgets[label_text] = (combobox, values)

    def clear_form(self) -> None:
        """Clear all input fields and the result area."""
        for key, var in self.entries.items():
            widget_info = self.widgets.get(key)
            if isinstance(widget_info, tuple):
                # Combobox: reset to the first option
                combobox, values = widget_info
                if values:
                    combobox.set(values[0])
                    var.set(values[0])
                else:
                    var.set("")
            else:
                var.set("")

        # Reset highlighting
        for entry in self._input_widgets.values():
            entry.configure(background="#ffffff")


        # Reset the show-defaulted toggle
        if hasattr(self, "show_defaulted_var"):
            self.show_defaulted_var.set(False)

        self._set_result_text("")
        if hasattr(self, "prediction_label"):
            self.prediction_label.configure(text="Prediction will appear here after clicking Predict Risk.")

    def _get_preset_values(self, diagnosis_value: int, percentile: float | None = None) -> dict:
        """Return a dict with default values for the given diagnosis class.

        - diagnosis_value: 1 for healthy/no-cancer, 3 for cancer
        - percentile: None for median, otherwise numeric quantile (0..1)
        """
        try:
            df = pd.read_csv(DATASET_PATH)
            subset = df[df["diagnosis"] == diagnosis_value]
            if subset.empty:
                raise ValueError("No rows found for diagnosis " + str(diagnosis_value))
            targets = ["LYVE1", "REG1B", "TFF1", "stage", "plasma_C", "creatinine", "CA19_9", "glucose"]
            if percentile is None:
                values = subset[targets].median()
            else:
                values = subset[targets].quantile(percentile)
            return {
                "LYVE1": float(values["LYVE1"]),
                "REG1B": float(values["REG1B"]),
                "TFF1": float(values["TFF1"]),
                "Stage": float(values["stage"]),
                "Plasma C": float(values["plasma_C"]),
                "Creatinine": float(values["creatinine"]),
                "CA19_9": float(values["CA19_9"]),
                "Glucose": float(values["glucose"]),
            }
        except Exception:
            # fallback to dataset defaults if anything fails
            metadata = getattr(self, "metadata", {})
            numeric_defaults = metadata.get("numeric_defaults", {})
            return {
                "LYVE1": numeric_defaults.get("LYVE1", 0),
                "REG1B": numeric_defaults.get("REG1B", 0),
                "TFF1": numeric_defaults.get("TFF1", 0),
                "Stage": numeric_defaults.get("stage", 1),
                "Plasma C": numeric_defaults.get("plasma_C", 0),
                "Creatinine": numeric_defaults.get("creatinine", 0),
                "CA19_9": numeric_defaults.get("CA19_9", 0),
                "Glucose": numeric_defaults.get("glucose", 0),
            }

    def apply_preset(self, preset: str) -> None:
        """Apply a preset set of values to the form controls."""
        if preset == "no_cancer":
            values = self._get_preset_values(1)
            cat_defaults = {
                "Smoking History": "Never",
                "Alcohol Consumption": "None",
                "Family History of Cancer": "No",
                "Diabetes": "No",
            }
        elif preset == "likely_cancer":
            values = self._get_preset_values(3, percentile=0.5)
            cat_defaults = {
                "Smoking History": "Current",
                "Alcohol Consumption": "Regular",
                "Family History of Cancer": "Yes",
                "Diabetes": "Yes",
            }
        else:
            return

        for field, value in values.items():
            if field in self.entries:
                self.entries[field].set(str(round(value, 3)))

        for field, value in cat_defaults.items():
            widget_info = self.widgets.get(field)
            if isinstance(widget_info, tuple):
                combobox, options = widget_info
                if value in options:
                    combobox.set(value)
                    self.entries[field].set(value)
                elif options:
                    combobox.set(options[0])
                    self.entries[field].set(options[0])

        self._set_result_text(f"Loaded preset: {preset.replace('_', ' ').title()}")

    def _set_result_text(self, text: str) -> None:
        """Set the result text in the result display area."""
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, text)
        self.result_text.configure(state="disabled")

    def _update_gauge(self, percent: float) -> None:
        """Update the mini gauge graphic with the given probability percentage."""
        # Clamp value to 0-100
        percent = max(0.0, min(100.0, percent))

        # Clear previous drawings
        self.gauge_canvas.delete("all")

        w = int(self.gauge_canvas.winfo_width() or self.gauge_canvas["width"])
        h = int(self.gauge_canvas.winfo_height() or self.gauge_canvas["height"])

        # Define gauge dimensions
        padding = 12
        radius = (w - padding * 2) / 2
        cx = w / 2
        cy = h + padding  # center below the visible top so only top half (semi-circle) shows

        bbox = (cx - radius, cy - radius, cx + radius, cy + radius)

        # Background arc
        self.gauge_canvas.create_arc(
            *bbox,
            start=180,
            extent=180,
            style="arc",
            width=18,
            outline="#d1d5db",
        )

        # Choose color based on risk level
        if percent <= 40:
            color = "#22c55e"  # green
        elif percent <= 70:
            color = "#f59e0b"  # amber
        else:
            color = "#ef4444"  # red

        # Foreground arc for value
        self.gauge_canvas.create_arc(
            *bbox,
            start=180,
            extent=(percent / 100) * 180,
            style="arc",
            width=18,
            outline=color,
        )

        # Draw the percentage text
        self.gauge_canvas.create_text(
            cx,
            h * 0.35,
            text=f"{percent:.1f}%",
            font=("Segoe UI", 14, "bold"),
            fill="#1f2937",
        )

        # Add a label below the value
        self.gauge_canvas.create_text(
            cx,
            h * 0.7,
            text="Risk Likelihood",
            font=("Segoe UI", 9),
            fill="#4b5563",
        )

    def _validate_numeric_field(self, label_text: str, min_val: float | None, max_val: float | None) -> bool:
        """Validate a numeric field and highlight it if invalid."""
        entry = self._input_widgets.get(label_text)
        if entry is None:
            return True

        val = self.entries.get(label_text).get().strip()
        if val == "":
            entry.configure(background="#ffffff")
            return True

        try:
            num = float(val)
        except ValueError:
            entry.configure(background="#ffe6e6")
            return False

        if (min_val is not None and num < min_val) or (max_val is not None and num > max_val):
            entry.configure(background="#ffe6e6")
            return False

        entry.configure(background="#ffffff")
        return True

    def _validate_all_numeric_fields(self) -> bool:
        """Validate all known numeric inputs and return True if all are valid."""
        valid = True
        numeric_fields = {
            "LYVE1": (0, None),
            "REG1B": (0, None),
            "TFF1": (0, None),
            "Stage": (0, 10),
            "Plasma C": (0, None),
            "Creatinine": (0, None),
            "CA19_9": (0, None),
            "Glucose": (0, None),
        }
        for label, (min_v, max_v) in numeric_fields.items():
            if not self._validate_numeric_field(label, min_v, max_v):
                valid = False
        return valid

    def _import_mysql_connector(self):
        try:
            import mysql.connector
            return mysql.connector
        except ImportError as exc:
            raise ImportError(
                "mysql-connector-python is required for database logging. "
                "Install it with: pip install mysql-connector-python"
            ) from exc

    def _connect_db(self, with_database: bool = True):
        mysql = self._import_mysql_connector()
        connect_kwargs = {
            "host": DB_CONFIG["host"],
            "port": DB_CONFIG["port"],
            "user": DB_CONFIG["user"],
            "password": DB_CONFIG["password"],
        }
        if with_database:
            connect_kwargs["database"] = DB_CONFIG["database"]
        return mysql.connect(**connect_kwargs)

    def _ensure_database(self) -> None:
        conn = None
        cursor = None
        try:
            conn = self._connect_db(with_database=False)
            cursor = conn.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_CONFIG['database']}`")
            conn.commit()
        finally:
            if cursor is not None:
                try:
                    cursor.close()
                except Exception:
                    pass
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def _ensure_db_tables(self, cursor) -> None:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS action_log (
                id INT AUTO_INCREMENT PRIMARY KEY,
                action_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                user_action VARCHAR(255) NOT NULL,
                customer_id VARCHAR(100),
                customer_name VARCHAR(255),
                details TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_log (
                id INT AUTO_INCREMENT PRIMARY KEY,
                action_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                user_action VARCHAR(255) NOT NULL,
                customer_id VARCHAR(100),
                customer_name VARCHAR(255),
                features TEXT,
                prediction VARCHAR(255) NOT NULL,
                probability DOUBLE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

    def _log_prediction_to_db(
        self,
        customer_id: str,
        customer_name: str,
        features: str,
        prediction: str,
        probability: float,
    ) -> None:
        sql = (
            "INSERT INTO prediction_log "
            "(action_time, user_action, customer_id, customer_name, features, prediction, probability) "
            "VALUES (NOW(), %s, %s, %s, %s, %s, %s)"
        )

        conn = None
        cursor = None
        try:
            self._ensure_database()
            conn = self._connect_db()
            cursor = conn.cursor()
            self._ensure_db_tables(cursor)
            cursor.execute(
                sql,
                (
                    "predict_risk",
                    customer_id,
                    customer_name,
                    features,
                    prediction,
                    probability,
                ),
            )
            conn.commit()
        except Exception as exc:
            print(f"Database log failed: {exc}")
            messagebox.showwarning("Database Log Error", f"Could not log prediction: {exc}")
        finally:
            if cursor is not None:
                try:
                    cursor.close()
                except Exception:
                    pass
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def on_predict(self) -> None:
        """Event handler for the Predict Risk button."""
        if not self._validate_all_numeric_fields():
            messagebox.showwarning("Validation", "Please fix highlighted fields before predicting.")
            return

        try:
            X, debug_info = self._collect_inputs()
        except ValueError as exc:
            messagebox.showerror("Input Error", str(exc))
            return

        self._update_gauge(0)

        try:
            X_scaled = self.scaler.transform(X)
            selected_alg = self.algorithm_var.get()
            model = self.models.get(selected_alg, self.model)
            prediction = model.predict(X_scaled)[0]

            # KNN predict_proba requires probability support on the model; handle gracefully.
            prob_text = "N/A"
            try:
                proba = self.model.predict_proba(X_scaled)[0]
                # Assume binary classification and use the "positive" class probability
                positive_prob = max(proba)
                prob_text = f"{positive_prob * 100:.1f}%"
            except Exception:
                prob_text = "(probability not available)"

            # Map the predicted label to a clinical-style diagnosis description.
            label = int(prediction)
            risk_text = DIAGNOSIS_LABELS.get(
                label,
                f"Unknown diagnosis ({label})",
            )

            # Show a succinct prediction summary at the top of the results panel.
            if hasattr(self, "prediction_label"):
                self.prediction_label.configure(
                    text=f"{risk_text} ({prob_text})"
                )

            provided_items = "\n".join(f"- {k}: {v}" for k, v in debug_info.get("provided", {}).items())
            defaulted_items = "\n".join(f"- {k}: {v}" for k, v in debug_info.get("defaulted", {}).items())

            # Update the gauge visualization
            try:
                percent = float(prob_text.strip('%'))
            except Exception:
                percent = 0
            self._update_gauge(percent)

            # Auto-scroll to bottom (so results and buttons remain visible)
            try:
                self.canvas.yview_moveto(1.0)
            except Exception:
                pass

            show_defaults = self.show_defaulted_var.get() if hasattr(self, "show_defaulted_var") else False

            result_text = (
                f"Clinical diagnosis (model prediction):\n{risk_text}\n\n"
                f"Estimated confidence: {prob_text}\n\n"
                "Provided inputs / values used for prediction:\n"
                f"{provided_items or ' - (none)'}\n\n"
            )

            if show_defaults:
                result_text += (
                    "Defaulted inputs (using dataset defaults):\n"
                    f"{defaulted_items or ' - (none)'}\n\n"
                )

            if debug_info.get("defaulted"):
                result_text += (
                    "\nNote: Several fields were filled from dataset defaults. "
                    "For best prediction accuracy, enter as many measured values as possible."
                )

            # Store the prediction in the database with customer details.
            customer_id = self.entries.get("Customer ID").get().strip() if self.entries.get("Customer ID") else ""
            customer_name = self.entries.get("Customer Name").get().strip() if self.entries.get("Customer Name") else ""
            features = "; ".join(f"{k}={v}" for k, v in debug_info.get("provided", {}).items())
            try:
                self._log_prediction_to_db(customer_id, customer_name, features, risk_text, percent)
            except Exception as exc:
                print(f"Failed to log prediction to DB: {exc}")

            # Provide a clear clinical-style output while still reminding users that this is
            # a model-based prediction (not a certified medical diagnosis).
            result_text += (
                "Disclaimer: This is an automated prediction based on available data and is "
                "not a substitute for a medical diagnosis. Please consult a healthcare professional."
            )

            self._set_result_text(result_text)

        except NotFittedError:
            messagebox.showerror(
                "Model Error",
                "The loaded model is not fitted. Please retrain or replace the model artifacts."
            )
        except Exception as exc:
            messagebox.showerror("Prediction Error", f"An unexpected error occurred:\n{exc}")

    def _collect_inputs(self):
        """Collect inputs, validate them, and return feature vector + debug info.

        The model was trained using a dataset where categorical variables were one-hot encoded.
        This function builds a single-row DataFrame, applies the same encoding, and reorders
        the columns to match the feature order saved in metadata.

        Returns:
            tuple: (X, debug_info) where X is a numpy array ready for scaling/prediction and
            debug_info contains 'provided' and 'defaulted' input details.
        """

        # Start with defaults for all raw columns
        raw_cols = self.metadata.get("raw_columns", [])
        categorical_defaults = self.metadata.get("categorical_defaults", {})
        numeric_defaults = self.metadata.get("numeric_defaults", {})

        raw_input = {}
        for col in raw_cols:
            if col in numeric_defaults:
                raw_input[col] = numeric_defaults[col]
            else:
                raw_input[col] = categorical_defaults.get(col, 0)

        provided = {}

        def parse_float(field_key: str):
            value = self.entries.get(field_key)
            if value is None:
                return None
            txt = value.get().strip()
            if not txt:
                return None
            try:
                return float(txt)
            except ValueError:
                raise ValueError(f"{field_key} must be a number.")

        # Common numeric fields used in the dataset
        numeric_fields = [
            "LYVE1",
            "REG1B",
            "TFF1",
            "stage",
            "plasma_C",
            "creatinine",
            "CA19_9",
            "glucose",
        ]

        for field in numeric_fields:
            # The form labels may be slightly different from raw column names
            label_map = {
                "LYVE1": "LYVE1",
                "REG1B": "REG1B",
                "TFF1": "TFF1",
                "stage": "Stage",
                "plasma_C": "Plasma C",
                "creatinine": "Creatinine",
                "CA19_9": "CA19_9",
                "glucose": "Glucose",
            }
            label = label_map.get(field, field)
            value = parse_float(label)
            if value is not None and field in raw_input:
                raw_input[field] = value
                provided[field] = value

        # Build a DataFrame with a single row to apply the same encoding used during training.
        df_row = pd.DataFrame([raw_input], columns=raw_cols)
        df_encoded = pd.get_dummies(df_row, drop_first=True)

        # Align to the exact feature ordering used during training.
        feature_cols = self.metadata.get("feature_columns", [])
        df_aligned = df_encoded.reindex(columns=feature_cols, fill_value=0)

        # Determine which inputs were defaulted (versus provided by the user)
        defaulted = {k: v for k, v in raw_input.items() if k not in provided}

        debug_info = {
            "provided": provided,
            "defaulted": defaulted,
        }

        return df_aligned.values.astype(float).reshape(1, -1), debug_info


def main():
    root = tk.Tk()
    app = PancreaticCancerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()