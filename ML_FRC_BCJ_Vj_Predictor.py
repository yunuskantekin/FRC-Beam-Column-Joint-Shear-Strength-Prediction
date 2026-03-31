# FRC Beam-Column Joint Shear Strength Predictor

# Prediction tool using pretrained models

# Supplementary material for the manuscript titled
# "Machine learning based prediction of joint shear
# strength for FRC beam-column connections"

# Code by: Yunus Kantekin • VT


# Required imports

import os
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib
import warnings

warnings.filterwarnings('ignore')

FONT_FAMILY = 'Open Sans'


class FRCPredictorApp:

    def __init__(self, root):
        self.root = root
        self.root.title("FRC Beam-Column Joint Shear Strength Predictor")

        # Icon

        try:
            icon = tk.PhotoImage(width=16, height=16)
            icon.put('#861F41', to=(0, 0, 16, 16))
            self.root.iconphoto(True, icon)
        except Exception:
            pass

        self.root.geometry("1400x850")
        self.root.minsize(1100, 700)

        self.vt_maroon = '#861F41'
        self.vt_burnt_orange = '#E5751F'
        self.vt_impact_orange = '#CA4F00'
        self.vt_hokie_stone = '#75787B'
        self.vt_white = '#FFFFFF'

        self.bg_color = '#1a1d2e'
        self.fg_color = '#E5E1E6'
        self.input_bg = '#2b2d42'
        self.accent_color = '#D7D2CB'

        self.root.configure(bg=self.bg_color)

        self.feature_display = {
            'hb': 'hb', 'bb': 'bb', 'hc': 'hc', 'bc': 'bc',
            'fc': "f'c", 'fyv': 'fyv', 'rhob': 'ρb',
            'rhoc': 'ρc', 'rhov': 'ρv', 'vf': 'Vf',
            'ar': 'Lf/Df', 'ftf': 'ftf', 'n': 'n'
        }

        self.load_models()
        self.create_gui()

    def load_models(self):
        try:
            print("Initializing pretrained models.")
            _code_dir = os.path.dirname(os.path.abspath(__file__))
            # check root directory first, then fall back to latest Export_ folder
            _root_path = os.path.join(_code_dir, 'FRC_BCJ_Rep_ML_Models.joblib')
            if os.path.exists(_root_path):
                _joblib_path = _root_path
            else:
                _export_dirs = sorted([
                    d for d in os.listdir(_code_dir)
                    if d.startswith('Export_') and os.path.isdir(os.path.join(_code_dir, d))
                ])
                if not _export_dirs:
                    raise FileNotFoundError
                _joblib_path = os.path.join(_code_dir, _export_dirs[-1], 'FRC_BCJ_Rep_ML_Models.joblib')
            data = joblib.load(_joblib_path)

            self.models = data['models']
            self.X_train = data['X_train']
            self.X_train_oh = data['X_train_oh']
            self.selected_features = data['selected_features']
            self.categorical_cols = data['categorical_cols']
            self.feature_ranges = data['feature_ranges']

            print(f"Successfully loaded {len(self.models)} ML models.")

        except FileNotFoundError:
            messagebox.showerror("Error", "No export folder found. Please run main.py first.")
            self.root.destroy()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models:\n{str(e)}")
            self.root.destroy()

    def create_gui(self):
        title_frame = tk.Frame(self.root, bg=self.vt_maroon, height=90)
        title_frame.pack(fill='x', pady=(0, 10))
        title_frame.pack_propagate(False)

        tk.Label(title_frame,
                text="FRC BEAM-COLUMN JOINT SHEAR STRENGTH PREDICTOR",
                font=(FONT_FAMILY, 24, 'bold'),
                bg=self.vt_maroon,
                fg=self.vt_white).pack(pady=(15, 2))

        tk.Label(title_frame,
                text="Machine Learning Based Prediction Tool • VT • METU",
                font=(FONT_FAMILY, 12, 'bold'),
                bg=self.vt_maroon,
                fg=self.vt_burnt_orange).pack(pady=(0, 0))

        main_container = tk.Frame(self.root, bg=self.bg_color)
        main_container.pack(fill='both', expand=True, padx=20, pady=10)
        main_container.grid_rowconfigure(0, weight=1)
        main_container.grid_columnconfigure(0, weight=1, minsize=550)
        main_container.grid_columnconfigure(1, weight=0, minsize=300)
        main_container.grid_columnconfigure(2, weight=1, minsize=450)

        left_frame = tk.Frame(main_container, bg=self.input_bg, relief='flat', bd=0)
        left_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 10))

        main_frame = tk.Frame(left_frame, bg=self.input_bg)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.input_vars = {}

        # Fiber type mapping

        self.fiber_mapping = {
            'Aramid': 'Aramid',
            'Basalt': 'Basalt',
            'Hooked End Steel Fiber (HESF)': 'HESF',
            'Hybrid': 'Hybrid',
            'Polyethylene Terephthalate (PET)': 'PET',
            'Polyethylene (PE)': 'PE',
            'Polypropylene (PP)': 'PP',
            'Polyvinyl Alcohol (PVA)': 'PVA',
            'Straight Steel Fiber (S)': 'S'
        }

        # Input groups

        self.create_input_group(main_frame, "SPECIMEN INFORMATION", [
            ('jtype', 'Joint Type', None, 'categorical')
        ])

        self.create_input_group(main_frame, "COMPOSITE PROPERTIES", [
            ('fc', "Compressive Strength f'c (MPa)", None, 'numerical')
        ])

        self.create_input_group(main_frame, "FIBER PROPERTIES", [
            ('fbrtype', 'Fiber Type', None, 'categorical'),
            ('vf', 'Fiber Volume Fraction Vf', None, 'numerical'),
            ('ar', 'Aspect Ratio Lf/Df', None, 'numerical'),
            ('ftf', 'Fiber Tensile Strength ftf (MPa)', None, 'numerical')
        ])

        self.create_input_group(main_frame, "GEOMETRIC PROPERTIES", [
            ('hb', 'Beam Depth hb (mm)', None, 'numerical'),
            ('bb', 'Beam Width bb (mm)', None, 'numerical'),
            ('hc', 'Column Depth hc (mm)', None, 'numerical'),
            ('bc', 'Column Width bc (mm)', None, 'numerical')
        ])

        self.create_input_group(main_frame, "LONGITUDINAL REINFORCEMENT", [
            ('rhob', 'Beam Reinf. Ratio ρb', None, 'numerical'),
            ('rhoc', 'Column Reinf. Ratio ρc', None, 'numerical')
        ])

        self.create_input_group(main_frame, "AXIAL LOAD", [
            ('n', 'Axial Load Ratio n', None, 'numerical')
        ])

        self.create_input_group(main_frame, "JOINT TRANSVERSE REINFORCEMENT", [
            ('rhov', 'Volumetric Ratio ρv', None, 'numerical'),
            ('fyv', 'Yield Strength fyv (MPa)', None, 'numerical')
        ])

        middle_frame = tk.Frame(main_container, bg=self.bg_color)
        middle_frame.grid(row=0, column=1, sticky='ns', padx=10)

        tk.Frame(middle_frame, bg=self.bg_color).pack(side='top', expand=True)

        predict_btn = tk.Button(middle_frame,
                 text="\nPREDICT\nJOINT SHEAR\nSTRENGTH\n",
                 font=(FONT_FAMILY, 11, 'bold'),
                 bg=self.vt_impact_orange,
                 fg=self.vt_white,
                 activebackground=self.vt_burnt_orange,
                 activeforeground=self.vt_white,
                 command=self.predict,
                 cursor='hand2',
                 relief='raised',
                 bd=3,
                 width=18,
                 height=8)
        predict_btn.pack(pady=10)

        tk.Frame(middle_frame, bg=self.bg_color).pack(side='bottom', expand=True)

        right_frame = tk.Frame(main_container, bg=self.input_bg, relief='flat', bd=0)
        right_frame.grid(row=0, column=2, sticky='nsew', padx=(10, 0))

        result_header = tk.Frame(right_frame, bg=self.vt_maroon, height=60)
        result_header.pack(fill='x')

        result_header.pack_propagate(False)

        tk.Label(result_header,
                text="PREDICTION RESULTS",
                font=(FONT_FAMILY, 15, 'bold'),
                bg=self.vt_maroon,
                fg=self.vt_white).pack(pady=15)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("VT.Treeview",
                       background=self.input_bg,
                       foreground=self.fg_color,
                       fieldbackground=self.input_bg,
                       borderwidth=0,
                       font=(FONT_FAMILY, 10),
                       rowheight=28)
        style.configure("VT.Treeview.Heading",
                       background=self.vt_hokie_stone,
                       foreground=self.vt_white,
                       borderwidth=1,
                       relief='raised',
                       font=(FONT_FAMILY, 11, 'bold'))
        style.map('VT.Treeview',
                 background=[('selected', self.vt_burnt_orange)])

        columns = ('#', 'Model', 'Vj (MPa)')
        self.results_tree = ttk.Treeview(right_frame, columns=columns, show='headings',
                                        height=13, style="VT.Treeview")

        self.results_tree.heading('#', text='#')
        self.results_tree.heading('Model', text='Algorithm')
        self.results_tree.heading('Vj (MPa)', text='Vj (MPa)')

        self.results_tree.column('#', width=50, anchor='center', stretch=False)
        self.results_tree.column('Model', width=200, anchor='w', stretch=True)
        self.results_tree.column('Vj (MPa)', width=100, anchor='center', stretch=False)

        self.results_tree.pack(fill='both', expand=True, padx=10, pady=10)

        info_frame = tk.Frame(right_frame, bg=self.input_bg)
        info_frame.pack(fill='both', padx=10, pady=(0, 10))

        self.info_label = tk.Label(info_frame,
                                    text=self.get_range_info(),
                                    font=('Consolas', 8),
                                    bg=self.input_bg,
                                    fg=self.accent_color,
                                    justify='left',
                                    anchor='nw')
        self.info_label.pack(fill='both', expand=True)

    def create_input_group(self, parent, title, fields):
        group_frame = tk.LabelFrame(parent, text=f"  {title}  ",
                                   bg=self.input_bg,
                                   fg=self.vt_burnt_orange,
                                   font=(FONT_FAMILY, 10, 'bold'),
                                   relief='solid',
                                   bd=1)
        group_frame.pack(fill='both', padx=10, pady=2, expand=True)

        group_frame.columnconfigure(0, weight=0, minsize=250)
        group_frame.columnconfigure(1, weight=1)

        for i, (feature, label, unit, field_type) in enumerate(fields):

            lbl = tk.Label(group_frame,
                    text=label + ":",
                    font=(FONT_FAMILY, 9),
                    bg=self.input_bg,
                    fg=self.fg_color,
                    anchor='w')
            lbl.grid(row=i, column=0, sticky='w', padx=15, pady=2)

            if field_type == 'categorical':
                var = tk.StringVar()
                if feature == 'jtype':
                    values = ['Interior', 'Exterior']
                    var.set('Interior')
                elif feature == 'fbrtype':
                    values = list(self.fiber_mapping.keys())
                    var.set(values[0])
                else:
                    values = []

                combo = ttk.Combobox(group_frame, textvariable=var, values=values,
                                    state='readonly', font=(FONT_FAMILY, 9))
                combo.grid(row=i, column=1, padx=15, pady=2, sticky='ew')
                self.input_vars[feature] = var
            else:
                var = tk.StringVar()
                entry = tk.Entry(group_frame, textvariable=var,
                               bg='#f8f9fa', fg='#212529',
                               font=(FONT_FAMILY, 9),
                               relief='solid', bd=1)
                entry.grid(row=i, column=1, padx=15, pady=2, sticky='ew')
                self.input_vars[feature] = var

    def get_range_info(self):
        info_lines = ["Valid Input Ranges (Training Data):", ""]

        for feature in self.selected_features:
            if feature not in self.categorical_cols:
                min_val = self.feature_ranges[feature]['min']
                max_val = self.feature_ranges[feature]['max']

                display_name = self.feature_display.get(feature, feature)
                info_lines.append(f"{display_name:8} : [{min_val:7.3f} - {max_val:7.3f}]")

        return "\n".join(info_lines)

    def predict(self):
        try:
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)

            input_data = {}
            out_of_range = []

            for feature, var in self.input_vars.items():
                value = var.get().strip()

                if feature in self.categorical_cols:
                    if feature == 'fbrtype':
                        input_data[feature] = self.fiber_mapping.get(value, value)
                    else:
                        input_data[feature] = value
                else:
                    if not value:
                        messagebox.showerror("Input Error", f"Please enter a value for {feature}")
                        return

                    try:
                        num_value = float(value)
                        input_data[feature] = num_value

                        min_val = self.feature_ranges[feature]['min']
                        max_val = self.feature_ranges[feature]['max']

                        if num_value < min_val or num_value > max_val:
                            out_of_range.append(
                                f"{self.feature_display.get(feature, feature)}: {num_value:.3f} (valid: {min_val:.3f}-{max_val:.3f})"
                            )
                    except ValueError:
                        messagebox.showerror("Input Error", f"Invalid number for {feature}: {value}")
                        return

            if out_of_range:
                warning_msg = "⚠ WARNING! Inputs outside training data range!\n\n"
                warning_msg += "\n".join(out_of_range)
                warning_msg += "\n\nPredictions may be unreliable!\nContinue?"

                if not messagebox.askyesno("Range Warning", warning_msg):
                    return

            input_df = pd.DataFrame([input_data])

            input_df = input_df[self.selected_features]

            dummy_data = {feature: 0 if feature not in self.categorical_cols else 'DUMMY'
                         for feature in self.selected_features}
            dummy_df = pd.DataFrame([dummy_data])

            combined_df = pd.concat([input_df, dummy_df], ignore_index=True)

            # drop_first encoding — for MLR and SVR
            combined_encoded = pd.get_dummies(combined_df, columns=self.categorical_cols, drop_first=True)
            input_encoded = combined_encoded.iloc[[0]].copy()
            for col in self.X_train.columns:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            input_encoded = input_encoded[self.X_train.columns]

            # Full one-hot encoding — for KNN, DT, RF, GB, XGB, MLP
            combined_encoded_oh = pd.get_dummies(combined_df, columns=self.categorical_cols, drop_first=False)
            input_encoded_oh = combined_encoded_oh.iloc[[0]].copy()
            for col in self.X_train_oh.columns:
                if col not in input_encoded_oh.columns:
                    input_encoded_oh[col] = 0
            input_encoded_oh = input_encoded_oh[self.X_train_oh.columns]

            input_lgb = input_df.copy()
            for col in self.categorical_cols:
                if col in input_lgb.columns:
                    input_lgb[col] = input_lgb[col].astype('category')

            input_cb = input_df.copy()
            for col in self.categorical_cols:
                if col in input_cb.columns:
                    input_cb[col] = input_cb[col].astype(str)

            predictions = {}

            for name, model in self.models.items():
                try:
                    if name == 'LightGBM':
                        pred = model.predict(input_lgb)[0]
                    elif name == 'CatBoost':
                        pred = model.predict(input_cb)[0]
                    elif name in ('Multiple Linear Regression', 'Support Vector Regression'):
                        pred = model.predict(input_encoded)[0]
                    else:
                        pred = model.predict(input_encoded_oh)[0]

                    predictions[name] = pred
                except Exception as e:
                    predictions[name] = None

            display_order = [
                'XGBoost',
                'Gradient Boosting',
                'LightGBM',
                'Support Vector Regression',
                'CatBoost',
                'Random Forest',
                'K-Nearest Neighbors',
                'Multi-Layer Perceptron',
                'Multiple Linear Regression',
                'Decision Tree']

            sorted_predictions = []
            for model_name in display_order:
                if model_name in predictions and predictions[model_name] is not None:
                    sorted_predictions.append((model_name, predictions[model_name]))

            for i, (name, pred) in enumerate(sorted_predictions, 1):
                self.results_tree.insert('', 'end', values=(i, name, f'{pred:.2f}'))

            # Update the info

            if out_of_range:
                warning_text = "\n\n⚠ WARNING: Inputs outside training range!"
                self.info_label.config(text=self.get_range_info() + warning_text,
                                      fg='#ff6b6b')
            else:
                self.info_label.config(text=self.get_range_info() + "\n\n✓ All inputs valid",
                                      fg='#51cf66')

        except Exception as e:
            messagebox.showerror("Prediction Error", f"Failed:\n{str(e)}")


# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = FRCPredictorApp(root)
    root.mainloop()
