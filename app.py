from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
import gradio as gr
from preprocessing import data
from neighbourhoods import build_neighbourhood_maps
from visualization import plot_neighbourhoods

REQUIRED_COLS = [
    "CURRENT_LAND_VALUE",
    "CURRENT_IMPROVEMENT_VALUE",
    "BIG_IMPROVEMENT_YEAR",
    "ZONING_CLASSIFICATION",
    "LEGAL_TYPE",
    "NEIGHBOURHOOD_CODE",
    "REPORT_YEAR",
]

df = data.dropna(subset=[c for c in REQUIRED_COLS if c in data.columns]).copy()

df = df[df["LEGAL_TYPE"] != "OTHER"]

# drop these unless price of factories/malls/unbuyable buildings wanted
df = df[
    ~df["ZONING_CLASSIFICATION"].isin(
        [
            "Comprehensive Development",
            "Historical Area",
            "Industrial",
            "Other",
            "Commercial",
        ]
    )
]


df["BIG_IMPROVEMENT_YEAR"] = pd.to_numeric(df["BIG_IMPROVEMENT_YEAR"], errors="coerce")
df["REPORT_YEAR"] = pd.to_numeric(df["REPORT_YEAR"], errors="coerce")
df = df.dropna(subset=["BIG_IMPROVEMENT_YEAR", "REPORT_YEAR", "CURRENT_LAND_VALUE"])

# remove impossible / ultra-extreme
df = df[
    (df["BIG_IMPROVEMENT_YEAR"] > 1900)
    & (df["CURRENT_LAND_VALUE"] > 1e4)
    & (df["CURRENT_IMPROVEMENT_VALUE"] > 1e4)
    & (df["CURRENT_LAND_VALUE"] < 5e7)
]


df["AGE_SINCE_IMPROVEMENT"] = df["REPORT_YEAR"] - df["BIG_IMPROVEMENT_YEAR"]

CATEGORICALS = ["ZONING_CLASSIFICATION", "LEGAL_TYPE", "NEIGHBOURHOOD_CODE"]
cat_frame = df[CATEGORICALS].astype(str)
X_cat = pd.get_dummies(cat_frame, prefix=CATEGORICALS, drop_first=False)

X_num = df[["BIG_IMPROVEMENT_YEAR", "REPORT_YEAR", "AGE_SINCE_IMPROVEMENT"]]

X = pd.concat([X_num, X_cat], axis=1)

df["LOG_LAND_VALUE"] = np.log1p(df["CURRENT_LAND_VALUE"])

y = df["LOG_LAND_VALUE"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = HistGradientBoostingRegressor(random_state=1)
model.fit(X_train, y_train)

pred_test = model.predict(X_test)

rmse_log = root_mean_squared_error(y_test, pred_test)
rmse_original = root_mean_squared_error(np.expm1(y_test), np.expm1(pred_test))
print(f"RMSE (log scale): {rmse_log:.4f}")
print(f"RMSE (original): {rmse_original:,.2f}")
print(f"R^2: {r2_score(y_test, pred_test):.3f}")

FEATURE_COLUMNS = X.columns.tolist()

zoning_values = sorted(df["ZONING_CLASSIFICATION"].dropna().unique())
legal_values = sorted(df["LEGAL_TYPE"].dropna().unique())
neigh_values = sorted(df["NEIGHBOURHOOD_CODE"].dropna().unique())

neigh_display_map, neigh_code_to_name = build_neighbourhood_maps(neigh_values)

neigh_dropdown_choices = list(neigh_display_map.keys())


def predict_land_value(improve_year, zoning_choice, legal_choice, neighborhood_display):
    if improve_year is None:
        return "Provide improvement/build year."
    report_year = datetime.now().year
    if improve_year > report_year:
        return "Improvement year cannot exceed current year."

    row = pd.Series(0.0, index=FEATURE_COLUMNS)
    row["BIG_IMPROVEMENT_YEAR"] = improve_year
    row["REPORT_YEAR"] = report_year
    row["AGE_SINCE_IMPROVEMENT"] = report_year - improve_year

    n_raw = neigh_display_map[neighborhood_display]
    z_raw = zoning_choice
    l_raw = legal_choice

    for col in (
        f"ZONING_CLASSIFICATION_{z_raw}",
        f"LEGAL_TYPE_{l_raw}",
        f"NEIGHBOURHOOD_CODE_{n_raw}",
    ):
        if col in row.index:
            row[col] = 1.0

    input_df = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    pred_log = model.predict(input_df)[0]
    pred_val = np.expm1(pred_log)
    return f"Predicted Land Value: ${pred_val:,.2f}"


inputs = [
    gr.Number(label="Big Improvement / Build Year", value=1970, precision=0),
    gr.Dropdown(
        choices=zoning_values, label="Zoning Classification", value=zoning_values[2]
    ),
    gr.Dropdown(choices=legal_values, label="Legal Type", value=legal_values[0]),
    gr.Dropdown(
        choices=neigh_dropdown_choices,
        label="Neighbourhood",
        value=neigh_dropdown_choices[0],
    ),
]

demo1 = gr.Interface(
    fn=predict_land_value,
    inputs=inputs,
    outputs="text",
    title="Land Value Prediction",
    description=f"""
        Predicts Vancouver land values using machine learning. 
        Built with property tax data from Vancouver Open Data.
        \n 
        Accuracy: R² Score: {r2_score(y_test, pred_test):.3f}  
        RMSE: ${rmse_original:,.0f}  
        Training samples after cleaning+filtering: {len(X_train):,}
    """,
)

with gr.Blocks() as demo2:
    with gr.Row():
        input_col = gr.Column(scale=1)
        output_col = gr.Column(scale=3)
    
    with input_col:
        instructions = gr.Markdown("Adjust the z-score to visualize results at different amounts of filtering.")
        max_sd_input = gr.Number(label="Max z-score (# SD away from mean)", value=1.25, step=0.25)
        submit_btn = gr.Button("Plot")
    
    with output_col:
        disclaimer = gr.Markdown("Data is sourced from Vancouver OpenData/BCAssessment, and should serve as a rough estimate. No semantic definitions for the neighbourhood codes were provided. Estimations of neighborhood areas are based on this data.", visible=True)
        plot_out = gr.Plot()

    submit_btn.click(fn=plot_neighbourhoods, inputs=max_sd_input, outputs=plot_out)
    demo2.load(fn=plot_neighbourhoods, inputs=max_sd_input, outputs=plot_out)

demo = gr.TabbedInterface([demo1, demo2], ["Land Value", "Neighbourhood Bounds Plotting"])

demo.launch()
