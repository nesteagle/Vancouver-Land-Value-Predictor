import pandas as pd
from huggingface_hub import hf_hub_download

VANCOUVER_LON_BOUNDS = (-123.3, -123)
VANCOUVER_LAT_BOUNDS = (49.2, 49.3)

prop_path = hf_hub_download(repo_id="nesteagle/VanProperty", filename="vancouver-property-tax-report.csv", repo_type="dataset")
postal_path = hf_hub_download(repo_id="nesteagle/VanProperty", filename="CA_full.txt", repo_type="dataset")
data = (
    pd.read_csv(prop_path, delimiter=";")
      .dropna(subset=["NEIGHBOURHOOD_CODE", "PROPERTY_POSTAL_CODE", "LAND_COORDINATE"])
)
ca_postal_df = pd.read_csv(
    postal_path,
    delimiter="\t",
    header=None,
    names=[
        "country","postal_code","place_name","province","province_code",
        "c6","c7","c8","c9","latitude","longitude","accuracy"
    ],
    dtype={"postal_code": "string"},
    low_memory=False
)

bc_postal_df = ca_postal_df[ca_postal_df["postal_code"].str.startswith("V")]


for col in ["CURRENT_LAND_VALUE", "CURRENT_IMPROVEMENT_VALUE"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")

lat_dict = bc_postal_df.set_index("postal_code")["latitude"].to_dict()
lon_dict = bc_postal_df.set_index("postal_code")["longitude"].to_dict()

data["latitude"] = data["PROPERTY_POSTAL_CODE"].map(lat_dict)
data["longitude"] = data["PROPERTY_POSTAL_CODE"].map(lon_dict)

data = data.dropna(subset=["latitude", "longitude"])
data = data[
    data["latitude"].between(*VANCOUVER_LAT_BOUNDS)
    & data["longitude"].between(*VANCOUVER_LON_BOUNDS)
]
