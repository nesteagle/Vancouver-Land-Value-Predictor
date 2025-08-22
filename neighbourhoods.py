NEIGHBOURHOOD_CODE_NAME = {
    1: "West Point Grey",
    2: "Kitsilano",
    3: "Dunbar-Southlands",
    4: "Arbutus Ridge",
    5: "Kerrisdale",
    6: "Southlands",
    7: "Fairview",
    8: "Shaughnessy",
    9: "South Cambie-Main",
    10: "South Granville-Oakridge West",
    11: "East Oakridge",
    12: "Marpole",
    13: "Mount Pleasant, Strathcona",
    14: "Grandview-Woodland",
    15: "Dickens",
    16: "Main-Riley Park",
    17: "Sunset",
    18: "Marine Drive",
    19: "Kensington-Cedar Cottage",
    20: "North Hastings-Sunrise",
    21: "South Hastings-Sunrise",
    22: "Renfrew Heights",
    23: "Renfrew-Collingwood",
    24: "Killarney",
    25: "Champlain Heights",
    26: "Downtown",
    27: "West End",
    28: "Waterfront",
    29: "Yaletown North",
    30: "Yaletown"
}

def build_neighbourhood_maps(codes):
    display_to_code = {}
    for c in codes:
        name = NEIGHBOURHOOD_CODE_NAME.get(c, f"Neighbourhood {c}")
        display_to_code[name] = c
    code_to_display = {v: k for k, v in display_to_code.items()}
    return display_to_code, code_to_display