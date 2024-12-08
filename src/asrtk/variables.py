blacklist = [
    # "(jenerik.)",
    # "(jenerik)",
    ".",
    # "(müzik)",
    # "[jenerik.]",
    # "[jenerik]",
    # "[müzik]",
    # "müzik]",
    # "[müzik",
    # "(müzik",
    # "(...)",
    "*müzik*",
    "İNTRO",
    "-filmden sesler-",
]
# ♫ Müzik sesleri ♫

punctuation_dict = {
    "?": "soru işareti",
    "/": "bölü",
    # ";": "noktalıvirgül",
    # "(": "aç parantez",
    # ")": "kapa parantez",
    # Add more mappings as needed
}


unit_translations = {
    "cc": "si-si",
    "mm": "milimetre",
    "cm": "santimetre",
    "dm": "desimetre",
    "m": "metre",
    "km": "kilometre",
    "g": "gram",
    "kg": "kilogram",
    "ml": "mililitre",
    "in": "inç",
    "ft": "feet",
    "yd": "yard",
    "mg": "miligram",
    "oz": "ons",
    "lb": "pound",
    "st": "stone",
    "l": "litre",
    "dl": "desilitre",
    "cl": "santilitre",
    "gal": "galon",
    "pt": "pint",
    "fl oz": "sıvı ons",
    "sq mm": "milimetre kare",
    "sq cm": "santimetre kare",
    "sq m": "metre kare",
    "acre": "akre",
    "hectare": "hektar",
    "j": "jul",
    "kj": "kilojul",
    "cal": "kalori",
    "kcal": "kilokalori",
    "wh": "watt saat",
    "kwh": "kilowatt saat",
    "°c": "santigrat derece",
    "°C": "santigrat derece",
    "c°": "santigrat derece",
    "C°": "santigrat derece",
    "°f": "derece fahrenheit",
    "k": "kelvin",
    "mph": "mil/saat",
    "km/h": "kilometre/saat",
    "km/s": "kilometre/saat",
    "km": "kilometre",
    "knot": "düğüm",
    "pa": "paskal",
    "kpa": "kilopaskal",
    "mpa": "megapaskal",
    "bar": "bar",
    "psi": "pound/inç kare",
}

abbreviations_dict = {}

special_cases = {
    "°": " derece",
}

if __name__ == "__main__":
    from asrtk.normalizer import apply_normalizers

    test_sentences = [
        "Cümledeki kelime sayısı: 5",
    ]

    for ts in test_sentences:
        print(apply_normalizers(ts))
