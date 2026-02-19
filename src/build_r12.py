import pandas as pd
from pathlib import Path

RAW = Path(r"G:\Min disk\Børsdata\data\raw")
INFILE = RAW / "reports_quarter.csv"
OUTFILE = RAW / "reports_r12_from_quarters.csv"

df = pd.read_csv(INFILE)

# ---- 1) Finn/normaliser nøkkelkolonner ----
# InstrumentId
id_candidates = ["InstrumentId", "InstrumentID", "InstId", "instrumentid", "instrument_id"]
inst_col = next((c for c in id_candidates if c in df.columns), None)
if not inst_col:
    raise ValueError(f"Fant ikke InstrumentId-kolonne. Kolonner: {list(df.columns)[:30]}")

# Rapport sluttdato (best)
date_candidates = ["Rapport slutdatum", "Report end date", "ReportEndDate", "report_end_date", "Rapport slutdato", "Rapport slutt"]
date_col = next((c for c in date_candidates if c in df.columns), None)
if not date_col:
    # fallback: prøv År + Kvartal hvis finnes
    y_candidates = ["År", "Year", "year"]
    q_candidates = ["Kvartal", "Quarter", "quarter", "Period"]
    y_col = next((c for c in y_candidates if c in df.columns), None)
    q_col = next((c for c in q_candidates if c in df.columns), None)
    if not (y_col and q_col):
        raise ValueError("Fant ikke 'Rapport slutdatum' og heller ikke (År + Kvartal/Period). "
                         "Legg til en sluttdato-kolonne i eksporten (anbefalt).")
    # Lag en dato (slutten av kvartalet) grovt:
    q_map = {"Q1": "03-31", "Q2": "06-30", "Q3": "09-30", "Q4": "12-31"}
    def q_to_end(q):
        qs = str(q).upper()
        for k,v in q_map.items():
            if k in qs:
                return v
        return "12-31"
    df["_end_date"] = pd.to_datetime(df[y_col].astype(str) + "-" + df[q_col].apply(q_to_end), errors="coerce")
    date_col = "_end_date"
else:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

if df[date_col].isna().all():
    raise ValueError(f"Kunne ikke parse dato fra kolonnen '{date_col}'. Sjekk datoformat i CSV.")

# ---- 2) Velg hvilke felt som skal summeres (flow) og hvilke som tas som siste (stock) ----
# Du kan utvide denne listen etter behov. Scriptet bruker bare de som faktisk finnes i CSV.
flow_fields_candidates = [
    "Total intäkter", "Omsättning", "Bruttoresultat", "Rörelseresultat",
    "Resultat Före Skatt", "Åretskassaflöde", "FrittKassaflöde",
    "Kassaf LöpandeVerk", "Kassaf Investeringsverk", "Kassaf Finansieringsverk",
    "Resultat Hänföring Aktieägare"
]
stock_fields_candidates = [
    "Summa Eget Kapital", "Nettoskuld", "Summa Tillgångar",
    "Kassa/Bank", "Långfristiga skulder", "Kortfristiga skulder",
    "Antal Aktier "
]

flow_fields = [c for c in flow_fields_candidates if c in df.columns]
stock_fields = [c for c in stock_fields_candidates if c in df.columns]

if not flow_fields:
    raise ValueError("Fant ingen kjente 'flow'-felt (omsetning/EBIT/FCF osv.) i CSV. "
                     "Legg til minst Omsättning, Rörelseresultat eller FrittKassaflöde.")
# stock_fields kan være tomt – men da får du R12 uten balanseposter.

# Sikre numeriske felt (komma/space kan forekomme i noen exporter)
def to_num(s: pd.Series) -> pd.Series:
    return (s.astype(str)
              .str.replace("\u00a0", "", regex=False)   # non-breaking space
              .str.replace(" ", "", regex=False)
              .str.replace(",", ".", regex=False)
              .replace({"nan": None, "None": None})
              .astype(float))

for c in flow_fields + stock_fields:
    df[c] = to_num(df[c])

# ---- 3) Bygg R12: ta 4 siste kvartaler per selskap ----
df = df.sort_values([inst_col, date_col])

# Behold bare rader med dato og minst ett flow-felt
df = df.dropna(subset=[date_col])

def build_r12(group: pd.DataFrame) -> pd.Series:
    g = group.sort_values(date_col).tail(4)  # siste 4 kvartaler
    out = {}

    # sum flows
    for c in flow_fields:
        out[f"R12_{c}"] = g[c].sum(min_count=1)

    # last stocks (fra siste kvartal)
    last = g.iloc[-1]
    for c in stock_fields:
        out[c] = last[c]

    # metadata
    out["R12_end_date"] = last[date_col]
    out["quarters_used"] = len(g)
    return pd.Series(out)

r12 = df.groupby(inst_col, dropna=False).apply(build_r12).reset_index()

# Filtrer bort selskaper uten 4 kvartaler (valgfritt, men anbefalt)
r12 = r12[r12["quarters_used"] == 4].copy()

# Lagre
r12.to_csv(OUTFILE, index=False, encoding="utf-8")
print("OK ->", OUTFILE)
print("Selskaper med R12:", len(r12))
print("Kolonner:", list(r12.columns)[:15], "...")
