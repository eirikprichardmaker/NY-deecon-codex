import os
import json
import requests
from src.common.utils import get_first_env
from dotenv import load_dotenv

load_dotenv()

AUTHKEY = get_first_env(["BORSDATA_AUTHKEY", "BORSDATA_API_KEY", "BORSDATA_KEY"])
if not AUTHKEY:
    raise SystemExit("Mangler BORSDATA_AUTHKEY. Sjekk .env-filen.")

BASE = "https://apiservice.borsdata.se"
url = f"{BASE}/v1/instruments"

r = requests.get(url, params={"authKey": AUTHKEY}, timeout=30)
print("Status:", r.status_code)

# Skriv litt respons for feilsÃ¸king
if r.status_code != 200:
    print(r.text[:500])
    raise SystemExit("Kallet feilet. Sjekk authKey/abonnement/rate-limit.")

data = r.json()
print("NÃ¸kler i respons:", list(data.keys()) if isinstance(data, dict) else type(data))

# Lagre rÃ¥data lokalt (ikke i Git)
os.makedirs("data/raw", exist_ok=True)
with open("data/raw/instruments.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("OK: Lagret data/raw/instruments.json")


