"""
Diagnostikktester: verifiser at pipelinen KAN produsere en kandidat når gates slakkes.
Hvis disse feiler er problemet data/merge, ikke gates.

DoD: Alle tre tester kjører → du kan identifisere nøyaktig hvor "always cash" oppstår.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Hjelpefunksjon: repliserer fundamental_ok og technical_ok fra decision.py
# ---------------------------------------------------------------------------

def _compute_fundamental_ok(df: pd.DataFrame) -> pd.Series:
    """Repliserer fundamental_ok-logikken fra src/decision.py linje 3079–3087."""
    return (
        df["mos"].notna()
        & (df["mos"] >= df["mos_req"])
        & df["value_creation_ok"].fillna(False)
        & df["quality_gate_ok"].fillna(False)
        & df["dividend_min_score_ok"].fillna(True)
        & df["graham_min_score_ok"].fillna(True)
        & df["candidate_data_ok"].fillna(False)
        & ~df["data_quality_fail"].fillna(False)
    )


def _compute_technical_ok(df: pd.DataFrame) -> pd.Series:
    """Repliserer tech_ok (strict mode) fra src/decision.py linje 3005–3008."""
    tech_data_ready = (
        df["adj_close"].notna()
        & df["ma200"].notna()
        & df["index_ma200"].notna()
    )
    return (
        tech_data_ready
        & df["stock_ma200_ok"].astype(bool)
        & df["stock_mad_ok"].astype(bool)
        & df["index_ma200_ok"].astype(bool)
    )


# ---------------------------------------------------------------------------
# Fixture: syntetisk master-df med realistisk kolonndekning
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_master_df() -> pd.DataFrame:
    """
    Syntetisk master-df med 20 tickers.
    Simulerer en realistisk "always cash"-situasjon der ulike gates blokkerer
    ulike tickers. Brukes for å diagnostisere nøyaktig hvilken gate som dreper.
    """
    n = 20
    tickers = [f"T{i:02d}.OL" for i in range(n)]
    rng = np.random.default_rng(42)

    df = pd.DataFrame({
        "ticker": tickers,
        "yahoo_ticker": tickers,
        # Prisdata
        "adj_close": rng.uniform(50, 500, n),
        "ma200": rng.uniform(45, 480, n),
        "ma21": rng.uniform(48, 490, n),
        "mad": rng.uniform(-0.5, 2.0, n),
        # Indeksdata
        "index_ma200": rng.uniform(1000, 1500, n),
        "index_mad": rng.uniform(-0.3, 1.5, n),
        # Verdsettelse
        "intrinsic_value": rng.uniform(80, 600, n),
        "mos": rng.uniform(-0.1, 0.35, n),   # Mange under typisk krav på 15–30%
        "mos_req": np.full(n, 0.20),           # 20% krav
        # Gate-kolonner: satt til blokkerende verdier for å simulere "always cash"
        "value_creation_ok":      [True] * 12 + [False] * 8,
        "quality_gate_ok":        [True] * 15 + [False] * 5,
        "dividend_min_score_ok":  [True] * n,
        "graham_min_score_ok":    [True] * n,
        "candidate_data_ok":      [True] * 14 + [False] * 6,
        "data_quality_fail":      [False] * 16 + [True] * 4,
        # Tekniske gate-kolonner
        "stock_ma200_ok":   [True] * 13 + [False] * 7,
        "stock_mad_ok":     [True] * 14 + [False] * 6,
        "index_ma200_ok":   [True] * 15 + [False] * 5,
        "index_mad_ok":     [True] * n,
        "sector_group":     ["energy"] * 10 + ["bank"] * 5 + ["industrial"] * 5,
    })
    return df


# ---------------------------------------------------------------------------
# Test 1: Med alle gates åpne skal minst 1 ticker bli kandidat
# ---------------------------------------------------------------------------

def test_can_invest_with_all_gates_open(sample_master_df):
    """
    Test 1: MOS=0, alle gates=True → minst 1 kandidat.
    Hvis dette feiler: fundamental_ok-logikken er feil, ikke data.
    """
    df = sample_master_df.copy()

    # Slakk alle gates
    df["mos_req"] = 0.0                    # Ingen MOS-krav
    df["mos"] = df["mos"].abs() + 0.01     # Alle MOS positive
    df["value_creation_ok"] = True
    df["quality_gate_ok"] = True
    df["dividend_min_score_ok"] = True
    df["graham_min_score_ok"] = True
    df["candidate_data_ok"] = True
    df["data_quality_fail"] = False

    df["fundamental_ok"] = _compute_fundamental_ok(df)

    n_candidates = int(df["fundamental_ok"].sum())
    assert n_candidates >= 1, (
        f"DIAGNOSE: Ingen kandidater selv med alle gates åpne! "
        f"Problemet er i fundamental_ok-logikken eller kolonnene. "
        f"fundamental_ok-fordeling: {df['fundamental_ok'].value_counts().to_dict()}"
    )


def test_can_invest_gates_open_produces_all_candidates(sample_master_df):
    """Med alle gates åpne og positiv MOS skal alle 20 tickers bli kandidater."""
    df = sample_master_df.copy()
    df["mos_req"] = 0.0
    df["mos"] = 0.05
    df["value_creation_ok"] = True
    df["quality_gate_ok"] = True
    df["candidate_data_ok"] = True
    df["data_quality_fail"] = False
    df["dividend_min_score_ok"] = True
    df["graham_min_score_ok"] = True

    df["fundamental_ok"] = _compute_fundamental_ok(df)
    assert df["fundamental_ok"].all(), (
        f"Forventet alle kandidater men fikk: "
        f"{df[~df['fundamental_ok']]['ticker'].tolist()}"
    )


# ---------------------------------------------------------------------------
# Test 2: Teknisk filter produserer ≥10% technical_ok med gode prisdata
# ---------------------------------------------------------------------------

def test_technical_filter_passes_with_good_price_data(sample_master_df):
    """
    Test 2: Med gode prisdata skal tech-modulen produsere technical_ok=True
    for minst 10% av tickers.
    Hvis dette feiler: prisdata mangler, MA200 er ikke kalkulert, eller
    indeksdata er ikke hentet.
    """
    df = sample_master_df.copy()
    n = len(df)

    df["technical_ok"] = _compute_technical_ok(df)
    tech_ok_pct = df["technical_ok"].mean()

    assert tech_ok_pct >= 0.10, (
        f"DIAGNOSE: Kun {tech_ok_pct:.0%} av tickers passerer teknisk gate. "
        f"Sjekk: adj_close notna={df['adj_close'].notna().mean():.0%}, "
        f"ma200 notna={df['ma200'].notna().mean():.0%}, "
        f"index_ma200 notna={df['index_ma200'].notna().mean():.0%}, "
        f"stock_ma200_ok={df['stock_ma200_ok'].mean():.0%}, "
        f"stock_mad_ok={df['stock_mad_ok'].mean():.0%}, "
        f"index_ma200_ok={df['index_ma200_ok'].mean():.0%}"
    )


def test_technical_filter_all_ok_with_perfect_data(sample_master_df):
    """Med perfekte prisdata skal alle tickers passere teknisk gate."""
    df = sample_master_df.copy()
    df["adj_close"] = 300.0
    df["ma200"] = 280.0
    df["index_ma200"] = 1200.0
    df["stock_ma200_ok"] = True
    df["stock_mad_ok"] = True
    df["index_ma200_ok"] = True

    df["technical_ok"] = _compute_technical_ok(df)
    assert df["technical_ok"].all(), (
        f"Med perfekte data bør alle passere teknisk gate: "
        f"{df[~df['technical_ok']]['ticker'].tolist()}"
    )


# ---------------------------------------------------------------------------
# Test 3: Funnel-logging — teller per gate hvor mange som droppes
# ---------------------------------------------------------------------------

def test_funnel_logging_counts_per_gate(sample_master_df):
    """
    Test 3: For hver gate, tell hvor mange tickers som dropper ut og hvorfor.
    Output: {gate_name: {input_count, pass_count, drop_count, drop_pct}}
    Brukes for å diagnostisere nøyaktig hvilken gate som forårsaker "always cash".
    """
    df = sample_master_df.copy()
    n = len(df)

    # Sett realistisk mos_req (20%)
    df["fundamental_ok"] = _compute_fundamental_ok(df)
    df["technical_ok"] = _compute_technical_ok(df)

    funnel = _compute_gate_funnel(df)

    # Grunnleggende strukturkrav
    assert "mos_gate" in funnel
    assert "value_creation_gate" in funnel
    assert "quality_gate" in funnel
    assert "candidate_data_gate" in funnel
    assert "data_quality_gate" in funnel
    assert "technical_gate" in funnel
    assert "combined_gate" in funnel

    for gate_name, stats in funnel.items():
        assert "pass_count" in stats
        assert "drop_count" in stats
        assert "drop_pct" in stats
        assert 0 <= stats["pass_count"] <= n
        assert stats["drop_count"] >= 0
        assert 0.0 <= stats["drop_pct"] <= 1.0

    # combined_gate skal telle tickers der BEGGE fundamental_ok og technical_ok er True
    combined_expected = int((df["fundamental_ok"] & df["technical_ok"]).sum())
    assert funnel["combined_gate"]["pass_count"] == combined_expected, (
        f"combined_gate.pass_count={funnel['combined_gate']['pass_count']} "
        f"!= forventet {combined_expected}"
    )


def test_funnel_identifies_dominant_blocker(sample_master_df):
    """
    Funnelen skal identifisere hvilken gate som blokkerer flest tickers.
    I sample_master_df er MOS-gate (krav 20%) typisk den dominante blokkeren.
    """
    df = sample_master_df.copy()
    funnel = _compute_gate_funnel(df)

    # Finn gaten med flest drops
    dominant = max(
        (g for g in funnel if g != "combined_gate"),
        key=lambda g: funnel[g]["drop_count"],
    )
    # Bekrefter at en gate faktisk blokkerer noe (ikke tom analyse)
    assert funnel[dominant]["drop_count"] > 0, (
        "DIAGNOSE: Ingen gate blokkerer noe — sjekk om fixture-dataene er korrekte."
    )


def test_funnel_always_cash_diagnosis(sample_master_df):
    """
    Scenario: 'always cash' — combined_gate.pass_count == 0.
    Testen verifiserer at funnelen korrekt identifiserer hvilke gates som er årsak.
    """
    df = sample_master_df.copy()

    # Tving "always cash": sett mos under krav for alle
    df["mos"] = -0.50  # Alle langt under krav

    funnel = _compute_gate_funnel(df)

    assert funnel["combined_gate"]["pass_count"] == 0, (
        "Forventet 0 kandidater med mos=-50% for alle"
    )
    # MOS-gate bør identifiseres som blokkerer
    assert funnel["mos_gate"]["pass_count"] == 0, (
        "mos_gate bør ha 0 pass når mos=-50% for alle"
    )


# ---------------------------------------------------------------------------
# Hjelpefunksjon: beregn funnel per gate
# ---------------------------------------------------------------------------

def _compute_gate_funnel(df: pd.DataFrame) -> dict:
    """
    Beregner per-gate statistikk for å diagnostisere "always cash".

    Returns:
        dict[gate_name -> {pass_count, drop_count, drop_pct}]
    """
    n = len(df)

    def _stats(mask: pd.Series) -> dict:
        pass_count = int(mask.sum())
        drop_count = n - pass_count
        return {
            "pass_count": pass_count,
            "drop_count": drop_count,
            "drop_pct": drop_count / n if n > 0 else 0.0,
        }

    mos_ok = df["mos"].notna() & (df["mos"] >= df["mos_req"])
    vc_ok = df["value_creation_ok"].fillna(False).astype(bool)
    q_ok = df["quality_gate_ok"].fillna(False).astype(bool)
    div_ok = df["dividend_min_score_ok"].fillna(True).astype(bool)
    graham_ok = df["graham_min_score_ok"].fillna(True).astype(bool)
    cand_ok = df["candidate_data_ok"].fillna(False).astype(bool)
    dq_ok = ~df["data_quality_fail"].fillna(False).astype(bool)

    fundamental_ok = _compute_fundamental_ok(df)
    technical_ok = _compute_technical_ok(df)

    return {
        "mos_gate":             _stats(mos_ok),
        "value_creation_gate":  _stats(vc_ok),
        "quality_gate":         _stats(q_ok),
        "dividend_gate":        _stats(div_ok),
        "graham_gate":          _stats(graham_ok),
        "candidate_data_gate":  _stats(cand_ok),
        "data_quality_gate":    _stats(dq_ok),
        "fundamental_gate":     _stats(fundamental_ok),
        "technical_gate":       _stats(technical_ok),
        "combined_gate":        _stats(fundamental_ok & technical_ok),
    }
