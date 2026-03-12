"""
Agent C: Decision Dossier Writer (no-veto, no-math).
Produserer revisjonsvennlig beslutningsnarrativ. Kan ikke introdusere nye tall eller endre beslutninger.
Bruker Anthropic Claude API.
"""
import json
import logging

import anthropic

from src.agents.schemas import DossierInput, DossierOutput

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Du er Decision Dossier Writer for Deecon-systemet.

ROLLE: Skriv en revisjonsvennlig, menneskelig lesbar beslutningsrapport på norsk (bokmål).

BEGRENSNINGER (IKKE-FORHANDLBARE):
1. Du KAN IKKE endre eller overstyre beslutningen (KANDIDAT eller CASH).
2. Du KAN IKKE introdusere nye tall. Alle tall må komme fra input-dataene.
3. Du KAN IKKE beregne ratioer, multipler eller verdsettelser.
4. Hvert påstand må referere til et spesifikt datapunkt fra input.
5. Skriv på klart, profesjonelt norsk (bokmål).

STRUKTUR:
- Beslutningssammendrag (1 avsnitt)
- Verdsettelse og margin of safety (referer eksakte tall fra input)
- Risikofaktorer (fra skeptiker-funn hvis tilgjengelig)
- Kvalitetsvurdering (fra kvalitetsevaluering hvis tilgjengelig)
- Datakvalitet (sammendrag av DQ-status)
- Teknisk status (MA200/MAD/indeks)

OUTPUT: Svar KUN med gyldig JSON som matcher dette skjemaet:
{
  "agent": "decision_dossier_writer",
  "version": "1.0",
  "ticker": "<string>",
  "narrative": "<string, maks 5000 tegn — fullstendig rapport i markdown>",
  "key_risks": ["<string>"],
  "key_strengths": ["<string>"],
  "data_quality_summary": "<string>"
}"""


def run_dossier_writer(
    input_data: DossierInput,
    client: anthropic.Anthropic,
    model: str = "claude-sonnet-4-6",
    max_retries: int = 2,
) -> DossierOutput:
    """
    Kjør Decision Dossier Writer.
    Ved feil returneres minimalt gyldig output.
    """
    user_content = json.dumps(input_data.model_dump(), indent=2, ensure_ascii=False)

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4000,
                temperature=0.0,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )
            raw_text = response.content[0].text.strip()

            # Strip markdown code fences if present
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
                raw_text = raw_text.strip()

            output = DossierOutput.model_validate_json(raw_text)
            return output

        except Exception as e:
            logger.warning(f"Dossier writer forsøk {attempt + 1} feilet: {e}")

    logger.error(f"Dossier writer feilet for {input_data.ticker}")
    return DossierOutput(
        ticker=input_data.ticker,
        narrative=f"## {input_data.ticker} — {input_data.final_decision}\n\nAutomatisk rapport kunne ikke genereres.",
        key_risks=["Rapportgenerering feilet"],
        data_quality_summary="Ukjent — rapportgenerering feilet",
    )
