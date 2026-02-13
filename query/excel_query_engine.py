import os
import logging
from typing import Dict, Optional
import pandas as pd
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.experimental.query_engine import PandasQueryEngine
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Istruzioni di default per il PandasQueryEngine
DEFAULT_PNL_INSTRUCTION = (
    "You are a strict Python coding assistant working on a pandas DataFrame `df`.\n"
    "Columns: 'Categoria', 'Mese', 'Importo', 'Commenti del CFO'.\n"
    "\n"
    "RULES FOR RESPONSE:\n"
    "1. Output ONLY valid Python code.\n"
    "2. DO NOT use markdown code blocks (no ```).\n"
    "3. DO NOT type the word 'python' or 'code'.\n"
    "4. DO NOT add explanations. Just the code.\n"
    "5. The last line must be the result expression (no print).\n"
    "6. TEXT SEARCH: When the user asks for a word (e.g. 'sanzioni'), search for its stem "
    "(e.g. 'sanzion') to match both singular and plural.\n"
    "\n"
    "CORRECT EXAMPLE:\n"
    "df[df['Mese'] == 'Agosto 2024']['Importo'].sum()\n"
    "\n"
    "WRONG EXAMPLE:\n"
    "```python\n"
    "print(df...)\n"
    "```"
)


class ExcelQueryEngine:
    """
    Interroga DataFrame Excel tramite PandasQueryEngine + LLM.
    Riceve i DataFrame dal ExcelLoader (Layer 1), non li carica direttamente.
    """

    def __init__(self, llm: Optional[OpenAI] = None):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Manca la variabile OPENAI_API_KEY!")

        self.llm = llm or OpenAI(model="gpt-4o", temperature=0.0, api_key=api_key)
        self.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=api_key)

        # Configura LlamaIndex globalmente
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

        self.engines: Dict[str, PandasQueryEngine] = {}
        logger.info("ExcelQueryEngine pronto.")

    def register_sheet(
        self,
        name: str,
        df: pd.DataFrame,
        instruction: Optional[str] = None,
        synthesize: bool = True,
    ):
        """
        Registra un DataFrame come sorgente interrogabile.

        Args:
            name: Nome logico (es. 'pnl', 'summary').
            df: DataFrame da rendere interrogabile.
            instruction: Istruzioni custom per il PandasQueryEngine.
            synthesize: Se True, l'LLM sintetizza la risposta.
        """
        kwargs = {
            "df": df,
            "verbose": True,
            "synthesize_response": synthesize,
            "llm": self.llm,
        }
        if instruction:
            kwargs["instruction_str"] = instruction

        self.engines[name] = PandasQueryEngine(**kwargs)
        logger.info(f"  Registrato foglio '{name}' ({len(df)} righe)")

    def ask(self, question: str, scope: str = "pnl") -> Dict:
        """
        Interroga un foglio registrato.

        Args:
            question: La domanda dell'utente.
            scope: Nome del foglio da interrogare.

        Returns:
            {"answer": str, "scope": str}
        """
        if scope not in self.engines:
            available = ", ".join(self.engines.keys()) or "nessuno"
            return {"answer": f"Ambito '{scope}' non trovato. Disponibili: {available}", "scope": scope}

        try:
            response = self.engines[scope].query(question)
            return {"answer": str(response), "scope": scope}
        except Exception as e:
            logger.error(f"Errore query Excel ({scope}): {e}")
            return {"answer": "Errore nell'elaborazione della query.", "scope": scope}

    def list_scopes(self) -> list:
        """Restituisce i nomi dei fogli registrati."""
        return list(self.engines.keys())
