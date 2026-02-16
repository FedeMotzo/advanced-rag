import os
import logging
from typing import Dict, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class ExcelLoader:
    """
    Carica fogli Excel e li mantiene come DataFrame preprocessati in memoria.
    """

    def __init__(self, excel_path: str):
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"File non trovato: {excel_path}")

        self.excel_path = excel_path
        self.sheets: Dict[str, pd.DataFrame] = {}
        self._load_sheets()

    def _load_sheets(self):
        """Carica dinamicamente tutti i fogli dal file Excel."""
        logger.info(f"Caricamento dati: {self.excel_path}")

        xls = pd.ExcelFile(self.excel_path)
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            slug = sheet_name.lower().replace(" ", "_")
            self.sheets[slug] = df
            logger.info(f"  Foglio '{sheet_name}' → '{slug}': {len(df)} righe, {len(df.columns)} colonne")

        logger.info(f"✅ Caricati {len(self.sheets)} fogli da {os.path.basename(self.excel_path)}")

    def get_dataframe(self, sheet: str) -> Optional[pd.DataFrame]:
        """
        Restituisce il DataFrame preprocessato di un foglio.
        """
        return self.sheets.get(sheet)

    def list_sheets(self) -> list:
        """Restituisce i nomi dei fogli caricati."""
        return list(self.sheets.keys())
