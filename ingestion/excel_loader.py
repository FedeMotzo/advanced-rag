import os
import logging
from typing import Dict, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class ExcelLoader:
    """
    Carica fogli Excel e li mantiene come DataFrame preprocessati in memoria.
    Nessuna logica di query — solo loading e preprocessing.
    """

    def __init__(self, excel_path: str):
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"File non trovato: {excel_path}")

        self.excel_path = excel_path
        self.sheets: Dict[str, pd.DataFrame] = {}
        self._load_sheets()

    def _load_sheets(self):
        """Carica tutti i fogli rilevanti dal file Excel."""
        logger.info(f"Caricamento dati: {self.excel_path}")

        # Executive Summary — caricato così com'è
        try:
            df_summary = pd.read_excel(self.excel_path, sheet_name='Executive Summary')
            self.sheets['summary'] = df_summary
            logger.info(f"  Foglio 'Executive Summary': {len(df_summary)} righe")
        except Exception as e:
            logger.warning(f"  Foglio 'Executive Summary' non trovato: {e}")

        # P&L Dettagliato — preprocessato da wide a long
        try:
            df_pnl_raw = pd.read_excel(self.excel_path, sheet_name='P&L Dettagliato')
            self.sheets['pnl'] = self._preprocess_pnl(df_pnl_raw)
            logger.info(f"  Foglio 'P&L Dettagliato': {len(self.sheets['pnl'])} righe (dopo reshape)")
        except Exception as e:
            logger.warning(f"  Foglio 'P&L Dettagliato' non trovato: {e}")

        logger.info(f"✅ Caricati {len(self.sheets)} fogli da {os.path.basename(self.excel_path)}")

    def _preprocess_pnl(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trasforma il P&L da formato wide a formato long per l'interrogazione."""
        cols_to_keep = ['Categoria', 'Commenti del CFO']
        id_vars = [c for c in cols_to_keep if c in df.columns]
        value_vars = [c for c in df.columns if c not in id_vars]
        df_long = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='Mese', value_name='Importo')
        df_long['Importo'] = pd.to_numeric(df_long['Importo'], errors='coerce').fillna(0)
        return df_long

    def get_dataframe(self, sheet: str) -> Optional[pd.DataFrame]:
        """
        Restituisce il DataFrame preprocessato di un foglio.

        Args:
            sheet: Nome logico del foglio ('summary', 'pnl').

        Returns:
            DataFrame o None se il foglio non esiste.
        """
        return self.sheets.get(sheet)

    def list_sheets(self) -> list:
        """Restituisce i nomi dei fogli caricati."""
        return list(self.sheets.keys())
