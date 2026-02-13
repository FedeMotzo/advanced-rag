import logging
import re
from typing import Optional, Tuple, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

# Prompt base — gli scopes Excel vengono iniettati dinamicamente
ROUTER_SYSTEM = (
    "Sei un classificatore di intent. Data una domanda dell'utente, "
    "determina quale fonte dati è la più appropriata per rispondere.\n\n"
    "Fonti disponibili:\n"
    "- text: documenti normativi (regolamenti EU, direttive AI), verbali di riunioni, "
    "documenti legali, policy aziendali\n"
    "{excel_scopes_description}\n\n"
    "Rispondi ESCLUSIVAMENTE con una delle opzioni elencate sopra.\n"
    "Nessuna altra parola o spiegazione."
)

# Fallback quando non ci sono file Excel caricati
NO_EXCEL_DESCRIPTION = "Non ci sono fonti Excel disponibili."


class IntentRouter:
    """
    Classificatore di intent basato su LLM.
    Supporta scopes Excel dinamici (multi-file).
    """

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self._parser = StrOutputParser()
        logger.info("IntentRouter inizializzato.")

    def classify(
        self, question: str, available_scopes: Optional[List[dict]] = None
    ) -> Tuple[str, Optional[str]]:
        """
        Classifica la domanda.

        Args:
            question: La domanda dell'utente.
            available_scopes: Lista di scope disponibili, ognuno con:
                - name: nome dello scope (es. "financial_report_q3_pnl")
                - description: descrizione del contenuto
                Può essere None se non ci sono file Excel.

        Returns:
            Tupla (source, scope):
            - ("text", None) per documenti testuali
            - ("excel", "scope_name") per dati Excel
        """
        # Costruisci la descrizione degli scopes Excel
        if available_scopes:
            lines = []
            for s in available_scopes:
                lines.append(f"- {s['name']}: {s['description']}")
            excel_desc = "Fonti Excel disponibili:\n" + "\n".join(lines)
        else:
            excel_desc = NO_EXCEL_DESCRIPTION

        prompt = ChatPromptTemplate.from_messages([
            ("system", ROUTER_SYSTEM.format(excel_scopes_description=excel_desc)),
            ("human", "{question}"),
        ])

        chain = prompt | self.llm | self._parser

        try:
            result = chain.invoke({"question": question}).strip().lower()

            # Cerca match con uno scope disponibile
            if available_scopes:
                for s in available_scopes:
                    if s["name"].lower() in result:
                        return ("excel", s["name"])

            # Fallback: se contiene "excel" generico ma nessun match
            if "excel" in result:
                # Ritorna il primo scope disponibile come default
                if available_scopes:
                    return ("excel", available_scopes[0]["name"])
                return ("excel", None)

            return ("text", None)

        except Exception as e:
            logger.warning(f"Errore nel router, fallback a 'text': {e}")
            return ("text", None)
