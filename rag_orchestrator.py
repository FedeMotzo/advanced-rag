import os
import re
import logging
from typing import Optional, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from ingestion.text_loader import TextLoader
from ingestion.excel_loader import ExcelLoader
from ingestion.chunking_strategies import RecursiveChunkingStrategy, BaseChunkingStrategy
from query.rag_query_engine import RAGQueryEngine
from query.excel_query_engine import ExcelQueryEngine, DEFAULT_INSTRUCTION
from query.intent_router import IntentRouter
from vector_db.vector_store_manager import VectorStoreManager

load_dotenv()
logger = logging.getLogger(__name__)


class RAGOrchestrator:
    """
    Punto di ingresso unificato per il sistema RAG.
    Coordina ingestion e query su documenti testuali (PDF/DOCX) ed Excel.
    """

    def __init__(
        self,
        db_manager: Optional[VectorStoreManager] = None,
        chunking_strategy: Optional[BaseChunkingStrategy] = None,
        llm: Optional[ChatOpenAI] = None,
        reranker: bool = False,
        hybrid_search: bool = False,
    ):
        # Infrastruttura condivisa
        self.db_manager = db_manager or VectorStoreManager()
        self.chunking_strategy = chunking_strategy or RecursiveChunkingStrategy()

        # Layer 1 — Ingestion
        self.text_loader = TextLoader(self.db_manager, self.chunking_strategy)
        self.excel_loader: Optional[ExcelLoader] = None

        # Layer 2 — Query
        self.rag_engine = RAGQueryEngine(
            self.db_manager, 
            llm=llm, 
            reranker=reranker, 
            hybrid_search=hybrid_search
        )
        self.excel_engine: Optional[ExcelQueryEngine] = None

        # Intent Router (classificatore automatico)
        self.intent_router = IntentRouter()

        # Registro scopes Excel con descrizioni (per il router)
        self._scope_registry: List[dict] = []

        self.ingested_files: list = []
        logger.info("RAGOrchestrator inizializzato.")

    def ingest(self, file_path: str):
        """
        Carica un file nel sistema. Routing automatico per tipo:
        - PDF/DOCX → TextLoader → VectorStore
        - XLSX → ExcelLoader → ExcelQueryEngine
        """
        ext = os.path.splitext(file_path)[1].lower()

        if ext in TextLoader.SUPPORTED_EXTENSIONS:
            n = self.text_loader.process_file(file_path)
            if n > 0:
                self.ingested_files.append(file_path)
            return n

        elif ext in {".xlsx", ".xls"}:
            self.excel_loader = ExcelLoader(file_path)

            # Inizializza l'ExcelQueryEngine se non ancora fatto
            if self.excel_engine is None:
                self.excel_engine = ExcelQueryEngine()

            # Genera prefisso unico dal nome file
            file_slug = self._file_slug(file_path)

            # Registra tutti i fogli con scope prefissato
            for sheet_name in self.excel_loader.list_sheets():
                df = self.excel_loader.get_dataframe(sheet_name)
                scoped_name = f"{file_slug}_{sheet_name}"

                # Genera istruzione e descrizione per il router
                cols = list(df.columns)
                description = f"File '{os.path.basename(file_path)}', foglio '{sheet_name}' — Colonne: {', '.join(cols[:8])}"

                self.excel_engine.register_sheet(scoped_name, df, instruction=DEFAULT_INSTRUCTION)
                self._scope_registry.append({
                    "name": scoped_name,
                    "description": description,
                })
                logger.info(f"Scope registrato: {scoped_name}")

            self.ingested_files.append(file_path)
            return len(self.excel_loader.list_sheets())

        else:
            logger.warning(f"Formato non supportato: {ext}")
            return 0

    def ask(self, question: str, source: str = "auto", **kwargs) -> dict:
        if source == "auto":
            scopes = self._scope_registry if self._scope_registry else None
            source, auto_scope = self.intent_router.classify(question, available_scopes=scopes)
            if auto_scope and "scope" not in kwargs:
                kwargs["scope"] = auto_scope

        if source == "text":
            result = self.rag_engine.ask(question, k=kwargs.get("k", 4))
            result["routed_to"] = "text"

            if "contexts" not in result:
                result["contexts"] = result.get("sources", []) or []
            return result

        elif source == "excel":
            if self.excel_engine is None:
                return {"answer": "Nessun file Excel caricato.", "contexts": [], "scope": "N/A", "routed_to": "excel"}

            result = self.excel_engine.ask(question, scope=kwargs.get("scope", "pnl"))
            result["routed_to"] = "excel"
            result.setdefault("scope", kwargs.get("scope", "pnl"))
            result.setdefault("contexts", [])
            return result

        return {"answer": f"Sorgente '{source}' non supportata.", "contexts": [], "routed_to": source}

    @staticmethod
    def _file_slug(file_path: str) -> str:
        """Genera un slug leggibile dal nome file (es. 'Financial_Report_Q3_2024' → 'financial_report_q3_2024')."""
        name = os.path.splitext(os.path.basename(file_path))[0]
        # Normalizza: minuscolo, solo alfanumerici e underscore
        slug = re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')
        return slug

    def get_stats(self) -> dict:
        """Restituisce statistiche sul sistema."""
        stats = {
            "files_ingested": len(self.ingested_files),
            "files": self.ingested_files,
        }
        if self.excel_engine:
            stats["excel_scopes"] = self.excel_engine.list_scopes()
        return stats

# TEST
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    try:
        print("=" * 60)
        print("  RAG Orchestrator — Test")
        print("=" * 60)

        # Setup
        orchestrator = RAGOrchestrator()
        orchestrator.db_manager.clear()

        # Ingestion
        print("\n📂 Ingestion documenti...")
        orchestrator.ingest("files/EU_AI_Regulation_2024_77.pdf")
        orchestrator.ingest("files/Verbale_CdA_2024_07_20.docx")
        orchestrator.ingest("files/Financial_Report_Q3_2024.xlsx")

        print(f"\n📊 Stats: {orchestrator.get_stats()}")

        # Query testo
        print("\n" + "-" * 40)
        q1 = "Cosa dice il regolamento sull'AI?"
        print(f"📝 Domanda (testo): {q1}")
        r1 = orchestrator.ask(q1, source="text")
        print(f"Risposta: {r1['answer']}")
        print(f"Fonti: {r1['sources']}")

        # Query Excel
        print("\n" + "-" * 40)
        q2 = "Qual è la somma degli importi per la categoria 'Costi - Personale'?"
        print(f"📊 Domanda (excel): {q2}")
        r2 = orchestrator.ask(q2, source="excel", scope="pnl")
        print(f"Risposta: {r2['answer']}")

    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()
