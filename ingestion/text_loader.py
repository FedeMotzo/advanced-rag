import os
import logging
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

from vector_db.vector_store_manager import VectorStoreManager
try:
    from ingestion.chunking_strategies import BaseChunkingStrategy, RecursiveChunkingStrategy
except ImportError:
    from chunking_strategies import BaseChunkingStrategy, RecursiveChunkingStrategy

logger = logging.getLogger(__name__)


class TextLoader:
    """
    Carica documenti PDF/DOCX, li chunka e li salva nel vector store.
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".docx"}

    def __init__(self, db_manager: VectorStoreManager, chunking_strategy: BaseChunkingStrategy):
        self.db_manager = db_manager
        self.chunking_strategy = chunking_strategy
        logger.info(f"TextLoader pronto (Strategia: {self.chunking_strategy.name})")

    def process_file(self, file_path: str) -> int:
        """
        Carica, crea i chunk e indicizza un file nel vector store.
        """
        if not os.path.exists(file_path):
            logger.error(f"File non trovato: {file_path}")
            return 0

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            logger.warning(f"Formato non supportato: {ext}")
            return 0

        try:
            logger.info(f"Inizio elaborazione: {file_path}")

            # Selezioniamo il loader in base all'estensione del file
            loader = PyPDFLoader(file_path) if ext == ".pdf" else Docx2txtLoader(file_path)
            raw_docs = loader.load()

            if not raw_docs:
                logger.warning(f"Nessun testo trovato in {file_path}")
                return 0

            # Creazione dei chunk in base alla strategia passata in input
            chunked_docs = self.chunking_strategy.split_documents(raw_docs)

            # Aggiunta del nome del file ai metadati per ogni chunk
            filename = os.path.basename(file_path)
            for doc in chunked_docs:
                doc.metadata["source"] = filename

            # Salvataggio nel vector store
            self.db_manager.add_documents(chunked_docs)
            n = len(chunked_docs)
            logger.info(f"✅ Indicizzato {filename}: {n} chunks (Strategia: {self.chunking_strategy.name})")
            return n

        except Exception as e:
            logger.error(f"Errore ingestion {file_path}: {e}")
            return 0
