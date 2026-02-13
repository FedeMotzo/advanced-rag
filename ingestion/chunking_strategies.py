import numpy as np
from abc import ABC, abstractmethod
from typing import List, Any
from dataclasses import dataclass
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter

@dataclass
class ChunkMetrics:
    avg_length: float
    std_length: float
    total_chunks: int
    semantic_coherence: float = 0.0

class BaseChunkingStrategy(ABC):
    """
    Classe astratta per le strategie di chunking.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.name = self.__class__.__name__

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Prende testo e restituisce lista di stringhe."""
        pass

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Prende i documenti grezzi (intere pagine) e restituisce documenti chunkati,
        preservando e arricchendo i metadati.
        """
        chunked_docs = []
        
        for doc in documents:
            # 1. Splitta il testo usando la logica specifica della sottoclasse
            text_chunks = self.split_text(doc.page_content)
            
            # 2. Ricostruisce i Document object mantenendo i metadati originali
            for i, chunk_text in enumerate(text_chunks):
                new_metadata = doc.metadata.copy()
                new_metadata.update({
                    'chunk_strategy': self.name,
                    'chunk_index': i,
                    'parent_source': new_metadata.get('source', 'unknown')
                })
                
                chunked_docs.append(Document(
                    page_content=chunk_text,
                    metadata=new_metadata
                ))
                
        return chunked_docs

    def calculate_metrics(self, chunks: List[str], embeddings: np.ndarray = None) -> ChunkMetrics:
        """Calcola metriche sulla qualità dei chunk."""
        if not chunks:
            return ChunkMetrics(0, 0, 0)

        lengths = [len(chunk) for chunk in chunks]
        semantic_coherence = 0.0
        
        # Calcolo coerenza semantica
        if embeddings is not None and len(embeddings) > 1:
            similarities = []
            for i in range(len(embeddings) - 1):
                # Cosine similarity tra chunk adiacenti
                norm_i = np.linalg.norm(embeddings[i])
                norm_next = np.linalg.norm(embeddings[i + 1])
                
                if norm_i > 0 and norm_next > 0:
                    sim = np.dot(embeddings[i], embeddings[i + 1]) / (norm_i * norm_next)
                    similarities.append(sim)
            
            if similarities:
                semantic_coherence = np.mean(similarities)

        return ChunkMetrics(
            avg_length=float(np.mean(lengths)),
            std_length=float(np.std(lengths)),
            total_chunks=len(chunks),
            semantic_coherence=semantic_coherence
        )

# --- STRATEGIA 1: RECURSIVE ---
class RecursiveChunkingStrategy(BaseChunkingStrategy):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__(chunk_size, chunk_overlap)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def split_text(self, text: str) -> List[str]:
        return self.splitter.split_text(text)

# --- STRATEGIA 2: TOKEN BASED  ---
class TokenChunkingStrategy(BaseChunkingStrategy):
    """
    Chunking basato sui token (es. tiktoken) invece che sui caratteri.
    Utile per assicurarsi di non sforare mai la context window.
    """
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        super().__init__(chunk_size, chunk_overlap)
        self.splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split_text(self, text: str) -> List[str]:
        return self.splitter.split_text(text)

# --- STRATEGIA 3: SEMANTIC ---
class SemanticChunkingStrategy(BaseChunkingStrategy):
    """
    Chunking semantico: usa gli embeddings per trovare i punti naturali
    in cui il significato del testo cambia, creando chunk tematicamente coerenti.

    Non usa chunk_size/chunk_overlap — la dimensione è guidata dalla semantica.

    Args:
        breakpoint_threshold_type: Metodo per determinare i breakpoint.
            - "percentile": soglia basata sul percentile delle distanze (default)
            - "standard_deviation": soglia basata sulla deviazione standard
            - "interquartile": soglia basata sul range interquartile
            - "gradient": soglia basata sui cambiamenti di gradiente
        breakpoint_threshold_amount: Valore numerico della soglia.
            Per "percentile" il default è ~95 (più alto = chunk più grandi).
    """
    def __init__(
        self,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: float = None,
        embeddings=None,
    ):
        # chunk_size/overlap non usati, ma li passiamo per compatibilità con la base
        super().__init__(chunk_size=0, chunk_overlap=0)

        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_openai import OpenAIEmbeddings

        self.embeddings = embeddings or OpenAIEmbeddings(model="text-embedding-3-small")

        kwargs = {"embeddings": self.embeddings, "breakpoint_threshold_type": breakpoint_threshold_type}
        if breakpoint_threshold_amount is not None:
            kwargs["breakpoint_threshold_amount"] = breakpoint_threshold_amount

        self.splitter = SemanticChunker(**kwargs)

    def split_text(self, text: str) -> List[str]:
        docs = self.splitter.create_documents([text])
        return [doc.page_content for doc in docs]