import logging
from typing import Dict, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from vector_db.vector_store_manager import VectorStoreManager

logger = logging.getLogger(__name__)

# Prompt di default — può essere sovrascritto dall'utente
DEFAULT_RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Sei un assistente esperto che risponde alle domande basandosi esclusivamente "
     "sul contesto fornito. Se il contesto non contiene informazioni sufficienti, "
     "dichiaralo esplicitamente.\n\n"
     "Contesto:\n{context}"),
    ("human", "{input}"),
])

class RAGQueryEngine:
    """
    Interroga documenti testuali tramite semantic search + LLM.
    Supporta: ricerca semantica pura, hybrid search (BM25+embeddings), reranking.
    """

    def __init__(
        self,
        db_manager: VectorStoreManager,
        llm: Optional[ChatOpenAI] = None,
        prompt: Optional[ChatPromptTemplate] = None,
        reranker: bool = False,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        hybrid_search: bool = False,
        fetch_multiplier: int = 3,
    ):
        self.db_manager = db_manager
        self.llm = llm or ChatOpenAI(model="gpt-4o", temperature=0)
        self.prompt = prompt or DEFAULT_RAG_PROMPT
        self.reranker_enabled = reranker
        self.hybrid_search = hybrid_search
        self.fetch_multiplier = fetch_multiplier

        # BM25 retriever (inizializzato lazy al primo uso)
        self._bm25_retriever = None

        # Cross-encoder (solo se richiesto)
        self._cross_encoder = None
        if self.reranker_enabled:
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder(reranker_model)
            logger.info(f"Reranker attivato: {reranker_model}")

        logger.info(f"RAGQueryEngine pronto (hybrid={hybrid_search}, reranker={reranker}).")

    @staticmethod
    def _format_docs(docs) -> str:
        """Concatena i documenti recuperati in un'unica stringa di contesto."""
        return "\n\n".join(doc.page_content for doc in docs)

    @staticmethod
    def _reciprocal_rank_fusion(results_lists: List[List[Document]], k: int = 60) -> List[Document]:
        """
        Fonde più liste di documenti usando Reciprocal Rank Fusion (RRF).
        """
        doc_scores: dict = {}  # page_content -> (score, Document)
        
        for result_list in results_lists:
            for rank, doc in enumerate(result_list):
                key = doc.page_content
                if key not in doc_scores:
                    doc_scores[key] = (0.0, doc)
                current_score, existing_doc = doc_scores[key]
                doc_scores[key] = (current_score + 1.0 / (rank + k), existing_doc)
        
        # Ordina per score decrescente
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in sorted_docs]

    def _get_bm25_retriever(self, k: int):
        """Inizializza il BM25 retriever con i documenti dal vector store."""
        if self._bm25_retriever is None:
            from langchain_community.retrievers import BM25Retriever
            all_docs = self.db_manager.get_all_documents()
            if all_docs:
                self._bm25_retriever = BM25Retriever.from_documents(all_docs)
                logger.info(f"BM25 retriever inizializzato con {len(all_docs)} documenti.")
            else:
                logger.warning("Nessun documento per BM25.")
                return None
        self._bm25_retriever.k = k
        return self._bm25_retriever

    def _rerank(self, question: str, docs: list, top_k: int) -> list:
        """Ri-ordina i documenti con il cross-encoder e restituisce i top_k."""
        if not docs:
            return docs

        pairs = [(question, doc.page_content) for doc in docs]
        scores = self._cross_encoder.predict(pairs)

        scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:top_k]]

    def ask(self, question: str, k: int = 4) -> Dict:
        """
        Esegue una ricerca e genera una risposta.

        Flusso:
        1. Retrieval (embedding-only oppure hybrid BM25+embedding)
        2. Opzionale: reranking con cross-encoder
        3. Generazione con LLM
        """
        try:
            # --- STEP 1: Retrieval ---

            # Nel caso in cui sia attivo il reranking andiamo a recuperare dal DB
            # più documenti e successivamente con il reranking selezioniamo i top-k
            fetch_k = k * self.fetch_multiplier if self.reranker_enabled else k

            if self.hybrid_search:
                # Hybrid: BM25 + Embedding
                embedding_retriever = self.db_manager.get_retriever(k=fetch_k)
                embedding_docs = embedding_retriever.invoke(question)

                # Costruiamo l'oggetto BM25 Retriever
                bm25 = self._get_bm25_retriever(k=fetch_k)
                bm25_docs = bm25.invoke(question) if bm25 else []

                # Fonde con RRF e prende i top
                fused_docs = self._reciprocal_rank_fusion([embedding_docs, bm25_docs])
                source_docs = fused_docs[:fetch_k]
            else:
                # Semantic-only
                retriever = self.db_manager.get_retriever(k=fetch_k)
                source_docs = retriever.invoke(question)

            # --- STEP 2: Reranking ---
            if self.reranker_enabled and self._cross_encoder is not None:
                source_docs = self._rerank(question, source_docs, top_k=k)
            else:
                source_docs = source_docs[:k]

            # --- STEP 3: Generation ---
            context_str = self._format_docs(source_docs)

            rag_chain = (
                {
                    "context": lambda _: context_str,
                    "input": RunnablePassthrough(),
                }
                | self.prompt
                | self.llm
                | StrOutputParser()
            )

            answer = rag_chain.invoke(question)

            sources = list(set(
                doc.metadata.get("source", "Sconosciuto")
                for doc in source_docs
            ))

            return {
                "answer": answer,
                "sources": sources,
                "contexts": [doc.page_content for doc in source_docs],
            }

        except Exception as e:
            logger.error(f"Errore RAGQueryEngine: {e}")
            return {"answer": "Errore durante la generazione della risposta.", "sources": [], "contexts": []}
