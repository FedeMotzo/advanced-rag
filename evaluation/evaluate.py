import json
import logging
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.llms import llm_factory
from ragas.run_config import RunConfig
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings as LCOpenAIEmbeddings
from dotenv import load_dotenv

from rag_orchestrator import RAGOrchestrator
from vector_db.vector_store_manager import VectorStoreManager
from ingestion.chunking_strategies import RecursiveChunkingStrategy, SemanticChunkingStrategy

load_dotenv()
logging.basicConfig(level=logging.WARNING)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES_DIR = os.path.join(PROJECT_ROOT, "files")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DOC_FILES = [
    os.path.join(FILES_DIR, "EU_AI_Regulation_2024_77.pdf"),
    os.path.join(FILES_DIR, "Verbale_CdA_2024_07_20.docx"),
    os.path.join(FILES_DIR, "Financial_Report_Q3_2024.xlsx"),
]


def load_gold_dataset(path: str) -> list:
    """Carica il dataset gold standard da file JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline(orchestrator: RAGOrchestrator, questions: list) -> list:
    """Esegue ogni domanda e raccoglie risultati."""
    results = []
    for item in questions:
        question = item["question"]
        print(f"    🔍 {question[:70]}...")
        result = orchestrator.ask(question, source="auto")
        results.append({
            "question": question,
            "answer": result["answer"],
            "contexts": result.get("contexts", []),
            "ground_truth": item["ground_truth"],
        })
    return results


def build_ragas_dataset(results: list) -> EvaluationDataset:
    """Costruisce un EvaluationDataset RAGAS dai risultati."""
    samples = []
    for r in results:
        samples.append(SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["ground_truth"],
        ))
    return EvaluationDataset(samples=samples)


import time


def evaluate_strategy(strategy, strategy_name: str, gold_data: list, evaluator_llm, evaluator_embeddings, reranker: bool = False, hybrid_search: bool = False) -> pd.DataFrame:
    """Esegue ingestion + query + RAGAS evaluation per una singola configurazione."""
    retrieval_mode = "HybridRetrieve" if hybrid_search else "SemanticRetrieve"
    rerank_label = " + Reranker" if reranker else ""
    label = f"{strategy_name} + {retrieval_mode}{rerank_label}".strip()
    
    print(f"\n{'─' * 50}")
    print(f"  Config: {label}")
    print(f"{'─' * 50}")

    start_time = time.time()

    # 1. Setup orchestrator con collection specifica per strategia di chunking
    collection_name = f"datatrust_docs_{strategy_name.lower()}"
    db_manager = VectorStoreManager(collection_name=collection_name)
    
    orchestrator = RAGOrchestrator(
        db_manager=db_manager, 
        chunking_strategy=strategy, 
        reranker=reranker,
        hybrid_search=hybrid_search
    )

    # 2. Ingestion (solo se la collection è vuota)
    text_files = [f for f in DOC_FILES if not f.endswith((".xlsx", ".xls"))]
    excel_files = [f for f in DOC_FILES if f.endswith((".xlsx", ".xls"))]

    # 2a. Ingestion Testo (solo se la collection è vuota)
    n_chunks = db_manager.count()
    if n_chunks == 0:
        print(f"Ingestion documenti testo (collection vuota)...")
        for f in text_files:
            n_chunks += orchestrator.ingest(f)
        print(f"Chunks testo creati: {n_chunks}")
    else:
        print(f"Riutilizzo chunks testo esistenti: {n_chunks}")

    # 2b. Ingestion Excel (SEMPRE necessaria perché in-memory)
    if excel_files:
        print(f"  Caricamento file Excel (in-memory)...")
        for f in excel_files:
            orchestrator.ingest(f)
        if orchestrator.excel_engine:
            scopes = orchestrator.excel_engine.list_scopes()
            print(f"Scopes Excel registrati: {scopes}")


    # 3. Query
    print(f" Esecuzione pipeline RAG...")
    results = run_pipeline(orchestrator, gold_data)

    # 4. RAGAS evaluation
    print(f"Valutazione RAGAS...")
    eval_dataset = build_ragas_dataset(results)
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    score = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=RunConfig(max_workers=4)
    )

    end_time = time.time()
    duration = end_time - start_time

    print(f"Tempo: {duration:.2f}s")

    df = score.to_pandas()
    df.insert(0, "strategy", label)
    df.insert(1, "n_chunks", n_chunks)
    df["duration"] = duration
    return df


def print_comparison(all_results: pd.DataFrame, gold_data: list):
    """Stampa tabella comparativa tra strategie."""
    exclude_cols = ["strategy", "n_chunks", "user_input", "response", "retrieved_contexts", "reference", "duration"]
    metric_cols = [c for c in all_results.columns if c not in exclude_cols]

    # --- Tabella riassuntiva ---
    print("\n" + "=" * 90)
    print("  CONFRONTO CONFIGURAZIONI (Performance)")
    print("=" * 90)

    summary = all_results.groupby("strategy").agg(
        n_chunks=("n_chunks", "first"),
        duration=("duration", "mean"),
        **{col: (col, "mean") for col in metric_cols}
    ).round(3)

    # Ordina per Answer Relevancy
    if "answer_relevancy" in summary.columns:
        summary = summary.sort_values("answer_relevancy", ascending=False)

    print(f"\n{'Configurazione':<45} {'Sec':>6} {'Chunks':>6}", end="")
    for col in metric_cols:
        print(f"  {col:>18}", end="")
    print()
    print("─" * (60 + 20 * len(metric_cols)))

    for strategy_name, row in summary.iterrows():
        print(f"{strategy_name:<45} {row['duration']:>6.1f} {int(row['n_chunks']):>6}", end="")
        for col in metric_cols:
            val = row[col]
            print(f"  {val:.3f}", end="")
        print()
    
    # --- Dettaglio Migliore Configurazione ---
    best_strategy = summary.index[0]
    print(f"\n{'─' * 80}")
    print(f"  MIGLIORE CONFIGURAZIONE: {best_strategy}")
    print(f"{'─' * 80}")

    # Filtra i risultati per la strategia migliore
    best_results = all_results[all_results["strategy"] == best_strategy]

    for i, item in enumerate(gold_data):
        q = item["question"]
        row = best_results[best_results["user_input"] == q].iloc[0]
        
        print(f"\n  Q{i+1}: {q}")
        print(f"     Attesa:   {item['ground_truth']}")
        print(f"     Sistema:  {row['response']}")
        print("-" * 40)


def main():
    print("=" * 60)
    print("  RAGAS Evaluation — Confronto 8 Configurazioni")
    print("=" * 60)
    print("  1. Chunking: Recursive vs Semantic")
    print("  2. Retrieval: Semantic vs Hybrid (BM25+Emb)")
    print("  3. Reranking: Off vs On")
    print("=" * 60)

    # Carica dataset gold
    dataset_path = os.path.join(SCRIPT_DIR, "eval_dataset.json")
    gold_data = load_gold_dataset(dataset_path)
    
    # Filtra domande specifiche per maggiore velocità
    selected_indices = [0, 4, 8, 9]
    gold_data = [gold_data[i] for i in selected_indices if i < len(gold_data)]
    
    print(f"\nDataset gold (filtrato): {len(gold_data)} domande")

    # Configura LLM e embeddings per RAGAS
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    evaluator_llm = llm_factory("gpt-4o", client=openai_client)
    evaluator_embeddings = LCOpenAIEmbeddings(model="text-embedding-3-small")

    # Genera le 5 configurazioni selezionate
    configs = []
    
    # 1. Recursive
    configs.append((RecursiveChunkingStrategy(), "RecursiveChunking", False, True))  # Hybrid
    configs.append((RecursiveChunkingStrategy(), "RecursiveChunking", True, True))   # Hybrid + Reranker

    # 2. Semantic
    configs.append((SemanticChunkingStrategy(), "SemanticChunking", True, False))    # Semantic + Reranker
    configs.append((SemanticChunkingStrategy(), "SemanticChunking", False, True))    # Hybrid
    configs.append((SemanticChunkingStrategy(), "SemanticChunking", True, True))     # Hybrid + Reranker

    # Esegui evaluation per ogni configurazione
    all_dfs = []
    for strategy, name, use_reranker, use_hybrid in configs:
        df = evaluate_strategy(
            strategy, name, gold_data, evaluator_llm, evaluator_embeddings, 
            reranker=use_reranker, 
            hybrid_search=use_hybrid
        )
        all_dfs.append(df)

    # Combina e confronta
    all_results = pd.concat(all_dfs, ignore_index=True)
    print_comparison(all_results, gold_data)

    # Salva risultati
    output_path = os.path.join(SCRIPT_DIR, "results_comparison.csv")
    all_results.to_csv(output_path, index=False)
    print(f"\nRisultati salvati in: {output_path}")


if __name__ == "__main__":
    main()
