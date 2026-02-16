import json
import logging
import os
import sys
import pandas as pd
from typing import Tuple, List, Dict

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
            "routed_to": result.get("routed_to", "unknown"),
            "scope": result.get("scope"),
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


def evaluate_strategy(
    strategy,
    strategy_name: str,
    gold_data: list,
    evaluator_llm,
    evaluator_embeddings,
    reranker: bool = False,
    hybrid_search: bool = False,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """Esegue ingestion + query + RAGAS evaluation per una singola configurazione.
    
    Nota: split evaluation per task:
      - TEXT: faithfulness, answer_relevancy, context_precision, context_recall
      - EXCEL: answer_relevancy, faithfulness (niente context_* perché non c'è retrieval)
    """
    retrieval_mode = "HybridRetrieve" if hybrid_search else "SemanticRetrieve"
    rerank_label = " + Reranker" if reranker else ""
    label = f"{strategy_name} + {retrieval_mode}{rerank_label}".strip()

    print(f"\n{'─' * 50}")
    print(f"  Config: {label}")
    print(f"{'─' * 50}")

    start_time = time.time()

    # 1) Setup orchestrator con collection specifica per strategia di chunking
    collection_name = f"datatrust_docs_{strategy_name.lower()}"
    db_manager = VectorStoreManager(collection_name=collection_name)

    orchestrator = RAGOrchestrator(
        db_manager=db_manager,
        chunking_strategy=strategy,
        reranker=reranker,
        hybrid_search=hybrid_search,
    )

    # 2) Ingestion
    text_files = [f for f in DOC_FILES if not f.endswith((".xlsx", ".xls"))]
    excel_files = [f for f in DOC_FILES if f.endswith((".xlsx", ".xls"))]

    # 2a) Ingestion Testo (solo se collection vuota)
    n_chunks = db_manager.count()
    if n_chunks == 0:
        print("Ingestion documenti testo (collection vuota)...")
        for f in text_files:
            n_chunks += orchestrator.ingest(f)
        print(f"Chunks testo creati: {n_chunks}")
    else:
        print(f"Riutilizzo chunks testo esistenti: {n_chunks}")

    # 2b) Ingestion Excel (sempre, perché in-memory)
    if excel_files:
        print("Caricamento file Excel (in-memory)...")
        for f in excel_files:
            orchestrator.ingest(f)
        if orchestrator.excel_engine:
            scopes = orchestrator.excel_engine.list_scopes()
            print(f"Scopes Excel registrati: {scopes}")

    # 3) Query
    print("Esecuzione pipeline RAG...")
    results = run_pipeline(orchestrator, gold_data)

    # 4) Split risultati per task (text vs excel)
    text_results = [r for r in results if r.get("routed_to") == "text"]
    excel_results = [r for r in results if r.get("routed_to") == "excel"]

    print("Valutazione RAGAS...")
    dfs = []

    # 4a) TEXT-RAG evaluation (retrieval-based)
    if text_results:
        text_dataset = build_ragas_dataset(text_results)
        text_metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

        text_score = evaluate(
            dataset=text_dataset,
            metrics=text_metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            run_config=RunConfig(max_workers=4),
        )
        df_text = text_score.to_pandas()
        df_text["task"] = "text"
        dfs.append(df_text)

    # 4b) EXCEL evaluation (no retrieval metrics)
    if excel_results:
        excel_dataset = build_ragas_dataset(excel_results)
        excel_metrics = [answer_relevancy, faithfulness]

        excel_score = evaluate(
            dataset=excel_dataset,
            metrics=excel_metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            run_config=RunConfig(max_workers=4),
        )
        df_excel = excel_score.to_pandas()
        df_excel["task"] = "excel"

        # Uniforma colonne nel caso tu voglia un CSV unico
        for col in ("context_precision", "context_recall"):
            if col not in df_excel.columns:
                df_excel[col] = pd.NA

        dfs.append(df_excel)

    # Se nessun risultato (caso limite)
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    end_time = time.time()
    duration = end_time - start_time
    print(f"Tempo: {duration:.2f}s")

    # Metadata configurazione
    if not df.empty:
        df.insert(0, "strategy", label)
        df.insert(1, "n_chunks", n_chunks)
        df["duration"] = duration

        # utile per debug/report
        df["n_text_questions"] = len(text_results)
        df["n_excel_questions"] = len(excel_results)

    return df, results


def print_comparison(all_results: pd.DataFrame, gold_data: list, raw_results_by_strategy: dict):
    """Stampa tabella comparativa tra strategie + dettaglio Q/A per la migliore config per task."""
    if all_results.empty:
        print("\nNessun risultato da mostrare.")
        return

    exclude_cols = {
        "strategy", "task", "n_chunks", "duration",
        "user_input", "response", "retrieved_contexts", "reference",
        "n_text_questions", "n_excel_questions",
        "scope", "routed_to"
    }
    metric_cols = [c for c in all_results.columns if c not in exclude_cols]

    print("\n" + "=" * 90)
    print("  CONFRONTO CONFIGURAZIONI (Performance)")
    print("=" * 90)

    for task in ["text", "excel"]:
        task_df = all_results[all_results.get("task") == task] if "task" in all_results.columns else pd.DataFrame()
        if task_df.empty:
            print(f"\nNessun risultato per task: {task}")
            continue

        print("\n" + "-" * 90)
        print(f"  TASK: {task.upper()}")
        print("-" * 90)

        # metriche effettivamente presenti (no NaN-only)
        task_metric_cols = [c for c in metric_cols if c in task_df.columns and task_df[c].notna().any()]

        summary = task_df.groupby("strategy").agg(
            n_chunks=("n_chunks", "first"),
            duration=("duration", "mean"),
            **{col: (col, "mean") for col in task_metric_cols},
        ).round(3)

        if "answer_relevancy" in summary.columns:
            summary = summary.sort_values("answer_relevancy", ascending=False)
        elif "faithfulness" in summary.columns:
            summary = summary.sort_values("faithfulness", ascending=False)

        # tabella
        print(f"\n{'Configurazione':<45} {'Sec':>6} {'Chunks':>6}", end="")
        for col in task_metric_cols:
            print(f"  {col:>18}", end="")
        print()
        print("─" * (60 + 20 * len(task_metric_cols)))

        for strategy_name, row in summary.iterrows():
            print(f"{strategy_name:<45} {row['duration']:>6.1f} {int(row['n_chunks']):>6}", end="")
            for col in task_metric_cols:
                val = row[col]
                if pd.isna(val):
                    print(f"  {'-':>18}", end="")
                else:
                    print(f"  {val:.3f}".rjust(20), end="")
            print()

        # migliore per il task
        best_strategy = summary.index[0]
        print("\n" + "─" * 80)
        print(f"  MIGLIORE CONFIGURAZIONE ({task.upper()}): {best_strategy}")
        print("─" * 80)

        # stampa metriche best
        best_row = summary.loc[best_strategy]
        for col in task_metric_cols:
            val = best_row[col]
            if not pd.isna(val):
                print(f"  {col}: {val:.3f}")

        # dettaglio Q/A dalla pipeline (raw results)
        raw = raw_results_by_strategy.get(best_strategy, [])
        if not raw:
            print("\n  Dettaglio Q/A non disponibile (raw results mancanti).")
            continue

        # filtra solo le domande del task
        raw_task = [r for r in raw if r.get("routed_to") == task]
        if not raw_task:
            print("\n  Nessuna domanda instradata a questo task per la migliore configurazione.")
            continue

        print("\n  DETTAGLIO DOMANDE (migliore configurazione)")
        for i, r in enumerate(raw_task, start=1):
            q = r.get("question", "")
            gt = r.get("ground_truth", "")
            ans = r.get("answer", "")
            scope = r.get("scope")

            print(f"\n  Q{i}: {q}")
            if task == "excel" and scope:
                print(f"     Scope: {scope}")
            print(f"     Attesa:  {gt}")
            print(f"     Sistema: {ans}")

            # opzionale: per Excel stampa anche un estratto del contesto (pandas instruction)
            if task == "excel":
                ctx = r.get("contexts", []) or []
                # stampa al massimo 2 blocchi per non esplodere output
                for c in ctx[:2]:
                    # accorcia
                    c_short = c if len(c) <= 600 else c[:600] + "..."
                    print(f"     Context: {c_short}")


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

    # 3. Esegue evaluation per ogni configurazione
    all_dfs = []
    raw_results_by_strategy = {}

    for strategy, name, use_reranker, use_hybrid in configs:
        df, raw_results = evaluate_strategy(
            strategy, name, gold_data, evaluator_llm, evaluator_embeddings,
            reranker=use_reranker,
            hybrid_search=use_hybrid
        )
        all_dfs.append(df)

        label = df["strategy"].iloc[0] if not df.empty else f"{name}"
        raw_results_by_strategy[label] = raw_results

    # 4. Combina e confronta
    all_results = pd.concat(all_dfs, ignore_index=True)
    print_comparison(all_results, gold_data, raw_results_by_strategy)

    # 5. Salva risultati
    output_path = os.path.join(SCRIPT_DIR, "results_comparison.csv")
    all_results.to_csv(output_path, index=False)
    print(f"\nRisultati salvati in: {output_path}")


if __name__ == "__main__":
    main()
