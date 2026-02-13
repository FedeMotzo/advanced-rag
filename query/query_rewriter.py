import logging
from typing import Optional, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Sei un assistente che riscrive domande di follow-up in domande autonome.\n\n"
     "Data la cronologia della conversazione e una nuova domanda dell'utente, "
     "riscrivi la domanda in modo che sia comprensibile SENZA il contesto precedente.\n\n"
     "Regole:\n"
     "- Se la domanda è già autonoma, restituiscila invariata.\n"
     "- Mantieni la stessa lingua dell'utente.\n"
     "- Non aggiungere informazioni inventate.\n"
     "- Rispondi SOLO con la domanda riscritta, nient'altro."),
    ("human",
     "Cronologia conversazione:\n{history}\n\n"
     "Nuova domanda: {question}\n\n"
     "Domanda riscritta:"),
])


class QueryRewriter:
    """Riscrive domande di follow-up in domande autonome usando la chat history."""

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.chain = REWRITE_PROMPT | self.llm | StrOutputParser()
        logger.info("QueryRewriter inizializzato.")

    def rewrite(self, question: str, history: List[dict]) -> str:
        """
        Riscrive la domanda se necessario.

        Args:
            question: Domanda dell'utente.
            history: Lista di messaggi [{"role": "user"/"assistant", "content": str}].

        Returns:
            Domanda autonoma riscritta (o originale se già autonoma).
        """
        # Nessuna history → niente da riscrivere
        if not history:
            return question

        # Prendi solo gli ultimi 4 messaggi (2 turni) per efficienza
        recent = history[-4:]
        history_str = "\n".join(
            f"{'Utente' if m['role'] == 'user' else 'Assistente'}: {m['content'][:200]}"
            for m in recent
        )

        try:
            rewritten = self.chain.invoke({
                "history": history_str,
                "question": question,
            }).strip()

            if rewritten and rewritten != question:
                logger.info(f"Query rewrite: '{question}' → '{rewritten}'")
            return rewritten or question

        except Exception as e:
            logger.warning(f"Errore query rewrite, uso originale: {e}")
            return question
