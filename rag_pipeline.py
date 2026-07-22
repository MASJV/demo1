"""
TechTrinetra7 — Advanced RAG Pipeline for Enterprise Document Search
=====================================================================

Architecture (every stage is captured for the dashboard):

  1. INGESTION        -> multi-format loaders (document_loaders.py)
  2. CHUNKING         -> LangChain RecursiveCharacterTextSplitter
  3. INDEXING (dual)  -> (a) LangChain FAISS dense vector store
                         (b) LangChain BM25Retriever sparse index
                         (c) LlamaIndex VectorStoreIndex (second dense path,
                             demonstrates LangChain+LlamaIndex hybrid stack)
  4. QUERY REWRITE    -> LCEL chain (gpt-4o-mini) generates N query variants
  5. HYBRID RETRIEVAL -> dense + sparse + llamaindex retrievers run per variant
  6. RRF FUSION       -> Reciprocal Rank Fusion merges all ranked lists
  7. RERANKING        -> local cross-encoder (sentence-transformers, free,
                          no API cost) re-scores the fused candidates
  8. GENERATION       -> LCEL chain (gpt-4o-mini) grounded, cited answer

Cost discipline (hard constraints):
  - LLM:        gpt-4o-mini ONLY
  - Embeddings: text-embedding-3-small ONLY
  - Reranker:   local open-weight cross-encoder (no Cohere/paid rerank API)
  - Vector DB:  FAISS in-memory (no Pinecone/Weaviate cloud charges)
  - API key is supplied by the user at runtime — never hardcoded.
"""

from __future__ import annotations
import os
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---- cost-locked model identifiers -----------------------------------------
CHAT_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # local, open-weight, free


@dataclass
class StageTrace:
    """Captures everything the premium dashboard needs to visualize."""
    query: str = ""
    query_variants: List[str] = field(default_factory=list)
    dense_results: List[Dict[str, Any]] = field(default_factory=list)
    sparse_results: List[Dict[str, Any]] = field(default_factory=list)
    llamaindex_results: List[Dict[str, Any]] = field(default_factory=list)
    fused_results: List[Dict[str, Any]] = field(default_factory=list)
    reranked_results: List[Dict[str, Any]] = field(default_factory=list)
    final_context: str = ""
    answer: str = ""
    timings_ms: Dict[str, float] = field(default_factory=dict)
    token_estimate: Dict[str, int] = field(default_factory=dict)
    chunk_count: int = 0


class AdvancedRAGPipeline:
    def __init__(self, api_key: str, chunk_size: int = 800, chunk_overlap: int = 120,
                 top_k_per_retriever: int = 8, final_k: int = 5, num_query_variants: int = 3):
        if not api_key:
            raise ValueError("OpenAI API key is required (entered via UI, never hardcoded).")
        os.environ["OPENAI_API_KEY"] = api_key
        self.api_key = api_key
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k_per_retriever = top_k_per_retriever
        self.final_k = final_k
        self.num_query_variants = num_query_variants

        self.chunks: List[Document] = []
        self._dense_store = None
        self._bm25 = None
        self._li_index = None
        self._reranker = None

        self._llm = None
        self._embeddings = None

    # ---------------------------------------------------------------- setup
    def _get_llm(self, streaming: bool = False):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=CHAT_MODEL, temperature=0.2, api_key=self.api_key, streaming=streaming)

    def _get_embeddings(self):
        from langchain_openai import OpenAIEmbeddings
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=self.api_key)
        return self._embeddings

    def _get_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(RERANKER_MODEL)
        return self._reranker

    # ------------------------------------------------------------ chunking
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        for i, c in enumerate(chunks):
            c.metadata["chunk_id"] = i
        self.chunks = chunks
        return chunks

    # ------------------------------------------------------------- indexing
    def build_indexes(self, chunks: Optional[List[Document]] = None):
        chunks = chunks or self.chunks
        if not chunks:
            raise ValueError("No chunks to index. Run chunk_documents() first.")

        # (a) Dense — LangChain FAISS
        from langchain_community.vectorstores import FAISS
        self._dense_store = FAISS.from_documents(chunks, self._get_embeddings())

        # (b) Sparse — BM25
        from langchain_community.retrievers import BM25Retriever
        self._bm25 = BM25Retriever.from_documents(chunks)
        self._bm25.k = self.top_k_per_retriever

        # (c) LlamaIndex — second dense retrieval path (hybrid stack)
        from llama_index.core import VectorStoreIndex, Document as LIDocument, Settings
        from llama_index.embeddings.openai import OpenAIEmbedding as LIOpenAIEmbedding
        from llama_index.llms.openai import OpenAI as LIOpenAI

        Settings.embed_model = LIOpenAIEmbedding(model=EMBED_MODEL, api_key=self.api_key)
        Settings.llm = LIOpenAI(model=CHAT_MODEL, api_key=self.api_key)

        li_docs = [
            LIDocument(text=c.page_content, metadata={**c.metadata})
            for c in chunks
        ]
        self._li_index = VectorStoreIndex.from_documents(li_docs)
        return True

    # ------------------------------------------------------- query rewrite
    def _query_expansion_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You generate search query reformulations for enterprise document retrieval. "
             "Given a user question, produce {n} diverse reformulations that vary phrasing, "
             "synonyms, and specificity to maximize recall. Return ONLY the reformulations, "
             "one per line, no numbering, no extra text."),
            ("human", "{question}"),
        ])
        llm = self._get_llm()
        chain = prompt | llm | StrOutputParser() | RunnableLambda(
            lambda text: [line.strip() for line in text.split("\n") if line.strip()]
        )
        return chain

    # ----------------------------------------------------------- retrieval
    @staticmethod
    def _doc_to_dict(doc: Document, score: float = None, rank: int = None) -> Dict[str, Any]:
        return {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": score,
            "rank": rank,
        }

    def _dense_retrieve(self, query: str) -> List[Document]:
        return self._dense_store.similarity_search(query, k=self.top_k_per_retriever)

    def _sparse_retrieve(self, query: str) -> List[Document]:
        return self._bm25.invoke(query)

    def _llamaindex_retrieve(self, query: str) -> List[Document]:
        retriever = self._li_index.as_retriever(similarity_top_k=self.top_k_per_retriever)
        nodes = retriever.retrieve(query)
        return [
            Document(page_content=n.node.get_content(), metadata=dict(n.node.metadata))
            for n in nodes
        ]

    # -------------------------------------------------------------- fusion
    @staticmethod
    def _doc_key(doc: Document) -> str:
        return f"{doc.metadata.get('source','')}-{doc.metadata.get('chunk_id', hash(doc.page_content))}"

    def reciprocal_rank_fusion(self, ranked_lists: List[List[Document]], k: int = 60) -> List[Document]:
        """Standard RRF: score = sum(1 / (k + rank)) across all lists a doc appears in."""
        scores: Dict[str, float] = {}
        doc_lookup: Dict[str, Document] = {}
        for ranked_list in ranked_lists:
            for rank, doc in enumerate(ranked_list):
                key = self._doc_key(doc)
                doc_lookup[key] = doc
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_lookup[key] for key, _ in ordered], scores

    # ------------------------------------------------------------ reranking
    def rerank(self, query: str, docs: List[Document]) -> List[Dict[str, Any]]:
        if not docs:
            return []
        reranker = self._get_reranker()
        pairs = [[query, d.page_content] for d in docs]
        raw_scores = reranker.predict(pairs)
        scored = list(zip(docs, raw_scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            self._doc_to_dict(doc, score=float(score), rank=i + 1)
            for i, (doc, score) in enumerate(scored[: self.final_k])
        ]

    # ---------------------------------------------------------- generation
    def _generation_chain(self, streaming: bool = False):
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an enterprise document search assistant for TechTrinetra7. "
             "Answer ONLY using the provided context. If the answer is not in the "
             "context, say you could not find it in the indexed documents. "
             "Always cite sources inline like [source: filename, chunk N]."),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ])
        llm = self._get_llm(streaming=streaming)
        return prompt | llm | StrOutputParser()

    @staticmethod
    def _format_context(reranked: List[Dict[str, Any]]) -> str:
        blocks = []
        for r in reranked:
            meta = r["metadata"]
            tag = f"[source: {meta.get('source','?')}, chunk {meta.get('chunk_id','?')}]"
            blocks.append(f"{tag}\n{r['content']}")
        return "\n\n---\n\n".join(blocks)

    # --------------------------------------------------------------- run()
    def run(self, query: str) -> StageTrace:
        trace = StageTrace(query=query, chunk_count=len(self.chunks))
        t0 = time.time()

        # Stage: query expansion
        s = time.time()
        variants = self._query_expansion_chain().invoke({"question": query, "n": self.num_query_variants})
        variants = [query] + variants
        trace.query_variants = variants
        trace.timings_ms["query_rewrite"] = (time.time() - s) * 1000

        # Stage: hybrid retrieval across all variants
        s = time.time()
        dense_lists, sparse_lists, li_lists = [], [], []
        for v in variants:
            dense_lists.append(self._dense_retrieve(v))
            sparse_lists.append(self._sparse_retrieve(v))
            li_lists.append(self._llamaindex_retrieve(v))
        trace.dense_results = [self._doc_to_dict(d, rank=i + 1) for i, d in enumerate(dense_lists[0])]
        trace.sparse_results = [self._doc_to_dict(d, rank=i + 1) for i, d in enumerate(sparse_lists[0])]
        trace.llamaindex_results = [self._doc_to_dict(d, rank=i + 1) for i, d in enumerate(li_lists[0])]
        trace.timings_ms["hybrid_retrieval"] = (time.time() - s) * 1000

        # Stage: RRF fusion (across every list from every variant)
        s = time.time()
        all_lists = dense_lists + sparse_lists + li_lists
        fused_docs, fusion_scores = self.reciprocal_rank_fusion(all_lists)
        trace.fused_results = [
            self._doc_to_dict(d, score=fusion_scores[self._doc_key(d)], rank=i + 1)
            for i, d in enumerate(fused_docs[: self.top_k_per_retriever * 2])
        ]
        trace.timings_ms["rrf_fusion"] = (time.time() - s) * 1000

        # Stage: reranking
        s = time.time()
        candidates = fused_docs[: self.top_k_per_retriever * 2]
        trace.reranked_results = self.rerank(query, candidates)
        trace.timings_ms["reranking"] = (time.time() - s) * 1000

        # Stage: generation
        s = time.time()
        context = self._format_context(trace.reranked_results)
        trace.final_context = context
        answer = self._generation_chain().invoke({"context": context, "question": query})
        trace.answer = answer
        trace.timings_ms["generation"] = (time.time() - s) * 1000

        trace.timings_ms["total"] = (time.time() - t0) * 1000
        trace.token_estimate = {
            "context_tokens_est": len(context) // 4,
            "answer_tokens_est": len(answer) // 4,
        }
        return trace

    def stream_answer(self, context: str, query: str):
        """Generator for Streamlit st.write_stream — used after run() has built context."""
        chain = self._generation_chain(streaming=True)
        for chunk in chain.stream({"context": context, "question": query}):
            yield chunk
