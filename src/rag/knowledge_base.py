"""
Chroma Vector Knowledge Base for Clinical Triage RAG
AI-Assisted Pre-Hospital Patient Criticality Prediction System

Ingests authoritative healthcare reference documents (WHO, NIH, CDC, NHS)
into a local ChromaDB collection and provides similarity search retrieval.
"""

import os
import sys
import glob
import chromadb
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.abspath("."))


class ClinicalKnowledgeBase:
    """
    RAG Vector Store for authoritative triage guidelines and clinical definitions.
    """
    def __init__(self, persist_dir: str = "models/chroma_db", knowledge_dir: str = "knowledge"):
        self.persist_dir = persist_dir
        self.knowledge_dir = knowledge_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        
        # Initialize Persistent Chroma Client
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="clinical_triage_guidelines",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Ingest documents if collection is empty
        if self.collection.count() == 0:
            self.ingest_documents()

    def ingest_documents(self):
        """
        Parses Markdown files in knowledge_dir and indexes chunks into ChromaDB.
        """
        md_files = glob.glob(os.path.join(self.knowledge_dir, "*.md"))
        if not md_files:
            print(f"Warning: No markdown knowledge files found in '{self.knowledge_dir}'.")
            return

        documents = []
        metadatas = []
        ids = []
        chunk_idx = 0

        for filepath in md_files:
            filename = os.path.basename(filepath)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            lines = content.split("\n")
            title = filename.replace("_", " ").replace(".md", "").title()
            source_line = "Authoritative Clinical Reference"
            authority = "Standard Healthcare Guidelines"

            for line in lines:
                if line.startswith("Source:"):
                    source_line = line.replace("Source:", "").strip()
                elif line.startswith("Authority:"):
                    authority = line.replace("Authority:", "").strip()
                elif line.startswith("# ") and title == filename.replace("_", " ").replace(".md", "").title():
                    title = line.replace("# ", "").strip()

            # Split document into logical sections by "## " headers
            sections = content.split("\n## ")
            for i, sec in enumerate(sections):
                sec_text = sec.strip()
                if not sec_text:
                    continue
                if i > 0:
                    sec_text = "## " + sec_text

                # Avoid indexing headers alone
                if len(sec_text) < 40:
                    continue

                chunk_id = f"{filename}_{i}_{chunk_idx}"
                documents.append(sec_text)
                metadatas.append({
                    "title": title,
                    "source": source_line,
                    "authority": authority,
                    "filename": filename
                })
                ids.append(chunk_id)
                chunk_idx += 1

        if documents:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"[RAG] Successfully indexed {len(documents)} clinical knowledge chunks into ChromaDB.")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Searches the vector store for the most relevant clinical documentation chunks.
        """
        if self.collection.count() == 0:
            self.ingest_documents()

        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, max(1, self.collection.count()))
        )

        formatted_results = []
        if results and results.get("documents") and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

            for doc, meta, dist in zip(docs, metas, distances):
                formatted_results.append({
                    "content": doc,
                    "title": meta.get("title", "Clinical Reference"),
                    "source": meta.get("source", "Authoritative Clinical Reference"),
                    "authority": meta.get("authority", "Emergency Medicine Guidelines"),
                    "filename": meta.get("filename", "unknown.md"),
                    "similarity_score": round(1.0 - float(dist), 3) if dist is not None else 1.0
                })

        return formatted_results


# Global singleton instance for efficient reuse
_global_kb: Optional[ClinicalKnowledgeBase] = None

def get_knowledge_base() -> ClinicalKnowledgeBase:
    """Returns the singleton ClinicalKnowledgeBase instance."""
    global _global_kb
    if _global_kb is None:
        _global_kb = ClinicalKnowledgeBase()
    return _global_kb


if __name__ == "__main__":
    kb = ClinicalKnowledgeBase()
    sample_query = "What is the Glasgow Coma Scale and when is it critical?"
    chunks = kb.retrieve(sample_query, top_k=2)
    print(f"\nQuery: {sample_query}")
    for i, c in enumerate(chunks):
        print(f"\n--- Result {i+1} (Source: {c['source']} | Score: {c['similarity_score']}) ---")
        print(c['content'][:250] + "...")
