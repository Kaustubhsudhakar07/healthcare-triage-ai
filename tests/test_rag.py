"""
Unit Tests for Chroma Clinical Knowledge Base & RAG Retrieval
AI-Assisted Pre-Hospital Patient Criticality Prediction System
"""

import pytest
from src.rag.knowledge_base import ClinicalKnowledgeBase


def test_knowledge_base_indexing_and_retrieval():
    kb = ClinicalKnowledgeBase()
    assert kb.collection.count() > 0
    
    # Query Glasgow Coma Scale
    results = kb.retrieve("What is the Glasgow Coma Scale?", top_k=2)
    assert len(results) > 0
    top_doc = results[0]
    assert "glasgow" in top_doc["content"].lower() or "coma" in top_doc["content"].lower()
    assert "source" in top_doc
    assert "authority" in top_doc


def test_knowledge_base_retrieves_spo2():
    kb = ClinicalKnowledgeBase()
    results = kb.retrieve("What is pulse oximetry SpO2 hypoxia threshold?", top_k=2)
    assert len(results) > 0
    top_doc = results[0]
    assert "spo2" in top_doc["content"].lower() or "oxygen" in top_doc["content"].lower()
