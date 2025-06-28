# vector_store.py
import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import logging

class VectorStore:
    """Handle vector storage and retrieval using FAISS"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", index_path: str = "vector_index"):
        self.model_name = model_name
        self.index_path = index_path
        self.embedding_model = SentenceTransformer(model_name)
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        self.documents = []  # Store original documents
        self.document_metadata = []  # Store metadata for each document
        
        self.logger = logging.getLogger(__name__)
        
        # Load existing index if available
        self._load_index()
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms
    
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            return self._normalize_embeddings(embeddings)
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def add_documents(self, documents: List[str], metadata: List[Dict] = None) -> None:
        """Add documents to the vector store"""
        try:
            if not documents:
                raise ValueError("No documents provided")
            
            # Generate embeddings
            embeddings = self._generate_embeddings(documents)
            
            # Add to FAISS index
            self.index.add(embeddings.astype('float32'))
            
            # Store documents and metadata
            self.documents.extend(documents)
            
            if metadata:
                self.document_metadata.extend(metadata)
            else:
                # Create default metadata
                default_metadata = [{'id': len(self.documents) + i, 'source': 'unknown'} 
                                  for i in range(len(documents))]
                self.document_metadata.extend(default_metadata)
            
            self.logger.info(f"Added {len(documents)} documents to vector store")
            
            # Save index
            self._save_index()
            
        except Exception as e:
            self.logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def similarity_search(self, query: str, k: int = 5, threshold: float = 0.0) -> List[str]:
        """Search for similar documents"""
        try:
            if self.index.ntotal == 0:
                raise ValueError("No documents in vector store")
            
            # Generate query embedding
            query_embedding = self._generate_embeddings([query])
            
            # Search
            scores, indices = self.index.search(query_embedding.astype('float32'), k)
            
            # Filter by threshold and return results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if score >= threshold and idx != -1:
                    results.append(self.documents[idx])
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error in similarity search: {str(e)}")
            raise
    
    def similarity_search_with_scores(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar documents with similarity scores"""
        try:
            if self.index.ntotal == 0:
                raise ValueError("No documents in vector store")
            
            # Generate query embedding
            query_embedding = self._generate_embeddings([query])
            
            # Search
            scores, indices = self.index.search(query_embedding.astype('float32'), k)
            
            # Return results with scores
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1:
                    results.append((self.documents[idx], float(score)))
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error in similarity search with scores: {str(e)}")
            raise
    
    def get_relevant_context(self, query: str, max_length: int = 2000) -> str:
        """Get relevant context for a query, concatenated up to max_length"""
        try:
            relevant_docs = self.similarity_search(query, k=10)
            
            context = ""
            for doc in relevant_docs:
                if len(context) + len(doc) <= max_length:
                    context += doc + "\n\n"
                else:
                    # Add partial document if it fits
                    remaining_space = max_length - len(context)
                    if remaining_space > 100:  # Only add if meaningful space left
                        context += doc[:remaining_space] + "..."
                    break
            
            return context.strip()
        
        except Exception as e:
            self.logger.error(f"Error getting relevant context: {str(e)}")
            return ""
    
    def _save_index(self) -> None:
        """Save FAISS index and metadata to disk"""
        try:
            os.makedirs(self.index_path, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, os.path.join(self.index_path, "index.faiss"))
            
            # Save documents and metadata
            with open(os.path.join(self.index_path, "documents.pkl"), 'wb') as f:
                pickle.dump(self.documents, f)
            
            with open(os.path.join(self.index_path, "metadata.pkl"), 'wb') as f:
                pickle.dump(self.document_metadata, f)
            
            self.logger.info("Vector index saved successfully")
        
        except Exception as e:
            self.logger.error(f"Error saving index: {str(e)}")
    
    def _load_index(self) -> None:
        """Load FAISS index and metadata from disk"""
        try:
            index_file = os.path.join(self.index_path, "index.faiss")
            documents_file = os.path.join(self.index_path, "documents.pkl")
            metadata_file = os.path.join(self.index_path, "metadata.pkl")
            
            if os.path.exists(index_file) and os.path.exists(documents_file):
                # Load FAISS index
                self.index = faiss.read_index(index_file)
                
                # Load documents
                with open(documents_file, 'rb') as f:
                    self.documents = pickle.load(f)
                
                # Load metadata if exists
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'rb') as f:
                        self.document_metadata = pickle.load(f)
                
                self.logger.info(f"Loaded existing index with {len(self.documents)} documents")
        
        except Exception as e:
            self.logger.error(f"Error loading index: {str(e)}")
            # Reset to empty index if loading fails
            self.index = faiss.IndexFlatIP(self.dimension)
            self.documents = []
            self.document_metadata = []
    
    def clear_index(self) -> None:
        """Clear all documents from the vector store"""
        try:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.documents = []
            self.document_metadata = []
            
            # Remove saved files
            if os.path.exists(self.index_path):
                for file in os.listdir(self.index_path):
                    os.remove(os.path.join(self.index_path, file))
            
            self.logger.info("Vector store cleared")
        
        except Exception as e:
            self.logger.error(f"Error clearing index: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        return {
            'total_documents': len(self.documents),
            'index_size': self.index.ntotal,
            'dimension': self.dimension,
            'model_name': self.model_name,
            'average_doc_length': np.mean([len(doc) for doc in self.documents]) if self.documents else 0
        }
    
    def batch_similarity_search(self, queries: List[str], k: int = 5) -> List[List[str]]:
        """Perform batch similarity search for multiple queries"""
        try:
            if self.index.ntotal == 0:
                raise ValueError("No documents in vector store")
            
            # Generate embeddings for all queries
            query_embeddings = self._generate_embeddings(queries)
            
            # Batch search
            scores, indices = self.index.search(query_embeddings.astype('float32'), k)
            
            # Process results
            results = []
            for query_scores, query_indices in zip(scores, indices):
                query_results = []
                for score, idx in zip(query_scores, query_indices):
                    if idx != -1:
                        query_results.append(self.documents[idx])
                results.append(query_results)
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error in batch similarity search: {str(e)}")
            raise