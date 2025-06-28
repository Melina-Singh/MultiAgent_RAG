# summarizer.py
import logging
from typing import List, Dict, Any, Optional
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

class Summarizer:
    """Handle document and context summarization for the RAG system"""
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn", 
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the summarizer with specified models
        
        Args:
            model_name: HuggingFace model for summarization
            embedding_model: Model for semantic similarity calculations
        """
        self.logger = logging.getLogger(__name__)
        
        try:
            # Initialize summarization pipeline
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.summarizer = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            # Initialize embedding model for similarity calculations
            self.embedding_model = SentenceTransformer(embedding_model)
            
            # Configuration parameters
            self.max_input_length = 1024
            self.min_summary_length = 50
            self.max_summary_length = 200
            
            self.logger.info(f"Summarizer initialized with {model_name} on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Error initializing summarizer: {str(e)}")
            # Fallback to a simpler model
            self._initialize_fallback()
    
    def _initialize_fallback(self):
        """Initialize with fallback models if primary initialization fails"""
        try:
            self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            self.logger.info("Initialized with fallback models")
        except Exception as e:
            self.logger.error(f"Fallback initialization failed: {str(e)}")
            self.summarizer = None
    
    def summarize_text(self, text: str, max_length: Optional[int] = None, 
                      min_length: Optional[int] = None) -> str:
        """
        Summarize a single text document
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of summary
            min_length: Minimum length of summary
            
        Returns:
            Summarized text
        """
        try:
            if not text or not text.strip():
                return ""
            
            if self.summarizer is None:
                # CHANGED: Fixed method call - was calling non-existent _extractive_summary
                return self._extractive_summary(text, max_length or self.max_summary_length)
            
            # Clean and prepare text
            cleaned_text = self._clean_text(text)
            
            # Check if text is too short to summarize
            if len(cleaned_text.split()) < 50:
                return cleaned_text
            
            # Truncate if text is too long for the model
            if len(cleaned_text.split()) > self.max_input_length:
                cleaned_text = self._truncate_text(cleaned_text, self.max_input_length)
            
            # Set summary parameters
            max_len = max_length or min(self.max_summary_length, len(cleaned_text.split()) // 3)
            min_len = min_length or max(self.min_summary_length, max_len // 4)
            
            # Generate summary
            summary = self.summarizer(
                cleaned_text,
                max_length=max_len,
                min_length=min_len,
                do_sample=False,
                early_stopping=True
            )
            
            return summary[0]['summary_text'].strip()
            
        except Exception as e:
            self.logger.error(f"Error in text summarization: {str(e)}")
            # CHANGED: Fixed method call - was calling non-existent _extractive_summary
            return self._extractive_summary(text, max_length or self.max_summary_length)
    
    def summarize_chunks(self, chunks: List[str], query: Optional[str] = None) -> str:
        """
        Summarize multiple text chunks, optionally guided by a query
        
        Args:
            chunks: List of text chunks to summarize
            query: Optional query to guide summarization
            
        Returns:
            Combined summary of all chunks
        """
        try:
            if not chunks:
                return ""
            
            # If only one chunk, summarize directly
            if len(chunks) == 1:
                return self.summarize_text(chunks[0])
            
            # If query is provided, rank chunks by relevance
            if query:
                ranked_chunks = self._rank_chunks_by_relevance(chunks, query)
            else:
                ranked_chunks = chunks
            
            # Combine chunks intelligently
            combined_text = self._combine_chunks(ranked_chunks, query)
            
            # Summarize the combined text
            return self.summarize_text(combined_text, max_length=300)
            
        except Exception as e:
            self.logger.error(f"Error in chunk summarization: {str(e)}")
            return self._fallback_chunk_summary(chunks)
    
    def summarize_for_context(self, chunks: List[str], query: str, 
                            max_context_length: int = 2000) -> str:
        """
        Create a focused summary for RAG context
        
        Args:
            chunks: Retrieved chunks
            query: User query
            max_context_length: Maximum length of context
            
        Returns:
            Focused summary for RAG context
        """
        try:
            if not chunks:
                return ""
            
            # Rank chunks by relevance to query
            ranked_chunks = self._rank_chunks_by_relevance(chunks, query)
            
            # Select most relevant chunks that fit within context length
            selected_chunks = self._select_chunks_for_context(ranked_chunks, max_context_length)
            
            # Create focused summary
            if len(selected_chunks) == 1:
                return self._create_focused_summary(selected_chunks[0], query)
            else:
                combined = self._combine_chunks(selected_chunks, query)
                return self._create_focused_summary(combined, query)
                
        except Exception as e:
            self.logger.error(f"Error creating context summary: {str(e)}")
            return " ".join(chunks[:3])[:max_context_length]
    
    def generate_document_summary(self, document_chunks: List[str], 
                                document_title: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive document summary with metadata
        
        Args:
            document_chunks: All chunks from a document
            document_title: Optional document title
            
        Returns:
            Dictionary with summary and metadata
        """
        try:
            if not document_chunks:
                return {"summary": "", "word_count": 0, "key_topics": []}
            
            # Combine all chunks
            full_text = " ".join(document_chunks)
            
            # Generate main summary
            main_summary = self.summarize_text(full_text, max_length=300)
            
            # Extract key topics
            key_topics = self._extract_key_topics(full_text)
            
            # Generate section summaries if document is large
            section_summaries = []
            if len(document_chunks) > 10:
                section_summaries = self._generate_section_summaries(document_chunks)
            
            return {
                "title": document_title or "Untitled Document",
                "main_summary": main_summary,
                "word_count": len(full_text.split()),
                "chunk_count": len(document_chunks),
                "key_topics": key_topics,
                "section_summaries": section_summaries
            }
            
        except Exception as e:
            self.logger.error(f"Error generating document summary: {str(e)}")
            return {"summary": "", "word_count": 0, "key_topics": [], "error": str(e)}
    
    # NEW METHOD ADDED: This was missing and causing the error
    def _extractive_summary(self, text: str, max_length: int) -> str:
        """
        Create extractive summary by selecting important sentences
        This serves as a fallback when transformer-based summarization fails
        
        Args:
            text: Input text to summarize
            max_length: Maximum number of words in summary
            
        Returns:
            Extractive summary of the text
        """
        try:
            if not text or not text.strip():
                return ""
            
            # Split text into sentences
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not sentences:
                return text[:max_length * 5]  # Fallback to character truncation
            
            # If only one sentence, return it (truncated if necessary)
            if len(sentences) == 1:
                words = sentences[0].split()
                if len(words) <= max_length:
                    return sentences[0]
                else:
                    return ' '.join(words[:max_length])
            
            # Score sentences based on word frequency and position
            word_freq = self._calculate_word_frequencies(text)
            sentence_scores = []
            
            for i, sentence in enumerate(sentences):
                # Score based on word frequencies
                words = sentence.lower().split()
                score = sum(word_freq.get(word, 0) for word in words) / len(words) if words else 0
                
                # Boost score for sentences at the beginning (often more important)
                position_boost = 1.5 if i < 2 else 1.0
                
                sentence_scores.append((score * position_boost, i, sentence))
            
            # Sort by score (descending)
            sentence_scores.sort(reverse=True)
            
            # Select top sentences that fit within max_length
            selected_sentences = []
            current_word_count = 0
            
            for score, original_index, sentence in sentence_scores:
                sentence_word_count = len(sentence.split())
                
                if current_word_count + sentence_word_count <= max_length:
                    selected_sentences.append((original_index, sentence))
                    current_word_count += sentence_word_count
                elif current_word_count == 0:  # If first sentence is too long, truncate it
                    words = sentence.split()[:max_length]
                    selected_sentences.append((original_index, ' '.join(words)))
                    break
            
            # Sort selected sentences by original order
            selected_sentences.sort(key=lambda x: x[0])
            
            # Join sentences
            summary = ' '.join([sentence for _, sentence in selected_sentences])
            
            return summary if summary else text[:max_length * 5]
            
        except Exception as e:
            self.logger.error(f"Error in extractive summarization: {str(e)}")
            # Ultimate fallback: simple truncation
            words = text.split()
            return ' '.join(words[:max_length]) if len(words) > max_length else text
    
    # NEW METHOD ADDED: Helper method for extractive summarization
    def _calculate_word_frequencies(self, text: str) -> Dict[str, float]:
        """
        Calculate normalized word frequencies for extractive summarization
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of word frequencies
        """
        # Clean text and get words
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        # Filter out common stop words (simple list)
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        }
        
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        if not filtered_words:
            return {}
        
        # Calculate frequencies
        word_count = {}
        for word in filtered_words:
            word_count[word] = word_count.get(word, 0) + 1
        
        # Normalize frequencies
        max_freq = max(word_count.values()) if word_count else 1
        normalized_freq = {word: count / max_freq for word, count in word_count.items()}
        
        return normalized_freq
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text for summarization"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^\w\s.,!?;:-]', '', text)
        
        # Ensure proper sentence endings
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        return text.strip()
    
    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limits"""
        words = text.split()
        if len(words) <= max_tokens:
            return text
        
        # Truncate to max_tokens and try to end at sentence boundary
        truncated = ' '.join(words[:max_tokens])
        
        # Find last sentence ending
        last_sentence = max(
            truncated.rfind('.'),
            truncated.rfind('!'),
            truncated.rfind('?')
        )
        
        if last_sentence > len(truncated) * 0.8:  # If we can keep 80% of text
            return truncated[:last_sentence + 1]
        
        return truncated
    
    def _rank_chunks_by_relevance(self, chunks: List[str], query: str) -> List[str]:
        """Rank chunks by semantic similarity to query"""
        try:
            if not query:
                return chunks
            
            query_embedding = self.embedding_model.encode([query])
            chunk_embeddings = self.embedding_model.encode(chunks)
            
            similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
            
            # Sort chunks by similarity (descending)
            ranked_indices = np.argsort(similarities)[::-1]
            
            return [chunks[i] for i in ranked_indices]
            
        except Exception as e:
            self.logger.error(f"Error ranking chunks: {str(e)}")
            return chunks
    
    def _combine_chunks(self, chunks: List[str], query: Optional[str] = None) -> str:
        """Intelligently combine chunks into coherent text"""
        if not chunks:
            return ""
        
        if len(chunks) == 1:
            return chunks[0]
        
        # Add transition phrases between chunks
        combined = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                combined.append("Additionally, ")
            combined.append(chunk.strip())
        
        return " ".join(combined)
    
    def _select_chunks_for_context(self, chunks: List[str], max_length: int) -> List[str]:
        """Select chunks that fit within context length"""
        selected = []
        current_length = 0
        
        for chunk in chunks:
            chunk_length = len(chunk.split())
            if current_length + chunk_length <= max_length:
                selected.append(chunk)
                current_length += chunk_length
            else:
                # Try to fit partial chunk
                remaining_length = max_length - current_length
                if remaining_length > 50:  # Only if we can fit meaningful content
                    words = chunk.split()[:remaining_length]
                    selected.append(' '.join(words))
                break
        
        return selected
    
    def _create_focused_summary(self, text: str, query: str) -> str:
        """Create a summary focused on the query"""
        try:
            # Add query context to the summarization
            prompt_text = f"Query: {query}\n\nContext: {text}"
            
            return self.summarize_text(prompt_text, max_length=250)
            
        except Exception as e:
            self.logger.error(f"Error creating focused summary: {str(e)}")
            return self.summarize_text(text, max_length=250)
    
    def _extract_key_topics(self, text: str) -> List[str]:
        """Extract key topics from text using simple keyword extraction"""
        try:
            # Simple keyword extraction (can be enhanced with more sophisticated methods)
            words = re.findall(r'\b[A-Z][a-z]+\b', text)  # Capitalized words
            
            # Count frequency
            word_freq = {}
            for word in words:
                if len(word) > 3:  # Skip short words
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Return top keywords
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            
            return [word for word, freq in sorted_words[:10]] # Top 10 keywords
        except Exception as e:
            self.logger.error(f"Error extracting key topics: {str(e)}")
            return []
    
    # NEW METHOD ADDED: This was referenced but missing
    def _fallback_chunk_summary(self, chunks: List[str]) -> str:
        """
        Fallback method for summarizing chunks when main summarization fails
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Simple summary of chunks
        """
        try:
            if not chunks:
                return ""
            
            # Take first few chunks and create simple summary
            combined_text = " ".join(chunks[:3])  # Limit to first 3 chunks
            
            # Use extractive summarization as fallback
            return self._extractive_summary(combined_text, 200)
            
        except Exception as e:
            self.logger.error(f"Error in fallback chunk summary: {str(e)}")
            # Ultimate fallback: return first chunk truncated
            if chunks:
                words = chunks[0].split()
                return ' '.join(words[:100])  # First 100 words
            return ""
    
    # NEW METHOD ADDED: This was referenced but missing
    def _generate_section_summaries(self, document_chunks: List[str]) -> List[Dict[str, str]]:
        """
        Generate summaries for different sections of a large document
        
        Args:
            document_chunks: All chunks from the document
            
        Returns:
            List of section summaries with metadata
        """
        try:
            section_summaries = []
            chunk_per_section = max(3, len(document_chunks) // 5)  # Divide into ~5 sections
            
            for i in range(0, len(document_chunks), chunk_per_section):
                section_chunks = document_chunks[i:i + chunk_per_section]
                section_text = " ".join(section_chunks)
                
                # Generate summary for this section
                section_summary = self.summarize_text(section_text, max_length=150)
                
                section_summaries.append({
                    "section_number": len(section_summaries) + 1,
                    "chunk_range": f"{i + 1}-{min(i + chunk_per_section, len(document_chunks))}",
                    "summary": section_summary,
                    "word_count": len(section_text.split())
                })
            
            return section_summaries
            
        except Exception as e:
            self.logger.error(f"Error generating section summaries: {str(e)}")
            return []