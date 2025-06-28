# evaluation_metrics.py
import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
import logging

class EvaluationMetrics:
    """Handle evaluation metrics for the RAG system"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
        self.logger = logging.getLogger(__name__)
    
    def evaluate_answer(self, question: str, answer: str, context_chunks: List[str]) -> Dict[str, float]:
        """Comprehensive evaluation of an answer"""
        try:
            metrics = {}
            
            # Relevance score
            metrics['relevance_score'] = self._calculate_relevance_score(question, answer)
            
            # Context utilization score
            metrics['context_utilization'] = self._calculate_context_utilization(answer, context_chunks)
            
            # Answer completeness
            metrics['completeness_score'] = self._calculate_completeness_score(question, answer)
            
            # Factual consistency (semantic similarity with context)
            metrics['factual_consistency'] = self._calculate_factual_consistency(answer, context_chunks)
            
            # Answer length appropriateness
            metrics['length_appropriateness'] = self._calculate_length_appropriateness(answer)
            
            # Confidence score (aggregate measure)
            metrics['confidence_score'] = self._calculate_confidence_score(metrics)
            
            # Overall quality score
            metrics['quality_score'] = self._calculate_overall_quality(metrics)
            
            return metrics
        
        except Exception as e:
            self.logger.error(f"Error evaluating answer: {str(e)}")
            return self._get_default_metrics()
    
    def _calculate_relevance_score(self, question: str, answer: str) -> float:
        """Calculate how relevant the answer is to the question"""
        try:
            # Use semantic similarity
            question_embedding = self.embedding_model.encode([question])
            answer_embedding = self.embedding_model.encode([answer])
            
            similarity = cosine_similarity(question_embedding, answer_embedding)[0][0]
            
            # Normalize to 0-1 range (cosine similarity can be negative)
            relevance_score = max(0, (similarity + 1) / 2)
            
            return float(relevance_score)
        
        except Exception as e:
            self.logger.error(f"Error calculating relevance score: {str(e)}")
            return 0.5
    
    def _calculate_context_utilization(self, answer: str, context_chunks: List[str]) -> float:
        """Calculate how well the answer utilizes the provided context"""
        try:
            if not context_chunks:
                return 0.0
            
            answer_embedding = self.embedding_model.encode([answer])
            context_embeddings = self.embedding_model.encode(context_chunks)
            
            # Calculate similarity with each context chunk
            similarities = cosine_similarity(answer_embedding, context_embeddings)[0]
            
            # Use maximum similarity as utilization score
            max_similarity = np.max(similarities)
            
            # Normalize to 0-1 range
            utilization_score = max(0, (max_similarity + 1) / 2)
            
            return float(utilization_score)
        
        except Exception as e:
            self.logger.error(f"Error calculating context utilization: {str(e)}")
            return 0.5
    
    def _calculate_completeness_score(self, question: str, answer: str) -> float:
        """Calculate completeness of the answer"""
        try:
            # Heuristic-based completeness scoring
            score = 0.0
            
            # Length factor (longer answers are generally more complete, but with diminishing returns)
            answer_length = len(answer.split())
            if answer_length < 10:
                length_score = answer_length / 10
            elif answer_length < 50:
                length_score = 0.8 + (answer_length - 10) / 200  # Slower growth
            else:
                length_score = 1.0
            
            score += length_score * 0.4
            
            # Structure factor (presence of detailed explanations)
            if any(word in answer.lower() for word in ['because', 'therefore', 'however', 'additionally', 'furthermore']):
                score += 0.2
            
            # Specificity factor (presence of specific details)
            if re.search(r'\d+', answer) or any(word in answer.lower() for word in ['specifically', 'particularly', 'example']):
                score += 0.2
            
            # Question addressing factor
            question_words = set(re.findall(r'\b\w+\b', question.lower()))
            answer_words = set(re.findall(r'\b\w+\b', answer.lower()))
            overlap_ratio = len(question_words & answer_words) / max(len(question_words), 1)
            score += overlap_ratio * 0.2
            
            return min(1.0, score)
        
        except Exception as e:
            self.logger.error(f"Error calculating completeness score: {str(e)}")
            return 0.5
    
    def _calculate_factual_consistency(self, answer: str, context_chunks: List[str]) -> float:
        """Calculate factual consistency between answer and context"""
        try:
            if not context_chunks:
                return 0.5
            
            # Combine all context
            full_context = " ".join(context_chunks)
            
            # Use semantic similarity as a proxy for factual consistency
            answer_embedding = self.embedding_model.encode([answer])
            context_embedding = self.embedding_model.encode([full_context])
            
            similarity = cosine_similarity(answer_embedding, context_embedding)[0][0]
            
            # Normalize to 0-1 range
            consistency_score = max(0, (similarity + 1) / 2)
            
            return float(consistency_score)
        
        except Exception as e:
            self.logger.error(f"Error calculating factual consistency: {str(e)}")
            return 0.5
    
    def _calculate_length_appropriateness(self, answer: str) -> float:
        """Calculate if the answer length is appropriate"""
        try:
            word_count = len(answer.split())
            
            # Optimal range: 20-200 words
            if 20 <= word_count <= 200:
                return 1.0
            elif word_count < 20:
                return word_count / 20  # Penalty for too short
            else:  # word_count > 200
                return max(0.5, 1.0 - (word_count - 200) / 300)  # Penalty for too long
        
        except Exception as e:
            self.logger.error(f"Error calculating length appropriateness: {str(e)}")
            return 0.5
    
    def _calculate_confidence_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall confidence in the answer"""
        try:
            # Weighted average of key metrics
            weights = {
                'relevance_score': 0.3,
                'context_utilization': 0.25,
                'factual_consistency': 0.25,
                'completeness_score': 0.2
            }
            
            confidence = 0.0
            total_weight = 0.0
            
            for metric, weight in weights.items():
                if metric in metrics:
                    confidence += metrics[metric] * weight
                    total_weight += weight
            
            if total_weight > 0:
                confidence /= total_weight
            
            return float(confidence)
        
        except Exception as e:
            self.logger.error(f"Error calculating confidence score: {str(e)}")
            return 0.5
    
    def _calculate_overall_quality(self, metrics: Dict[str, float]) -> float:
        """Calculate overall quality score"""
        try:
            # Weighted combination of all metrics
            weights = {
                'relevance_score': 0.25,
                'context_utilization': 0.2,
                'completeness_score': 0.2,
                'factual_consistency': 0.2,
                'length_appropriateness': 0.1,
                'confidence_score': 0.05
            }
            
            quality = 0.0
            total_weight = 0.0
            
            for metric, weight in weights.items():
                if metric in metrics:
                    quality += metrics[metric] * weight
                    total_weight += weight
            
            if total_weight > 0:
                quality /= total_weight
            
            return float(quality)
        
        except Exception as e:
            self.logger.error(f"Error calculating overall quality: {str(e)}")
            return 0.5
    
    def _get_default_metrics(self) -> Dict[str, float]:
        """Return default metrics when evaluation fails"""
        return {
            'relevance_score': 0.5,
            'context_utilization': 0.5,
            'completeness_score': 0.5,
            'factual_consistency': 0.5,
            'length_appropriateness': 0.5,
            'confidence_score': 0.5,
            'quality_score': 0.5
        }
    
    def evaluate_retrieval_quality(self, query: str, retrieved_chunks: List[str], 
                                 ground_truth_chunks: List[str] = None) -> Dict[str, float]:
        """Evaluate the quality of retrieved chunks"""
        try:
            metrics = {}
            
            if not retrieved_chunks:
                return {'retrieval_precision': 0.0, 'retrieval_recall': 0.0, 'retrieval_f1': 0.0}
            
            # Query-chunk relevance
            query_embedding = self.embedding_model.encode([query])
            chunk_embeddings = self.embedding_model.encode(retrieved_chunks)
            
            similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
            
            # Average relevance score
            avg_relevance = np.mean(similarities)
            metrics['avg_chunk_relevance'] = float(max(0, (avg_relevance + 1) / 2))
            
            # Diversity score (how diverse are the retrieved chunks)
            if len(retrieved_chunks) > 1:
                chunk_similarities = cosine_similarity(chunk_embeddings)
                # Remove diagonal elements and calculate average
                mask = ~np.eye(chunk_similarities.shape[0], dtype=bool)
                avg_chunk_similarity = np.mean(chunk_similarities[mask])
                diversity_score = 1 - max(0, (avg_chunk_similarity + 1) / 2)
                metrics['chunk_diversity'] = float(diversity_score)
            else:
                metrics['chunk_diversity'] = 1.0
            
            # Coverage score (how well chunks cover the query)
            metrics['query_coverage'] = self._calculate_query_coverage(query, retrieved_chunks)
            
            # If ground truth is provided, calculate precision and recall
            if ground_truth_chunks:
                precision, recall, f1 = self._calculate_precision_recall(retrieved_chunks, ground_truth_chunks)
                metrics['retrieval_precision'] = precision
                metrics['retrieval_recall'] = recall
                metrics['retrieval_f1'] = f1
            
            return metrics
        
        except Exception as e:
            self.logger.error(f"Error evaluating retrieval quality: {str(e)}")
            return {'retrieval_precision': 0.5, 'retrieval_recall': 0.5, 'retrieval_f1': 0.5}
    
    def _calculate_query_coverage(self, query: str, chunks: List[str]) -> float:
        """Calculate how well the chunks cover the query topics"""
        try:
            query_words = set(re.findall(r'\b\w+\b', query.lower()))
            chunk_words = set()
            
            for chunk in chunks:
                chunk_words.update(re.findall(r'\b\w+\b', chunk.lower()))
            
            if not query_words:
                return 1.0
            
            coverage = len(query_words & chunk_words) / len(query_words)
            return float(coverage)
        
        except Exception as e:
            self.logger.error(f"Error calculating query coverage: {str(e)}")
            return 0.5
    
    def _calculate_precision_recall(self, retrieved: List[str], ground_truth: List[str]) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1 score for retrieval"""
        try:
            if not retrieved:
                return 0.0, 0.0, 0.0
            
            # Use semantic similarity to match chunks
            retrieved_embeddings = self.embedding_model.encode(retrieved)
            ground_truth_embeddings = self.embedding_model.encode(ground_truth)
            
            # Calculate similarity matrix
            similarities = cosine_similarity(retrieved_embeddings, ground_truth_embeddings)
            
            # Consider chunks as matching if similarity > threshold
            threshold = 0.8
            matches = (similarities > threshold).any(axis=1)
            
            true_positives = np.sum(matches)
            precision = true_positives / len(retrieved) if retrieved else 0.0
            recall = true_positives / len(ground_truth) if ground_truth else 0.0
            
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            return float(precision), float(recall), float(f1)
        
        except Exception as e:
            self.logger.error(f"Error calculating precision/recall: {str(e)}")
            return 0.5, 0.5, 0.5
    
    def evaluate_system_performance(self, queries: List[str], answers: List[str], 
                                  contexts: List[List[str]]) -> Dict[str, Any]:
        """Evaluate overall system performance across multiple queries"""
        try:
            if len(queries) != len(answers) or len(queries) != len(contexts):
                raise ValueError("Queries, answers, and contexts must have the same length")
            
            all_metrics = []
            
            for query, answer, context in zip(queries, answers, contexts):
                metrics = self.evaluate_answer(query, answer, context)
                all_metrics.append(metrics)
            
            # Calculate aggregate statistics
            aggregate_metrics = {}
            
            for metric_name in all_metrics[0].keys():
                values = [m[metric_name] for m in all_metrics]
                aggregate_metrics[f"{metric_name}_mean"] = float(np.mean(values))
                aggregate_metrics[f"{metric_name}_std"] = float(np.std(values))
                aggregate_metrics[f"{metric_name}_min"] = float(np.min(values))
                aggregate_metrics[f"{metric_name}_max"] = float(np.max(values))
            
            # Overall system score
            quality_scores = [m['quality_score'] for m in all_metrics]
            aggregate_metrics['overall_system_score'] = float(np.mean(quality_scores))
            
            # Performance consistency
            aggregate_metrics['performance_consistency'] = float(1 - np.std(quality_scores))
            
            return {
                'individual_metrics': all_metrics,
                'aggregate_metrics': aggregate_metrics,
                'total_queries': len(queries)
            }
        
        except Exception as e:
            self.logger.error(f"Error evaluating system performance: {str(e)}")
            return {'error': str(e)}
    
    def generate_evaluation_report(self, metrics: Dict[str, Any]) -> str:
        """Generate a human-readable evaluation report"""
        try:
            report = []
            report.append("=== RAG System Evaluation Report ===\n")
            
            if 'individual_metrics' in metrics:
                # System-level report
                agg = metrics['aggregate_metrics']
                report.append(f"Total Queries Evaluated: {metrics['total_queries']}")
                report.append(f"Overall System Score: {agg['overall_system_score']:.3f}")
                report.append(f"Performance Consistency: {agg['performance_consistency']:.3f}")
                report.append("\n--- Average Metrics ---")
                report.append(f"Relevance Score: {agg['relevance_score_mean']:.3f} (±{agg['relevance_score_std']:.3f})")
                report.append(f"Context Utilization: {agg['context_utilization_mean']:.3f} (±{agg['context_utilization_std']:.3f})")
                report.append(f"Completeness: {agg['completeness_score_mean']:.3f} (±{agg['completeness_score_std']:.3f})")
                report.append(f"Factual Consistency: {agg['factual_consistency_mean']:.3f} (±{agg['factual_consistency_std']:.3f})")
                report.append(f"Quality Score: {agg['quality_score_mean']:.3f} (±{agg['quality_score_std']:.3f})")
            else:
                # Single query report
                report.append("--- Individual Query Metrics ---")
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        report.append(f"{metric.replace('_', ' ').title()}: {value:.3f}")
            
            return "\n".join(report)
        
        except Exception as e:
            self.logger.error(f"Error generating evaluation report: {str(e)}")
            return f"Error generating report: {str(e)}"