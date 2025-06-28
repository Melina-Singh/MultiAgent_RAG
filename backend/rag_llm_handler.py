# llm_handler.py
import google.generativeai as genai
import os
from typing import List, Dict, Any, Optional
import logging
import time
from dotenv import load_dotenv

class LLMHandler:
    """Handle interactions with Google Gemini LLM"""
    
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        
        # Configure Gemini
        print("Current working directory:", os.getcwd())
        print("Loading .env file...")
        load_dotenv()
        
        api_key = os.getenv('GOOGLE_API_KEY')
        print(f"API key loaded: {'Yes' if api_key else 'No'}")
        if api_key:
            print(f"API key starts with: {api_key[:10]}...")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
        # Generation config
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
    
    def generate_answer(self, question: str, context_chunks: List[str], 
                       max_context_length: int = 4000) -> str:
        """Generate an answer based on question and context"""
        try:
            # Prepare context
            context = self._prepare_context(context_chunks, max_context_length)
            
            # Create prompt
            prompt = self._create_qa_prompt(question, context)
            
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            if response.text:
                return response.text.strip()
            else:
                return "I couldn't generate an answer based on the provided context."
        
        except Exception as e:
            self.logger.error(f"Error generating answer: {str(e)}")
            return f"Error generating answer: {str(e)}"
    
    def _prepare_context(self, chunks: List[str], max_length: int) -> str:
        """Prepare context from chunks, ensuring it doesn't exceed max length"""
        context = ""
        for i, chunk in enumerate(chunks):
            chunk_with_separator = f"\n[Context {i+1}]\n{chunk}\n"
            
            if len(context) + len(chunk_with_separator) <= max_length:
                context += chunk_with_separator
            else:
                # Add partial chunk if there's meaningful space left
                remaining_space = max_length - len(context) - 50  # Leave some buffer
                if remaining_space > 200:
                    context += f"\n[Context {i+1}]\n{chunk[:remaining_space]}...\n"
                break
        
        return context.strip()
    
    def _create_qa_prompt(self, question: str, context: str) -> str:
        """Create a prompt for question answering"""
        prompt = f"""Based on the following context, please answer the question. If the answer cannot be found in the context, please say so clearly.

Context:
{context}

Question: {question}

Instructions:
- Provide a clear, accurate answer based on the context
- If the information is not available in the context, state this clearly
- Use specific details from the context when possible
- Keep the answer concise but comprehensive

Answer:"""
        
        return prompt
    
    def generate_summary(self, text: str, summary_type: str = "abstractive") -> str:
        """Generate a summary of the given text"""
        try:
            if summary_type == "abstractive":
                prompt = f"""Please provide a comprehensive abstractive summary of the following text. The summary should:
- Capture the main ideas and key points
- Be written in your own words
- Be approximately 20% of the original length
- Maintain the logical flow of information

Text to summarize:
{text[:8000]}  # Limit text length for API

Summary:"""
            
            elif summary_type == "extractive":
                prompt = f"""Please provide an extractive summary by selecting the most important sentences from the following text:

Text:
{text[:8000]}

Please extract 5-10 key sentences that best represent the main content:"""
            
            else:  # key_points
                prompt = f"""Please extract the key points from the following text as a bullet list:

Text:
{text[:8000]}

Key Points:"""
            
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            if response.text:
                return response.text.strip()
            else:
                return "Could not generate summary."
        
        except Exception as e:
            self.logger.error(f"Error generating summary: {str(e)}")
            return f"Error generating summary: {str(e)}"
    
    def evaluate_answer_quality(self, question: str, answer: str, context: str) -> Dict[str, float]:
        """Evaluate the quality of an answer"""
        try:
            prompt = f"""Please evaluate the quality of the following answer based on these criteria:

Question: {question}

Answer: {answer}

Context: {context[:2000]}

Please rate each aspect on a scale of 0.0 to 1.0 and provide your ratings in this exact format:
RELEVANCE: [score]
ACCURACY: [score]
COMPLETENESS: [score]
CLARITY: [score]

Evaluation:"""
            
            response = self.model.generate_content(
                prompt,
                generation_config={**self.generation_config, "temperature": 0.3}
            )
            
            if response.text:
                return self._parse_evaluation_scores(response.text)
            else:
                return {"relevance": 0.5, "accuracy": 0.5, "completeness": 0.5, "clarity": 0.5}
        
        except Exception as e:
            self.logger.error(f"Error evaluating answer: {str(e)}")
            return {"relevance": 0.5, "accuracy": 0.5, "completeness": 0.5, "clarity": 0.5}
    
    def _parse_evaluation_scores(self, evaluation_text: str) -> Dict[str, float]:
        """Parse evaluation scores from LLM response"""
        scores = {"relevance": 0.5, "accuracy": 0.5, "completeness": 0.5, "clarity": 0.5}
        
        try:
            lines = evaluation_text.split('\n')
            for line in lines:
                line = line.strip().upper()
                if 'RELEVANCE:' in line:
                    score = float(line.split(':')[1].strip())
                    scores["relevance"] = max(0.0, min(1.0, score))
                elif 'ACCURACY:' in line:
                    score = float(line.split(':')[1].strip())
                    scores["accuracy"] = max(0.0, min(1.0, score))
                elif 'COMPLETENESS:' in line:
                    score = float(line.split(':')[1].strip())
                    scores["completeness"] = max(0.0, min(1.0, score))
                elif 'CLARITY:' in line:
                    score = float(line.split(':')[1].strip())
                    scores["clarity"] = max(0.0, min(1.0, score))
        except:
            pass  # Return default scores if parsing fails
        
        return scores
    
    def generate_questions(self, text: str, num_questions: int = 5) -> List[str]:
        """Generate questions based on the given text"""
        try:
            prompt = f"""Based on the following text, generate {num_questions} diverse and meaningful questions that can be answered using the information in the text. The questions should cover different aspects and levels of detail.

Text:
{text[:6000]}

Please generate exactly {num_questions} questions, one per line:"""
            
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            if response.text:
                questions = [q.strip() for q in response.text.split('\n') if q.strip()]
                return questions[:num_questions]  # Ensure we don't exceed requested number
            else:
                return []
        
        except Exception as e:
            self.logger.error(f"Error generating questions: {str(e)}")
            return []
    
    def check_answer_relevance(self, question: str, answer: str) -> float:
        """Check how relevant an answer is to a question"""
        try:
            prompt = f"""Rate how relevant the following answer is to the given question on a scale of 0.0 to 1.0, where:
- 1.0 = Perfectly relevant and directly answers the question
- 0.5 = Somewhat relevant but incomplete or tangential
- 0.0 = Not relevant at all

Question: {question}
Answer: {answer}

Please respond with only a number between 0.0 and 1.0:"""
            
            response = self.model.generate_content(
                prompt,
                generation_config={**self.generation_config, "temperature": 0.1}
            )
            
            if response.text:
                try:
                    score = float(response.text.strip())
                    return max(0.0, min(1.0, score))
                except:
                    return 0.5
            else:
                return 0.5
        
        except Exception as e:
            self.logger.error(f"Error checking answer relevance: {str(e)}")
            return 0.5
    
    def batch_generate_answers(self, questions: List[str], context: str) -> List[str]:
        """Generate answers for multiple questions using the same context"""
        answers = []
        for question in questions:
            try:
                answer = self.generate_answer(question, [context])
                answers.append(answer)
                # Small delay to avoid rate limiting
                time.sleep(0.5)
            except Exception as e:
                self.logger.error(f"Error generating answer for question '{question}': {str(e)}")
                answers.append(f"Error generating answer: {str(e)}")
        
        return answers
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "model_name": self.model_name,
            "generation_config": self.generation_config,
            "api_configured": bool(os.getenv("GOOGLE_API_KEY"))
        }