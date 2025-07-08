"""
LLM pipeline using OpenAI API with enhanced context processing
"""
from openai import OpenAI
from typing import List, Dict, Any
import json
import os
import re
from dataclasses import dataclass

@dataclass
class QAResponse:
    """Structured response from QA system"""
    answer: str
    context: List[str]
    confidence: float
    sources: List[str]

class OpenAILLM:
    """OpenAI LLM wrapper with enhanced fallback processing"""
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.2, max_tokens: int = 800):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=api_key)
    
    def generate_answer(self, question: str, context: List[str]) -> QAResponse:
        """Generate answer using OpenAI API with intelligent fallback"""
        
        # Create enhanced prompt
        context_text = "\n\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(context)])
        
        prompt = f"""
You are an expert on "Alice's Adventures in Wonderland" by Lewis Carroll. Based on the provided context from the book, answer the following question accurately and comprehensively.

Context from the book:
{context_text}

Question: {question}

Please provide your answer in the following JSON format:
{{
    "answer": "Your detailed answer here",
    "confidence": 0.95,
    "key_points": ["point1", "point2", "point3"],
    "relevant_characters": ["character1", "character2"],
    "relevant_themes": ["theme1", "theme2"]
}}

Guidelines:
- Base your answer strictly on the provided context
- If the context doesn't contain enough information, mention this limitation
- Provide specific details and quotes when possible
- Rate your confidence from 0.0 to 1.0
- Identify key characters and themes relevant to the question
"""

        try:
            # Add retry logic and better error handling
            import time
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant expert in literature analysis."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        timeout=30
                    )
                    break  # Success, exit retry loop
                except Exception as e:
                    if attempt == max_retries - 1:  # Last attempt
                        raise e
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in 2 seconds...")
                    time.sleep(2)
            
            # Parse response
            content = response.choices[0].message.content.strip()
            
            # Try to parse JSON response
            try:
                parsed_response = json.loads(content)
                return QAResponse(
                    answer=parsed_response.get("answer", content),
                    context=context,
                    confidence=parsed_response.get("confidence", 0.8),
                    sources=[f"Alice's Adventures in Wonderland - Context {i+1}" for i in range(len(context))]
                )
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return QAResponse(
                    answer=content,
                    context=context,
                    confidence=0.7,
                    sources=[f"Alice's Adventures in Wonderland - Context {i+1}" for i in range(len(context))]
                )
                
        except Exception as e:
            # Enhanced fallback: process context intelligently
            print(f"OpenAI API error: {e}")
            print("Using intelligent context processing...")
            
            if context:
                fallback_answer = self._process_context_intelligently(question, context)
                confidence = self._calculate_confidence(question, context, fallback_answer)
            else:
                fallback_answer = "I couldn't find relevant information in the knowledge base to answer your question."
                confidence = 0.1
            
            return QAResponse(
                answer=fallback_answer,
                context=context,
                confidence=confidence,
                sources=[f"Alice's Adventures in Wonderland - Context {i+1}" for i in range(len(context))]
            )
    
    def _process_context_intelligently(self, question: str, context: List[str]) -> str:
        """Process context using NLP techniques to create intelligent answers"""
        # Combine all context
        full_context = " ".join(context)
        
        # Extract question keywords
        question_words = self._extract_keywords(question)
        
        # Find most relevant sentences
        relevant_sentences = self._find_relevant_sentences(full_context, question_words)
        
        # Process specific question types
        answer = self._generate_contextual_answer(question, relevant_sentences, full_context)
        
        return answer
    
    def _extract_keywords(self, question: str) -> List[str]:
        """Extract important keywords from question"""
        # Remove common stop words
        stop_words = {'is', 'are', 'was', 'were', 'what', 'who', 'where', 'when', 'why', 'how', 
                     'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
        
        words = re.findall(r'\b\w+\b', question.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    def _find_relevant_sentences(self, context: str, keywords: List[str]) -> List[str]:
        """Find sentences most relevant to the question"""
        sentences = re.split(r'[.!?]+', context)
        relevant_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
                
            # Count keyword matches
            sentence_lower = sentence.lower()
            keyword_count = sum(1 for keyword in keywords if keyword in sentence_lower)
            
            if keyword_count > 0:
                relevant_sentences.append((sentence, keyword_count))
        
        # Sort by relevance (keyword count) and return top sentences
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        return [sentence for sentence, _ in relevant_sentences[:3]]
    
    def _generate_contextual_answer(self, question: str, relevant_sentences: List[str], full_context: str) -> str:
        """Generate answer based on relevant sentences and context"""
        if not relevant_sentences:
            return "I found some relevant text but couldn't extract a clear answer to your specific question."
        
        question_lower = question.lower()
        
        # Question type detection
        if any(word in question_lower for word in ['who is', 'who was']):
            return self._answer_who_question(relevant_sentences, question_lower)
        elif any(word in question_lower for word in ['what is', 'what was', 'what does', 'what did']):
            return self._answer_what_question(relevant_sentences, question_lower)
        elif any(word in question_lower for word in ['how does', 'how did', 'how was']):
            return self._answer_how_question(relevant_sentences, question_lower)
        elif any(word in question_lower for word in ['where', 'when']):
            return self._answer_where_when_question(relevant_sentences, question_lower)
        else:
            # General answer
            return self._create_general_answer(relevant_sentences)
    
    def _answer_who_question(self, sentences: List[str], question: str) -> str:
        """Answer 'who' questions by finding character descriptions"""
        for sentence in sentences:
            # Look for character descriptions or introductions
            if any(word in sentence.lower() for word in ['is', 'was', 'character', 'girl', 'boy', 'cat', 'rabbit', 'queen', 'hatter']):
                return f"Based on the text: {sentence.strip()}."
        
        return f"According to the story: {sentences[0].strip()}."
    
    def _answer_what_question(self, sentences: List[str], question: str) -> str:
        """Answer 'what' questions by finding descriptions or explanations"""
        for sentence in sentences:
            # Look for descriptive content
            if any(word in sentence.lower() for word in ['is', 'was', 'like', 'made', 'does', 'did']):
                return f"Based on the text: {sentence.strip()}."
        
        return f"The story describes: {sentences[0].strip()}."
    
    def _answer_how_question(self, sentences: List[str], question: str) -> str:
        """Answer 'how' questions by finding process or method descriptions"""
        for sentence in sentences:
            # Look for action or process descriptions
            if any(word in sentence.lower() for word in ['by', 'through', 'then', 'first', 'after', 'began', 'started']):
                return f"According to the story: {sentence.strip()}."
        
        return f"The text explains: {sentences[0].strip()}."
    
    def _answer_where_when_question(self, sentences: List[str], question: str) -> str:
        """Answer 'where' and 'when' questions by finding location/time references"""
        for sentence in sentences:
            # Look for location or time indicators
            if any(word in sentence.lower() for word in ['in', 'at', 'on', 'under', 'over', 'near', 'when', 'while', 'during']):
                return f"Based on the text: {sentence.strip()}."
        
        return f"The story mentions: {sentences[0].strip()}."
    
    def _create_general_answer(self, sentences: List[str]) -> str:
        """Create a general answer from relevant sentences"""
        if len(sentences) == 1:
            return f"According to the story: {sentences[0].strip()}."
        else:
            # Combine multiple sentences intelligently
            combined = ". ".join([s.strip() for s in sentences[:2]])
            return f"Based on the text: {combined}."
    
    def _calculate_confidence(self, question: str, context: List[str], answer: str) -> float:
        """Calculate confidence score based on context quality and answer relevance"""
        if not context:
            return 0.1
        
        # Base confidence
        confidence = 0.5
        
        # Increase confidence if answer is longer and more detailed
        if len(answer) > 50:
            confidence += 0.1
        
        # Increase confidence if we found multiple relevant sentences
        full_context = " ".join(context)
        question_words = self._extract_keywords(question)
        relevant_sentences = self._find_relevant_sentences(full_context, question_words)
        
        if len(relevant_sentences) >= 2:
            confidence += 0.1
        if len(relevant_sentences) >= 3:
            confidence += 0.1
        
        # Increase confidence if answer contains question keywords
        answer_lower = answer.lower()
        question_words_in_answer = sum(1 for word in question_words if word in answer_lower)
        confidence += min(question_words_in_answer * 0.05, 0.2)
        
        return min(confidence, 0.9)  # Cap at 0.9