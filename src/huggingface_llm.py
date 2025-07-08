"""
HuggingFace LLM pipeline for local model inference
Supports lightweight models optimized for Q&A tasks
"""
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    BitsAndBytesConfig
)
from typing import List, Dict, Any
import json
import re
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

@dataclass
class QAResponse:
    """Structured response from QA system"""
    answer: str
    context: List[str]
    confidence: float
    sources: List[str]

class HuggingFaceLLM:
    """HuggingFace LLM wrapper with support for lightweight models"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", 
                 temperature: float = 0.2, max_tokens: int = 800,
                 device: str = "auto", load_in_4bit: bool = False):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.device = device
        self.load_in_4bit = load_in_4bit
        
        print(f"Loading {model_name}...")
        self._setup_model()
    
    def _setup_model(self):
        """Setup the model and tokenizer"""
        try:
            # For smaller models, we don't need quantization
            quantization_config = None
            if self.load_in_4bit and torch.cuda.is_available() and "7b" in self.model_name.lower():
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with appropriate settings
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float32,  # Use float32 for smaller models
            }
            
            # Only use device_map for larger models
            if "7b" in self.model_name.lower() or "13b" in self.model_name.lower():
                model_kwargs["device_map"] = "auto" if torch.cuda.is_available() else None
                model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Move to device if not using device_map
            if "device_map" not in model_kwargs:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = self.model.to(device)
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() and "device_map" not in model_kwargs else -1,
            )
            
            print(f"✅ Model {self.model_name} loaded successfully!")
            print(f"Device: {next(self.model.parameters()).device}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e
    
    def _create_prompt(self, question: str, context: List[str]) -> str:
        """Create a structured prompt for the model"""
        context_text = "\n\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(context)])
        
        if "flan" in self.model_name.lower():
            # FLAN-T5 format (instruction following)
            prompt = f"""Answer the following question based on the given context from "Alice's Adventures in Wonderland":

Context:
{context_text}

Question: {question}

Answer:"""
        
        elif "distilgpt2" in self.model_name.lower() or "dialogpt" in self.model_name.lower():
            # GPT-style models
            prompt = f"""Based on Alice's Adventures in Wonderland:

{context_text}

Q: {question}
A:"""
        
        else:
            # Generic format
            prompt = f"""Context from Alice's Adventures in Wonderland:
{context_text}

Question: {question}
Answer:"""
        
        return prompt
    
    def generate_answer(self, question: str, context: List[str]) -> QAResponse:
        """Generate answer using HuggingFace model"""
        try:
            # Create prompt
            prompt = self._create_prompt(question, context)
            
            # Generate response
            print("🤖 Generating response...")
            
            # Configure generation parameters for smaller models
            generation_kwargs = {
                "max_new_tokens": min(self.max_tokens, 200),  # Limit for smaller models
                "temperature": self.temperature,
                "do_sample": True if self.temperature > 0 else False,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.1,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "return_full_text": False,
            }
            
            # Generate
            outputs = self.pipeline(prompt, **generation_kwargs)
            generated_text = outputs[0]["generated_text"].strip()
            
            # Clean up the response
            answer = self._clean_response(generated_text)
            
            # Calculate confidence
            confidence = self._calculate_confidence(answer, context)
            
            return QAResponse(
                answer=answer,
                context=context,
                confidence=confidence,
                sources=[f"Alice's Adventures in Wonderland - Context {i+1}" for i in range(len(context))]
            )
            
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            # Fallback to simple context-based answer
            return self._fallback_answer(question, context)
    
    def _clean_response(self, response: str) -> str:
        """Clean and format the model response"""
        # Remove prompt artifacts
        response = re.sub(r'Context from.*?Answer:', '', response, flags=re.DOTALL)
        response = re.sub(r'Q:.*?A:', '', response)
        response = re.sub(r'Question:.*?Answer:', '', response, flags=re.DOTALL)
        
        # Remove duplicate whitespace
        response = re.sub(r'\s+', ' ', response)
        
        # Remove common artifacts
        response = response.replace("Based on the context:", "")
        response = response.replace("According to the text:", "")
        
        return response.strip()
    
    def _calculate_confidence(self, answer: str, context: List[str]) -> float:
        """Calculate confidence score"""
        if not answer or not context:
            return 0.1
        
        # Simple keyword overlap-based confidence
        answer_words = set(answer.lower().split())
        context_words = set()
        for ctx in context:
            context_words.update(ctx.lower().split())
        
        if len(answer_words) == 0:
            return 0.1
        
        overlap = len(answer_words.intersection(context_words))
        confidence = min(0.85, 0.4 + (overlap / len(answer_words)) * 0.3)
        
        # Adjust based on answer quality
        if len(answer) > 50:
            confidence += 0.1
        if any(word in answer.lower() for word in ['alice', 'wonderland', 'rabbit', 'queen']):
            confidence += 0.05
        
        return max(0.1, min(0.85, confidence))
    
    def _fallback_answer(self, question: str, context: List[str]) -> QAResponse:
        """Fallback answer generation"""
        if not context:
            return QAResponse(
                answer="I couldn't find relevant information to answer your question.",
                context=[],
                confidence=0.0,
                sources=[]
            )
        
        # Extract relevant sentences
        sentences = []
        for ctx in context:
            sentences.extend([s.strip() for s in ctx.split('.') if len(s.strip()) > 10])
        
        # Simple relevance scoring
        question_words = set(question.lower().split())
        scored_sentences = []
        
        for sentence in sentences[:10]:  # Limit to first 10 sentences
            sentence_words = set(sentence.lower().split())
            score = len(question_words.intersection(sentence_words))
            if score > 0:
                scored_sentences.append((sentence, score))
        
        if scored_sentences:
            # Sort by relevance and take top sentence
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            answer = scored_sentences[0][0]
        else:
            answer = context[0][:200] + "..."
        
        return QAResponse(
            answer=answer,
            context=context,
            confidence=0.3,
            sources=[f"Alice's Adventures in Wonderland - Context {i+1}" for i in range(len(context))]
        )

class ModelFactory:
    """Factory class for creating lightweight HuggingFace models"""
    
    AVAILABLE_MODELS = {
        # Lightweight models (< 1GB)
        "distilgpt2": "distilgpt2",  # ~82MB
        "gpt2-small": "gpt2",  # ~500MB
        "dialogpt-small": "microsoft/DialoGPT-small",  # ~117MB
        "dialogpt-medium": "microsoft/DialoGPT-medium",  # ~345MB
        "flan-t5-small": "google/flan-t5-small",  # ~80MB
        "flan-t5-base": "google/flan-t5-base",  # ~250MB
        
        # Medium models (1-3GB) - only if you have space
        "flan-t5-large": "google/flan-t5-large",  # ~780MB
        "gpt2-medium": "gpt2-medium",  # ~1.5GB
        "dialogpt-large": "microsoft/DialoGPT-large",  # ~775MB
        
        # Large models (> 3GB) - use with caution!
        "phi3-mini": "microsoft/Phi-3-mini-4k-instruct",  # ~2.2GB
        "llama2-7b": "meta-llama/Llama-2-7b-chat-hf",  # ~13GB ⚠️
    }
    
    MODEL_SIZES = {
        "distilgpt2": "82MB",
        "gpt2-small": "500MB", 
        "dialogpt-small": "117MB",
        "dialogpt-medium": "345MB",
        "flan-t5-small": "80MB",
        "flan-t5-base": "250MB",
        "flan-t5-large": "780MB",
        "gpt2-medium": "1.5GB",
        "dialogpt-large": "775MB",
        "phi3-mini": "2.2GB",
        "llama2-7b": "13GB ⚠️",
    }
    
    @classmethod
    def create_model(cls, model_type: str, **kwargs) -> HuggingFaceLLM:
        """Create a model instance based on type"""
        if model_type not in cls.AVAILABLE_MODELS:
            raise ValueError(f"Model type '{model_type}' not supported. Available: {list(cls.AVAILABLE_MODELS.keys())}")
        
        model_name = cls.AVAILABLE_MODELS[model_type]
        return HuggingFaceLLM(model_name=model_name, **kwargs)
    
    @classmethod
    def list_models(cls):
        """List all available models with sizes"""
        print("Available models (with approximate sizes):")
        print("\n🟢 Lightweight models (recommended):")
        for key in ["distilgpt2", "gpt2-small", "dialogpt-small", "dialogpt-medium", "flan-t5-small", "flan-t5-base"]:
            if key in cls.AVAILABLE_MODELS:
                size = cls.MODEL_SIZES.get(key, "Unknown")
                print(f"  {key}: {cls.AVAILABLE_MODELS[key]} ({size})")
        
        print("\n🟡 Medium models:")
        for key in ["flan-t5-large", "gpt2-medium", "dialogpt-large", "phi3-mini"]:
            if key in cls.AVAILABLE_MODELS:
                size = cls.MODEL_SIZES.get(key, "Unknown")
                print(f"  {key}: {cls.AVAILABLE_MODELS[key]} ({size})")
        
        print("\n🔴 Large models (use with caution):")
        for key in ["llama2-7b"]:
            if key in cls.AVAILABLE_MODELS:
                size = cls.MODEL_SIZES.get(key, "Unknown")
                print(f"  {key}: {cls.AVAILABLE_MODELS[key]} ({size})")