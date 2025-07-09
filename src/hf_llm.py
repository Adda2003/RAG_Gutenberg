"""HuggingFace LLM integration for Phi-2 and Llama models"""
from typing import List
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from src.llm_pipeline import QAResponse


@dataclass
class HFConfig:
    model_name: str = "microsoft/phi-2"
    device: str = "cpu"
    max_new_tokens: int = 512
    temperature: float = 0.2


class HFLLM:
    """Simple wrapper around HuggingFace text-generation pipeline"""

    def __init__(self, config: HFConfig = HFConfig()):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # Fix padding token issue
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name)
        device_index = 0 if config.device == "cuda" else -1
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device_index,
        )

    def generate_answer(self, question: str, context: List[str]) -> QAResponse:
        """Generate answer using the loaded model"""
        try:
            # Create prompt
            context_text = "\n\n".join(context[:3])  # Limit context
            prompt = f"""Based on the following context from Alice's Adventures in Wonderland, answer the question.

Context: {context_text}

Question: {question}
Answer:"""

            # Use the pipeline instead of manual generation
            response = self.generator(
                prompt,
                max_new_tokens=150,
                temperature=self.config.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                return_full_text=False,  # Only return the generated part
                truncation=True
            )
            
            # Extract the generated text
            generated_text = response[0]['generated_text'].strip()
            
            # Clean up the response - take only the first line/sentence
            answer = generated_text.split('\n')[0].strip()
            if not answer:
                answer = "I couldn't generate a proper answer based on the context."
            
            return QAResponse(
                answer=answer,
                context=context,
                confidence=0.7,  # Default confidence
                sources=[]
            )
            
        except Exception as e:
            print(f"Generation error: {e}")
            return QAResponse(
                answer="I encountered an error while generating the answer.",
                context=context,
                confidence=0.0,
                sources=[]
            )