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
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name)
        device_index = 0 if config.device == "cuda" else -1
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device_index,
        )

    def generate_answer(self, question: str, context: List[str]) -> QAResponse:
        context_text = "\n\n".join(context)
        prompt = f"{context_text}\n\nQuestion: {question}\nAnswer:"
        outputs = self.generator(
            prompt,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            num_return_sequences=1,
        )
        generated = outputs[0]["generated_text"]
        answer = generated.split("Answer:", 1)[-1].strip()
        return QAResponse(
            answer=answer,
            context=context,
            confidence=0.6,
            sources=[f"Context {i+1}" for i in range(len(context))],
        )

