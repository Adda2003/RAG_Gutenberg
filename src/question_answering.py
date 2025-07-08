"""
Main Question Answering interface with support for multiple LLM backends
"""
import yaml
from typing import Dict, Any, List, Optional
from src.data_loader import GutenbergLoader, TextProcessor
from src.embeddings import HuggingFaceEmbeddings
from src.vector_store import FAISSVectorStore
from src.llm_pipeline import OpenAILLM, QAResponse
from src.huggingface_llm import HuggingFaceLLM, ModelFactory

import os

class QASystem:
    """Main Question Answering System with multiple LLM backend support"""
    
    def __init__(self, config_path: str = "config/config.yaml", 
                 model_type: str = "openai", 
                 model_name: Optional[str] = None,
                 model_config: Optional[Dict[str, Any]] = None):
        self.config = self._load_config(config_path)
        self.model_type = model_type
        self.model_name = model_name
        self.model_config = model_config or {}
        
        self.embeddings = None
        self.vector_store = None
        self.llm = None
        self._setup_components()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_components(self):
        """Initialize all components"""
        print("🔧 Setting up QA system components...")
        
        # Setup embeddings
        print("📊 Loading embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config['embeddings']['model_name'],
            device=self.config['embeddings']['device']
        )
        
        # Setup vector store
        print("🗂️ Setting up vector store...")
        self.vector_store = FAISSVectorStore(self.embeddings)
        
        # Setup LLM based on type
        self._setup_llm()
    
    def _setup_llm(self):
        """Setup the LLM based on specified type"""
        print(f"🤖 Setting up {self.model_type} LLM...")
        
        if self.model_type == "openai":
            # Use OpenAI LLM
            openai_config = self.config['llm'].copy()
            openai_config.update(self.model_config)
            
            self.llm = OpenAILLM(
                model_name=openai_config.get('model_name', 'gpt-4o-mini'),
                temperature=openai_config.get('temperature', 0.2),
                max_tokens=openai_config.get('max_tokens', 800)
            )
            
        elif self.model_type == "huggingface":
            # Use HuggingFace LLM
            if self.model_name:
                # Use specific model via factory
                hf_config = {
                    'temperature': self.model_config.get('temperature', 0.2),
                    'max_tokens': self.model_config.get('max_tokens', 800),
                    'load_in_4bit': self.model_config.get('load_in_4bit', True)
                }
                
                self.llm = ModelFactory.create_model(self.model_name, **hf_config)
            else:
                # Use default Phi-3
                hf_config = {
                    'temperature': self.model_config.get('temperature', 0.2),
                    'max_tokens': self.model_config.get('max_tokens', 800),
                    'load_in_4bit': self.model_config.get('load_in_4bit', True)
                }
                
                self.llm = HuggingFaceLLM(**hf_config)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def build_knowledge_base(self):
        """Build the knowledge base from Gutenberg text"""
        print("📚 Loading text from Project Gutenberg...")
        
        # Load document
        loader = GutenbergLoader(self.config['data']['gutenberg_url'])
        documents = loader.load()
        
        print(f"✅ Loaded {len(documents)} documents")
        
        # Process and chunk text
        processor = TextProcessor(
            chunk_size=self.config['data']['chunk_size'],
            chunk_overlap=self.config['data']['chunk_overlap']
        )
        chunks = processor.split_documents(documents)
        
        print(f"✅ Created {len(chunks)} text chunks")
        
        # Add to vector store
        print("🔍 Building vector store...")
        self.vector_store.add_documents(chunks)
        
        # Save vector store
        self.vector_store.save(self.config['vector_store']['index_path'])
        print("✅ Knowledge base built and saved successfully!")
    
    def load_knowledge_base(self):
        """Load existing knowledge base"""
        index_path = self.config['vector_store']['index_path']
        if os.path.exists(index_path):
            print("📁 Loading existing knowledge base...")
            self.vector_store.load(index_path)
            print("✅ Knowledge base loaded successfully!")
        else:
            print("❌ No existing knowledge base found. Please build it first with --build")
            raise FileNotFoundError(f"Knowledge base not found at {index_path}")
    
    def ask_question(self, question: str) -> QAResponse:
        """Ask a question and get an answer"""
        if self.vector_store.index is None:
            raise ValueError("Knowledge base not loaded. Please build or load it first.")
        
        print(f"🔍 Searching for relevant context...")
        
        # Retrieve relevant context
        results = self.vector_store.similarity_search(
            question, 
            k=self.config['retrieval']['top_k']
        )
        
        # Filter by score threshold if configured
        if 'score_threshold' in self.config['retrieval']:
            threshold = self.config['retrieval']['score_threshold']
            results = [(doc, score) for doc, score in results if score >= threshold]
        
        if not results:
            print("⚠️ No relevant context found")
            return QAResponse(
                answer="I couldn't find relevant information to answer your question in Alice's Adventures in Wonderland.",
                context=[],
                confidence=0.0,
                sources=[]
            )
        
        print(f"✅ Found {len(results)} relevant text passages")
        
        # Extract context
        context = [doc.page_content for doc, _ in results]
        
        # Generate answer using the configured LLM
        print(f"💭 Generating answer using {self.model_type} model...")
        response = self.llm.generate_answer(question, context)
        
        print(f"✅ Answer generated with confidence: {response.confidence:.2f}")
        return response
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration"""
        return {
            "model_type": self.model_type,
            "model_name": self.model_name,
            "model_config": self.model_config,
            "llm_class": type(self.llm).__name__
        }