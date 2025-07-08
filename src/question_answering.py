"""
Main Question Answering interface
"""
import yaml
from typing import Dict, Any, List
from src.data_loader import GutenbergLoader, TextProcessor
from src.embeddings import HuggingFaceEmbeddings
from src.vector_store import FAISSVectorStore
from src.llm_pipeline import OpenAILLM, QAResponse

import os

class QASystem:
    """Main Question Answering System"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
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
        # Setup embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config['embeddings']['model_name'],
            device=self.config['embeddings']['device']
        )
        
        # Setup vector store
        self.vector_store = FAISSVectorStore(self.embeddings)
        
        # Setup LLM
        self.llm = OpenAILLM(
            model_name=self.config['llm']['model_name'],
            temperature=self.config['llm']['temperature'],
            max_tokens=self.config['llm']['max_tokens']
        )
    
    def build_knowledge_base(self):
        """Build the knowledge base from Gutenberg text"""
        print("Loading text from Project Gutenberg...")
        
        # Load document
        loader = GutenbergLoader(self.config['data']['gutenberg_url'])
        documents = loader.load()
        
        print(f"Loaded {len(documents)} documents")
        
        # Process and chunk text
        processor = TextProcessor(
            chunk_size=self.config['data']['chunk_size'],
            chunk_overlap=self.config['data']['chunk_overlap']
        )
        chunks = processor.split_documents(documents)
        
        print(f"Created {len(chunks)} text chunks")
        
        # Add to vector store
        print("Building vector store...")
        self.vector_store.add_documents(chunks)
        
        # Save vector store
        self.vector_store.save(self.config['vector_store']['index_path'])
        print("Knowledge base built and saved successfully!")
    
    def load_knowledge_base(self):
        """Load existing knowledge base"""
        if os.path.exists(self.config['vector_store']['index_path']):
            self.vector_store.load(self.config['vector_store']['index_path'])
            print("Knowledge base loaded successfully!")
        else:
            print("No existing knowledge base found. Please build it first.")
    
    def ask_question(self, question: str) -> QAResponse:
        """Ask a question and get an answer"""
        if self.vector_store.index is None:
            raise ValueError("Knowledge base not loaded. Please build or load it first.")
        
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
            return QAResponse(
                answer="I couldn't find relevant information to answer your question.",
                context=[],
                confidence=0.0,
                sources=[]
            )
        
        # Extract context
        context = [doc.page_content for doc, _ in results]
        
        # Generate answer
        response = self.llm.generate_answer(question, context)
        
        return response
    
    def get_sample_questions(self) -> List[str]:
        """Get sample questions for different categories"""
        return [
            # Character questions
            "Who is Alice and what are her main characteristics?",
            "Describe the Cheshire Cat and its role in the story.",
            "What is the Queen of Hearts like?",
            "Who is the Mad Hatter and what makes him mad?",
            
            # Plot questions
            "How does Alice fall down the rabbit hole?",
            "What happens at the Mad Tea Party?",
            "Describe the Queen's croquet game.",
            "How does the story end?",
            "What is the trial scene about?",
            
            # Thematic questions
            "What are the main themes in Alice's Adventures in Wonderland?",
            "How does the story portray the concept of growing up?",
            "What does Wonderland represent symbolically?",
            "How does Carroll use nonsense and wordplay in the story?",
            "What social commentary can be found in the story?"
        ]