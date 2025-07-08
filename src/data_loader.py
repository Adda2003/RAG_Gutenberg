"""
Data loading module using LangChain GutenbergLoader
"""
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
import re

class GutenbergLoader:
    """Custom Gutenberg loader for Project Gutenberg texts"""
    
    def __init__(self, url: str):
        self.url = url
    
    def load(self) -> List[Document]:
        """Load and clean the Gutenberg text"""
        try:
            response = requests.get(self.url)
            response.raise_for_status()
            text = response.text
            
            # Clean the text
            cleaned_text = self._clean_gutenberg_text(text)
            
            # Create document
            doc = Document(
                page_content=cleaned_text,
                metadata={"source": self.url, "title": "Alice's Adventures in Wonderland"}
            )
            
            return [doc]
            
        except Exception as e:
            raise Exception(f"Error loading Gutenberg text: {e}")
    
    def _clean_gutenberg_text(self, text: str) -> str:
        """Clean Project Gutenberg specific formatting"""
        # Remove Project Gutenberg header and footer
        lines = text.split('\n')
        start_idx = 0
        end_idx = len(lines)
        
        # Find start of actual content
        for i, line in enumerate(lines):
            if "*** START OF" in line or "CHAPTER I" in line:
                start_idx = i + 1
                break
        
        # Find end of actual content
        for i in range(len(lines) - 1, -1, -1):
            if "*** END OF" in lines[i] or "THE END" in lines[i]:
                end_idx = i
                break
        
        # Join the content
        content = '\n'.join(lines[start_idx:end_idx])
        
        # Clean up extra whitespace and formatting
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)  # Multiple newlines
        content = re.sub(r'[ \t]+', ' ', content)  # Multiple spaces/tabs
        
        return content.strip()

class TextProcessor:
    """Process and chunk text for vector storage"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        return self.text_splitter.split_documents(documents)