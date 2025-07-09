# Alice in Wonderland Question Answering System with RAG

A Retrieval Augmented Generation (RAG) system that answers questions about "Alice's Adventures in Wonderland" using Project Gutenberg text data, with a Flask web interface and intelligent fallback processing.

### Approach

This system implements a **Retrieval-Augmented Generation (RAG)** architecture to answer questions about Lewis Carroll's classic novel. Here's how it works:

#### 1. Document Processing & Chunking
- The complete text of "Alice's Adventures in Wonderland" is downloaded from Project Gutenberg
- Text is split into manageable chunks (400 characters with 50-character overlap) to preserve context
- Each chunk maintains semantic coherence while being small enough for efficient processing

#### 2. Vector Embeddings & Storage
- Text chunks are converted into high-dimensional vector representations using HuggingFace's `all-MiniLM-L6-v2` model
- These embeddings capture semantic meaning, allowing similar concepts to be grouped together
- Vectors are stored in a **FAISS (Facebook AI Similarity Search)** database for fast retrieval

#### 3. Question Processing & Retrieval
- When a user asks a question, it's converted into the same vector space as the document chunks
- The system performs **semantic similarity search** to find the most relevant text passages
- Top-k most similar chunks (default: 7) are retrieved based on cosine similarity scores

#### 4. Context-Aware Answer Generation
- Retrieved text chunks serve as **context** for the language model
- The system constructs a prompt combining the user's question with relevant context
- Multiple LLM options generate coherent answers:
  - **OpenAI GPT-4o-mini**: Cloud-based, highest quality
  - **Phi-2**: Local 2.2GB model, good balance of speed/quality
  - **Llama 2**: Local 13GB model, high quality offline inference

#### 5. Response Enhancement
- Generated answers include confidence scores based on retrieval quality
- Source passages are provided for transparency and verification
- The system handles edge cases with fallback responses when context is insufficient

#### Key Advantages of RAG Architecture:
- **Grounded Responses**: Answers are based on actual text content, reducing hallucinations
- **Scalable Knowledge**: Can work with any text corpus without model retraining
- **Transparency**: Users can see source passages used for each answer
- **Flexibility**: Multiple LLM backends support different use cases and constraints

## Features

- **Multiple LLM Support**: OpenAI GPT, HuggingFace Phi-2, and Llama 2 models
- **RAG Architecture**: Combines vector search with language generation
- **Interactive Mode**: Chat-like interface for asking questions
- **Evaluation Framework**: Built-in metrics for answer quality assessment
- **Web Interface**: Flask-based web UI for easy interaction
- **Local Model Support**: Run completely offline with HuggingFace models

## Setup

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Environment Variables**:
Create a `.env` file:
```
OPENAI_API_KEY=your_openai_api_key_here
```

3. **Build Knowledge Base**:
```bash
python main.py --build
```

## Usage

### Web Interface (Primary Method)
1. **Start the Flask server**:
```bash
# Using OpenAI
python run_server.py

# Using Phi-2 model
python run_server.py --phi

# Using Llama 2 model
python run_server.py --llama
```

2. **Open your browser** and go to:
```
http://localhost:5000
```

3. **Ask questions** in the text box!

### Command Line Interface (Alternative)
```bash
# Interactive mode
python main.py --interactive

# Ask single question
python main.py --question "Who is Alice?"

# Run evaluation
python main.py --evaluate
```

## Web Interface Features

- **Simple Text Input**: Clean, user-friendly interface with single text box
- **Sample Questions**: Click to load example questions
- **Real-time Answers**: Get answers with confidence scores
- **Source Context**: View the text passages used to generate answers
- **Responsive Design**: Works on desktop and mobile devices
- **Error Handling**: Clear error messages for troubleshooting
- **Loading States**: Visual feedback during processing

## API Endpoints

- `GET /` - Main web interface
- `POST /ask` - Ask a question (JSON)
- `GET /samples` - Get sample questions
- `GET /health` - Health check

### API Usage Example
```bash
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Who is Alice?"}'
```

## Architecture

### Components
1. **Flask App** (`app.py`): Web server and API endpoints
2. **HTML Interface** (`templates/index.html`): User interface
3. **QA System** (`src/question_answering.py`): Main orchestration
4. **Document Processing** (`src/document_processing.py`): Text processing and chunking
5. **Vector Store** (`src/vector_store.py`): FAISS similarity search
6. **LLM Pipeline** (`src/llm_pipeline.py`): OpenAI integration
7. **HuggingFace LLM** (`src/hf_llm.py`): Local model integration (Phi-2, Llama)
8. **Evaluation** (`src/evaluation.py`): Performance analysis and metrics
9. **Server Runner** (`run_server.py`): Production server with model selection

### Process Flow
1. User enters question in web interface
2. Frontend sends AJAX request to Flask backend
3. Flask calls QA system with the question
4. QA system retrieves relevant text chunks using FAISS
5. Context sent to selected LLM (OpenAI/Phi-2/Llama) for answer generation
6. **If using local models**: HuggingFace pipeline processes context locally
7. **Model selection**: Automatic routing based on command-line flags
8. Structured response returned to user with confidence score



## Configuration

### Model Selection

The system supports multiple language models with different capabilities:

```bash
# OpenAI Models (default, requires API key)
python main.py --question "Who is Alice?"
python run_server.py

# Local HuggingFace Models (no API key required)
python main.py --phi --question "Who is Alice?"        # Phi-2 (2.2GB)
python main.py --llama --question "Who is Alice?"      # Llama 2 (13GB)
python run_server.py --phi                             # Web server with Phi-2
```

### System Configuration

Edit `config/config.yaml` to customize retrieval and processing settings:
```yaml
embeddings:
  model_name: "all-MiniLM-L6-v2"  # HuggingFace embedding model
  device: "cpu"                   # Use "cuda" for GPU acceleration

llm:
  # OpenAI configuration (when using --openai or default)
  model_name: "gpt-4o-mini"       # OpenAI model
  temperature: 0.2                # Response creativity (0.0-1.0)
  max_tokens: 800                 # Maximum response length

retrieval:
  top_k: 7                        # Number of context passages to retrieve
  score_threshold: 0.3            # Minimum similarity score for relevance
  
data:
  chunk_size: 400                 # Text chunk size for vector storage
  chunk_overlap: 50               # Overlap between adjacent chunks
```

### Model Comparison

| Model Type | Command Flag | Size | Speed | Quality | API Required | Cost |
|------------|--------------|------|-------|---------|--------------|------|
| **OpenAI GPT-4o-mini** | `--openai` (default) | N/A | Fast | Excellent | Yes | Low |
| **Phi-2** | `--phi` | 2.2GB | Medium | Good | No | Free |
| **Llama 2** | `--llama` | 13GB | Slow | Very Good | No | Free |

### Model Configuration Details

#### OpenAI Models (Default)
- **Advantages**: Best quality, fastest inference, no local storage
- **Requirements**: API key, internet connection
- **Best for**: Production use, highest quality answers
- **Configuration**: Uses `gpt-4o-mini` for cost-effectiveness

#### Phi-2 Local Model
- **Advantages**: Completely offline, moderate size, good performance
- **Requirements**: 2.2GB storage, 4GB+ RAM recommended
- **Best for**: Privacy-sensitive applications, moderate quality needs
- **First run**: Downloads model automatically

#### Llama 2 Local Model
- **Advantages**: High quality local inference, completely offline
- **Requirements**: 13GB storage, 8GB+ RAM recommended  
- **Best for**: High-quality offline inference
- **Note**: Large download, slower inference

### Runtime Configuration

You can override default settings via command line:

```bash
# Custom temperature and token limits
python main.py --phi --question "Who is Alice?" --temperature 0.5 --max-tokens 500

# Custom server port
python run_server.py --phi --port 8080

# Custom cache directory for local models
python run_server.py --llama --cache-dir /path/to/custom/cache
```

### Local Model Management

#### Cache Location
Local models are stored in:
```bash
~/.cache/huggingface/hub/
```

#### Cache Management
```bash
# Check cache size
du -sh ~/.cache/huggingface/hub/

# Clear all cached models
rm -rf ~/.cache/huggingface/hub/

# Clear specific model
rm -rf ~/.cache/huggingface/hub/models--microsoft--phi-2
```

#### Download Behavior
- **First run**: Models download automatically (one-time)
- **Subsequent runs**: Use cached models (fast startup)
- **Network**: Only required for initial download

### Performance Tuning

#### For Better Quality
```yaml
retrieval:
  top_k: 10                    # More context
  score_threshold: 0.2         # Lower threshold for more passages

llm:
  temperature: 0.1             # More focused responses
  max_tokens: 1000            # Longer responses
```

#### For Better Speed
```yaml
retrieval:
  top_k: 5                     # Less context
  score_threshold: 0.4         # Higher threshold for fewer passages

llm:
  temperature: 0.3             # Faster generation
  max_tokens: 500             # Shorter responses
```

#### For Resource-Constrained Systems
```bash
# Use smallest model
python main.py --phi --question "Who is Alice?"

# Reduce context
# Edit config.yaml: top_k: 3, max_tokens: 300
```

### Hybrid Configuration

You can run different models for different use cases:

```bash
# Development/testing with fast local model
python main.py --phi --interactive

# Production web server with OpenAI
python run_server.py --openai

# High-quality analysis with Llama
python main.py --llama --evaluate
```

### Environment Variables

```bash
# Required for OpenAI
OPENAI_API_KEY=your_api_key_here

# Optional: Custom cache directory
HF_HOME=/path/to/custom/huggingface/cache

# Optional: Disable tokenizer warnings
TOKENIZERS_PARALLELISM=false
```

### Troubleshooting Configuration

#### Model Loading Issues
```bash
# Check available models
python -c "from src.huggingface_llm import ModelFactory; ModelFactory.list_models()"

# Test specific model
python main.py --phi --question "test" --verbose
```

#### Memory Issues
```bash
# Use smaller model
python main.py --phi --question "Who is Alice?"

# Reduce chunk size in config.yaml
data:
  chunk_size: 200
  chunk_overlap: 25
```

#### API Issues
```bash
# Verify OpenAI key
python -c "import openai; openai.api_key='your_key'; print(openai.Model.list())"

# Use local model as backup
python main.py --phi --question "Who is Alice?"
```

## System Performance

### Evaluation Metrics
The system includes comprehensive evaluation with:
- **Retrieval Score**: Quality of context retrieval (avg: 0.85)
- **Answer Quality**: Relevance and accuracy of responses
- **Confidence Score**: System confidence in answers (avg: 0.85)
- **Response Time**: Speed of answer generation (avg: 11.5s)

### Question Categories
- **Character Questions**: 87.5% retrieval accuracy
- **Plot Questions**: 88% retrieval accuracy  
- **Factual Questions**: 90% retrieval accuracy
- **Thematic Questions**: 71% confidence (more challenging)

### Multi-Model Operation
- **OpenAI Mode**: Full AI-powered responses with GPT-4o-mini
- **Phi-2 Mode**: Local inference with Microsoft's Phi-2 model (2.2GB)
- **Llama 2 Mode**: High-quality local inference with Llama 2 (13GB)
- **Automatic Selection**: Model chosen via command-line flags
- **Seamless Switching**: Easy model swapping for different use cases


## Evaluation

Run comprehensive evaluation:
```bash
python main.py --evaluate

```

## Requirements

- Python 3.8+
- OpenAI API key
- Internet connection for initial text download
- ~500MB disk space for vector storage

## File Structure
```
gutenberg-qa-rag/
├── app.py                      # Flask web application
├── run_server.py              # Production server runner
├── main.py                    # CLI interface
├── templates/
│   └── index.html             # Web interface
├── src/                       # Core modules
│   ├── data_loader.py         # Text loading and processing
│   ├── embeddings.py          # HuggingFace embeddings
│   ├── vector_store.py        # FAISS vector database
│   ├── llm_pipeline.py        # OpenAI integration
│   ├── question_answering.py  # Main QA system
│   └── evaluation.py          # Evaluation framework
├── config/
│   └── config.yaml           # Configuration settings
├── data/
│   └── vector_store/         # Vector database storage
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables
└── README.md                # Documentation
```

## Quick Start

1. **Clone and setup**:
```bash
git clone https://github.com/Adda2003/gutenberg-qa-rag.git
cd gutenberg-qa-rag
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure environment**:
```bash
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

4. **Build knowledge base**:
```bash
python main.py --build
```

5. **Start web server**:
```bash
python run_server.py
```

6. **Open browser**: http://localhost:5000

