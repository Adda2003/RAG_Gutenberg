# Alice in Wonderland Question Answering System with RAG

A Retrieval Augmented Generation (RAG) system that answers questions about "Alice's Adventures in Wonderland" using Project Gutenberg text data, with a Flask web interface and intelligent fallback processing.

## Features

- **Text Loading**: Uses LangChain's GutenbergLoader to load text from Project Gutenberg
- **Vector Database**: FAISS-based vector store with HuggingFace embeddings
- **Dual-Mode Operation**: 
  - **Primary**: OpenAI GPT-4o-mini for AI-powered responses
  - **Fallback**: Intelligent NLP-based context processing when OpenAI is unavailable
- **Question Types**: Supports character, plot, thematic, and factual questions
- **Web Interface**: Clean HTML interface with Flask backend
- **Evaluation Framework**: Comprehensive evaluation with detailed metrics
- **Robust Design**: Graceful degradation when external services are unavailable

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
python run_server.py
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

## Example Questions

### Character Questions
- Who is Alice and what are her main characteristics?
- Describe the Cheshire Cat and its role in the story.
- What is the Queen of Hearts like?
- Who is the Mad Hatter and what makes him mad?

### Plot Questions
- How does Alice fall down the rabbit hole?
- What happens at the Mad Tea Party?
- Describe the Queen's croquet game.
- How does the story end?

### Thematic Questions
- What are the main themes in Alice's Adventures in Wonderland?
- How does the story portray the concept of growing up?
- What does Wonderland represent symbolically?

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
4. **Data Loader** (`src/data_loader.py`): Text processing and chunking
5. **Vector Store** (`src/vector_store.py`): FAISS similarity search
6. **LLM Pipeline** (`src/llm_pipeline.py`): OpenAI integration with NLP fallback
7. **Evaluation** (`src/evaluation.py`): Performance analysis and metrics

### Process Flow
1. User enters question in web interface
2. Frontend sends AJAX request to Flask backend
3. Flask calls QA system with the question
4. QA system retrieves relevant text chunks using FAISS
5. Context sent to OpenAI for answer generation
6. **If OpenAI fails**: Intelligent NLP-based fallback processing
7. Structured response returned to user

### Fallback Processing
When OpenAI API is unavailable, the system uses intelligent context processing:
- **Keyword Extraction**: Identifies important terms from questions
- **Sentence Relevance**: Scores sentences based on keyword matches
- **Question Type Detection**: Handles "who", "what", "how", "where" questions
- **Answer Generation**: Creates responses from relevant context

## Configuration

Edit `config/config.yaml` to customize:
```yaml
embeddings:
  model_name: "all-MiniLM-L6-v2"  # HuggingFace model
  device: "cpu"
llm:
  model_name: "gpt-4o-mini"       # OpenAI model
  temperature: 0.2
  max_tokens: 800
retrieval:
  top_k: 7                        # Number of context passages
  score_threshold: 0.3            # Similarity threshold (lowered for better recall)
data:
  chunk_size: 400                 # Smaller chunks for better granularity
  chunk_overlap: 50               # Overlap between chunks
```

### Model Configuration
The system uses **GPT-4o-mini** by default, which is:
- More cost-effective than GPT-4
- Faster response times
- Still highly capable for literature analysis

### Fallback Mode
If OpenAI is unavailable, the system automatically switches to:
- **NLP-based processing**: Uses keyword extraction and pattern matching
- **Context analysis**: Finds relevant sentences from retrieved chunks
- **Question type detection**: Adapts answers based on question patterns
- **Confidence scoring**: Provides realistic confidence estimates

## Troubleshooting

### Common Issues

1. **"Knowledge base not found"**:
   ```bash
   python main.py --build
   ```

2. **"OPENAI_API_KEY not set"**:
   - Create `.env` file with your API key
   - Make sure the file is in the project root

3. **"QA system not initialized"**:
   - Check that knowledge base was built successfully
   - Verify OpenAI API key is valid

4. **OpenAI Connection Issues**:
   - System automatically falls back to NLP processing
   - Check network connectivity or firewall settings
   - Verify API key is correct and has credits

5. **Port already in use**:
   - Change port in `run_server.py` (default: 5000)
   - Or kill existing process: `lsof -ti:5000 | xargs kill`

6. **HuggingFace Tokenizers Warning**:
   - Set environment variable: `export TOKENIZERS_PARALLELISM=false`

### Performance Optimization

- **Chunk Size**: Smaller chunks (400) provide better granularity
- **Score Threshold**: Lower threshold (0.3) improves recall
- **Top K**: More context (7) provides better answers
- **Fallback Mode**: Enables offline operation when OpenAI fails

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

### Dual-Mode Operation
- **OpenAI Mode**: Full AI-powered responses with JSON structured output
- **Fallback Mode**: NLP-based processing using keyword extraction and pattern matching
- **Seamless Transition**: Automatic fallback with user notification

## Development Mode
To run in development mode with auto-reload:
```bash
export FLASK_ENV=development
python app.py
```

## Evaluation
Run comprehensive evaluation:
```bash
python main.py --evaluate
```

This provides detailed analysis including:
- Individual question performance
- Category-based statistics
- Retrieval effectiveness
- Answer quality metrics
- Response time analysis

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

## License

This project is for educational purposes. The Alice in Wonderland text is in the public domain.