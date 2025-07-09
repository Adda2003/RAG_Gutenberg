"""
Flask web application for Alice in Wonderland QA System
"""
from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from src.question_answering import QASystem
import logging

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'alice-in-wonderland-qa-system'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global QA system instance
qa_system = None

def initialize_qa_system(model_type="openai", model_name=None, model_config=None):
    """Initialize the QA system with specified model configuration"""
    global qa_system
    try:
        logger.info("Initializing QA system...")
        qa_system = QASystem(
            model_type=model_type,
            model_name=model_name,
            model_config=model_config or {}
        )
        qa_system.load_knowledge_base()
        logger.info("QA system initialized successfully!")
        return True
    except Exception as e:
        logger.error(f"Error initializing QA system: {e}")
        return False

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    """Handle question asking"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({
                'success': False,
                'error': 'Please enter a question'
            }), 400
        
        if qa_system is None:
            return jsonify({
                'success': False,
                'error': 'QA system not initialized. Please check server logs.'
            }), 500
        
        # Get answer from QA system
        response = qa_system.ask_question(question)
        
        return jsonify({
            'success': True,
            'question': question,
            'answer': response.answer,
            'confidence': round(response.confidence, 2),
            'num_sources': len(response.context),
            'sources': [context[:200] + "..." if len(context) > 200 else context 
                       for context in response.context[:3]]  # Show first 3 sources, truncated
        })
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return jsonify({
            'success': False,
            'error': f'Error processing your question: {str(e)}'
        }), 500

@app.route('/samples')
def get_samples():
    """Get sample questions"""
    try:
        if qa_system is None:
            return jsonify({
                'success': False,
                'error': 'QA system not initialized'
            }), 500
        
        samples = qa_system.get_sample_questions()
        return jsonify({
            'success': True,
            'samples': samples
        })
        
    except Exception as e:
        logger.error(f"Error getting samples: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    status = 'healthy' if qa_system is not None else 'unhealthy'
    return jsonify({
        'status': status,
        'qa_system_loaded': qa_system is not None
    })

if __name__ == '__main__':
    # Initialize QA system on startup
    if initialize_qa_system():
        logger.info("Starting Flask server...")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        logger.error("Failed to initialize QA system. Please run 'python main.py --build' first.")