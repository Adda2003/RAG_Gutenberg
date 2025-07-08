"""
Production server runner for Alice in Wonderland QA System
"""
import os
import sys
from dotenv import load_dotenv
from app import app, initialize_qa_system
import logging

# Load environment variables
load_dotenv()

def check_prerequisites():
    """Check if all prerequisites are met"""
    # Check if OpenAI API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please create a .env file with your OpenAI API key:")
        print("OPENAI_API_KEY=your_api_key_here")
        return False
    
    # Check if knowledge base exists
    knowledge_base_path = "data/vector_store"
    if not os.path.exists(knowledge_base_path):
        print("❌ Error: Knowledge base not found")
        print("Please build the knowledge base first by running:")
        print("python main.py --build")
        return False
    
    return True

def main():
    """Main function to run the server"""
    print("🐰 Alice in Wonderland QA System")
    print("=" * 40)
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Initialize QA system
    print("🔄 Initializing QA system...")
    if not initialize_qa_system():
        print("❌ Failed to initialize QA system")
        sys.exit(1)
    
    print("✅ QA system initialized successfully!")
    print("\n🌐 Starting Flask server...")
    print("📍 Server will be available at: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 40)
    
    # Run the Flask app
    try:
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,  # Set to False for production
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")

if __name__ == '__main__':
    main()