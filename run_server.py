"""
Production server runner for Alice in Wonderland QA System
"""
import os
import sys
import argparse
from dotenv import load_dotenv
from app import app, initialize_qa_system
import logging

# Load environment variables
load_dotenv()

def check_prerequisites(model_type="openai"):
    """Check if all prerequisites are met"""
    # Check model-specific requirements
    if model_type == "openai":
        if not os.getenv('OPENAI_API_KEY'):
            print("❌ Error: OPENAI_API_KEY environment variable not set")
            print("Please create a .env file with your OpenAI API key:")
            print("OPENAI_API_KEY=your_api_key_here")
            return False
    elif model_type == "huggingface":
        print("✅ Using HuggingFace model - no API key required")
    
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
    parser = argparse.ArgumentParser(description="Run Alice in Wonderland QA System Server")
    
    # Model selection arguments
    parser.add_argument("--openai", action="store_true", help="Use OpenAI model (default)")
    parser.add_argument("--small", action="store_true", help="Use small lightweight model (distilgpt2)")
    parser.add_argument("--medium", action="store_true", help="Use medium model (dialogpt-medium)")
    parser.add_argument("--phi", action="store_true", help="Use Phi-3 model (2.2GB)")
    parser.add_argument("--llama", action="store_true", help="Use Llama 2 model (13GB - WARNING!)")
    parser.add_argument("--model", type=str, 
                       choices=['distilgpt2', 'gpt2-small', 'dialogpt-small', 'dialogpt-medium', 
                               'flan-t5-small', 'flan-t5-base', 'flan-t5-large', 'gpt2-medium', 
                               'dialogpt-large', 'phi3-mini', 'llama2-7b'], 
                       help="Specify exact model to use")
    
    # Model configuration
    parser.add_argument("--temperature", type=float, default=0.2, help="Model temperature (default: 0.2)")
    parser.add_argument("--max-tokens", type=int, default=800, help="Max tokens for generation (default: 800)")
    parser.add_argument("--cache-dir", type=str, help="Custom cache directory for models")
    parser.add_argument("--port", type=int, default=5000, help="Port to run server on (default: 5000)")
    
    args = parser.parse_args()
    
    # Set custom cache directory if provided
    if args.cache_dir:
        os.environ['HF_HOME'] = args.cache_dir
        print(f"📁 Using custom cache directory: {args.cache_dir}")
    
    # Determine model type and name
    model_type = "openai"  # default
    model_name = None
    
    if args.small:
        model_type = "huggingface"
        model_name = "distilgpt2"
        print("📦 Using lightweight model (82MB)")
    elif args.medium:
        model_type = "huggingface"
        model_name = "dialogpt-medium"
        print("📦 Using medium model (345MB)")
    elif args.phi:
        model_type = "huggingface"
        model_name = "phi3-mini"
        print("⚠️  Using Phi-3 model (2.2GB) - this will download ~2.2GB")
        confirm = input("Continue? (y/n): ")
        if confirm.lower() != 'y':
            print("Cancelled.")
            return
    elif args.llama:
        model_type = "huggingface"
        model_name = "llama2-7b"
        print("🚨 WARNING: Llama 2 model is ~13GB!")
        print("This will download a very large model to ~/.cache/huggingface/hub/")
        confirm = input("Are you sure you want to continue? (y/n): ")
        if confirm.lower() != 'y':
            print("Cancelled. Try --small or --medium for lightweight options.")
            return
    elif args.model:
        model_type = "huggingface"
        model_name = args.model
        
        # Warn about large models
        large_models = ["llama2-7b", "phi3-mini"]
        if model_name in large_models:
            from src.huggingface_llm import ModelFactory
            size = ModelFactory.MODEL_SIZES.get(model_name, "Unknown")
            print(f"⚠️  This will download {size} to your cache directory")
            confirm = input("Continue? (y/n): ")
            if confirm.lower() != 'y':
                print("Cancelled.")
                return
    elif args.openai:
        model_type = "openai"
    
    print("🐰 Alice in Wonderland QA System")
    print("=" * 40)
    
    # Model configuration
    model_config = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    
    if model_type == "huggingface":
        print(f"🤖 Using HuggingFace model: {model_name}")
        from src.huggingface_llm import ModelFactory
        size = ModelFactory.MODEL_SIZES.get(model_name, "Unknown")
        print(f"📊 Model size: {size}")
        print(f"📊 Configuration: {model_config}")
    else:
        print("🔗 Using OpenAI model")
    
    # Check prerequisites
    if not check_prerequisites(model_type):
        sys.exit(1)
    
    # Set model configuration for the app
    os.environ['QA_MODEL_TYPE'] = model_type
    if model_name:
        os.environ['QA_MODEL_NAME'] = model_name
    os.environ['QA_TEMPERATURE'] = str(args.temperature)
    os.environ['QA_MAX_TOKENS'] = str(args.max_tokens)
    
    # Initialize QA system
    print("🔄 Initializing QA system...")
    if not initialize_qa_system(model_type=model_type, model_name=model_name, model_config=model_config):
        print("❌ Failed to initialize QA system")
        sys.exit(1)
    
    print("✅ QA system initialized successfully!")
    print(f"\n🌐 Starting Flask server on port {args.port}...")
    print(f"📍 Server will be available at: http://localhost:{args.port}")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 40)
    
    # Run the Flask app
    try:
        app.run(
            host='0.0.0.0',
            port=args.port,
            debug=False,  # Set to False for production
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")

if __name__ == '__main__':
    main()