"""
Main application entry point
"""
import os
import argparse
from dotenv import load_dotenv
from src.question_answering import QASystem
from src.evaluation import EvaluationFramework

# Load environment variables
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Alice in Wonderland QA System")
    parser.add_argument("--build", action="store_true", help="Build knowledge base")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation")
    parser.add_argument("--interactive", action="store_true", help="Run interactive mode")
    parser.add_argument("--question", type=str, help="Ask a specific question")
    
    # Model selection arguments - updated with lightweight options
    parser.add_argument("--small", action="store_true", help="Use small lightweight model (distilgpt2)")
    parser.add_argument("--medium", action="store_true", help="Use medium model (dialogpt-medium)")
    parser.add_argument("--phi", action="store_true", help="Use Phi-3 model (2.2GB)")
    parser.add_argument("--llama", action="store_true", help="Use Llama 2 model (13GB - WARNING!)")
    parser.add_argument("--model", type=str, 
                       choices=['distilgpt2', 'gpt2-small', 'dialogpt-small', 'dialogpt-medium', 
                               'flan-t5-small', 'flan-t5-base', 'flan-t5-large', 'gpt2-medium', 
                               'dialogpt-large', 'phi3-mini', 'llama2-7b'], 
                       help="Specify exact model to use")
    parser.add_argument("--openai", action="store_true", help="Use OpenAI model (default)")
    parser.add_argument("--list-models", action="store_true", help="List all available models with sizes")
    
    # Model configuration
    parser.add_argument("--temperature", type=float, default=0.2, help="Model temperature (default: 0.2)")
    parser.add_argument("--max-tokens", type=int, default=800, help="Max tokens for generation (default: 800)")
    parser.add_argument("--cache-dir", type=str, help="Custom cache directory for models")
    
    args = parser.parse_args()
    
    # Set custom cache directory if provided
    if args.cache_dir:
        os.environ['HF_HOME'] = args.cache_dir
        print(f"📁 Using custom cache directory: {args.cache_dir}")
    
    # Handle list models command
    if args.list_models:
        from src.huggingface_llm import ModelFactory
        ModelFactory.list_models()
        print(f"\n💡 Default cache location: ~/.cache/huggingface/hub/")
        print(f"💡 Use --cache-dir to specify custom location")
        return
    
    # Determine model type
    model_type = "huggingface"  # default
    model_name = "dialogpt-medium"
    
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
    
    # Initialize QA system with specified model
    qa_system = QASystem(model_type=model_type, model_name=model_name, model_config=model_config)
    
    if args.build:
        print("Building knowledge base...")
        qa_system.build_knowledge_base()
        return
    
    # Load existing knowledge base
    qa_system.load_knowledge_base()
    
    if args.evaluate:
        print("Running evaluation...")
        evaluator = EvaluationFramework(qa_system)
        results = evaluator.run_evaluation()
        print(f"Evaluation Results:")
        print(f"Average Faithfulness: {results.get('avg_faithfulness', 0):.3f}")
        print(f"Average Relevance: {results.get('avg_relevance', 0):.3f}")
        print(f"Average Confidence: {results.get('avg_confidence', 0):.3f}")
        return
    
    if args.question:
        print(f"🤔 Processing question: {args.question}")
        response = qa_system.ask_question(args.question)
        print(f"\n📝 Question: {args.question}")
        print(f"💡 Answer: {response.answer}")
        print(f"🎯 Confidence: {response.confidence:.2f}")
        print(f"📚 Sources: {len(response.context)} context passages")
        return
    
    if args.interactive:
        print("=== Alice in Wonderland QA System ===")
        print(f"🤖 Model: {model_type} ({model_name if model_name else 'default'})")
        print("Ask questions about the story. Type 'quit' to exit, 'samples' for example questions.")
        
        while True:
            question = input("\n🐰 Your question: ").strip()
            
            if question.lower() == 'quit':
                break
            elif question.lower() == 'samples':
                samples = [
                    "Who is Alice and what are her main characteristics?",
                    "Describe the Cheshire Cat and its role in the story.",
                    "What happens at the Mad Tea Party?",
                    "How does Alice fall down the rabbit hole?",
                    "What is the Queen of Hearts like?",
                ]
                print("\n📋 Sample questions:")
                for i, q in enumerate(samples, 1):
                    print(f"{i}. {q}")
                continue
            elif not question:
                continue
            
            try:
                print("🔍 Searching for relevant context...")
                response = qa_system.ask_question(question)
                print(f"\n💡 Answer: {response.answer}")
                print(f"🎯 Confidence: {response.confidence:.2f}")
                
                if response.context:
                    print(f"📚 Based on {len(response.context)} relevant text passages from the book.")
                
            except Exception as e:
                print(f"❌ Error: {e}")

    else:
        # Show help with recommended usage
        parser.print_help()
        print(f"\n🚀 Recommended commands:")
        print(f"  python main.py --build                           # Build knowledge base first")
        print(f"  python main.py --small --interactive             # Use tiny model (82MB)")
        print(f"  python main.py --medium --interactive            # Use medium model (345MB)")
        print(f"  python main.py --list-models                     # Show all models with sizes")
        print(f"  python main.py --model flan-t5-base --interactive # Use specific model")

if __name__ == "__main__":
    main()