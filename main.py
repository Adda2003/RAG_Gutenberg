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
    parser.add_argument("--llama", action="store_true", help="Use Llama 2 model via HuggingFace")
    parser.add_argument("--phi", action="store_true", help="Use Phi-2 model via HuggingFace")
    
    args = parser.parse_args()

    # Determine which LLM to use
    llm_type = "openai"
    if args.phi:
        llm_type = "phi"
    elif args.llama:
        llm_type = "llama"

    # Initialize QA system
    qa_system = QASystem(llm_type=llm_type)
    
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
        response = qa_system.ask_question(args.question)
        print(f"\nQuestion: {args.question}")
        print(f"Answer: {response.answer}")
        print(f"Confidence: {response.confidence:.2f}")
        return
    
    if args.interactive:
        print("=== Alice in Wonderland QA System ===")
        print("Ask questions about the story. Type 'quit' to exit, 'samples' for example questions.")
        
        while True:
            question = input("\nYour question: ").strip()
            
            if question.lower() == 'quit':
                break
            elif question.lower() == 'samples':
                samples = qa_system.get_sample_questions()
                print("\nSample questions:")
                for i, q in enumerate(samples, 1):
                    print(f"{i}. {q}")
                continue
            elif not question:
                continue
            
            try:
                response = qa_system.ask_question(question)
                print(f"\nAnswer: {response.answer}")
                print(f"Confidence: {response.confidence:.2f}")
                
                if response.context:
                    print(f"\nBased on {len(response.context)} relevant text passages from the book.")
                
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    main()