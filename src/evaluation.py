"""
Enhanced evaluation module for RAG system performance analysis
"""
from typing import List, Dict, Any
import json
from dataclasses import dataclass
from src.question_answering import QASystem, QAResponse
import time

@dataclass
class EvaluationResult:
    """Individual evaluation result"""
    question: str
    answer: str
    confidence: float
    context_count: int
    retrieval_score: float
    answer_length: int
    response_time: float
    category: str

class EvaluationFramework:
    """Enhanced evaluation framework for the QA system"""
    
    def __init__(self, qa_system: QASystem):
        self.qa_system = qa_system
        self.test_questions = self._create_test_questions()
    
    def _create_test_questions(self) -> List[Dict[str, str]]:
        """Create comprehensive test questions"""
        return [
            # Character questions
            {"question": "Who is Alice?", "category": "character", "expected_keywords": ["alice", "girl", "main", "character"]},
            {"question": "Who is Dinah?", "category": "character", "expected_keywords": ["dinah", "cat", "alice"]},
            {"question": "What is the Cheshire Cat like?", "category": "character", "expected_keywords": ["cheshire", "cat", "grin", "disappear"]},
            {"question": "Who is the Queen of Hearts?", "category": "character", "expected_keywords": ["queen", "hearts", "off", "heads"]},
            {"question": "What is the Mad Hatter like?", "category": "character", "expected_keywords": ["hatter", "mad", "tea", "party"]},
            {"question": "Who is the White Rabbit?", "category": "character", "expected_keywords": ["rabbit", "white", "late", "watch"]},
            
            # Plot questions
            {"question": "How does Alice enter Wonderland?", "category": "plot", "expected_keywords": ["rabbit", "hole", "fall", "follow"]},
            {"question": "What happens at the tea party?", "category": "plot", "expected_keywords": ["tea", "party", "hatter", "riddles"]},
            {"question": "How does the story end?", "category": "plot", "expected_keywords": ["dream", "wake", "sister", "end"]},
            {"question": "What is the trial about?", "category": "plot", "expected_keywords": ["trial", "knave", "tarts", "stolen"]},
            {"question": "What happens with the croquet game?", "category": "plot", "expected_keywords": ["croquet", "flamingo", "hedgehog", "queen"]},
            
            # Thematic questions
            {"question": "What are the main themes?", "category": "theme", "expected_keywords": ["growing", "identity", "nonsense", "logic"]},
            {"question": "How does Alice change size?", "category": "theme", "expected_keywords": ["size", "grow", "shrink", "drink", "eat"]},
            {"question": "What does growing up mean in the story?", "category": "theme", "expected_keywords": ["growing", "up", "adult", "child"]},
            
            # Factual questions
            {"question": "What does Alice drink?", "category": "factual", "expected_keywords": ["drink", "bottle", "potion"]},
            {"question": "What game does the Queen play?", "category": "factual", "expected_keywords": ["croquet", "game", "flamingo"]},
        ]
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive evaluation with detailed analysis"""
        print("🔄 Starting Enhanced RAG Evaluation...")
        print("=" * 50)
        
        results = []
        category_performance = {}
        
        for i, test_case in enumerate(self.test_questions, 1):
            print(f"\n📝 Question {i}/{len(self.test_questions)}: {test_case['question']}")
            print(f"   Category: {test_case['category']}")
            
            # Measure response time
            start_time = time.time()
            
            try:
                # Get response from QA system
                response = self.qa_system.ask_question(test_case['question'])
                response_time = time.time() - start_time
                
                # Evaluate retrieval quality
                retrieval_score = self._evaluate_retrieval(test_case, response)
                
                # Evaluate answer quality
                answer_score = self._evaluate_answer_quality(test_case, response)
                
                # Create evaluation result
                eval_result = EvaluationResult(
                    question=test_case['question'],
                    answer=response.answer,
                    confidence=response.confidence,
                    context_count=len(response.context),
                    retrieval_score=retrieval_score,
                    answer_length=len(response.answer),
                    response_time=response_time,
                    category=test_case['category']
                )
                
                results.append(eval_result)
                
                # Print results
                print(f"   ✅ Answer: {response.answer[:100]}{'...' if len(response.answer) > 100 else ''}")
                print(f"   📊 Confidence: {response.confidence:.2f}")
                print(f"   🔍 Context pieces: {len(response.context)}")
                print(f"   📈 Retrieval score: {retrieval_score:.2f}")
                print(f"   📝 Answer score: {answer_score:.2f}")
                print(f"   ⏱️  Response time: {response_time:.2f}s")
                
                # Track category performance
                if test_case['category'] not in category_performance:
                    category_performance[test_case['category']] = []
                category_performance[test_case['category']].append({
                    'confidence': response.confidence,
                    'retrieval_score': retrieval_score,
                    'answer_score': answer_score,
                    'response_time': response_time
                })
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                continue
        
        # Calculate overall statistics
        overall_stats = self._calculate_statistics(results, category_performance)
        
        # Print summary
        self._print_evaluation_summary(overall_stats, category_performance)
        
        return overall_stats
    
    def _evaluate_retrieval(self, test_case: Dict[str, str], response: QAResponse) -> float:
        """Evaluate how well the retrieval system performed"""
        if not response.context:
            return 0.0
        
        score = 0.0
        expected_keywords = test_case.get('expected_keywords', [])
        
        # Check if expected keywords appear in retrieved context
        all_context = " ".join(response.context).lower()
        keyword_matches = sum(1 for keyword in expected_keywords if keyword in all_context)
        
        if expected_keywords:
            score += (keyword_matches / len(expected_keywords)) * 0.6
        
        # Bonus for having multiple context pieces
        if len(response.context) >= 3:
            score += 0.2
        elif len(response.context) >= 2:
            score += 0.1
        
        # Bonus for reasonable context length
        avg_context_length = sum(len(ctx) for ctx in response.context) / len(response.context)
        if 100 <= avg_context_length <= 800:
            score += 0.2
        
        return min(score, 1.0)
    
    def _evaluate_answer_quality(self, test_case: Dict[str, str], response: QAResponse) -> float:
        """Evaluate the quality of the generated answer"""
        answer = response.answer.lower()
        expected_keywords = test_case.get('expected_keywords', [])
        
        score = 0.0
        
        # Check keyword presence in answer
        if expected_keywords:
            keyword_matches = sum(1 for keyword in expected_keywords if keyword in answer)
            score += (keyword_matches / len(expected_keywords)) * 0.5
        
        # Answer length appropriateness
        if 50 <= len(response.answer) <= 300:
            score += 0.2
        elif len(response.answer) > 300:
            score += 0.1
        
        # Check for question words in answer (should be minimal)
        question_words = set(test_case['question'].lower().split())
        answer_words = set(answer.split())
        question_overlap = len(question_words.intersection(answer_words))
        if question_overlap <= 3:  # Some overlap is good, too much is bad
            score += 0.1
        
        # Bonus for coherent structure
        if any(phrase in answer for phrase in ['based on', 'according to', 'the story', 'the text']):
            score += 0.1
        
        # Penalty for obviously bad answers
        if answer.startswith("i couldn't find") or answer.startswith("i found some relevant"):
            score -= 0.3
        
        return max(0.0, min(score, 1.0))
    
    def _calculate_statistics(self, results: List[EvaluationResult], category_performance: Dict) -> Dict[str, Any]:
        """Calculate comprehensive statistics"""
        if not results:
            return {}
        
        return {
            'total_questions': len(results),
            'avg_confidence': sum(r.confidence for r in results) / len(results),
            'avg_retrieval_score': sum(r.retrieval_score for r in results) / len(results),
            'avg_context_count': sum(r.context_count for r in results) / len(results),
            'avg_answer_length': sum(r.answer_length for r in results) / len(results),
            'avg_response_time': sum(r.response_time for r in results) / len(results),
            'category_performance': category_performance,
            'detailed_results': results
        }
    
    def _print_evaluation_summary(self, stats: Dict[str, Any], category_performance: Dict):
        """Print comprehensive evaluation summary"""
        print("\n" + "=" * 50)
        print("📊 EVALUATION SUMMARY")
        print("=" * 50)
        
        if not stats:
            print("❌ No results to analyze")
            return
        
        print(f"📈 Overall Performance:")
        print(f"   • Total Questions: {stats['total_questions']}")
        print(f"   • Average Confidence: {stats['avg_confidence']:.3f}")
        print(f"   • Average Retrieval Score: {stats['avg_retrieval_score']:.3f}")
        print(f"   • Average Context Count: {stats['avg_context_count']:.1f}")
        print(f"   • Average Answer Length: {stats['avg_answer_length']:.0f} chars")
        print(f"   • Average Response Time: {stats['avg_response_time']:.2f}s")
        
        print(f"\n📊 Performance by Category:")
        for category, performances in category_performance.items():
            avg_conf = sum(p['confidence'] for p in performances) / len(performances)
            avg_retr = sum(p['retrieval_score'] for p in performances) / len(performances)
            avg_ans = sum(p['answer_score'] for p in performances) / len(performances)
            avg_time = sum(p['response_time'] for p in performances) / len(performances)
            
            print(f"   🔹 {category.upper()}:")
            print(f"      Confidence: {avg_conf:.3f} | Retrieval: {avg_retr:.3f} | Answer: {avg_ans:.3f} | Time: {avg_time:.2f}s")
        
        # Performance recommendations
        print(f"\n💡 Recommendations:")
        if stats['avg_retrieval_score'] < 0.6:
            print("   • Consider lowering score threshold or improving embeddings")
        if stats['avg_confidence'] < 0.6:
            print("   • Review context processing logic")
        if stats['avg_response_time'] > 2.0:
            print("   • Consider optimizing retrieval or reducing context size")
        if stats['avg_context_count'] < 3:
            print("   • Increase top_k in retrieval configuration")
        
        print("\n✅ Evaluation Complete!")

    def run_quick_test(self, question: str) -> Dict[str, Any]:
        """Run a quick test on a single question with detailed analysis"""
        print(f"🔍 Quick Test: {question}")
        print("-" * 40)
        
        start_time = time.time()
        response = self.qa_system.ask_question(question)
        response_time = time.time() - start_time
        
        print(f"📝 Answer: {response.answer}")
        print(f"📊 Confidence: {response.confidence}")
        print(f"🔍 Context pieces: {len(response.context)}")
        print(f"⏱️  Response time: {response_time:.2f}s")
        
        if response.context:
            print(f"\n📚 Retrieved Context:")
            for i, ctx in enumerate(response.context[:3], 1):
                print(f"   {i}. {ctx[:150]}{'...' if len(ctx) > 150 else ''}")
        
        return {
            'question': question,
            'answer': response.answer,
            'confidence': response.confidence,
            'context_count': len(response.context),
            'response_time': response_time
        }