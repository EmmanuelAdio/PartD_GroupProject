"""
Example Usage of the AI Agent System
Demonstrates different ways to use the three-agent system.
"""

import json
import sys
import os

# Add parent directory to path to import agents
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import AIAgentSystem


def example_basic_usage():
    """Example 1: Basic usage with default configuration."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Usage")
    print("="*80)
    
    system = AIAgentSystem()
    
    question = "What is machine learning?"
    result = system.process_question(question, return_full_pipeline=False)
    
    print(f"\nQuestion: {question}")
    print(f"Answer: {result['answer']}")
    print(f"Evaluation Score: {result['evaluation_score']}/100")


def example_full_pipeline():
    """Example 2: Get full pipeline output."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Full Pipeline Output")
    print("="*80)
    
    system = AIAgentSystem()
    
    question = "How can I improve my Python programming skills?"
    result = system.process_question(question, return_full_pipeline=True)
    
    print(f"\nQuestion: {question}\n")
    
    print("Processor Output:")
    print(f"  - Question Type: {result['processor_output']['question_type']}")
    print(f"  - Intent: {result['processor_output']['intent']}")
    print(f"  - Complexity: {result['processor_output']['complexity']}")
    print(f"  - Keywords: {result['processor_output']['keywords']}")
    
    print("\nAnswerer Output:")
    print(f"  - Confidence: {result['answerer_output']['confidence']}")
    print(f"  - Sources: {result['answerer_output']['sources']}")
    print(f"  - Answer: {result['answerer_output']['answer'][:100]}...")
    
    print("\nEvaluator Output:")
    print(f"  - Overall Score: {result['evaluator_output']['overall_score']}/100")
    print(f"  - Relevance: {result['evaluator_output']['relevance_score']}/100")
    print(f"  - Completeness: {result['evaluator_output']['completeness_score']}/100")
    print(f"  - Clarity: {result['evaluator_output']['clarity_score']}/100")
    print(f"  - Accuracy: {result['evaluator_output']['accuracy_score']}/100")
    print(f"  - Passed: {result['evaluator_output']['passed']}")


def example_custom_config():
    """Example 3: Using custom configuration."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Custom Configuration")
    print("="*80)
    
    # Custom configuration with stricter evaluation
    custom_config = {
        'evaluator': {
            'quality_threshold': 80,  # Stricter threshold
            'scoring_weights': {
                'relevance': 0.40,      # More weight on relevance
                'completeness': 0.30,   # More weight on completeness
                'clarity': 0.15,
                'accuracy': 0.15
            }
        }
    }
    
    system = AIAgentSystem(custom_config)
    
    question = "Why is version control important in software development?"
    result = system.process_question(question, return_full_pipeline=False)
    
    print(f"\nQuestion: {question}")
    print(f"Answer: {result['answer']}")
    print(f"Evaluation Score: {result['evaluation_score']}/100")
    print(f"Quality Check (80% threshold): {'✓ PASSED' if result['evaluation_passed'] else '✗ FAILED'}")
    print(f"Feedback: {result['feedback']}")


def example_multiple_questions():
    """Example 4: Processing multiple questions."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Processing Multiple Questions")
    print("="*80)
    
    system = AIAgentSystem()
    
    questions = [
        "What is the capital of France?",
        "How do neural networks work?",
        "When was the internet invented?",
        "Why should I learn programming?",
        "Can AI replace human creativity?"
    ]
    
    results = []
    for question in questions:
        result = system.process_question(question, return_full_pipeline=False)
        results.append(result)
    
    print("\n" + "-"*80)
    print("Summary of Results:")
    print("-"*80)
    
    for i, result in enumerate(results, 1):
        status = "✓" if result['evaluation_passed'] else "✗"
        print(f"{i}. {status} Score: {result['evaluation_score']}/100 - {result['question'][:50]}...")


def example_evaluation_summary():
    """Example 5: Get evaluation summary."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Evaluation Summary")
    print("="*80)
    
    system = AIAgentSystem()
    
    question = "What are the benefits of cloud computing?"
    summary = system.get_evaluation_summary(question)
    
    print(f"\nQuestion: {question}\n")
    print(f"Overall Score: {summary['overall_score']}/100")
    print(f"Quality Check: {'✓ PASSED' if summary['passed'] else '✗ FAILED'}")
    print(f"\nFeedback: {summary['feedback']}")
    print(f"\nSuggestions for Improvement:")
    for i, suggestion in enumerate(summary['suggestions'], 1):
        print(f"  {i}. {suggestion}")


def example_convenience_method():
    """Example 6: Using convenience method."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Quick Answer (Convenience Method)")
    print("="*80)
    
    system = AIAgentSystem()
    
    question = "What is the difference between AI and ML?"
    answer = system.get_answer(question)
    
    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("AI AGENT SYSTEM - USAGE EXAMPLES")
    print("="*80)
    
    examples = [
        ("Basic Usage", example_basic_usage),
        ("Full Pipeline", example_full_pipeline),
        ("Custom Configuration", example_custom_config),
        ("Multiple Questions", example_multiple_questions),
        ("Evaluation Summary", example_evaluation_summary),
        ("Convenience Method", example_convenience_method)
    ]
    
    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\nError in {name}: {e}")
        print("\n")
    
    print("="*80)
    print("All examples completed!")
    print("="*80)


if __name__ == "__main__":
    main()
