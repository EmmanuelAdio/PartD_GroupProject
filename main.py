"""
AI Agent System
Main orchestrator for the Processor, Answerer, and Evaluator agents.
"""

from typing import Dict, Any
from agents import ProcessorAgent, AnswererAgent, EvaluatorAgent


class AIAgentSystem:
    """
    Main system that orchestrates the three AI agents.
    
    The system follows this workflow:
    1. Processor analyzes the user's question
    2. Answerer generates a response based on the analysis
    3. Evaluator assesses the quality and relevance of the response
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the AI Agent System.
        
        Args:
            config: Configuration dictionary for the system and agents
        """
        self.config = config or {}
        
        # Initialize the three agents
        self.processor = ProcessorAgent(self.config.get('processor', {}))
        self.answerer = AnswererAgent(self.config.get('answerer', {}))
        self.evaluator = EvaluatorAgent(self.config.get('evaluator', {}))
        
        print("AI Agent System initialized with Processor, Answerer, and Evaluator")
    
    def process_question(self, question: str, return_full_pipeline: bool = True) -> Dict[str, Any]:
        """
        Process a question through the complete agent pipeline.
        
        Args:
            question: The user's question
            return_full_pipeline: If True, returns data from all agents; 
                                 if False, returns only the final answer and evaluation
        
        Returns:
            Dictionary containing results from all three agents
        """
        print(f"\n{'='*80}")
        print(f"Processing Question: {question}")
        print(f"{'='*80}\n")
        
        # Step 1: Processor analyzes the question
        print("[1/3] Processor Agent analyzing question...")
        processed_data = self.processor.process(question)
        print(f"✓ Analysis complete - Type: {processed_data['question_type']}, "
              f"Complexity: {processed_data['complexity']}")
        
        # Step 2: Answerer generates response
        print("\n[2/3] Answerer Agent generating response...")
        answer_data = self.answerer.process(processed_data)
        print(f"✓ Answer generated with {answer_data['confidence']} confidence")
        
        # Step 3: Evaluator assesses the response
        print("\n[3/3] Evaluator Agent assessing response quality...")
        evaluation_input = {
            'original_question': question,
            'processed_data': processed_data,
            'answer_data': answer_data
        }
        evaluation_data = self.evaluator.process(evaluation_input)
        print(f"✓ Evaluation complete - Score: {evaluation_data['overall_score']}/100, "
              f"Passed: {evaluation_data['passed']}")
        
        # Compile results
        if return_full_pipeline:
            result = {
                'question': question,
                'processor_output': processed_data,
                'answerer_output': answer_data,
                'evaluator_output': evaluation_data
            }
        else:
            result = {
                'question': question,
                'answer': answer_data['answer'],
                'confidence': answer_data['confidence'],
                'evaluation_score': evaluation_data['overall_score'],
                'evaluation_passed': evaluation_data['passed'],
                'feedback': evaluation_data['feedback']
            }
        
        print(f"\n{'='*80}")
        print("Processing Complete")
        print(f"{'='*80}\n")
        
        return result
    
    def get_answer(self, question: str) -> str:
        """
        Convenience method to get just the answer to a question.
        
        Args:
            question: The user's question
            
        Returns:
            The answer string
        """
        result = self.process_question(question, return_full_pipeline=False)
        return result['answer']
    
    def get_evaluation_summary(self, question: str) -> Dict[str, Any]:
        """
        Get a summary of the evaluation for a question.
        
        Args:
            question: The user's question
            
        Returns:
            Dictionary with evaluation summary
        """
        result = self.process_question(question, return_full_pipeline=True)
        evaluation = result['evaluator_output']
        
        return {
            'overall_score': evaluation['overall_score'],
            'passed': evaluation['passed'],
            'feedback': evaluation['feedback'],
            'suggestions': evaluation['suggestions']
        }


def main():
    """
    Main function demonstrating the AI Agent System.
    """
    # Initialize the system
    config = {
        'evaluator': {
            'quality_threshold': 70,
            'scoring_weights': {
                'relevance': 0.35,
                'completeness': 0.25,
                'clarity': 0.20,
                'accuracy': 0.20
            }
        }
    }
    
    system = AIAgentSystem(config)
    
    # Example questions
    example_questions = [
        "What is artificial intelligence?",
        "How do I train a machine learning model?",
        "Why is data preprocessing important in machine learning?"
    ]
    
    print("\n" + "="*80)
    print("AI AGENT SYSTEM DEMONSTRATION")
    print("="*80)
    
    for question in example_questions:
        result = system.process_question(question, return_full_pipeline=False)
        
        print("\n" + "-"*80)
        print(f"Question: {result['question']}")
        print(f"\nAnswer: {result['answer']}")
        print(f"\nConfidence: {result['confidence']}")
        print(f"Evaluation Score: {result['evaluation_score']}/100")
        print(f"Quality Check: {'✓ PASSED' if result['evaluation_passed'] else '✗ FAILED'}")
        print(f"\nFeedback: {result['feedback']}")
        print("-"*80)


if __name__ == "__main__":
    main()
