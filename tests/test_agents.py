"""
Unit Tests for the AI Agent System
Tests for Processor, Answerer, and Evaluator agents.
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import ProcessorAgent, AnswererAgent, EvaluatorAgent
from main import AIAgentSystem


class TestProcessorAgent(unittest.TestCase):
    """Test cases for the ProcessorAgent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = ProcessorAgent()
    
    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.name, "Processor")
        self.assertIsNotNone(self.agent.config)
    
    def test_process_basic_question(self):
        """Test processing a basic question."""
        question = "What is Python?"
        result = self.agent.process(question)
        
        self.assertIn('original_question', result)
        self.assertIn('processed_question', result)
        self.assertIn('keywords', result)
        self.assertIn('question_type', result)
        self.assertIn('intent', result)
        self.assertIn('complexity', result)
        
        self.assertEqual(result['original_question'], question)
    
    def test_question_type_identification(self):
        """Test question type identification."""
        test_cases = {
            "What is AI?": "factual",
            "How do I code?": "procedural",
            "Why is this important?": "reasoning",
            "When was it invented?": "temporal",
            "Where is it located?": "location",
            "Can I do this?": "capability"
        }
        
        for question, expected_type in test_cases.items():
            result = self.agent.process(question)
            self.assertEqual(result['question_type'], expected_type,
                           f"Failed for question: {question}")
    
    def test_keyword_extraction(self):
        """Test keyword extraction."""
        question = "How do I train a machine learning model?"
        result = self.agent.process(question)
        
        self.assertIsInstance(result['keywords'], list)
        self.assertIn('train', result['keywords'])
        self.assertIn('machine', result['keywords'])
        self.assertIn('learning', result['keywords'])
    
    def test_complexity_estimation(self):
        """Test complexity estimation."""
        simple = "What is AI?"
        moderate = "How do I train a machine learning model?"
        complex = "Why is data preprocessing so incredibly important in machine learning when dealing with very complex datasets?"
        
        self.assertEqual(self.agent.process(simple)['complexity'], 'simple')
        self.assertEqual(self.agent.process(moderate)['complexity'], 'moderate')
        self.assertEqual(self.agent.process(complex)['complexity'], 'complex')


class TestAnswererAgent(unittest.TestCase):
    """Test cases for the AnswererAgent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = AnswererAgent()
        self.processor = ProcessorAgent()
    
    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.name, "Answerer")
        self.assertIsNotNone(self.agent.config)
    
    def test_process_with_processed_data(self):
        """Test generating an answer from processed data."""
        question = "What is Python?"
        processed_data = self.processor.process(question)
        result = self.agent.process(processed_data)
        
        self.assertIn('answer', result)
        self.assertIn('confidence', result)
        self.assertIn('sources', result)
        self.assertIn('supporting_details', result)
        self.assertIn('question_metadata', result)
        
        self.assertIsInstance(result['answer'], str)
        self.assertIn(result['confidence'], ['low', 'medium', 'high'])
    
    def test_confidence_levels(self):
        """Test confidence level calculation."""
        # Simple questions with keywords should have reasonable confidence
        simple_question = "What is AI?"
        processed = self.processor.process(simple_question)
        result = self.agent.process(processed)
        
        # Confidence should be one of the valid levels
        self.assertIn(result['confidence'], ['low', 'medium', 'high'])


class TestEvaluatorAgent(unittest.TestCase):
    """Test cases for the EvaluatorAgent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = EvaluatorAgent()
        self.processor = ProcessorAgent()
        self.answerer = AnswererAgent()
    
    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.name, "Evaluator")
        self.assertIsNotNone(self.agent.config)
        self.assertIn('relevance', self.agent.scoring_weights)
    
    def test_process_evaluation(self):
        """Test evaluating an answer."""
        question = "What is machine learning?"
        processed_data = self.processor.process(question)
        answer_data = self.answerer.process(processed_data)
        
        evaluation_input = {
            'original_question': question,
            'processed_data': processed_data,
            'answer_data': answer_data
        }
        
        result = self.agent.process(evaluation_input)
        
        self.assertIn('overall_score', result)
        self.assertIn('relevance_score', result)
        self.assertIn('completeness_score', result)
        self.assertIn('clarity_score', result)
        self.assertIn('accuracy_score', result)
        self.assertIn('feedback', result)
        self.assertIn('suggestions', result)
        self.assertIn('passed', result)
        
        # Scores should be between 0 and 100
        for score_key in ['overall_score', 'relevance_score', 'completeness_score', 
                         'clarity_score', 'accuracy_score']:
            self.assertGreaterEqual(result[score_key], 0)
            self.assertLessEqual(result[score_key], 100)
    
    def test_quality_threshold(self):
        """Test quality threshold checking."""
        # Test with custom threshold
        config = {'quality_threshold': 90}
        evaluator = EvaluatorAgent(config)
        
        question = "What is AI?"
        processed_data = self.processor.process(question)
        answer_data = self.answerer.process(processed_data)
        
        evaluation_input = {
            'original_question': question,
            'processed_data': processed_data,
            'answer_data': answer_data
        }
        
        result = evaluator.process(evaluation_input)
        self.assertIn('passed', result)
        self.assertIsInstance(result['passed'], bool)


class TestAIAgentSystem(unittest.TestCase):
    """Test cases for the complete AI Agent System."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.system = AIAgentSystem()
    
    def test_initialization(self):
        """Test system initialization."""
        self.assertIsNotNone(self.system.processor)
        self.assertIsNotNone(self.system.answerer)
        self.assertIsNotNone(self.system.evaluator)
    
    def test_process_question_full_pipeline(self):
        """Test processing a question through the full pipeline."""
        question = "What is artificial intelligence?"
        result = self.system.process_question(question, return_full_pipeline=True)
        
        self.assertIn('question', result)
        self.assertIn('processor_output', result)
        self.assertIn('answerer_output', result)
        self.assertIn('evaluator_output', result)
        
        self.assertEqual(result['question'], question)
    
    def test_process_question_simplified(self):
        """Test processing with simplified output."""
        question = "How does Python work?"
        result = self.system.process_question(question, return_full_pipeline=False)
        
        self.assertIn('question', result)
        self.assertIn('answer', result)
        self.assertIn('confidence', result)
        self.assertIn('evaluation_score', result)
        self.assertIn('evaluation_passed', result)
        self.assertIn('feedback', result)
    
    def test_get_answer_convenience_method(self):
        """Test the convenience method for getting answers."""
        question = "What is Python?"
        answer = self.system.get_answer(question)
        
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)
    
    def test_get_evaluation_summary(self):
        """Test getting evaluation summary."""
        question = "Why is programming important?"
        summary = self.system.get_evaluation_summary(question)
        
        self.assertIn('overall_score', summary)
        self.assertIn('passed', summary)
        self.assertIn('feedback', summary)
        self.assertIn('suggestions', summary)
    
    def test_custom_configuration(self):
        """Test system with custom configuration."""
        config = {
            'evaluator': {
                'quality_threshold': 85,
                'scoring_weights': {
                    'relevance': 0.40,
                    'completeness': 0.30,
                    'clarity': 0.15,
                    'accuracy': 0.15
                }
            }
        }
        
        system = AIAgentSystem(config)
        result = system.process_question("What is AI?", return_full_pipeline=True)
        
        self.assertEqual(
            system.evaluator.config['quality_threshold'], 
            85
        )


if __name__ == '__main__':
    # Run tests with verbosity
    unittest.main(verbosity=2)
