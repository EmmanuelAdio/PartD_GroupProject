"""
Evaluator Agent
Assesses the quality and relevance of the Answerer's response.
"""

from typing import Dict, Any
from .base_agent import BaseAgent


class EvaluatorAgent(BaseAgent):
    """
    Evaluator Agent assesses the quality and relevance of answers.
    
    This agent is responsible for:
    - Evaluating answer quality and completeness
    - Assessing relevance to the original question
    - Providing feedback and improvement suggestions
    - Scoring the overall response
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the Evaluator Agent."""
        super().__init__("Evaluator", config)
        self.scoring_weights = config.get('scoring_weights', {
            'relevance': 0.35,
            'completeness': 0.25,
            'clarity': 0.20,
            'accuracy': 0.20
        })
    
    def process(self, evaluation_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the answer's quality and relevance.
        
        Args:
            evaluation_input: Dictionary containing:
                - original_question: The user's original question
                - processed_data: Data from ProcessorAgent
                - answer_data: Response from AnswererAgent
                
        Returns:
            Dictionary containing:
                - overall_score: Overall quality score (0-100)
                - relevance_score: How relevant the answer is (0-100)
                - completeness_score: How complete the answer is (0-100)
                - clarity_score: How clear the answer is (0-100)
                - accuracy_score: Estimated accuracy (0-100)
                - feedback: Detailed feedback
                - suggestions: Improvement suggestions
                - passed: Boolean indicating if answer meets quality threshold
        """
        original_question = evaluation_input.get('original_question', '')
        processed_data = evaluation_input.get('processed_data', {})
        answer_data = evaluation_input.get('answer_data', {})
        
        self.log_info(f"Evaluating answer for: {original_question}")
        
        # Evaluate different aspects
        relevance_score = self._evaluate_relevance(
            original_question, 
            processed_data, 
            answer_data
        )
        
        completeness_score = self._evaluate_completeness(
            processed_data, 
            answer_data
        )
        
        clarity_score = self._evaluate_clarity(answer_data)
        
        accuracy_score = self._evaluate_accuracy(
            processed_data, 
            answer_data
        )
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(
            relevance_score,
            completeness_score,
            clarity_score,
            accuracy_score
        )
        
        # Generate feedback and suggestions
        feedback = self._generate_feedback(
            relevance_score,
            completeness_score,
            clarity_score,
            accuracy_score
        )
        
        suggestions = self._generate_suggestions(
            relevance_score,
            completeness_score,
            clarity_score,
            accuracy_score
        )
        
        # Determine if answer passes quality threshold
        threshold = self.config.get('quality_threshold', 70)
        passed = overall_score >= threshold
        
        result = {
            'overall_score': overall_score,
            'relevance_score': relevance_score,
            'completeness_score': completeness_score,
            'clarity_score': clarity_score,
            'accuracy_score': accuracy_score,
            'feedback': feedback,
            'suggestions': suggestions,
            'passed': passed,
            'quality_threshold': threshold
        }
        
        self.log_info(f"Evaluation complete - Overall Score: {overall_score}/100, Passed: {passed}")
        return result
    
    def _evaluate_relevance(self, question: str, processed_data: Dict, 
                           answer_data: Dict) -> float:
        """
        Evaluate how relevant the answer is to the question.
        
        Checks if keywords from the question appear in the answer
        and if the answer addresses the question type appropriately.
        """
        answer = answer_data.get('answer', '').lower()
        keywords = processed_data.get('keywords', [])
        question_type = processed_data.get('question_type', '')
        
        # Check keyword presence
        keyword_matches = sum(1 for kw in keywords if kw in answer)
        keyword_score = (keyword_matches / len(keywords) * 100) if keywords else 50
        
        # Check if answer addresses question type
        type_appropriate = answer_data.get('question_metadata', {}).get('type') == question_type
        type_score = 100 if type_appropriate else 70
        
        # Combine scores
        relevance = (keyword_score * 0.6 + type_score * 0.4)
        return min(100, max(0, relevance))
    
    def _evaluate_completeness(self, processed_data: Dict, answer_data: Dict) -> float:
        """
        Evaluate how complete the answer is.
        
        Checks for presence of key components and supporting details.
        """
        score = 0
        
        # Check if answer exists and has content
        answer = answer_data.get('answer', '')
        if answer and len(answer) > 20:
            score += 40
        
        # Check for supporting details
        if answer_data.get('supporting_details'):
            score += 20
        
        # Check for sources
        if answer_data.get('sources'):
            score += 20
        
        # Check confidence level
        confidence = answer_data.get('confidence', 'low')
        if confidence in ['high', 'medium']:
            score += 20
        
        return min(100, score)
    
    def _evaluate_clarity(self, answer_data: Dict) -> float:
        """
        Evaluate how clear and well-structured the answer is.
        
        Checks for readability and structure.
        """
        answer = answer_data.get('answer', '')
        
        if not answer:
            return 0
        
        score = 50  # Base score
        
        # Check length - not too short, not too long
        word_count = len(answer.split())
        if 10 <= word_count <= 200:
            score += 25
        elif word_count > 5:
            score += 15
        
        # Check for proper punctuation
        if any(p in answer for p in ['.', '!', '?', ':']):
            score += 15
        
        # Check for structure indicators
        if any(indicator in answer.lower() for indicator in ['first', 'second', 'step', 'because', 'however']):
            score += 10
        
        return min(100, score)
    
    def _evaluate_accuracy(self, processed_data: Dict, answer_data: Dict) -> float:
        """
        Evaluate the estimated accuracy of the answer.
        
        Based on confidence level and complexity alignment.
        """
        confidence = answer_data.get('confidence', 'low')
        complexity = processed_data.get('complexity', 'moderate')
        
        # Base score on confidence
        confidence_scores = {'high': 85, 'medium': 70, 'low': 50}
        score = confidence_scores.get(confidence, 50)
        
        # Adjust based on complexity match
        if complexity == 'simple' and confidence in ['high', 'medium']:
            score += 10
        elif complexity == 'complex' and confidence == 'low':
            score -= 10
        
        return min(100, max(0, score))
    
    def _calculate_overall_score(self, relevance: float, completeness: float,
                                 clarity: float, accuracy: float) -> float:
        """Calculate weighted overall score."""
        weights = self.scoring_weights
        overall = (
            relevance * weights['relevance'] +
            completeness * weights['completeness'] +
            clarity * weights['clarity'] +
            accuracy * weights['accuracy']
        )
        return round(overall, 2)
    
    def _generate_feedback(self, relevance: float, completeness: float,
                           clarity: float, accuracy: float) -> str:
        """Generate detailed feedback based on scores."""
        feedback_parts = []
        
        # Relevance feedback
        if relevance >= 80:
            feedback_parts.append("The answer is highly relevant to the question.")
        elif relevance >= 60:
            feedback_parts.append("The answer is reasonably relevant but could be more focused.")
        else:
            feedback_parts.append("The answer lacks relevance to the original question.")
        
        # Completeness feedback
        if completeness >= 80:
            feedback_parts.append("The answer is comprehensive and well-supported.")
        elif completeness >= 60:
            feedback_parts.append("The answer covers the main points but lacks some details.")
        else:
            feedback_parts.append("The answer is incomplete and needs more information.")
        
        # Clarity feedback
        if clarity >= 80:
            feedback_parts.append("The answer is clear and well-structured.")
        elif clarity >= 60:
            feedback_parts.append("The answer is understandable but could be clearer.")
        else:
            feedback_parts.append("The answer lacks clarity and structure.")
        
        # Accuracy feedback
        if accuracy >= 80:
            feedback_parts.append("The answer appears to be accurate and reliable.")
        elif accuracy >= 60:
            feedback_parts.append("The answer seems reasonably accurate but needs verification.")
        else:
            feedback_parts.append("The accuracy of the answer is questionable.")
        
        return " ".join(feedback_parts)
    
    def _generate_suggestions(self, relevance: float, completeness: float,
                             clarity: float, accuracy: float) -> list:
        """Generate improvement suggestions based on scores."""
        suggestions = []
        
        if relevance < 70:
            suggestions.append("Include more keywords from the original question in the answer")
        
        if completeness < 70:
            suggestions.append("Add more supporting details and cite sources")
        
        if clarity < 70:
            suggestions.append("Improve structure and readability of the answer")
        
        if accuracy < 70:
            suggestions.append("Verify information and provide more confident responses")
        
        if not suggestions:
            suggestions.append("The answer meets quality standards")
        
        return suggestions
