"""
Processor Agent
Analyzes and processes user questions to extract key information.
"""

from typing import Dict, Any
from .base_agent import BaseAgent


class ProcessorAgent(BaseAgent):
    """
    Processor Agent analyzes user questions to extract intent, keywords, and context.
    
    This agent is responsible for:
    - Understanding the user's question
    - Extracting key entities and topics
    - Identifying the question type and intent
    - Preparing structured data for the Answerer agent
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the Processor Agent."""
        super().__init__("Processor", config)
    
    def process(self, question: str) -> Dict[str, Any]:
        """
        Analyze the user's question and extract relevant information.
        
        Args:
            question: The user's question as a string
            
        Returns:
            Dictionary containing:
                - original_question: The original question
                - processed_question: Cleaned/normalized question
                - keywords: List of extracted keywords
                - question_type: Type of question (e.g., 'what', 'how', 'why')
                - intent: Identified intent
                - complexity: Estimated complexity level
        """
        self.log_info(f"Processing question: {question}")
        
        # Clean and normalize the question
        processed_question = question.strip()
        
        # Extract basic question type
        question_lower = processed_question.lower()
        question_type = self._identify_question_type(question_lower)
        
        # Extract keywords (simple implementation)
        keywords = self._extract_keywords(processed_question)
        
        # Estimate complexity
        complexity = self._estimate_complexity(processed_question)
        
        # Identify intent
        intent = self._identify_intent(question_lower, question_type)
        
        result = {
            'original_question': question,
            'processed_question': processed_question,
            'keywords': keywords,
            'question_type': question_type,
            'intent': intent,
            'complexity': complexity
        }
        
        self.log_info(f"Processed result - Type: {question_type}, Complexity: {complexity}")
        return result
    
    def _identify_question_type(self, question: str) -> str:
        """Identify the type of question."""
        question_words = {
            'what': 'factual',
            'who': 'identity',
            'where': 'location',
            'when': 'temporal',
            'why': 'reasoning',
            'how': 'procedural',
            'is': 'boolean',
            'can': 'capability',
            'should': 'recommendation'
        }
        
        for word, qtype in question_words.items():
            if question.startswith(word):
                return qtype
        
        return 'general'
    
    def _extract_keywords(self, question: str) -> list:
        """Extract keywords from the question."""
        # Remove common stop words
        stop_words = {'is', 'are', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 
                      'of', 'and', 'or', 'but', 'what', 'how', 'why', 'when', 'where',
                      'who', 'which', 'can', 'could', 'should', 'would', 'do', 'does'}
        
        words = question.lower().replace('?', '').split()
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords[:10]  # Limit to top 10 keywords
    
    def _estimate_complexity(self, question: str) -> str:
        """Estimate the complexity of the question."""
        word_count = len(question.split())
        
        if word_count <= 5:
            return 'simple'
        elif word_count <= 15:
            return 'moderate'
        else:
            return 'complex'
    
    def _identify_intent(self, question: str, question_type: str) -> str:
        """Identify the user's intent."""
        intent_map = {
            'factual': 'seeking_information',
            'identity': 'seeking_identification',
            'location': 'seeking_location',
            'temporal': 'seeking_timing',
            'reasoning': 'seeking_explanation',
            'procedural': 'seeking_instructions',
            'boolean': 'seeking_confirmation',
            'capability': 'seeking_possibility',
            'recommendation': 'seeking_advice'
        }
        
        return intent_map.get(question_type, 'general_inquiry')
