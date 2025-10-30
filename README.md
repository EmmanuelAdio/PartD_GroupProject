# AI Agent System - Three-Agent Question Answering Framework

A comprehensive Python framework implementing three specialized AI agents that work together to process user questions, generate answers, and evaluate response quality.

## Overview

This system implements a three-agent architecture:

1. **Processor Agent** - Analyzes and processes user questions to extract key information
2. **Answerer Agent** - Generates appropriate responses based on the processed question data
3. **Evaluator Agent** - Assesses the quality and relevance of the generated answers

## Architecture

```
User Question
     ↓
[Processor Agent] → Analyzes question, extracts keywords, identifies intent
     ↓
[Answerer Agent] → Generates response based on analysis
     ↓
[Evaluator Agent] → Evaluates answer quality and relevance
     ↓
Final Result with Quality Score
```

## Features

### Processor Agent
- Question type identification (factual, procedural, reasoning, etc.)
- Keyword extraction
- Intent recognition
- Complexity estimation
- Question normalization

### Answerer Agent
- Context-aware answer generation
- Confidence level assessment
- Source identification
- Supporting details provision
- Metadata tracking

### Evaluator Agent
- Multi-dimensional quality scoring:
  - Relevance (35% weight)
  - Completeness (25% weight)
  - Clarity (20% weight)
  - Accuracy (20% weight)
- Detailed feedback generation
- Improvement suggestions
- Configurable quality thresholds

## Installation

### Prerequisites
- Python 3.8 or higher

### Setup

1. Clone the repository:
```bash
git clone https://github.com/EmmanuelAdio/PartD_GroupProject.git
cd PartD_GroupProject
```

2. (Optional) Install dependencies:
```bash
pip install -r requirements.txt
```

Note: The base framework uses only Python standard library. Additional dependencies are optional for future enhancements.

## Usage

### Basic Usage

```python
from main import AIAgentSystem

# Initialize the system
system = AIAgentSystem()

# Process a question
question = "What is machine learning?"
result = system.process_question(question, return_full_pipeline=False)

print(f"Answer: {result['answer']}")
print(f"Score: {result['evaluation_score']}/100")
```

### Full Pipeline Output

```python
# Get detailed output from all three agents
result = system.process_question(question, return_full_pipeline=True)

print(f"Processor Output: {result['processor_output']}")
print(f"Answerer Output: {result['answerer_output']}")
print(f"Evaluator Output: {result['evaluator_output']}")
```

### Custom Configuration

```python
config = {
    'evaluator': {
        'quality_threshold': 80,  # Set quality threshold to 80%
        'scoring_weights': {
            'relevance': 0.40,
            'completeness': 0.30,
            'clarity': 0.15,
            'accuracy': 0.15
        }
    }
}

system = AIAgentSystem(config)
```

### Convenience Methods

```python
# Get just the answer
answer = system.get_answer("What is AI?")

# Get evaluation summary
summary = system.get_evaluation_summary("How does AI work?")
print(f"Score: {summary['overall_score']}")
print(f"Feedback: {summary['feedback']}")
```

## Running Examples

Run the main demonstration:
```bash
python main.py
```

Run comprehensive usage examples:
```bash
python examples/usage_examples.py
```

## Project Structure

```
PartD_GroupProject/
├── agents/
│   ├── __init__.py
│   ├── base_agent.py       # Base class for all agents
│   ├── processor.py        # Processor Agent implementation
│   ├── answerer.py         # Answerer Agent implementation
│   └── evaluator.py        # Evaluator Agent implementation
├── config/
│   └── default_config.json # Default configuration
├── examples/
│   └── usage_examples.py   # Comprehensive usage examples
├── tests/                  # Unit tests (to be implemented)
├── main.py                 # Main orchestrator and demo
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Configuration

The system can be configured through a JSON configuration file or by passing a dictionary to the `AIAgentSystem` constructor.

### Configuration Options

```json
{
  "processor": {
    "enabled": true,
    "max_keywords": 10,
    "stop_words_enabled": true
  },
  "answerer": {
    "enabled": true,
    "default_confidence": "medium",
    "include_sources": true,
    "include_supporting_details": true
  },
  "evaluator": {
    "enabled": true,
    "quality_threshold": 70,
    "scoring_weights": {
      "relevance": 0.35,
      "completeness": 0.25,
      "clarity": 0.20,
      "accuracy": 0.20
    }
  }
}
```

## Extending the Framework

### Adding Custom Logic

Each agent can be extended with custom logic:

```python
from agents import ProcessorAgent

class CustomProcessor(ProcessorAgent):
    def process(self, question: str):
        # Add custom processing logic
        result = super().process(question)
        # Add additional processing
        return result
```

### Integrating AI Models

The framework is designed to easily integrate with AI models:

```python
# Example: Integrating with OpenAI
from agents import AnswererAgent
import openai

class AIAnswerer(AnswererAgent):
    def _generate_answer(self, question, question_type, intent, keywords):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": question}]
        )
        return response.choices[0].message.content
```

## Future Enhancements

- Integration with external AI APIs (OpenAI, Anthropic, etc.)
- Support for multiple languages
- Persistent storage for questions and answers
- Advanced natural language processing
- Machine learning model integration
- REST API interface
- Web interface
- Database integration for knowledge base
- User feedback loop
- A/B testing capabilities

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is available for educational and research purposes.

## Authors

- Emmanuel Adio and Team

## Support

For questions or issues, please open an issue on the GitHub repository.