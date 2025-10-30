"""
API Extension Example
Shows how to extend the AI Agent System with a REST API interface.

This is a template/example file showing how to create a simple API.
To use this, you would need to install Flask:
    pip install flask

Then run:
    python examples/api_example.py
"""

# Note: This requires Flask to be installed
# pip install flask

from flask import Flask, request, jsonify
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import AIAgentSystem

# Initialize Flask app
app = Flask(__name__)

# Initialize AI Agent System
system = AIAgentSystem()


@app.route('/api/question', methods=['POST'])
def process_question():
    """
    Process a question through the AI Agent System.
    
    Expected JSON payload:
    {
        "question": "Your question here",
        "full_pipeline": false  # optional, defaults to false
    }
    
    Returns:
    {
        "success": true,
        "data": { ... },
        "error": null
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({
                'success': False,
                'data': None,
                'error': 'Missing required field: question'
            }), 400
        
        question = data['question']
        full_pipeline = data.get('full_pipeline', False)
        
        # Process the question
        result = system.process_question(question, return_full_pipeline=full_pipeline)
        
        return jsonify({
            'success': True,
            'data': result,
            'error': None
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'data': None,
            'error': str(e)
        }), 500


@app.route('/api/answer', methods=['POST'])
def get_answer():
    """
    Get just the answer for a question.
    
    Expected JSON payload:
    {
        "question": "Your question here"
    }
    
    Returns:
    {
        "success": true,
        "answer": "The answer text",
        "error": null
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({
                'success': False,
                'answer': None,
                'error': 'Missing required field: question'
            }), 400
        
        question = data['question']
        answer = system.get_answer(question)
        
        return jsonify({
            'success': True,
            'answer': answer,
            'error': None
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'answer': None,
            'error': str(e)
        }), 500


@app.route('/api/evaluate', methods=['POST'])
def evaluate_question():
    """
    Get evaluation summary for a question.
    
    Expected JSON payload:
    {
        "question": "Your question here"
    }
    
    Returns:
    {
        "success": true,
        "evaluation": { ... },
        "error": null
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({
                'success': False,
                'evaluation': None,
                'error': 'Missing required field: question'
            }), 400
        
        question = data['question']
        evaluation = system.get_evaluation_summary(question)
        
        return jsonify({
            'success': True,
            'evaluation': evaluation,
            'error': None
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'evaluation': None,
            'error': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'AI Agent System API'
    }), 200


@app.route('/', methods=['GET'])
def index():
    """API documentation."""
    return jsonify({
        'name': 'AI Agent System API',
        'version': '1.0.0',
        'endpoints': {
            '/api/question': {
                'method': 'POST',
                'description': 'Process a question through all three agents',
                'payload': {
                    'question': 'string (required)',
                    'full_pipeline': 'boolean (optional, default: false)'
                }
            },
            '/api/answer': {
                'method': 'POST',
                'description': 'Get just the answer for a question',
                'payload': {
                    'question': 'string (required)'
                }
            },
            '/api/evaluate': {
                'method': 'POST',
                'description': 'Get evaluation summary for a question',
                'payload': {
                    'question': 'string (required)'
                }
            },
            '/api/health': {
                'method': 'GET',
                'description': 'Health check endpoint'
            }
        }
    }), 200


if __name__ == '__main__':
    print("\n" + "="*80)
    print("AI AGENT SYSTEM - REST API")
    print("="*80)
    print("\nStarting Flask server...")
    print("\nAvailable endpoints:")
    print("  - GET  /              : API documentation")
    print("  - GET  /api/health    : Health check")
    print("  - POST /api/question  : Process question (full or simplified)")
    print("  - POST /api/answer    : Get answer only")
    print("  - POST /api/evaluate  : Get evaluation summary")
    print("\nExample curl commands:")
    print('  curl -X POST http://localhost:5000/api/answer \\')
    print('       -H "Content-Type: application/json" \\')
    print('       -d \'{"question": "What is AI?"}\'')
    print("\n" + "="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
