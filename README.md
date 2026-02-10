# PartD_GroupProject

## Testing the Processor Agent

### Prerequisites

1. **Set up MongoDB credentials**
   
   Create a `.env` file in the project root with your MongoDB connection details:
   ```
   MONGODB_URI="your_mongodb_connection_string"
   MONGODB_DB="partd_group"
   ```

2. **Install dependencies**
   ```bash
   pip install python-dotenv pymongo
   ```

### Running the Test

Execute the processor agent test from the project root:

```bash
python tests/test_processor.py
```

This will:
- Load example questions from `ExampleQuestions.txt`
- Process each question through the processor agent
- Validate the output structure contains all required fields
- Print the processed output for each question

### Expected Output

Each processed question returns a dictionary with:
- `raw_text` / `clean_text` - Original and normalised input
- `domain` - Detected domain (e.g., `location`, `course_info`)
- `intent` - Classified intent (e.g., `ask_directions`, `ask_fees`)
- `slots` - Extracted entities
- `retrieval_query` - Query string for the retrieval system
- `confidence` - Confidence scores for intent and domain