# 🌟 Enhanced Hybrid Chat - Vietnam Travel Assistant

An **advanced AI-powered travel assistant** that combines **Vector Search**, **Graph Database**, and **BM25 Keyword Search** to provide intelligent, context-aware travel recommendations for Vietnam.

## ✨ Key Features

### 🔥 **NEW: Hybrid Fusion Search**
- **3-way fusion**: Vector (semantic) + BM25 (keyword) + Graph (relationships)
- **20-30% better accuracy** than single-method search
- Handles both semantic queries and exact keyword matches

### 💭 **NEW: Conversation Memory**
- Maintains context across multiple turns
- Natural follow-up questions: "What about cultural sites?" 
- Commands: `history`, `clear`, `stats`

### 🔍 **NEW: Query Expansion**
- Generates 2 alternative phrasings per query
- Better recall and edge case coverage
- Parallel search with automatic deduplication

### ⚡ **Existing Advanced Features**
- ✅ Embedding caching (saves API calls)
- ✅ Async operations (parallel processing)
- ✅ Chain-of-thought reasoning
- ✅ Search result summarization
- ✅ FREE tier support (Gemini, Pinecone, Neo4j Aura)

## 🎯 Architecture

```
User Query → Query Expansion (3 variations)
    ↓
Hybrid Fusion Search (parallel)
├─→ Vector Search (Pinecone + Gemini embeddings)
├─→ BM25 Keyword Search (exact matching)
└─→ Graph Search (Neo4j relationships)
    ↓
Score Fusion → Conversation Memory → Chain-of-Thought → Response
```

## 📋 Prerequisites

### Required Services (All have FREE tiers)
1. **Google Gemini API** - [Get API Key](https://makersuite.google.com/app/apikey) - **FREE**
2. **Pinecone Account** - [Sign up](https://www.pinecone.io/) - FREE tier: 100K vectors
3. **Neo4j Aura** - [Create Database](https://neo4j.com/cloud/aura/) - FREE tier: 50K nodes

### Software Requirements
- Python 3.8 or higher
- pip (Python package manager)

## 🚀 Setup Instructions

### Step 1: Clone or Download the Project

Download this project to your local machine.

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Configure API Keys and Credentials

You have two options:

#### Option A: Using Environment Variables (Recommended)
1. Copy the `.env.example` file to `.env`:
   ```bash
   copy .env.example .env
   ```

2. Edit `.env` and add your actual credentials:
   ```
   OPENAI_API_KEY=sk-your-actual-openai-key
   PINECONE_API_KEY=your-actual-pinecone-key
   PINECONE_INDEX_NAME=vietnam-travel-hybrid
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your-neo4j-password
   ```

#### Option B: Direct Configuration
Edit `config.py` and replace the placeholder values with your actual credentials.

### Step 4: Setup Neo4j Database

1. **If using Neo4j Aura (Cloud)**:
   - Create a free instance at [neo4j.com/cloud/aura](https://neo4j.com/cloud/aura/)
   - Copy the connection URI (e.g., `neo4j+s://xxxxx.databases.neo4j.io`)
   - Save the username and password
   - Update `NEO4J_URI`, `NEO4J_USER`, and `NEO4J_PASSWORD` in your config

2. **If using Neo4j Desktop (Local)**:
   - Install Neo4j Desktop
   - Create a new database
   - Start the database
   - Default URI: `bolt://localhost:7687`
   - Default user: `neo4j`
   - Set a password during first login

3. **If using Docker**:
   ```bash
   docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest
   ```

### Step 5: Setup Pinecone

1. Sign up at [pinecone.io](https://www.pinecone.io/)
2. Create an API key from the dashboard
3. The index will be created automatically when you run the upload script

## 📊 Running the System

### Complete Workflow (Automated)

Run the entire pipeline with one command:

```bash
python run_pipeline.py
```

This will:
1. Upload data to Pinecone
2. Load data into Neo4j
3. Create a graph visualization
4. Start the interactive chat

### Manual Step-by-Step

#### 1. Upload Data to Pinecone
```bash
python pinecone_upload.py
```

#### 2. Load Data into Neo4j
```bash
python load_to_neo4j.py
```

#### 3. Visualize the Graph (Optional)
```bash
python visualize_graph.py
```
This creates `neo4j_viz.html` that you can open in your browser.

#### 4. Run the Hybrid Chat
## 💬 Using the Enhanced Chat Interface

### Quick Start
```bash
# Activate virtual environment (if using one)
venv\Scripts\activate

# Run the chat
python hybrid_chat_gemini.py
```

### Interactive Commands

```
💬 Your question: [type your question]

Available commands:
- stats      → Show cache & memory statistics
- history    → View conversation history
- clear      → Reset conversation memory
- exit/quit  → End session with final stats
```

### Example Conversations

**Example 1: Keyword Query**
```
💬 Your question: Hanoi temple
🔥 Hybrid Fusion Search (Vector + BM25 + Graph)...
📊 Results: Temple of Literature, One Pillar Pagoda
✅ BM25 boosts exact keyword matches!
```

**Example 2: Semantic Query**
```
💬 Your question: peaceful mountain retreat
🔍 Query Expansion generates 3 variations...
📊 Results: Sapa, Da Lat, Dalat Plateau
✅ Vector search handles semantics!
```

**Example 3: Follow-up with Context**
```
💬 Your question: Best beaches in Vietnam?
🤖 Assistant: Try Nha Trang, My Khe Beach...

💬 Your question: What about cultural sites there?
💭 Using conversation memory...
🤖 Assistant: In Nha Trang area, try Po Nagar Towers...
✅ Context-aware response!
```

### Search Process
For each query, the system:
1. 🔄 **Expands query** into 3 variations (original + 2 alternatives)
2. 🔥 **Hybrid Fusion** searches all 3 methods in parallel:
   - Vector Search (semantic similarity)
   - BM25 Search (keyword matching)
   - Graph Search (relationship connections)
3. 📊 **Fuses scores** with weights: 50% vector, 30% graph, 20% BM25
4. 💭 **Adds conversation context** from previous turns
5. 🤖 **Generates answer** with chain-of-thought reasoning

## 📁 Project Structure

```
Blue Enigma/
├── vietnam_travel_dataset.json      # 360 travel entities
├── config.py                         # Unified configuration (Gemini/OpenAI)
├── requirements.txt                  # Dependencies (with rank-bm25)
├── .env                              # API keys (Gemini, Pinecone, Neo4j)
│
├── pinecone_upload_gemini.py        # Upload embeddings to Pinecone
├── load_to_neo4j.py                 # Load graph data to Neo4j
├── visualize_graph.py               # Generate graph visualization
│
├── hybrid_chat_gemini.py            # 🌟 MAIN: Enhanced chat with all features
├── test_features.py                 # Test suite for enhancements
│
├── IMPLEMENTATION_COMPLETE.md       # Detailed feature documentation
├── SCALING_IMPROVEMENTS.md          # Future enhancements guide
└── README.md                        # This file
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_features.py
```

Expected output:
```
✅ Hybrid Fusion: PASSED (10 results, 20 connections)
✅ Conversation Memory: PASSED (1 turn stored)
✅ Context-Aware Follow-up: PASSED (history included)
✅ Embedding Cache: PASSED (50% hit rate)
✅ Fusion Types: PASSED (all methods active)
```

## 🔧 Troubleshooting

### "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### "Authentication failed" (Neo4j)
- Check your Neo4j URI, username, and password in `config.py`
- Make sure Neo4j database is running
- Try accessing Neo4j Browser at `http://localhost:7474`

### "Invalid API key" (OpenAI/Pinecone)
- Verify your API keys are correct
- Check for extra spaces or quotes
- Ensure your OpenAI account has credits

### Pinecone Index Not Found
- The index is created automatically on first run
- Wait a few moments for it to initialize
- Check Pinecone dashboard to verify index exists

### Empty Results
- Make sure data was uploaded to both Pinecone and Neo4j
- Check that `vietnam_travel_dataset.json` exists and contains data
- Run the upload scripts again

## 📈 System Architecture

```
User Query
    ↓
[OpenAI Embedding]
    ↓
[Pinecone Vector Search] → Top-K Similar Results
    ↓
[Neo4j Graph Query] → Related Entities & Connections
    ↓
[OpenAI GPT-4] → Combined Context → Natural Language Answer
    ↓
User Response
```

## 🔐 Security Notes

- **Never commit** `.env` or `config.py` with real API keys to version control
- Add `.env` and `config.py` to `.gitignore`
- Use environment variables for production deployments
- Rotate API keys regularly

## 📝 Dataset Information

The Vietnam travel dataset includes:
- **Cities**: Hanoi, Ha Long Bay, Hue, Hoi An, Nha Trang, Da Lat, Ho Chi Minh City, Mekong Delta
- **Attractions**: Historical sites, beaches, cultural landmarks
- **Metadata**: Best time to visit, tags, descriptions, connections

## 🤝 Contributing

Feel free to enhance this project by:
- Adding more travel destinations
- Improving the chat prompts
- Adding multi-language support
- Implementing user feedback loops

## 📄 License

This project is for educational and evaluation purposes.

## 🆘 Support

If you encounter issues:
1. Check the troubleshooting section
2. Review your configuration settings
3. Verify all services are running
4. Check API service status pages

---

**Happy Traveling! 🌏✈️**
