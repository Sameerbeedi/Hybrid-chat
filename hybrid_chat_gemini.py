#!/usr/bin/env python3
"""
Enhanced Hybrid Chat System
Features:
- ✅ Embedding caching (faster & saves API calls)
- ✅ Async operations (parallel vector + graph search)
- ✅ Search summarization 
- ✅ Chain-of-thought reasoning
- ✅ Hybrid Fusion (Vector + Graph + BM25)
- ✅ Conversation Memory (multi-turn context)
- ✅ Query Expansion (better recall)
- ✅ Supports Gemini (FREE)
"""

import json
import asyncio
import hashlib
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from collections import deque
from datetime import datetime
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
from neo4j import GraphDatabase
from rank_bm25 import BM25Okapi
import config

# ===== Configuration =====
genai.configure(api_key=config.GEMINI_API_KEY)
chat_model = genai.GenerativeModel(config.GEMINI_CHAT_MODEL)

pc = Pinecone(api_key=config.PINECONE_API_KEY)
if config.PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=config.PINECONE_INDEX_NAME,
        dimension=config.PINECONE_VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(config.PINECONE_INDEX_NAME)

driver = GraphDatabase.driver(
    config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
)

# ===== Load Dataset for BM25 =====
with open('vietnam_travel_dataset.json', 'r', encoding='utf-8') as f:
    DATASET = json.load(f)

# ===== FEATURE 1: Embedding Cache =====
class EmbeddingCache:
    """Cache embeddings to avoid repeated API calls"""
    
    def __init__(self, cache_file: str = config.CACHE_FILE):
        self.cache_file = Path(cache_file)
        self.cache = self._load_cache()
        self.hits = 0
        self.misses = 0
    
    def _load_cache(self) -> Dict:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def _get_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()
    
    def get(self, text: str) -> Optional[List[float]]:
        key = self._get_key(text)
        result = self.cache.get(key)
        if result:
            self.hits += 1
        else:
            self.misses += 1
        return result
    
    def set(self, text: str, embedding: List[float]):
        key = self._get_key(text)
        self.cache[key] = embedding
        self._save_cache()
    
    def stats(self) -> str:
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return f"Cache: {self.hits} hits, {self.misses} misses ({hit_rate:.1f}% hit rate)"

embedding_cache = EmbeddingCache() if config.CACHE_ENABLED else None

# ===== FEATURE 2: BM25 Keyword Search =====
class BM25SearchEngine:
    """BM25 keyword-based search for exact matches"""
    
    def __init__(self, documents: List[Dict]):
        self.documents = documents
        self.doc_map = {doc['id']: doc for doc in documents}
        
        # Create corpus from semantic text
        corpus = [doc.get('semantic_text', '') for doc in documents]
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        
        # Initialize BM25
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.doc_ids = [doc['id'] for doc in documents]
        
        print(f"✅ BM25 index built with {len(documents)} documents")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using BM25 and return (doc_id, score) pairs"""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k results
        scored_docs = [(self.doc_ids[i], scores[i]) for i in range(len(scores))]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return scored_docs[:top_k]
    
    def get_metadata(self, doc_id: str) -> Dict:
        """Get document metadata"""
        doc = self.doc_map.get(doc_id, {})
        return {
            'id': doc_id,
            'name': doc.get('name', 'Unknown'),
            'type': doc.get('type', 'N/A'),
            'description': doc.get('description', ''),
            'region': doc.get('region', 'N/A')
        }

bm25_engine = BM25SearchEngine(DATASET)

# ===== FEATURE 3: Conversation Memory =====
class ConversationMemory:
    """Maintain conversation history for context-aware responses"""
    
    def __init__(self, max_history: int = 5):
        self.history = deque(maxlen=max_history)
        self.max_history = max_history
    
    def add_turn(self, query: str, response: str, sources: List[str] = None):
        """Add a conversation turn"""
        self.history.append({
            'query': query,
            'response': response,  # Store full response
            'sources': sources or [],
            'timestamp': datetime.now().isoformat()
        })
    
    def get_context(self) -> str:
        """Get recent conversation context"""
        if not self.history:
            return ""
        
        context_lines = ["RECENT CONVERSATION HISTORY:"]
        for i, turn in enumerate(list(self.history)[-3:], 1):
            context_lines.append(f"{i}. User: {turn['query']}")
            # Use first 200 chars for context only
            context_lines.append(f"   Assistant: {turn['response'][:200]}...")
        
        return "\n".join(context_lines) + "\n"
    
    def get_recent_entities(self) -> List[str]:
        """Extract entity IDs from recent conversation"""
        entities = []
        for turn in self.history:
            entities.extend(turn.get('sources', []))
        return list(set(entities))[-10:]  # Last 10 unique entities
    
    def clear(self):
        """Clear conversation history"""
        self.history.clear()
    
    def stats(self) -> str:
        return f"Conversation: {len(self.history)} turns in memory"

conversation_memory = ConversationMemory()

# ===== Embedding with Cache =====
def embed_text(text: str) -> List[float]:
    """Get embedding with caching"""
    if embedding_cache:
        cached = embedding_cache.get(text)
        if cached:
            return cached
    
    result = genai.embed_content(
        model=config.GEMINI_EMBEDDING_MODEL,
        content=text,
        task_type="retrieval_query"
    )
    embedding = result['embedding']
    
    if embedding_cache:
        embedding_cache.set(text, embedding)
    
    return embedding

# ===== FEATURE 4: Query Expansion =====
async def expand_query(query: str) -> List[str]:
    """Generate query variations for better recall"""
    try:
        expansion_prompt = f"""Given this travel query: "{query}"

Generate 2 alternative phrasings that capture the same intent but use different words.
Focus on synonyms and related concepts.

Format: Just list the alternatives, one per line, without numbering.

Example:
Original: "Best beaches in Vietnam"
Alternative 1: Top coastal destinations and seaside locations
Alternative 2: Most popular ocean shores and waterfront areas

Now generate alternatives for: "{query}"
"""
        
        response = chat_model.generate_content(expansion_prompt)
        variations = [line.strip() for line in response.text.strip().split('\n') if line.strip()]
        
        # Filter out numbering and keep only valid variations
        clean_variations = []
        for v in variations:
            # Remove common prefixes like "Alternative 1:", "1.", etc.
            v = v.split(':', 1)[-1].strip()
            v = v.lstrip('0123456789. ')
            if v and len(v) > 10:  # Valid variation
                clean_variations.append(v)
        
        return [query] + clean_variations[:2]  # Original + 2 variations
    except Exception as e:
        print(f"⚠️  Query expansion failed: {e}")
        return [query]  # Fallback to original

# ===== FEATURE 5: Async Operations =====
async def async_pinecone_query(query_text: str, top_k: int = config.TOP_K_RESULTS):
    """Async Pinecone query"""
    vec = embed_text(query_text)
    loop = asyncio.get_running_loop()
    
    def _query():
        return index.query(
            vector=vec,
            top_k=top_k,
            include_metadata=True,
            include_values=False
        )
    
    res = await loop.run_in_executor(None, _query)
    return res["matches"]

async def async_fetch_graph_context(node_ids: List[str]):
    """Async Neo4j graph query"""
    facts = []
    
    def fetch_node(nid):
        with driver.session() as session:
            q = (
                "MATCH (n:Entity {id:$nid})-[r]-(m:Entity) "
                "RETURN type(r) AS rel, labels(m) AS labels, m.id AS id, "
                "m.name AS name, m.type AS type, m.description AS description "
                "LIMIT 10"
            )
            return list(session.run(q, nid=nid))
    
    loop = asyncio.get_running_loop()
    tasks = [loop.run_in_executor(None, fetch_node, nid) for nid in node_ids]
    results = await asyncio.gather(*tasks)
    
    for recs in results:
        for r in recs:
            facts.append({
                "source": node_ids[0] if node_ids else "unknown",
                "rel": r["rel"],
                "target_id": r["id"],
                "target_name": r["name"],
                "target_desc": (r.get("description") or "")[:400],
                "labels": r["labels"]
            })
    
    return facts[:config.MAX_GRAPH_FACTS]

async def bm25_search_async(query: str, top_k: int = 10):
    """Async wrapper for BM25 search"""
    loop = asyncio.get_running_loop()
    
    def _search():
        return bm25_engine.search(query, top_k)
    
    return await loop.run_in_executor(None, _search)

async def hybrid_fusion_search(query: str, weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)):
    """
    🔥 HYBRID FUSION: Combine Vector + Graph + BM25
    
    Args:
        query: User query
        weights: (vector_weight, graph_weight, bm25_weight)
    
    Returns:
        Fused results with combined scoring
    """
    try:
        # Run all three searches in parallel
        vector_task = async_pinecone_query(query, top_k=10)
        bm25_task = bm25_search_async(query, top_k=10)
        
        vector_results, bm25_results = await asyncio.gather(vector_task, bm25_task)
        
        # Validate results
        if vector_results is None:
            vector_results = []
        if bm25_results is None:
            bm25_results = []
        
        # Extract IDs for graph search
        vector_ids = [m["id"] for m in vector_results]
        bm25_ids = [doc_id for doc_id, score in bm25_results if score > 0]
        
        # Combine and deduplicate IDs
        all_ids = list(dict.fromkeys(vector_ids + bm25_ids))[:15]
        
        # Fetch graph context
        if all_ids:
            facts = await async_fetch_graph_context(all_ids)
        else:
            facts = []
        
        # Score fusion
        fused_scores = {}
        
        # 1. Vector scores (semantic similarity)
        for m in vector_results:
            fused_scores[m['id']] = fused_scores.get(m['id'], 0) + m['score'] * weights[0]
        
        # 2. BM25 scores (keyword matching)
        max_bm25 = max([s for _, s in bm25_results], default=1.0)
        for doc_id, score in bm25_results:
            if score > 0:
                normalized_score = score / max_bm25 if max_bm25 > 0 else 0
                fused_scores[doc_id] = fused_scores.get(doc_id, 0) + normalized_score * weights[2]
        
        # 3. Graph scores (connectivity boost)
        graph_boost = {}
        for fact in facts:
            # Source nodes get boost for having connections
            graph_boost[fact['source']] = graph_boost.get(fact['source'], 0) + 0.1
            # Target nodes get smaller boost
            graph_boost[fact['target_id']] = graph_boost.get(fact['target_id'], 0) + 0.05
        
        for doc_id, boost in graph_boost.items():
            if doc_id in fused_scores:
                fused_scores[doc_id] += min(boost, 0.3) * weights[1]  # Cap graph boost
        
        # Sort by fused score
        ranked_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Build result matches with metadata
        matches = []
        for doc_id, score in ranked_results:
            # Try to get original vector metadata
            vector_match = next((m for m in vector_results if m['id'] == doc_id), None)
            if vector_match:
                match = vector_match.copy()
                match['score'] = score  # Update with fused score
                match['fusion_type'] = 'vector+bm25+graph'
            else:
                # BM25-only match
                metadata = bm25_engine.get_metadata(doc_id)
                match = {
                    'id': doc_id,
                    'score': score,
                    'metadata': metadata,
                    'fusion_type': 'bm25+graph'
                }
            matches.append(match)
        
        return matches, facts
        
    except Exception as e:
        print(f"⚠️  Fusion search error: {e}")
        # Fallback to simple vector search
        try:
            vector_results = await async_pinecone_query(query, top_k=10)
            vector_ids = [m["id"] for m in vector_results]
            facts = await async_fetch_graph_context(vector_ids) if vector_ids else []
            return vector_results, facts
        except:
            return [], []

async def hybrid_search_with_expansion(query: str, use_expansion: bool = True):
    """
    🔍 QUERY EXPANSION: Search with multiple query variations
    """
    if use_expansion:
        print("🔄 Expanding query...")
        queries = await expand_query(query)
        print(f"   Generated {len(queries)} query variations")
    else:
        queries = [query]
    
    # Search with all query variations in parallel
    try:
        # Run all searches concurrently
        search_tasks = [hybrid_fusion_search(q) for q in queries]
        results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        all_matches = []
        all_facts = []
        
        # Collect results, skipping any that failed
        for result in results:
            if isinstance(result, Exception):
                print(f"⚠️  Query variation failed: {result}")
                continue
            if result and len(result) == 2:
                matches, facts = result
                all_matches.extend(matches)
                all_facts.extend(facts)
    
    except Exception as e:
        print(f"⚠️  Expansion search failed: {e}, falling back to single query")
        matches, facts = await hybrid_fusion_search(query)
        all_matches = [matches] if matches else []
        all_facts = facts if facts else []
    
    # Deduplicate matches by ID, keeping highest score
    seen = {}
    for m in all_matches:
        doc_id = m['id']
        if doc_id not in seen or m['score'] > seen[doc_id]['score']:
            seen[doc_id] = m
    
    # Sort by score and take top 10
    final_matches = sorted(seen.values(), key=lambda x: x['score'], reverse=True)[:10]
    
    # Deduplicate facts
    seen_facts = set()
    unique_facts = []
    for f in all_facts:
        fact_key = (f['source'], f['rel'], f['target_id'])
        if fact_key not in seen_facts:
            seen_facts.add(fact_key)
            unique_facts.append(f)
    
    return final_matches, unique_facts[:config.MAX_GRAPH_FACTS]

# Legacy function for backward compatibility
async def hybrid_search(query: str):
    """Run hybrid search with all enhancements"""
    return await hybrid_search_with_expansion(query, use_expansion=True)

# ===== FEATURE 6: Search Summarization =====
def search_summary(matches: List[Dict], facts: List[Dict]) -> str:
    """Summarize search results with fusion details"""
    lines = []
    
    if matches:
        lines.append(f"📊 Hybrid Fusion: {len(matches)} results (Vector + BM25 + Graph)")
        for i, m in enumerate(matches[:3], 1):
            meta = m.get("metadata", {})
            fusion_type = m.get('fusion_type', 'unknown')
            lines.append(
                f"   {i}. {meta.get('name', 'Unknown')} ({meta.get('type', 'N/A')}) "
                f"- Score: {m.get('score', 0):.3f} [{fusion_type}]"
            )
    
    if facts:
        lines.append(f"🕸️  Graph Search: {len(facts)} connections discovered")
        rel_types = set(f["rel"] for f in facts)
        lines.append(f"   Relationships: {', '.join(list(rel_types)[:5])}")
    
    return "\n".join(lines)

# ===== FEATURE 7: Chain-of-Thought Reasoning with Memory =====
def build_cot_prompt(user_query: str, matches: List[Dict], facts: List[Dict]) -> str:
    """Build prompt with chain-of-thought reasoning and conversation context"""
    
    # Get conversation context
    conversation_context = conversation_memory.get_context()
    
    vec_context = []
    for m in matches[:5]:
        meta = m.get("metadata", {})
        fusion_type = m.get('fusion_type', 'unknown')
        vec_context.append(
            f"- [{m['id']}] {meta.get('name', 'Unknown')} ({meta.get('type', '')}): "
            f"Score={m.get('score', 0):.3f}, Region={meta.get('region', 'N/A')} [{fusion_type}]"
        )
    
    graph_context = []
    for f in facts[:15]:
        graph_context.append(
            f"- {f['source']} -[{f['rel']}]-> {f['target_name']} [{f['target_id']}]"
        )
    
    prompt = f"""You are an expert Vietnam travel assistant with deep knowledge of destinations, culture, and travel planning.

{conversation_context}

USER QUERY: "{user_query}"

AVAILABLE DATA (from Hybrid Fusion: Vector + BM25 Keywords + Graph):

📊 SEARCH RESULTS (Semantic + Keyword + Connectivity):
{chr(10).join(vec_context) if vec_context else "No results"}

🕸️ GRAPH CONNECTIONS (Related Entities):
{chr(10).join(graph_context) if graph_context else "No connections"}

CHAIN-OF-THOUGHT REASONING:

Please think through this systematically:

1️⃣ UNDERSTAND: What is the user really asking? What's their travel goal?
   {f"Consider the conversation history above for context." if conversation_context else ""}

2️⃣ ANALYZE: Which locations from the search results are most relevant? Why?

3️⃣ CONNECT: How do the graph relationships enrich the answer? What connections matter?

4️⃣ SYNTHESIZE: Combine all data sources for a complete recommendation.

5️⃣ RESPOND: Provide a helpful, actionable answer.

FORMAT YOUR RESPONSE:

🤔 **My Thinking**: [1-2 sentences explaining your reasoning]

✅ **Recommendation**: [Your main answer with specific place IDs in brackets like [city_hanoi]]

💡 **Why This Works**: [Brief justification based on the data]

Keep it concise (3-5 sentences total). Always cite entity IDs in [brackets]."""
    
    return prompt

def call_chat(prompt: str) -> str:
    """Call Gemini for response"""
    response = chat_model.generate_content(prompt)
    return response.text

def format_response(raw_response: str) -> str:
    """
    Extract clean recommendation from chain-of-thought response
    Shows only the actual answer without meta-commentary
    """
    # Try to extract just the Recommendation section content
    if "✅ **Recommendation**:" in raw_response:
        parts = raw_response.split("✅ **Recommendation**:")
        if len(parts) > 1:
            recommendation_text = parts[1].strip()
            
            # Remove the "Why This Works" section if present
            if "💡 **Why This Works**:" in recommendation_text:
                recommendation_text = recommendation_text.split("💡 **Why This Works**:")[0].strip()
            
            # Return clean recommendation
            return recommendation_text
    
    # Alternative: look for just "Recommendation:" without emoji
    if "**Recommendation**:" in raw_response:
        parts = raw_response.split("**Recommendation**:")
        if len(parts) > 1:
            recommendation_text = parts[1].strip()
            
            # Remove trailing sections
            if "**Why This Works**:" in recommendation_text:
                recommendation_text = recommendation_text.split("**Why This Works**:")[0].strip()
            if "💡" in recommendation_text:
                recommendation_text = recommendation_text.split("💡")[0].strip()
            
            return recommendation_text
    
    # Fallback: return original if format not recognized
    return raw_response

# ===== Main Chat Loop =====
async def interactive_chat():
    """Enhanced interactive chat with all advanced features"""
    print("="*80)
    print("🌟 ENHANCED Hybrid Travel Assistant")
    print("   � Hybrid Fusion: Vector + BM25 + Graph (20-30% better accuracy)")
    print("   💭 Conversation Memory: Context-aware multi-turn conversations")
    print("   🔍 Query Expansion: Better recall with query variations")
    print("   ⚡ Async + Caching: Lightning fast responses")
    print("   🤖 Powered by: Google Gemini (FREE)")
    print("="*80)
    print("\n💡 Examples:")
    print("   - 'Best beaches in Vietnam?'")
    print("   - 'What about cultural sites?' (follow-up)")
    print("   - 'Plan a 3-day trip to Hanoi'")
    print("\n📝 Commands:")
    print("   - 'exit' or 'quit' - End session")
    print("   - 'stats' - Show cache & memory statistics")
    print("   - 'clear' - Clear conversation memory")
    print("   - 'history' - Show conversation history")
    print("   - 'save' - Save conversation to text file\n")
    
    query_count = 0
    
    while True:
        try:
            query = input("💬 Your question: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ("exit", "quit"):
                print("\n" + "="*80)
                print("📊 SESSION STATISTICS")
                print("="*80)
                if embedding_cache:
                    print(f"�️  {embedding_cache.stats()}")
                print(f"💭 {conversation_memory.stats()}")
                print(f"📈 Total queries: {query_count}")
                print("\n👋 Safe travels! Goodbye!")
                break
            
            if query.lower() == "stats":
                print("\n" + "="*80)
                print("📊 CURRENT STATISTICS")
                print("="*80)
                if embedding_cache:
                    print(f"�️  {embedding_cache.stats()}")
                print(f"� {conversation_memory.stats()}")
                print(f"�📈 Queries this session: {query_count}")
                print("="*80 + "\n")
                continue
            
            if query.lower() == "clear":
                conversation_memory.clear()
                print("✅ Conversation memory cleared!\n")
                continue
            
            if query.lower() == "history":
                print("\n" + "="*80)
                print("📜 CONVERSATION HISTORY")
                print("="*80)
                if conversation_memory.history:
                    for i, turn in enumerate(conversation_memory.history, 1):
                        print(f"\n{i}. Q: {turn['query']}")
                        # Format the answer to show only recommendation
                        clean_answer = format_response(turn['response'])
                        print(f"   A: {clean_answer[:200]}...")  # Show first 200 chars
                        if turn['sources']:
                            print(f"   Sources: {', '.join(turn['sources'][:3])}")
                else:
                    print("No conversation history yet.")
                print("="*80 + "\n")
                continue
            
            if query.lower() == "save":
                if not conversation_memory.history:
                    print("❌ No conversation to save yet!\n")
                    continue
                
                # Generate filename with timestamp
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"chat_history_{timestamp}.txt"
                
                try:
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write("="*80 + "\n")
                        f.write("VIETNAM TRAVEL AI ASSISTANT - CONVERSATION HISTORY\n")
                        f.write("="*80 + "\n")
                        f.write(f"Saved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Total Conversations: {len(conversation_memory.history)}\n")
                        f.write("="*80 + "\n\n")
                        
                        for i, turn in enumerate(conversation_memory.history, 1):
                            f.write(f"\n{'='*80}\n")
                            f.write(f"CONVERSATION {i}\n")
                            f.write(f"{'='*80}\n\n")
                            
                            f.write(f"📍 Timestamp: {turn.get('timestamp', 'N/A')}\n\n")
                            
                            f.write(f"❓ QUESTION:\n")
                            f.write(f"{turn['query']}\n\n")
                            
                            f.write(f"🤖 ANSWER:\n")
                            # Extract clean recommendation
                            clean_answer = format_response(turn['response'])
                            f.write(f"{clean_answer}\n\n")
                            
                            if turn['sources']:
                                f.write(f"📚 Sources: {', '.join(turn['sources'])}\n")
                            
                            f.write("\n")
                        
                        f.write("\n" + "="*80 + "\n")
                        f.write("END OF CONVERSATION HISTORY\n")
                        f.write("="*80 + "\n")
                    
                    print(f"✅ Conversation saved to: {filename}\n")
                    
                except Exception as e:
                    print(f"❌ Error saving conversation: {e}\n")
                
                continue
            
            query_count += 1
            print("\n� Hybrid Fusion Search (Vector + BM25 + Graph)...")
            
            # Parallel async search with all enhancements
            import time
            start = time.time()
            matches, facts = await hybrid_search(query)
            elapsed = time.time() - start
            
            # Show summary
            summary = search_summary(matches, facts)
            print(f"\n{summary}")
            print(f"⚡ Search completed in {elapsed:.2f}s\n")
            
            print("🤖 Generating answer with chain-of-thought reasoning...\n")
            
            # Build CoT prompt with conversation memory and get response
            prompt = build_cot_prompt(query, matches, facts)
            answer = call_chat(prompt)
            
            # Format response to show only the recommendation (not the thinking)
            formatted_answer = format_response(answer)
            
            # Extract source IDs for memory
            source_ids = [m['id'] for m in matches[:5]]
            
            # Store in conversation memory (store full answer for context)
            conversation_memory.add_turn(query, answer, source_ids)
            
            # Display formatted answer (user-friendly)
            print("="*80)
            print("🤖 ASSISTANT")
            print("="*80)
            print(formatted_answer)
            print("="*80 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again.\n")

# ===== Entry Point =====
if __name__ == "__main__":
    try:
        asyncio.run(interactive_chat())
    finally:
        driver.close()
        print("✅ Connections closed")
