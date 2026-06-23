import os
import httpx
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()
groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# --- 1. The CRAG Bouncer ---
async def evaluate_context(query: str, documents: list) -> str:
    """Evaluates if the retrieved documents actually contain the answer."""
    context_text = "\n\n".join([f"Doc {i+1}: {doc.get('title', '')}\n{doc.get('text', '')}" for i, doc in enumerate(documents)])
    
    prompt = f"""
    You are a strict relevance evaluator. 
    User Query: "{query}"
    Provided Context: 
    {context_text}
    
    Does the provided context contain sufficient information to accurately answer the query?
    Output exactly ONE WORD: "Correct" if it does, "Incorrect" if it does not, or "Ambiguous" if it partially does.
    """
    
    try:
        res = await groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.0,
            max_tokens=10
        )
        judgment = res.choices[0].message.content.strip().replace('"', '')
        return judgment if judgment in ["Correct", "Incorrect", "Ambiguous"] else "Ambiguous"
    except:
        return "Ambiguous"

# --- 2. Fallback Web Search ---
async def fallback_web_search(query: str) -> list:
    print(f"\n🌐 CRAG Fallback Triggered: Searching the web for '{query}'...")
    if not TAVILY_API_KEY or TAVILY_API_KEY == "dummy_key_for_testing":
        print("⚠️ Dummy Tavily key detected. Simulating web search fallback...")
        return [{"title": "Simulated Web Result", "text": "This is a simulated web search result because a dummy API key is in use.", "source": "Web Fallback"}]
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://api.tavily.com/search",
                json={"api_key": TAVILY_API_KEY, "query": query, "search_depth": "basic"},
                timeout=5.0
            )
            if response.status_code != 200:
                print(f"⚠️ Tavily API Error: {response.status_code}. Using empty web context.")
                return []
                
            results = response.json().get("results", [])
            return [{"title": r["title"], "text": r["content"], "source": "Web Fallback"} for r in results]
        except Exception as e:
            print(f"❌ Web search failed: {e}")
            return []

# --- 3. The Streaming Synthesizer ---
async def generate_final_answer_stream(query: str, final_context: list):
    """Streams the final answer token-by-token."""
    context_text = "\n\n".join([f"Source: {doc.get('title', '')}\n{doc.get('text', '')}" for doc in final_context])
    
    prompt = f"""
    You are a highly intelligent research assistant (like Perplexity).
    Answer the user's query using ONLY the provided context. 
    Cite your sources inline using brackets, e.g., [Source Name].
    If the context doesn't contain the answer, say "I cannot answer this based on the retrieved documents."
    
    Context:
    {context_text}
    
    User Query: "{query}"
    """
    
    stream = await groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile", # Using Groq's heavy 70B model for final synthesis
        temperature=0.3,
        stream=True 
    )
    
    async for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content