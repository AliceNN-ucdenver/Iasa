# Comprehensive LLM Engineering Learning Notes

## LangChain
**Core Concept:** Framework for building applications with LLMs through composable components.

**Detailed Explanation:**
- **Chain Architecture:** Components connect via input/output interfaces, allowing complex workflows
- **Abstractions:** Provides unified interfaces for different LLM providers (OpenAI, Anthropic, local models)
- **Memory Systems:** Built-in conversation memory options (ConversationBufferMemory, ConversationSummaryMemory)
- **Callback System:** Enables logging, tracing, and streaming across the entire pipeline

**Code Example:**
```python
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate

# Define a custom prompt
template = """
Answer the question based only on the following context:
{context}

Question: {question}
Answer: """

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# Create a chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

# Run the chain
result = qa_chain({"query": "How does photosynthesis work?"})
```

**Advanced Implementation Pattern:**
- The hybrid router chain pattern allows dynamically selecting different sub-chains based on query classification:
```python
from langchain.chains.router import MultiPromptRouter
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.prompts import PromptTemplate

destinations = [
    {"name": "technical", "description": "Questions about technical specifications or how technology works"},
    {"name": "pricing", "description": "Questions about product pricing, discounts, or payment plans"},
    {"name": "support", "description": "Questions about customer support or technical issues"}
]

router_prompt = PromptTemplate(
    template="Route this query to the appropriate destination:\n\nQuery: {query}\n\n{destinations}",
    input_variables=["query"],
    partial_variables={"destinations": destinations}
)

router_chain = LLMRouterChain.from_llm(llm, router_prompt, RouterOutputParser())

chains = {
    "technical": technical_chain,
    "pricing": pricing_chain,
    "support": support_chain
}

routing_chain = MultiPromptRouter(
    router_chain=router_chain,
    destination_chains=chains,
    default_chain=general_chain
)

response = routing_chain.run("How much does the premium plan cost?")
```

## LlamaIndex
**Core Concept:** Framework specialized for building LLM applications with custom data sources.

**Detailed Explanation:**
- **Indexing Pipeline:** Documents → Chunking → Embedding → Storage
- **Query Pipeline:** Query → Retrieval → Response Synthesis
- **Document Processing:** Advanced chunking strategies (semantic, fixed-size, overlapping)
- **Index Types:**
  - Vector Store Index: Dense vector similarity search
  - Summary Index: Hierarchical summaries of documents
  - Tree Index: Hierarchical organization for traversal
  - Knowledge Graph Index: Structured relationships between entities

**Code Example:**
```python
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.node_parser import SimpleNodeParser
from llama_index.llms import OpenAI

# Custom chunking strategy
node_parser = SimpleNodeParser.from_defaults(
    chunk_size=1024,
    chunk_overlap=200
)

# Load and process documents
documents = SimpleDirectoryReader('data/').load_data()
nodes = node_parser.get_nodes_from_documents(documents)

# Create service context with specific model
service_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-4", temperature=0.1),
    embed_model="local:BAAI/bge-small-en-v1.5"
)

# Build index
index = VectorStoreIndex(nodes, service_context=service_context)

# Create query engine with metadata filtering
query_engine = index.as_query_engine(
    similarity_top_k=3,
    filters={"source": "authoritative_document.pdf"},
    response_mode="compact"
)

# Query
response = query_engine.query("What are the key requirements for compliance?")
```

**Advanced Implementation Pattern:**
- Router query engine pattern for handling different document domains:
```python
from llama_index.query_engine import RouterQueryEngine
from llama_index.selectors import LLMSingleSelector

# Create query engines for different domains
finance_engine = finance_index.as_query_engine()
technical_engine = technical_index.as_query_engine()
policy_engine = policy_index.as_query_engine()

# Define selector with descriptions
selector = LLMSingleSelector.from_defaults(
    OpenAI(temperature=0),
    [
        {"name": "finance", "description": "Questions about financial data, revenue, costs, or budgets"},
        {"name": "technical", "description": "Questions about technical specifications, implementations, or how systems work"},
        {"name": "policy", "description": "Questions about company policies, procedures, or guidelines"}
    ]
)

# Create router
router_engine = RouterQueryEngine(
    selector=selector,
    query_engines={
        "finance": finance_engine,
        "technical": technical_engine,
        "policy": policy_engine
    },
    default_engine=general_engine
)

response = router_engine.query("What was the revenue growth in Q2 2024?")
```

## Retrieval Augmented Generation (RAG)
**Core Concept:** Paradigm for enhancing LLM responses with external knowledge sources.

**Detailed Explanation:**
- **Basic RAG Pipeline:** 
  1. Index documents into vector database
  2. Embed user query
  3. Retrieve relevant documents
  4. Create prompt with retrieved context
  5. Generate response with LLM

- **Advanced RAG Patterns:**
  - **Hypothetical Document Embeddings (HyDE):** Generate a hypothetical answer, embed it, then search
  - **Recursive Retrieval:** Use initial response to refine and perform additional searches
  - **Multi-Query Retrieval:** Generate multiple query variations to improve recall
  - **Re-ranking:** Two-stage retrieval with initial broad search followed by precision ranking

**Code Example:**
```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma

# Create vector database
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(documents, embeddings)

# Create multi-query retriever (generates query variations)
llm = OpenAI(temperature=0)
retriever = MultiQueryRetriever.from_llm(
    vectordb.as_retriever(search_kwargs={"k": 2}),
    llm
)

# Create QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

# Run query
response = qa.run("What are the environmental impacts of solar panel production?")
```

**Advanced Implementation Pattern:**
- Parent-child chunking with hybrid search:
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

# Create text splitters for different granularities
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

# Create vector store and document store
vectorstore = Chroma(embedding_function=embeddings)
docstore = InMemoryStore()

# Create parent-document retriever
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    parent_splitter=parent_splitter,
    child_splitter=child_splitter,
    search_kwargs={"k": 5}
)

# Add documents to retriever
retriever.add_documents(documents)

# Hybrid search with BM25 and vector search
results = retriever.get_relevant_documents(
    "renewable energy policy",
    search_type="hybrid",
    alpha=0.5  # Weight between keyword and semantic search
)
```

## Microservices
**Core Concept:** Architecture pattern for building scalable applications as independent, deployable services.

**Detailed Explanation:**
- **Service Boundaries:** Each service focuses on specific functionality (embedding, retrieval, inference)
- **Communication Patterns:**
  - Synchronous: REST, gRPC
  - Asynchronous: Kafka, RabbitMQ
- **Data Consistency:** Eventual consistency across services
- **Deployment:** Docker containers orchestrated with Kubernetes
- **Scaling Strategies:** Horizontal scaling for handling variable load

**Code Example (Docker Compose):**
```yaml
version: '3'
services:
  embedding-service:
    build: ./embedding-service
    ports:
      - "8001:8001"
    environment:
      - MODEL_NAME=BAAI/bge-base-en-v1.5
    volumes:
      - ./models:/app/models
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 8G

  vector-db:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  llm-inference:
    build: ./inference-service
    ports:
      - "8000:8000"
    environment:
      - MODEL_ID=gpt-4-turbo
      - API_KEY=${OPENAI_API_KEY}
    deploy:
      replicas: 3

  api-gateway:
    build: ./api-gateway
    ports:
      - "80:80"
    depends_on:
      - embedding-service
      - vector-db
      - llm-inference

volumes:
  qdrant_data:
```

**Advanced Implementation Pattern:**
- Circuit breaker pattern for handling service failures:
```python
from fastapi import FastAPI, HTTPException
from circuitbreaker import circuit
import httpx

app = FastAPI()
client = httpx.AsyncClient()

@circuit(failure_threshold=3, recovery_timeout=30, fallback_function=lambda x: {"error": "Service temporarily unavailable"})
async def call_llm_service(prompt):
    response = await client.post(
        "http://llm-inference:8000/generate",
        json={"prompt": prompt, "max_tokens": 500},
        timeout=10.0
    )
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="LLM service error")
    return response.json()

@app.post("/ask")
async def ask_question(query: str):
    try:
        # Get embeddings
        embed_response = await client.post(
            "http://embedding-service:8001/embed",
            json={"text": query},
            timeout=5.0
        )
        embeddings = embed_response.json()["embedding"]
        
        # Retrieve documents
        retrieval_response = await client.post(
            "http://vector-db:6333/retrieve",
            json={"vector": embeddings, "limit": 3},
            timeout=5.0
        )
        documents = retrieval_response.json()["documents"]
        
        # Generate response with LLM
        context = "\n".join([doc["text"] for doc in documents])
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        
        llm_response = await call_llm_service(prompt)
        return {"answer": llm_response["text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## State-of-the-art LLMs (SOTA)
**Core Concept:** Latest cutting-edge large language models pushing capabilities forward.

**Detailed Explanation:**
- **Architectures:** Predominantly decoder-only transformers with increasing parameter counts
- **Key Capabilities:**
  - In-context learning
  - Chain-of-thought reasoning
  - Instruction following
  - Tool use and function calling
  - Multimodal understanding (text, images, audio)
- **Performance Dimensions:**
  - Reasoning
  - Knowledge
  - Instruction-following
  - Factuality
  - Safety

**Comparison of Major Models:**

| Model | Parameters | Architecture | Key Strengths | Limitations |
|-------|------------|--------------|---------------|-------------|
| GPT-4 | ~1.7T (estimated) | Decoder-only | Reasoning, instruction following, multimodal | Cost, closed-source |
| Claude 3 | Not disclosed | Decoder-only | Long context, helpfulness, safety | Less tooling support |
| Llama 3 | 8B-70B+ | Decoder-only | Open-weights, fine-tunable | Resource requirements |
| Gemini | Not disclosed | Unknown (likely decoder) | Multimodal understanding | Less developer ecosystem |
| Mixtral | 8x7B (mixture) | Mixture of Experts | Efficiency, strong reasoning | Narrow context window |

**Code Example (Using Multiple Models):**
```python
import anthropic
import openai
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Claude
claude_client = anthropic.Anthropic()
claude_response = claude_client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1000,
    temperature=0.1,
    system="You are a helpful assistant with expertise in quantum physics.",
    messages=[
        {"role": "user", "content": "Explain quantum entanglement and its applications"}
    ]
)

# GPT-4
openai_client = openai.OpenAI()
gpt_response = openai_client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant with expertise in quantum physics."},
        {"role": "user", "content": "Explain quantum entanglement and its applications"}
    ],
    temperature=0.1,
    max_tokens=1000
)

# Llama 3 (local)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-70b-chat-hf")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-70b-chat-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

prompt = "[INST] Explain quantum entanglement and its applications [/INST]"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    inputs.input_ids,
    max_new_tokens=1000,
    temperature=0.1,
    do_sample=True
)

llama_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Compare responses across models
print("Claude response:", claude_response.content[0].text)
print("GPT-4 response:", gpt_response.choices[0].message.content)
print("Llama 3 response:", llama_response)
```

**Advanced Implementation Pattern:**
- Model fallback chain with performance monitoring:
```python
import time
import logging
from functools import wraps

# Performance metrics decorator
def log_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logging.info(f"{func.__name__} completed in {execution_time:.2f}s")
            return result, execution_time, None
        except Exception as e:
            execution_time = time.time() - start_time
            logging.error(f"{func.__name__} failed after {execution_time:.2f}s: {str(e)}")
            return None, execution_time, e
    return wrapper

# Model interfaces
@log_performance
def query_primary_model(prompt, max_tokens=1000):
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.1
    )
    return response.choices[0].message.content

@log_performance
def query_fallback_model(prompt, max_tokens=1000):
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens
    )
    return response.content[0].text

@log_performance
def query_last_resort_model(prompt, max_tokens=1000):
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.1
    )
    return response.choices[0].message.content

# Fallback chain
def resilient_query(prompt, max_tokens=1000, timeout=10.0):
    # Try primary model
    result, exec_time, error = query_primary_model(prompt, max_tokens)
    if result and exec_time < timeout:
        return {"response": result, "model": "gpt-4-turbo", "time": exec_time}
    
    logging.warning("Primary model failed or timed out, trying fallback")
    
    # Try fallback model
    result, exec_time, error = query_fallback_model(prompt, max_tokens)
    if result:
        return {"response": result, "model": "claude-3-sonnet", "time": exec_time}
    
    logging.warning("Fallback model failed, trying last resort")
    
    # Try last resort model
    result, exec_time, error = query_last_resort_model(prompt, max_tokens)
    if result:
        return {"response": result, "model": "gpt-3.5-turbo", "time": exec_time}
    
    # All models failed
    return {"response": "Unable to generate response at this time.", "model": "none", "time": 0}

# Example usage
response = resilient_query("Explain the theory of relativity in simple terms")
print(f"Response from {response['model']} in {response['time']:.2f}s:")
print(response["response"])
```

## Instruction Fine-tuning
**Core Concept:** Training pre-trained models on instruction-specific datasets to improve performance on practical tasks.

**Detailed Explanation:**
- **Process:**
  1. Start with pre-trained model
  2. Create instruction-following dataset
  3. Fine-tune with specialized learning objectives
  4. Evaluate on practical tasks

- **Dataset Types:**
  - Human-created instruction pairs
  - Self-instruct (model generated instructions)
  - Preference datasets (chosen vs. rejected responses)

- **Methods:**
  - Supervised Fine-tuning (SFT)
  - Reinforcement Learning from Human Feedback (RLHF)
  - Direct Preference Optimization (DPO)
  - Constitutional AI (CAI)

**Code Example (Using TRL Library):**
```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer
from peft import LoraConfig

# Load base model
model_name = "meta-llama/Llama-3-8b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Format instruction dataset
def formatting_func(example):
    output_text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
    return output_text

# Configure LoRA for parameter-efficient fine-tuning
peft_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

# Load dataset
dataset = load_dataset("tatsu-lab/alpaca", split="train")

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
    formatting_func=formatting_func,
    peft_config=peft_config
)

# Train
trainer.train()

# Save model
trainer.save_model("./llama3-instruction-tuned")
```

**Advanced Implementation Pattern:**
- Constitutional AI filtering with preference data:
```python
from trl import DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Load base model
model_name = "meta-llama/Llama-3-8b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load constitutional guidelines
guidelines = """
1. Provide helpful, accurate, truthful, and harmless information.
2. Refuse to generate content that could cause harm or illegal activities.
3. Avoid political bias or extreme viewpoints.
4. Respect user privacy and do not ask for personal information.
5. Acknowledge limitations in knowledge or capabilities.
"""

# Create function to filter harmful responses
def constitutional_filter(prompt, response):
    rejection_prompt = f"""
    Evaluate if the following AI response violates any constitutional guidelines:
    
    Guidelines:
    {guidelines}
    
    User prompt: {prompt}
    AI response: {response}
    
    Does this response comply with the guidelines? Answer 'compliant' or 'non-compliant'.
    """
    
    # Here you would actually evaluate with another model call
    # For this example, we'll simulate with a placeholder
    return "compliant"  # Return actual evaluation in real implementation

# Load preference dataset (chosen vs rejected pairs)
dataset = load_dataset("your-dataset-name")

# Filter dataset based on constitutional guidelines
filtered_dataset = {
    "prompt": [],
    "chosen": [],
    "rejected": []
}

for item in dataset:
    # Check if chosen response is compliant
    chosen_status = constitutional_filter(item["prompt"], item["chosen"])
    rejected_status = constitutional_filter(item["prompt"], item["rejected"])
    
    # Only keep examples where chosen is compliant and rejected is non-compliant
    if chosen_status == "compliant" and rejected_status == "non-compliant":
        filtered_dataset["prompt"].append(item["prompt"])
        filtered_dataset["chosen"].append(item["chosen"])
        filtered_dataset["rejected"].append(item["rejected"])

# Fine-tune with DPO
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=model,
    tokenizer=tokenizer,
    train_dataset=filtered_dataset,
    beta=0.1,
    max_length=512,
    max_prompt_length=256,
    per_device_train_batch_size=4
)

# Train model
dpo_trainer.train()

# Save model
dpo_trainer.save_model("./constitutional-llama3")
```

## LangChain Expression Language (LCEL)
**Core Concept:** Declarative language for building composable chains and workflows in LangChain.

**Detailed Explanation:**
- **Core Operations:**
  - Piping (|): Sequential execution
  - Dictionary composition ({...}): Parallel execution
  - RunnablePassthrough: Pass values unchanged
  - RunnableLambda: Apply custom functions

- **Building Blocks:**
  - Prompt templates
  - LLMs/Chat models
  - Retrievers
  - Output parsers
  - Tools and agents

- **Execution Modes:**
  - Synchronous (.invoke())
  - Asynchronous (.ainvoke())
  - Batch (.batch())
  - Streaming (.stream())

**Code Example:**
```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
from langchain.schema.output_parser import StrOutputParser
from langchain.retrievers import WikipediaRetriever

# Create components
llm = ChatOpenAI(model="gpt-4")
retriever = WikipediaRetriever(top_k_results=3)

# Create template
template = """
Answer the question based on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Create chain using LCEL
# Step 1: Create parallel retrieval & question passing
retrieval_chain = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)

# Step 2: Complete RAG chain with pipe operator
rag_chain = (
    retrieval_chain
    | prompt
    | llm
    | StrOutputParser()
)

# Run chain
result = rag_chain.invoke("What is quantum computing?")
print(result)
```

**Advanced Implementation Pattern:**
- Multi-agent collaboration with LCEL:
```python
from langchain.agents import Tool
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import (
    RunnablePassthrough, 
    RunnableBranch, 
    RunnableLambda
)
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional

# Define output schemas
class SubQuestion(BaseModel):
    question: str = Field(description="A sub-question to be answered")
    requires_research: bool = Field(description="Whether this question requires research")

class QuestionDecomposition(BaseModel):
    sub_questions: List[SubQuestion] = Field(description="List of sub-questions")
    main_topic: str = Field(description="The main topic of the question")

# Create components
llm = ChatOpenAI(model="gpt-4", temperature=0)
research_llm = ChatOpenAI(model="gpt-4", temperature=0.1)
synthesis_llm = ChatOpenAI(model="gpt-4", temperature=0.2)

# Create tools
wikipedia = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=WikipediaQueryRun(api_wrapper=wikipedia).run,
    description="Search Wikipedia for factual information"
)

# Create decomposition parser
decomposition_parser = PydanticOutputParser(pydantic_object=QuestionDecomposition)

# Create decomposition prompt
decomposition_prompt = ChatPromptTemplate.from_template("""
You are an expert at breaking down complex questions into simpler sub-questions.
Decompose the following question into 2-4 sub-questions that would help answer the original question.
For each sub-question, indicate whether it requires research.

Question: {question}

{format_instructions}
""")

# Create research prompt
research_prompt = ChatPromptTemplate.from_template("""
You are a research assistant. Use the provided context to answer the question.
If the context doesn't contain the answer, say "I don't have enough information".

Context: {research_results}

Question: {sub_question}
""")

# Create synthesis prompt
synthesis_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant synthesizing information to answer a complex question.
Original question: {original_question}

Here are the answers to sub-questions:
{sub_question_answers}

Provide a comprehensive answer to the original question based on this information.
""")

# Define research function
def conduct_research(sub_question):
    if not sub_question.requires_research:
        return {"sub_question": sub_question.question, "answer": "No research needed"}
    
    research_results = wikipedia_tool.run(sub_question.question)
    return {
        "sub_question": sub_question.question,
        "research_results": research_results,
        "answer": research_prompt.invoke({
            "research_results": research_results,
            "sub_question": sub_question.question
        }) | research_llm | (lambda x: x.content)
    }

# Define decomposition chain
decomposition_chain = (
    RunnablePassthrough.assign(
        format_instructions=lambda _: decomposition_parser.get_format_instructions()
    )
    | decomposition_prompt
    | llm
    | (lambda x: x.content)
    | decomposition_parser.parse
)

# Define research chain
research_chain = (
    RunnableLambda(lambda x: [conduct_research(sq) for sq in x["decomposition"].sub_questions])
    | (lambda results: {"original_question": results[0]["original_question"], 
                        "sub_question_answers": "\n\n".join([f"Q: {r['sub_question']}\nA: {r['answer']}" for r in results])})
)

# Define synthesis chain
synthesis_chain = (
    synthesis_prompt
    | synthesis_llm
    | (lambda x: x.content)
)

# Create full chain
full_chain = (
    RunnablePassthrough.assign(decomposition=decomposition_chain)
    | research_chain
    | synthesis_chain
)

# Run the chain
response = full_chain.invoke({"question": "What are the environmental and economic impacts of renewable energy adoption?"})
print(response)
```

## Zero-shot Classification
**Core Concept:** Using LLMs to classify text without explicit training examples by providing instructions in natural language.

**Detailed Explanation:**
- **Key Benefits:**
  - No training data requirement
  - Adaptable to new categories
  - Natural language specification

- **Prompt Engineering Techniques:**
  - Direct classification: "Classify X as either A, B, or C"
  - Chain-of-thought: "Think step by step about why X belongs to category Y"
  - Few-shot examples in prompt

- **Common Applications:**
  - Sentiment analysis
  - Intent classification
  - Content moderation
  - Topic categorization

**Code Example:**
```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

# Define output schema
class Classification(BaseModel):
    category: str = Field(description="The category the text belongs to")
    confidence: float = Field(description="Confidence score between 0 and 1")
    reasoning: str = Field(description="Explanation for the classification")

# Create parser
parser = PydanticOutputParser(pydantic_object=Classification)

# Create prompt template
template = """
Classify the following text into one of these
