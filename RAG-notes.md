for i, n_neighbors in enumerate(n_neighbors_values):
        key = f"umap_neighbors_{n_neighbors}"
        result = results[key]["result"]
        time_taken = results[key]["time"]
        
        scatter = axes[i].scatter(result[:, 0], result[:, 1], c=pd.factorize(labels)[0], cmap='viridis', alpha=0.8)
        axes[i].set_title(f'UMAP (n_neighbors: {n_neighbors}, Time: {time_taken:.2f}s)')
    
    plt.tight_layout()
    plt.savefig("umap_neighbors_tuning.png")
    
    # Visualize UMAP min_dist results
    fig, axes = plt.subplots(1, len(min_dist_values), figsize=(20, 5))
    
    for i, min_dist in enumerate(min_dist_values):
        key = f"umap_mindist_{min_dist}"
        result = results[key]["result"]
        time_taken = results[key]["time"]
        
        scatter = axes[i].scatter(result[:, 0], result[:, 1], c=pd.factorize(labels)[0], cmap='viridis', alpha=0.8)
        axes[i].set_title(f'UMAP (min_dist: {min_dist}, Time: {time_taken:.2f}s)')
    
    plt.tight_layout()
    plt.savefig("umap_mindist_tuning.png")
    
    return results

# Run parameter tuning
# tuning_results = tune_parameters(embeddings, categories)
```

This detailed guide covers all the critical topics for LLM engineering with practical code examples, implementation patterns, and detailed explanations. Each section provides both foundational knowledge and advanced implementation techniques that will help you master these concepts for your certification interview.

Key takeaways across all topics:

1. **Framework Understanding**: Master LangChain, LlamaIndex, and LCEL for building modular, maintainable LLM applications
2. **Embedding Techniques**: Understand bi-encoders vs. cross-encoders and how to optimize vector retrieval
3. **RAG Implementation**: Learn advanced patterns beyond basic retrieval, including reranking, query transformation, and hybrid search
4. **Evaluation Methods**: Implement comprehensive evaluation frameworks to measure and improve system performance
5. **Optimization Strategies**: Apply techniques like chunking optimization, vector database tuning, and model cascading for production systems

The examples provide a mix of basic implementations that demonstrate core concepts and advanced patterns that show how to solve real-world challenges in production environments.
Blockchain technology works as a distributed database or ledger shared among computer network nodes. It stores information in digital format, with the key innovation being that it guarantees the fidelity and security of data records without requiring a trusted third party. The blockchain consists of blocks, each containing a number of transactions. Once a block is filled with transactions, it's closed and linked to the previously filled block, creating a chain of blocks (hence 'blockchain'). New transactions are verified by network nodes through cryptography and added to the chain. This system is secure because altering any piece of information would require changing all subsequent blocks and the majority of the network would need to agree to the change."
    },
    {
        "query": "What are the benefits of intermittent fasting?",
        "reference_answer": "Intermittent fasting may offer several health benefits, including weight loss, improved metabolic health, reduced inflammation, potential increased longevity, enhanced heart health, improved brain function, and possible cancer prevention. It works primarily by giving the body time to process nutrients and repair cells.",
        "model_answer": "Intermittent fasting has gained popularity for several potential health benefits. Research suggests it may help with weight loss by limiting calorie intake and improving metabolic rates. It appears to improve insulin sensitivity, which can help manage blood sugar levels and potentially reduce risk of type 2 diabetes. Some studies indicate it may reduce inflammation markers and oxidative stress in the body. There's also evidence it promotes cellular repair through autophagy, where cells remove damaged components. Additionally, it may support heart health by improving blood pressure, cholesterol levels, and triglycerides. However, benefits vary between individuals, and it's not suitable for everyone, especially those with certain medical conditions."
    }
]

# Initialize evaluator models
llm = ChatOpenAI(model="gpt-4", temperature=0)

# 1. LLM as judge evaluation
def llm_judge_evaluation(data, criteria):
    # Create prompt template
    prompt_template = PromptTemplate(
        input_variables=["query", "answer", "reference", "criterion"],
        template="""
        You are an expert evaluator for language model outputs. Your task is to evaluate a model's answer against a reference answer based on a specific criterion.

        Query: {query}
        
        Model Answer: {answer}
        
        Reference Answer: {reference}
        
        Criterion: {criterion}
        
        First, analyze both the model answer and reference answer in relation to the criterion.
        Then, score the model answer on a scale of 1-5, where:
        1: Poor - Fails to meet the criterion
        2: Fair - Partially meets the criterion with significant issues
        3: Good - Mostly meets the criterion with minor issues
        4: Very Good - Fully meets the criterion with minimal issues
        5: Excellent - Exceeds expectations for this criterion
        
        Return your response in the following format:
        Analysis: [Your detailed analysis]
        Score: [Your score as a number between 1 and 5]
        """
    )
    
    results = {}
    
    for item in data:
        query = item["query"]
        reference = item["reference_answer"]
        answer = item["model_answer"]
        
        item_results = {}
        
        for criterion in criteria:
            prompt = prompt_template.format(
                query=query,
                answer=answer,
                reference=reference,
                criterion=criterion
            )
            
            response = llm.predict(prompt)
            
            # Extract score
            try:
                score_line = [line for line in response.split('\n') if line.startswith('Score:')][0]
                score = float(score_line.replace('Score:', '').strip())
                
                analysis_lines = [line for line in response.split('\n') if line.startswith('Analysis:')]
                analysis = analysis_lines[0].replace('Analysis:', '').strip() if analysis_lines else ""
                
                item_results[criterion] = {
                    "score": score,
                    "analysis": analysis
                }
            except:
                item_results[criterion] = {
                    "score": 0,
                    "analysis": "Failed to parse score"
                }
        
        results[query] = item_results
    
    return results

# 2. Automated metrics evaluation
def automated_metrics_evaluation(data):
    from nltk.translate.bleu_score import sentence_bleu
    from nltk.tokenize import word_tokenize
    from rouge import Rouge
    from bert_score import score as bert_score
    
    results = {}
    rouge = Rouge()
    
    for item in data:
        query = item["query"]
        reference = item["reference_answer"]
        answer = item["model_answer"]
        
        # BLEU score
        reference_tokens = [word_tokenize(reference)]
        answer_tokens = word_tokenize(answer)
        bleu_score = sentence_bleu(reference_tokens, answer_tokens)
        
        # ROUGE scores
        try:
            rouge_scores = rouge.get_scores(answer, reference)[0]
        except:
            rouge_scores = {
                "rouge-1": {"f": 0},
                "rouge-2": {"f": 0},
                "rouge-l": {"f": 0}
            }
        
        # BERTScore
        try:
            P, R, F1 = bert_score([answer], [reference], lang='en')
            bert_f1 = F1.item()
        except:
            bert_f1 = 0
        
        results[query] = {
            "bleu": bleu_score,
            "rouge-1": rouge_scores["rouge-1"]["f"],
            "rouge-2": rouge_scores["rouge-2"]["f"],
            "rouge-l": rouge_scores["rouge-l"]["f"],
            "bert_score": bert_f1
        }
    
    return results

# 3. RAGAS-inspired evaluation (simplified)
def ragas_inspired_evaluation(data, embeddings_model):
    results = {}
    
    for item in data:
        query = item["query"]
        reference = item["reference_answer"]
        answer = item["model_answer"]
        
        # Embed texts
        query_embedding = embeddings_model.embed_query(query)
        answer_embedding = embeddings_model.embed_query(answer)
        reference_embedding = embeddings_model.embed_query(reference)
        
        # Calculate faithfulness (similarity between answer and reference)
        faithfulness = cosine_similarity([answer_embedding], [reference_embedding])[0][0]
        
        # Calculate answer relevance (similarity between query and answer)
        answer_relevance = cosine_similarity([query_embedding], [answer_embedding])[0][0]
        
        # Calculate reference relevance (similarity between query and reference)
        reference_relevance = cosine_similarity([query_embedding], [reference_embedding])[0][0]
        
        # Length ratio (to check conciseness)
        answer_tokens = len(answer.split())
        reference_tokens = len(reference.split())
        length_ratio = min(1.0, reference_tokens / max(1, answer_tokens))
        
        # Combined RAGAS-inspired score
        ragas_score = (faithfulness * 0.4 + answer_relevance * 0.3 + reference_relevance * 0.1 + length_ratio * 0.2)
        
        results[query] = {
            "faithfulness": faithfulness,
            "answer_relevance": answer_relevance,
            "reference_relevance": reference_relevance,
            "conciseness": length_ratio,
            "ragas_score": ragas_score
        }
    
    return results

# Define evaluation criteria
criteria = [
    "Factual Accuracy: Does the answer provide factually correct information that aligns with the reference answer?",
    "Completeness: Does the answer cover all the key points mentioned in the reference answer?",
    "Relevance: Does the answer directly address the query?",
    "Coherence: Is the answer well-structured, logical, and easy to follow?",
    "Conciseness: Does the answer provide the information efficiently without unnecessary information?"
]

# Run LLM judge evaluation
llm_judge_results = llm_judge_evaluation(evaluation_data, criteria)

# Visualize LLM judge results
def visualize_llm_judge_results(results):
    # Prepare data for visualization
    data = []
    
    for query, query_results in results.items():
        for criterion, criterion_result in query_results.items():
            data.append({
                "Query": query[:30] + "...",
                "Criterion": criterion.split(":")[0],
                "Score": criterion_result["score"]
            })
    
    df = pd.DataFrame(data)
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    pivot_table = df.pivot_table(index="Query", columns="Criterion", values="Score")
    sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Score (1-5)'})
    plt.title("LLM Judge Evaluation Results")
    plt.tight_layout()
    plt.savefig("llm_judge_results.png")
    
    # Calculate average scores per criterion
    avg_scores = df.groupby("Criterion")["Score"].mean().reset_index()
    avg_scores = avg_scores.sort_values("Score", ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Score", y="Criterion", data=avg_scores, palette="YlGnBu")
    plt.title("Average Scores by Criterion")
    plt.xlim(0, 5)
    plt.tight_layout()
    plt.savefig("avg_criterion_scores.png")
    
    return df

# Visualize results
evaluation_df = visualize_llm_judge_results(llm_judge_results)
print(evaluation_df)

# Advanced: Multi-LLM consensus evaluation
def multi_llm_consensus(data, criteria, llms):
    all_results = {}
    
    for llm_name, llm_model in llms.items():
        # Create evaluator with this LLM
        prompt_template = PromptTemplate(
            input_variables=["query", "answer", "reference", "criterion"],
            template="""
            You are an expert evaluator for language model outputs. Your task is to evaluate a model's answer against a reference answer based on a specific criterion.

            Query: {query}
            
            Model Answer: {answer}
            
            Reference Answer: {reference}
            
            Criterion: {criterion}
            
            Score the model answer on a scale of 1-5, where:
            1: Poor - Fails to meet the criterion
            2: Fair - Partially meets the criterion with significant issues
            3: Good - Mostly meets the criterion with minor issues
            4: Very Good - Fully meets the criterion with minimal issues
            5: Excellent - Exceeds expectations for this criterion
            
            Return only a number between 1 and 5 as your evaluation score.
            """
        )
        
        results = defaultdict(dict)
        
        for item in data:
            query = item["query"]
            reference = item["reference_answer"]
            answer = item["model_answer"]
            
            for criterion in criteria:
                prompt = prompt_template.format(
                    query=query,
                    answer=answer,
                    reference=reference,
                    criterion=criterion
                )
                
                response = llm_model.predict(prompt)
                
                try:
                    # Try to extract score
                    score = float(''.join(c for c in response if c.isdigit() or c == '.'))
                    if score < 1 or score > 5:
                        score = max(1, min(5, score))  # Clamp to range
                except:
                    score = 0
                
                results[query][criterion] = score
        
        all_results[llm_name] = results
    
    # Calculate consensus scores
    consensus_results = {}
    
    for query in data[0]["query"]:  # Assuming all items have the same queries
        consensus_results[query] = {}
        
        for criterion in criteria:
            scores = [results[query][criterion] for llm_name, results in all_results.items()]
            
            consensus_results[query][criterion] = {
                "mean": np.mean(scores),
                "median": np.median(scores),
                "min": np.min(scores),
                "max": np.max(scores),
                "std": np.std(scores),
                "raw_scores": {llm_name: results[query][criterion] for llm_name, results in all_results.items()}
            }
    
    return consensus_results

# Define multiple LLMs for consensus (demo - would use different models in practice)
# llms = {
#     "gpt-4": ChatOpenAI(model="gpt-4", temperature=0),
#     "gpt-3.5": ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
#     "claude": AnthropicLLM(temperature=0)
# }

# RAGAS full implementation
def ragas_evaluation(data):
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_relevancy,
        context_recall
    )
    from datasets import Dataset
    
    # Prepare data for RAGAS
    ragas_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truths": []
    }
    
    for item in data:
        ragas_data["question"].append(item["query"])
        ragas_data["answer"].append(item["model_answer"])
        # Assume we split reference into contexts for this example
        contexts = [item["reference_answer"]]
        ragas_data["contexts"].append(contexts)
        ragas_data["ground_truths"].append([item["reference_answer"]])
    
    # Create dataset
    dataset = Dataset.from_dict(ragas_data)
    
    # Evaluate
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_relevancy,
            context_recall
        ]
    )
    
    return result

# Comprehensive evaluation framework
def comprehensive_evaluation(data, model_name):
    """Run all evaluation methods and compile results"""
    results = {
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "sample_size": len(data),
        "evaluations": {}
    }
    
    # LLM judge evaluation
    try:
        print("Running LLM judge evaluation...")
        results["evaluations"]["llm_judge"] = llm_judge_evaluation(data, criteria)
    except Exception as e:
        print(f"LLM judge evaluation failed: {str(e)}")
    
    # Automated metrics
    try:
        print("Running automated metrics evaluation...")
        results["evaluations"]["automated_metrics"] = automated_metrics_evaluation(data)
    except Exception as e:
        print(f"Automated metrics evaluation failed: {str(e)}")
    
    # RAGAS-inspired metrics
    try:
        print("Running RAGAS-inspired evaluation...")
        results["evaluations"]["ragas_inspired"] = ragas_inspired_evaluation(
            data, 
            OpenAIEmbeddings()
        )
    except Exception as e:
        print(f"RAGAS-inspired evaluation failed: {str(e)}")
    
    # Calculate aggregate scores
    aggregate_scores = {}
    
    # Aggregate LLM judge scores
    if "llm_judge" in results["evaluations"]:
        llm_scores = []
        for query, query_results in results["evaluations"]["llm_judge"].items():
            query_score = np.mean([res["score"] for res in query_results.values()])
            llm_scores.append(query_score)
        
        aggregate_scores["llm_judge_mean"] = np.mean(llm_scores)
    
    # Aggregate RAGAS-inspired scores
    if "ragas_inspired" in results["evaluations"]:
        ragas_scores = [res["ragas_score"] for res in results["evaluations"]["ragas_inspired"].values()]
        aggregate_scores["ragas_mean"] = np.mean(ragas_scores)
    
    # Add aggregate scores to results
    results["aggregate_scores"] = aggregate_scores
    
    return results

# Save evaluation results
def save_evaluation_results(results, filename):
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation results to {filename}")

# Run comprehensive evaluation
# evaluation_results = comprehensive_evaluation(evaluation_data, "gpt-4")
# save_evaluation_results(evaluation_results, "evaluation_results.json")
```

## Dimensionality reduction techniques
**Core Concept:** Methods for visualizing and simplifying high-dimensional embedding vectors.

**Detailed Explanation:**
- **Common Techniques:**
  - **PCA (Principal Component Analysis):** Linear dimensionality reduction
  - **t-SNE (t-distributed Stochastic Neighbor Embedding):** Non-linear dimensionality reduction focused on local structure
  - **UMAP (Uniform Manifold Approximation and Projection):** Balance of global and local structure preservation
  - **SOM (Self-Organizing Maps):** Neural network-based dimensionality reduction

- **Applications:**
  - Visualizing embedding spaces
  - Analyzing document clusters
  - Identifying data patterns
  - Reducing computational complexity

**Code Example:**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sentence_transformers import SentenceTransformer
import seaborn as sns
import time
from typing import List, Dict
import hdbscan

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample documents with categories
documents = [
    {"text": "Machine learning algorithms learn patterns from data.", "category": "ML Basics"},
    {"text": "Neural networks are inspired by the human brain.", "category": "Neural Networks"},
    {"text": "Deep learning uses multiple layers of neural networks.", "category": "Deep Learning"},
    {"text": "Convolutional neural networks excel at image recognition.", "category": "CNN"},
    {"text": "Recurrent neural networks process sequential data.", "category": "RNN"},
    {"text": "Transformers use self-attention mechanisms.", "category": "Transformers"},
    {"text": "BERT is a transformer-based language model.", "category": "BERT"},
    {"text": "GPT models are generative pre-trained transformers.", "category": "GPT"},
    {"text": "Reinforcement learning uses reward signals.", "category": "RL"},
    {"text": "Q-learning is a value-based reinforcement learning algorithm.", "category": "Q-Learning"},
    {"text": "Policy gradients directly optimize the policy function.", "category": "Policy Gradients"},
    {"text": "Supervised learning uses labeled training data.", "category": "Supervised"},
    {"text": "Unsupervised learning finds patterns in unlabeled data.", "category": "Unsupervised"},
    {"text": "Clustering groups similar data points together.", "category": "Clustering"},
    {"text": "K-means is a popular clustering algorithm.", "category": "K-means"},
    {"text": "Decision trees split data based on feature values.", "category": "Decision Trees"},
    {"text": "Random forests combine multiple decision trees.", "category": "Random Forests"},
    {"text": "Support vector machines find optimal decision boundaries.", "category": "SVM"},
    {"text": "Logistic regression is used for binary classification.", "category": "Regression"},
    {"text": "Natural language processing analyzes human language.", "category": "NLP"}
]

# Create embeddings
texts = [doc["text"] for doc in documents]
categories = [doc["category"] for doc in documents]

# Generate embeddings
start_time = time.time()
embeddings = model.encode(texts)
print(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]} in {time.time() - start_time:.2f}s")

# Function to apply and compare dimensionality reduction techniques
def compare_dim_reduction(embeddings, labels, perplexity=5, n_neighbors=15):
    results = {}
    
    # 1. PCA
    start_time = time.time()
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings)
    pca_time = time.time() - start_time
    
    variance_explained = pca.explained_variance_ratio_.sum()
    print(f"PCA completed in {pca_time:.2f}s, explained variance: {variance_explained:.2%}")
    
    results["pca"] = {
        "result": pca_result,
        "time": pca_time,
        "variance_explained": variance_explained
    }
    
    # 2. t-SNE
    start_time = time.time()
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, random_state=42)
    tsne_result = tsne.fit_transform(embeddings)
    tsne_time = time.time() - start_time
    
    print(f"t-SNE completed in {tsne_time:.2f}s")
    
    results["tsne"] = {
        "result": tsne_result,
        "time": tsne_time
    }
    
    # 3. UMAP
    start_time = time.time()
    umap_reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, n_components=2, random_state=42)
    umap_result = umap_reducer.fit_transform(embeddings)
    umap_time = time.time() - start_time
    
    print(f"UMAP completed in {umap_time:.2f}s")
    
    results["umap"] = {
        "result": umap_result,
        "time": umap_time
    }
    
    # Visualize results
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot PCA
    scatter = axes[0].scatter(pca_result[:, 0], pca_result[:, 1], c=pd.factorize(labels)[0], cmap='viridis', alpha=0.8)
    axes[0].set_title(f'PCA (Time: {pca_time:.2f}s, Var: {variance_explained:.2%})')
    
    # Plot t-SNE
    scatter = axes[1].scatter(tsne_result[:, 0], tsne_result[:, 1], c=pd.factorize(labels)[0], cmap='viridis', alpha=0.8)
    axes[1].set_title(f't-SNE (Time: {tsne_time:.2f}s, Perplexity: {perplexity})')
    
    # Plot UMAP
    scatter = axes[2].scatter(umap_result[:, 0], umap_result[:, 1], c=pd.factorize(labels)[0], cmap='viridis', alpha=0.8)
    axes[2].set_title(f'UMAP (Time: {umap_time:.2f}s, Neighbors: {n_neighbors})')
    
    plt.tight_layout()
    plt.savefig("dim_reduction_comparison.png")
    
    return results

# Compare techniques
reduction_results = compare_dim_reduction(embeddings, categories)

# Cluster analysis with HDBSCAN
def cluster_analysis(embeddings, reduced_data, true_labels):
    results = {}
    
    # Cluster in original space
    start_time = time.time()
    original_clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1)
    original_labels = original_clusterer.fit_predict(embeddings)
    original_time = time.time() - start_time
    
    # Get unique labels
    unique_original = len(set(original_labels)) - (1 if -1 in original_labels else 0)
    
    print(f"Original space clustering: {unique_original} clusters, {original_time:.2f}s")
    
    results["original"] = {
        "labels": original_labels,
        "n_clusters": unique_original,
        "time": original_time
    }
    
    # Cluster in reduced spaces
    for name, data in reduced_data.items():
        reduced_space = data["result"]
        
        start_time = time.time()
        reduced_clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1)
        reduced_labels = reduced_clusterer.fit_predict(reduced_space)
        reduced_time = time.time() - start_time
        
        # Get unique labels
        unique_reduced = len(set(reduced_labels)) - (1 if -1 in reduced_labels else 0)
        
        print(f"{name.upper()} space clustering: {unique_reduced} clusters, {reduced_time:.2f}s")
        
        results[name] = {
            "labels": reduced_labels,
            "n_clusters": unique_reduced,
            "time": reduced_time
        }
    
    # Visualize clustering results
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot UMAP with original labels
    scatter = axes[0].scatter(
        reduction_results["umap"]["result"][:, 0],
        reduction_results["umap"]["result"][:, 1],
        c=pd.factorize(true_labels)[0],
        cmap='viridis',
        alpha=0.8
    )
    axes[0].set_title('UMAP with True Categories')
    
    # Plot UMAP with original space clustering
    scatter = axes[1].scatter(
        reduction_results["umap"]["result"][:, 0],
        reduction_results["umap"]["result"][:, 1],
        c=results["original"]["labels"],
        cmap='viridis',
        alpha=0.8
    )
    axes[1].set_title(f'UMAP with Original Space Clustering ({results["original"]["n_clusters"]} clusters)')
    
    # Plot UMAP with reduced space clustering
    scatter = axes[2].scatter(
        reduction_results["umap"]["result"][:, 0],
        reduction_results["umap"]["result"][:, 1],
        c=results["umap"]["labels"],
        cmap='viridis',
        alpha=0.8
    )
    axes[2].set_title(f'UMAP with UMAP Space Clustering ({results["umap"]["n_clusters"]} clusters)')
    
    plt.tight_layout()
    plt.savefig("clustering_comparison.png")
    
    return results

# Run cluster analysis
cluster_results = cluster_analysis(embeddings, reduction_results, categories)

# Interactive 3D visualization with Plotly
def create_3d_visualization(embeddings, labels):
    import plotly.express as px
    from sklearn.decomposition import PCA
    
    # Reduce to 3D
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(embeddings)
    
    # Create DataFrame
    df = pd.DataFrame({
        'x': embeddings_3d[:, 0],
        'y': embeddings_3d[:, 1],
        'z': embeddings_3d[:, 2],
        'category': labels,
        'text': texts
    })
    
    # Create 3D scatter plot
    fig = px.scatter_3d(
        df, x='x', y='y', z='z',
        color='category',
        hover_data=['text'],
        opacity=0.7,
        title="3D PCA Visualization of Document Embeddings"
    )
    
    fig.update_layout(
        scene=dict(
            xaxis_title='PCA Component 1',
            yaxis_title='PCA Component 2',
            zaxis_title='PCA Component 3'
        ),
        width=900,
        height=700
    )
    
    # Save as HTML
    fig.write_html("embeddings_3d.html")
    return fig

# Create 3D visualization
# visualization = create_3d_visualization(embeddings, categories)

# Parameter tuning for dimensionality reduction
def tune_parameters(embeddings, labels):
    results = {}
    
    # t-SNE perplexity values
    perplexities = [5, 10, 30, 50]
    
    for perplexity in perplexities:
        start_time = time.time()
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, random_state=42)
        tsne_result = tsne.fit_transform(embeddings)
        tsne_time = time.time() - start_time
        
        results[f"tsne_perplexity_{perplexity}"] = {
            "result": tsne_result,
            "time": tsne_time,
            "perplexity": perplexity
        }
    
    # UMAP parameters
    n_neighbors_values = [5, 15, 30, 50]
    min_dist_values = [0.0, 0.1, 0.5, 0.99]
    
    # Fixed min_dist, vary n_neighbors
    for n_neighbors in n_neighbors_values:
        start_time = time.time()
        umap_reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, n_components=2, random_state=42)
        umap_result = umap_reducer.fit_transform(embeddings)
        umap_time = time.time() - start_time
        
        results[f"umap_neighbors_{n_neighbors}"] = {
            "result": umap_result,
            "time": umap_time,
            "n_neighbors": n_neighbors,
            "min_dist": 0.1
        }
    
    # Fixed n_neighbors, vary min_dist
    for min_dist in min_dist_values:
        start_time = time.time()
        umap_reducer = umap.UMAP(n_neighbors=15, min_dist=min_dist, n_components=2, random_state=42)
        umap_result = umap_reducer.fit_transform(embeddings)
        umap_time = time.time() - start_time
        
        results[f"umap_mindist_{min_dist}"] = {
            "result": umap_result,
            "time": umap_time,
            "n_neighbors": 15,
            "min_dist": min_dist
        }
    
    # Visualize t-SNE perplexity results
    fig, axes = plt.subplots(1, len(perplexities), figsize=(20, 5))
    
    for i, perplexity in enumerate(perplexities):
        key = f"tsne_perplexity_{perplexity}"
        result = results[key]["result"]
        time_taken = results[key]["time"]
        
        scatter = axes[i].scatter(result[:, 0], result[:, 1], c=pd.factorize(labels)[0], cmap='viridis', alpha=0.8)
        axes[i].set_title(f't-SNE (Perplexity: {perplexity}, Time: {time_taken:.2f}s)')
    
    plt.tight_layout()
    plt.savefig("tsne_parameter_tuning.png")
    
    # Visualize UMAP n_neighbors results
    fig, axes = plt.subplots(1, len(n_neighbors_values), figsize=(20, 5))
    
    for i, n_neighbors in enumerate    # Save to file
    output_dir = "evaluation_sets"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/{eval_set.name}.json", "w") as f:
        json.dump(eval_set.dict(), f, indent=2)
    
    print(f"Saved evaluation set to {output_dir}/{eval_set.name}.json")

# Analyze evaluation sets
for eval_set in evaluation_sets:
    print(f"\nAnalysis for {eval_set.name}:")
    print(f"Documents: {len(eval_set.documents)}")
    print(f"Questions: {len(eval_set.questions)}")
    print(f"Ground truth answers: {len(eval_set.ground_truth)}")
    
    # Question type distribution
    question_types = {}
    for q in eval_set.questions:
        if q.type not in question_types:
            question_types[q.type] = 0
        question_types[q.type] += 1
    
    print("\nQuestion type distribution:")
    for q_type, count in question_types.items():
        print(f"- {q_type}: {count} ({count/len(eval_set.questions)*100:.1f}%)")
    
    # Difficulty distribution
    difficulty_levels = {}
    for q in eval_set.questions:
        if q.difficulty not in difficulty_levels:
            difficulty_levels[q.difficulty] = 0
        difficulty_levels[q.difficulty] += 1
    
    print("\nDifficulty distribution:")
    for level, count in difficulty_levels.items():
        print(f"- {level}: {count} ({count/len(eval_set.questions)*100:.1f}%)")
```

## Vector stores
**Core Concept:** Specialized databases optimized for storing and querying high-dimensional vector embeddings.

**Detailed Explanation:**
- **Core Functionality:**
  - Efficient similarity search
  - Support for various distance metrics
  - Approximate nearest neighbor algorithms
  - Metadata filtering
  - Index optimization

- **Common Vector Databases:**
  - Pinecone: Fully managed vector database service
  - Weaviate: Knowledge graph + vector search hybrid
  - Chroma: Lightweight, embedded solution
  - FAISS: In-memory, high-performance library
  - Milvus: Distributed, scalable vector database

**Code Example:**
```python
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS, Chroma, Pinecone
from langchain.docstore.document import Document
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from typing import List, Dict, Any
import faiss

# Initialize embedding model
embeddings = OpenAIEmbeddings()

# Sample documents
documents = [
    Document(page_content="Machine learning models can make predictions based on data patterns.", metadata={"category": "ai", "difficulty": "beginner"}),
    Document(page_content="Neural networks consist of layers of interconnected nodes.", metadata={"category": "ai", "difficulty": "intermediate"}),
    Document(page_content="Transformers use self-attention mechanisms to process sequential data.", metadata={"category": "ai", "difficulty": "advanced"}),
    Document(page_content="Python is a popular programming language for data science.", metadata={"category": "programming", "difficulty": "beginner"}),
    Document(page_content="Object-oriented programming uses classes and inheritance.", metadata={"category": "programming", "difficulty": "intermediate"}),
    Document(page_content="Functional programming emphasizes immutability and pure functions.", metadata={"category": "programming", "difficulty": "advanced"}),
    Document(page_content="Data visualization helps understand patterns in information.", metadata={"category": "data", "difficulty": "beginner"}),
    Document(page_content="Statistical analysis involves hypothesis testing and confidence intervals.", metadata={"category": "data", "difficulty": "intermediate"}),
    Document(page_content="Principal component analysis reduces dimensionality of data.", metadata={"category": "data", "difficulty": "advanced"}),
]

# Create FAISS vector store
faiss_index = FAISS.from_documents(documents, embeddings)
print(f"Created FAISS index with {len(documents)} documents")

# Create Chroma vector store
chroma_db = Chroma.from_documents(documents, embeddings)
print(f"Created Chroma DB with {len(documents)} documents")

# Basic similarity search
query = "How do neural networks work?"
print(f"\nQuery: {query}")

# Search FAISS
start_time = time.time()
faiss_results = faiss_index.similarity_search(query, k=3)
faiss_time = time.time() - start_time

print(f"\nFAISS results (took {faiss_time:.4f}s):")
for i, doc in enumerate(faiss_results):
    print(f"{i+1}. {doc.page_content} (Category: {doc.metadata['category']}, Difficulty: {doc.metadata['difficulty']})")

# Search Chroma
start_time = time.time()
chroma_results = chroma_db.similarity_search(query, k=3)
chroma_time = time.time() - start_time

print(f"\nChroma results (took {chroma_time:.4f}s):")
for i, doc in enumerate(chroma_results):
    print(f"{i+1}. {doc.page_content} (Category: {doc.metadata['category']}, Difficulty: {doc.metadata['difficulty']})")

# Metadata filtering with FAISS
filter_query = "What are some advanced concepts?"
print(f"\nFilter query: {filter_query}")

start_time = time.time()
advanced_docs = [d for d in documents if d.metadata["difficulty"] == "advanced"]
advanced_faiss = FAISS.from_documents(advanced_docs, embeddings)
filter_results = advanced_faiss.similarity_search(filter_query, k=3)
filter_time = time.time() - start_time

print(f"\nFAISS with pre-filtering (took {filter_time:.4f}s):")
for i, doc in enumerate(filter_results):
    print(f"{i+1}. {doc.page_content} (Category: {doc.metadata['category']}, Difficulty: {doc.metadata['difficulty']})")

# Metadata filtering with Chroma
start_time = time.time()
chroma_filter_results = chroma_db.similarity_search(
    filter_query, 
    k=3,
    filter={"difficulty": "advanced"}
)
chroma_filter_time = time.time() - start_time

print(f"\nChroma with metadata filtering (took {chroma_filter_time:.4f}s):")
for i, doc in enumerate(chroma_filter_results):
    print(f"{i+1}. {doc.page_content} (Category: {doc.metadata['category']}, Difficulty: {doc.metadata['difficulty']})")

# MMR (Maximum Marginal Relevance) search for diversity
mmr_query = "Tell me about programming concepts"
print(f"\nMMR query for diverse results: {mmr_query}")

start_time = time.time()
mmr_results = faiss_index.max_marginal_relevance_search(mmr_query, k=3, fetch_k=6)
mmr_time = time.time() - start_time

print(f"\nMMR search results (took {mmr_time:.4f}s):")
for i, doc in enumerate(mmr_results):
    print(f"{i+1}. {doc.page_content} (Category: {doc.metadata['category']}, Difficulty: {doc.metadata['difficulty']})")

# Advanced FAISS index configuration
def create_optimized_faiss_index(documents, embedding_size=1536, index_type="flat"):
    # Get embeddings
    doc_embeddings = embeddings.embed_documents([d.page_content for d in documents])
    
    # Create index based on type
    if index_type == "flat":
        index = faiss.IndexFlatL2(embedding_size)
    elif index_type == "ivf":
        # IVF index for faster search at slight accuracy cost
        nlist = min(50, len(documents))  # Number of clusters
        quantizer = faiss.IndexFlatL2(embedding_size)
        index = faiss.IndexIVFFlat(quantizer, embedding_size, nlist)
        index.train(np.array(doc_embeddings, dtype=np.float32))
    elif index_type == "hnsw":
        # HNSW index for very fast search
        index = faiss.IndexHNSWFlat(embedding_size, 32)  # 32 neighbors per node
    else:
        raise ValueError(f"Unknown index type: {index_type}")
    
    # Add vectors
    index.add(np.array(doc_embeddings, dtype=np.float32))
    
    return index, doc_embeddings

# Generate larger document set for benchmarking
def generate_larger_document_set(size=1000):
    categories = ["ai", "programming", "data", "security", "cloud", "mobile", "web", "iot"]
    difficulties = ["beginner", "intermediate", "advanced"]
    
    larger_docs = []
    for i in range(size):
        category = random.choice(categories)
        difficulty = random.choice(difficulties)
        content = f"Sample document {i+1} about {category} at {difficulty} level."
        
        doc = Document(
            page_content=content,
            metadata={"category": category, "difficulty": difficulty, "id": str(i+1)}
        )
        larger_docs.append(doc)
    
    return larger_docs

# Benchmark different FAISS index types
def benchmark_indices(docs, queries, embedding_size=1536):
    index_types = ["flat", "ivf", "hnsw"]
    results = {}
    
    doc_contents = [d.page_content for d in docs]
    
    # Get query embeddings once
    query_embeddings = [embeddings.embed_query(q) for q in queries]
    
    for index_type in index_types:
        print(f"Testing {index_type} index...")
        
        # Create and train index
        start_time = time.time()
        index, doc_embeddings = create_optimized_faiss_index(docs, embedding_size, index_type)
        build_time = time.time() - start_time
        
        # Search times
        search_times = []
        for query_embedding in query_embeddings:
            start_time = time.time()
            D, I = index.search(np.array([query_embedding], dtype=np.float32), k=5)
            search_time = time.time() - start_time
            search_times.append(search_time)
        
        # Store results
        results[index_type] = {
            "build_time": build_time,
            "avg_search_time": np.mean(search_times),
            "min_search_time": np.min(search_times),
            "max_search_time": np.max(search_times)
        }
    
    return results

# Generate test data
large_docs = generate_larger_document_set(1000)
test_queries = [
    "How do neural networks work?",
    "Explain object-oriented programming",
    "What is data visualization?",
    "Tell me about cloud security",
    "Mobile app development best practices"
]

# Run benchmark
benchmark_results = benchmark_indices(large_docs, test_queries)

# Display results
print("\nBenchmark Results:")
results_df = pd.DataFrame(benchmark_results).T
print(results_df)

# Plot results
plt.figure(figsize=(12, 6))

# Build time
plt.subplot(1, 2, 1)
sns.barplot(x=results_df.index, y="build_time", data=results_df)
plt.title("Index Build Time (seconds)")
plt.xticks(rotation=45)

# Search time
plt.subplot(1, 2, 2)
sns.barplot(x=results_df.index, y="avg_search_time", data=results_df)
plt.title("Average Search Time (seconds)")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("vector_db_benchmark.png")
```

**Advanced Implementation Pattern:**
- Hybrid search with multiple vector indices and reranking:
```python
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from sentence_transformers import CrossEncoder
import numpy as np
from typing import List, Dict, Any, Optional
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import time

# Document class with multiple embeddings
class EnhancedDocument:
    def __init__(self, content, metadata=None):
        self.content = content
        self.metadata = metadata or {}
        self.embeddings = {}
        self.sparse_vectors = {}
    
    def add_embedding(self, name, embedding):
        self.embeddings[name] = embedding
    
    def add_sparse_vector(self, name, vector):
        self.sparse_vectors[name] = vector

# Multi-index vector store
class HybridVectorStore:
    def __init__(self):
        self.documents = []
        self.embedding_models = {}
        self.sparse_models = {}
        self.cross_encoders = {}
        self.indices = {}
    
    def add_embedding_model(self, name, model):
        self.embedding_models[name] = model
    
    def add_sparse_model(self, name, model):
        self.sparse_models[name] = model
    
    def add_cross_encoder(self, name, model):
        self.cross_encoders[name] = model
    
    def add_documents(self, documents):
        # Add documents to store
        for doc in documents:
            enhanced_doc = EnhancedDocument(doc.page_content, doc.metadata)
            self.documents.append(enhanced_doc)
        
        # Generate embeddings for all documents
        self._generate_embeddings()
        
        # Generate sparse vectors for all documents
        self._generate_sparse_vectors()
        
        # Build indices
        self._build_indices()
    
    def _generate_embeddings(self):
        for model_name, model in self.embedding_models.items():
            print(f"Generating {model_name} embeddings...")
            contents = [doc.content for doc in self.documents]
            
            if hasattr(model, 'embed_documents'):
                # LangChain embedding model
                embeddings = model.embed_documents(contents)
            else:
                # Sentence-transformers model
                embeddings = model.encode(contents)
            
            # Add embeddings to documents
            for i, doc in enumerate(self.documents):
                doc.add_embedding(model_name, embeddings[i])
    
    def _generate_sparse_vectors(self):
        for model_name, model in self.sparse_models.items():
            print(f"Generating {model_name} sparse vectors...")
            contents = [doc.content for doc in self.documents]
            
            if model_name == "tfidf":
                # TF-IDF Vectorizer
                if not hasattr(model, 'vocabulary_'):
                    model.fit(contents)
                
                sparse_vectors = model.transform(contents)
                
                # Add sparse vectors to documents
                for i, doc in enumerate(self.documents):
                    doc.add_sparse_vector(model_name, sparse_vectors[i])
            
            elif model_name == "bm25":
                # BM25 model needs tokenized docs
                tokenized_docs = [content.split() for content in contents]
                model.fit(tokenized_docs)
                
                # Store tokenized docs for later search
                self.bm25_tokenized_docs = tokenized_docs
    
    def _build_indices(self):
        # Build vector indices
        for model_name in self.embedding_models.keys():
            print(f"Building index for {model_name}...")
            embeddings = [doc.embeddings[model_name] for doc in self.documents]
            
            # Create Chroma DB
            documents = [
                Document(page_content=doc.content, metadata=doc.metadata)
                for doc in self.documents
            ]
            
            self.indices[model_name] = Chroma.from_documents(
                documents, 
                OpenAIEmbeddings(),  # Placeholder, we'll use our pre-computed embeddings
                embedding_function=None  # Will use pre-computed embeddings
            )
    
    def search(self, query, top_k=5, strategy="ensemble", weights=None, rerank=True):
        start_time = time.time()
        
        if strategy == "ensemble":
            return self._ensemble_search(query, top_k, weights, rerank)
        elif strategy == "cascade":
            return self._cascade_search(query, top_k, rerank)
        else:
            raise ValueError(f"Unknown search strategy: {strategy}")
    
    def _ensemble_search(self, query, top_k=5, weights=None, rerank=True):
        # Default equal weights if not provided
        if weights is None:
            weights = {name: 1.0 for name in self.embedding_models}
            # Add sparse models
            for name in self.sparse_models:
                weights[name] = 1.0
        
        all_results = {}
        scored_documents = {}
        
        # Get results from each dense embedding model
        for model_name, model in self.embedding_models.items():
            if model_name not in weights:
                continue
                
            weight = weights[model_name]
            if weight <= 0:
                continue
                
            # Embed query
            if hasattr(model, 'embed_query'):
                # LangChain embedding model
                query_embedding = model.embed_query(query)
            else:
                # Sentence-transformers model
                query_embedding = model.encode(query)
            
            # Search in corresponding index
            results = self.indices[model_name].similarity_search_by_vector(
                query_embedding, 
                k=top_k
            )
            
            # Score documents
            for i, doc in enumerate(results):
                doc_id = doc.metadata.get('id', str(i))
                if doc_id not in scored_documents:
                    scored_documents[doc_id] = {
                        "document": doc,
                        "score": 0,
                        "sources": []
                    }
                
                # Add score (inversely weighted by position)
                score = weight * (top_k - i) / top_k
                scored_documents[doc_id]["score"] += score
                scored_documents[doc_id]["sources"].append(f"{model_name}:{i+1}")
        
        # Get results from sparse models
        for model_name, model in self.sparse_models.items():
            if model_name not in weights:
                continue
                
            weight = weights[model_name]
            if weight <= 0:
                continue
            
            if model_name == "tfidf":
                # Transform query
                query_vector = model.transform([query])[0]
                
                # Calculate similarity
                similarities = []
                for doc in self.documents:
                    if model_name in doc.sparse_vectors:
                        similarity = cosine_similarity(
                            query_vector, 
                            doc.sparse_vectors[model_name]
                        )[0][0]
                        similarities.append((doc, similarity))
                
                # Sort by similarity
                similarities.sort(key=lambda x: x[1], reverse=True)
                
                # Add top results
                for i, (doc, similarity) in enumerate(similarities[:top_k]):
                    doc_id = doc.metadata.get('id', str(i))
                    if doc_id not in scored_documents:
                        scored_documents[doc_id] = {
                            "document": Document(
                                page_content=doc.content, 
                                metadata=doc.metadata
                            ),
                            "score": 0,
                            "sources": []
                        }
                    
                    # Add score (directly using similarity * weight)
                    score = weight * similarity
                    scored_documents[doc_id]["score"] += score
                    scored_documents[doc_id]["sources"].append(f"{model_name}:{similarity:.3f}")
            
            elif model_name == "bm25":
                # Tokenize query
                tokenized_query = query.split()
                
                # Get BM25 scores
                scores = model.get_scores(tokenized_query)
                
                # Get top documents
                top_indices = np.argsort(scores)[::-1][:top_k]
                
                for i, idx in enumerate(top_indices):
                    doc = self.documents[idx]
                    doc_id = doc.metadata.get('id', str(idx))
                    if doc_id not in scored_documents:
                        scored_documents[doc_id] = {
                            "document": Document(
                                page_content=doc.content, 
                                metadata=doc.metadata
                            ),
                            "score": 0,
                            "sources": []
                        }
                    
                    # Add score (normalized BM25 score * weight)
                    score = weight * scores[idx] / max(1, max(scores))
                    scored_documents[doc_id]["score"] += score
                    scored_documents[doc_id]["sources"].append(f"{model_name}:{scores[idx]:.3f}")
        
        # Convert to list and sort
        results = list(scored_documents.values())
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Take top-k
        results = results[:top_k]
        
        # Cross-encoder reranking
        if rerank and self.cross_encoders:
            # Use first cross-encoder for reranking
            encoder_name = list(self.cross_encoders.keys())[0]
            cross_encoder = self.cross_encoders[encoder_name]
            
            # Prepare pairs
            pairs = [(query, doc["document"].page_content) for doc in results]
            
            # Get scores
            ce_scores = cross_encoder.predict(pairs)
            
            # Add cross-encoder scores
            for i, score in enumerate(ce_scores):
                results[i]["ce_score"] = float(score)
                results[i]["sources"].append(f"{encoder_name}:{score:.3f}")
            
            # Re-sort based on cross-encoder score
            results.sort(key=lambda x: x["ce_score"], reverse=True)
        
        search_time = time.time() - start_time
        
        # Return results with timing information
        return {
            "results": [r["document"] for r in results],
            "detailed": results,
            "time": search_time
        }
    
    def _cascade_search(self, query, top_k=5, rerank=True):
        """Cascade search uses faster methods first, then more accurate methods."""
        start_time = time.time()
        
        # Step 1: Use BM25 for initial broad retrieval
        initial_k = min(top_k * 3, len(self.documents))
        
        if "bm25" in self.sparse_models:
            model = self.sparse_models["bm25"]
            tokenized_query = query.split()
            scores = model.get_scores(tokenized_query)
            top_indices = np.argsort(scores)[::-1][:initial_k]
            
            candidate_docs = [self.documents[idx] for idx in top_indices]
        else:
            # Fallback to all documents if no BM25
            candidate_docs = self.documents[:initial_k]
        
        # Step 2: Use embeddings for semantic ranking
        if self.embedding_models:
            # Use first embedding model
            model_name = list(self.embedding_models.keys())[0]
            model = self.embedding_models[model_name]
            
            # Embed query
            if hasattr(model, 'embed_query'):
                query_embedding = model.embed_query(query)
            else:
                query_embedding = model.encode(query)
            
            # Get embeddings for candidate docs
            similarities = []
            for doc in candidate_docs:
                if model_name in doc.embeddings:
                    similarity = np.dot(query_embedding, doc.embeddings[model_name])
                    similarities.append((doc, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Take top results
            candidate_docs = [doc for doc, _ in similarities[:top_k*2]]
        
        # Step 3: Cross-encoder reranking
        if rerank and self.cross_encoders:
            # Use first cross-encoder
            encoder_name = list(self.cross_encoders.keys())[0]
            cross_encoder = self.cross_encoders[encoder_name]
            
            # Prepare pairs
            pairs = [(query, doc.content) for doc in candidate_docs]
            
            # Get scores
            ce_scores = cross_encoder.predict(pairs)
            
            # Combine docs and scores
            scored_docs = [(candidate_docs[i], float(score)) for i, score in enumerate(ce_scores)]
            
            # Sort by cross-encoder score
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Take top-k
            final_results = scored_docs[:top_k]
            
            # Format results
            results = []
            for doc, score in final_results:
                results.append({
                    "document": Document(page_content=doc.content, metadata=doc.metadata),
                    "score": score,
                    "sources": [f"{encoder_name}:{score:.3f}"]
                })
        else:
            # No reranking, just take top-k candidates
            results = []
            for i, doc in enumerate(candidate_docs[:top_k]):
                results.append({
                    "document": Document(page_content=doc.content, metadata=doc.metadata),
                    "score": 1.0 - (i / top_k),  # Assign decreasing scores
                    "sources": ["cascade"]
                })
        
        search_time = time.time() - start_time
        
        # Return results with timing information
        return {
            "results": [r["document"] for r in results],
            "detailed": results,
            "time": search_time
        }

# Example usage
# Create hybrid vector store
hybrid_db = HybridVectorStore()

# Add embedding models
embedding_model = OpenAIEmbeddings()
hybrid_db.add_embedding_model("openai", embedding_model)

sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
hybrid_db.add_embedding_model("minilm", sentence_model)

# Add sparse models
tfidf = TfidfVectorizer()
hybrid_db.add_sparse_model("tfidf", tfidf)

bm25 = BM25Okapi([])  # Will be initialized when documents are added
hybrid_db.add_sparse_model("bm25", bm25)

# Add cross-encoder for reranking
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
hybrid_db.add_cross_encoder("minilm-cross", cross_encoder)

# Add documents
hybrid_db.add_documents(documents)

# Search with different strategies
query = "How do neural networks work?"
print(f"\nQuery: {query}")

# Ensemble search
ensemble_results = hybrid_db.search(
    query, 
    top_k=3, 
    strategy="ensemble",
    weights={
        "openai": 1.0,
        "minilm": 0.8,
        "tfidf": 0.5,
        "bm25": 0.7
    }
)

print(f"\nEnsemble search results (took {ensemble_results['time']:.4f}s):")
for i, doc in enumerate(ensemble_results['results']):
    detail = ensemble_results['detailed'][i]
    print(f"{i+1}. {doc.page_content} (Score: {detail['score']:.4f}, Sources: {', '.join(detail['sources'])})")

# Cascade search
cascade_results = hybrid_db.search(
    query,
    top_k=3,
    strategy="cascade"
)

print(f"\nCascade search results (took {cascade_results['time']:.4f}s):")
for i, doc in enumerate(cascade_results['results']):
    detail = cascade_results['detailed'][i]
    print(f"{i+1}. {doc.page_content} (Score: {detail['score']:.4f}, Sources: {', '.join(detail['sources'])})")
```

## Evaluating chatbot performance
**Core Concept:** Methods and metrics to assess the effectiveness and quality of conversational AI systems.

**Detailed Explanation:**
- **Evaluation Dimensions:**
  - **Accuracy:** Factual correctness
  - **Relevance:** Appropriateness to query
  - **Groundedness:** Connection to source material
  - **Coherence:** Logical flow and organization
  - **Helpfulness:** Practical utility of response

- **Evaluation Methods:**
  - **Human evaluation:** Manual review and scoring
  - **LLM-as-judge:** Using another LLM to evaluate responses
  - **Automated metrics:** ROUGE, BLEU, BERTScore
  - **Specialized frameworks:** RAGAS for RAG evaluation

**Code Example:**
```python
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.evaluation import load_evaluator
from langchain.evaluation.criteria import LabeledCriterion
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any
import numpy as np
import json
from collections import defaultdict

# Sample data to evaluate
evaluation_data = [
    {
        "query": "What causes climate change and what are the main effects?",
        "reference_answer": "Climate change is primarily caused by greenhouse gas emissions from human activities like burning fossil fuels, deforestation, and industrial processes. The main effects include rising global temperatures, sea level rise, more frequent and severe weather events, disruption of ecosystems, and threats to food security and human health.",
        "model_answer": "Climate change is mainly caused by the increase in greenhouse gases in Earth's atmosphere, primarily from burning fossil fuels like coal, oil, and natural gas. When we burn these fuels, they release carbon dioxide and other gases that trap heat in the atmosphere. Deforestation also contributes by reducing trees that would otherwise absorb CO2. The main effects include rising global temperatures, melting ice caps and glaciers leading to sea level rise, more extreme weather events including hurricanes and droughts, disruption of ecosystems and wildlife habitats, and threats to agriculture and human health."
    },
    {
        "query": "How does blockchain technology work?",
        "reference_answer": "Blockchain is a decentralized digital ledger that records transactions across many computers. Each block contains transaction data and a cryptographic hash of the previous block, creating a chain. This structure makes the ledger tamper-evident, as changing any block would require changing all subsequent blocks. Consensus mechanisms like Proof of Work or Proof of Stake validate transactions without a central authority.",
        "model_answer": "Blockchain technology works as a distributed database or ledger shared among computer network nodes. It stores information in digital format, with the key innovation being that it guarantees the fidelity and security of data records without requiring a trusted third party. The blockchain consists of blocks, each containing a number of transactions. Once a block is filled with transactions, it's closed and linked to the previously filled block, creating a chain of blocks (hence 'blockchain'). New transactions are verified by network nodes through cryptography and added to the chain. This system is secure    document_store,
#     kg,
#     llm
# )
```

## Document embeddings
**Core Concept:** Vector representations that capture semantic meaning of text for similarity search.

**Detailed Explanation:**
- **Creation Process:**
  - Text preprocessing (tokenization, normalization)
  - Model encoding (transformer-based, typically)
  - Dimensionality manipulation (optional)

- **Key Properties:**
  - High-dimensional vectors (typically 384-1536 dimensions)
  - Semantic similarity via cosine or dot product
  - Language-agnostic representations
  - Preservation of contextual meaning

- **Model Types:**
  - Generic models (e.g., BERT, MPNet)
  - Domain-specific models (e.g., clinical, legal)
  - Instruction-tuned models (e.g., BGE, E5)
  - Asymmetric models (different encoders for queries vs. documents)

**Code Example:**
```python
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import time

# Load embedding model
model_name = "BAAI/bge-small-en-v1.5"
model = SentenceTransformer(model_name)

# Sample documents
documents = [
    "Machine learning is a branch of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural language processing helps computers understand human language.",
    "Computer vision enables machines to interpret visual information.",
    "Reinforcement learning involves training agents through rewards.",
    "Supervised learning uses labeled data for training.",
    "Clustering is an unsupervised learning technique.",
    "Transfer learning leverages knowledge from pre-trained models.",
    "Generative AI can create new content like text and images.",
    "The Turing test evaluates a machine's ability to exhibit human-like intelligence."
]

# Create document embeddings
start_time = time.time()
document_embeddings = model.encode(documents)
encoding_time = time.time() - start_time
print(f"Encoding time for {len(documents)} documents: {encoding_time:.4f} seconds")
print(f"Embedding dimensions: {document_embeddings.shape}")

# Compute similarity matrix
similarity_matrix = cosine_similarity(document_embeddings)

# Visualize similarity matrix
plt.figure(figsize=(10, 8))
plt.imshow(similarity_matrix, cmap='viridis')
plt.colorbar()
plt.title("Document Similarity Matrix")
plt.xticks(np.arange(len(documents)), [f"Doc {i+1}" for i in range(len(documents))], rotation=45)
plt.yticks(np.arange(len(documents)), [f"Doc {i+1}" for i in range(len(documents))])
plt.tight_layout()
plt.savefig("similarity_matrix.png")

# Perform PCA for visualization
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(document_embeddings)

# Plot in 2D space
plt.figure(figsize=(12, 8))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], s=100)

# Add labels
for i, doc in enumerate(documents):
    short_doc = doc[:30] + "..." if len(doc) > 30 else doc
    plt.annotate(f"Doc {i+1}: {short_doc}", 
                 (reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
                 xytext=(5, 5), textcoords='offset points')

plt.title("Document Embeddings in 2D Space")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.tight_layout()
plt.savefig("document_embeddings_2d.png")

# Demonstrate search functionality
def search_documents(query, documents, model, top_k=3):
    # Encode query
    query_embedding = model.encode([query])[0]
    
    # Compute similarities
    similarities = [cosine_similarity([query_embedding], [doc_embedding])[0][0] 
                   for doc_embedding in document_embeddings]
    
    # Get top-k results
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for i in top_indices:
        results.append({
            "document": documents[i],
            "similarity": similarities[i]
        })
    
    return results

# Example searches
queries = [
    "How do neural networks work?",
    "What is unsupervised learning?",
    "How can computers understand human language?"
]

for query in queries:
    print(f"\nQuery: {query}")
    results = search_documents(query, documents, model)
    for i, result in enumerate(results):
        print(f"{i+1}. {result['document']} (Similarity: {result['similarity']:.4f})")

# Compare different embedding models
model_names = [
    "sentence-transformers/all-MiniLM-L6-v2",  # Small, general-purpose
    "BAAI/bge-small-en-v1.5",                  # Instruction-tuned retrieval
    "BAAI/bge-base-en-v1.5",                   # Larger instruction-tuned
    "sentence-transformers/all-mpnet-base-v2"  # High-quality, slower
]

# Benchmark function
def benchmark_models(query, documents, model_names):
    results = []
    
    for model_name in model_names:
        print(f"Testing model: {model_name}")
        
        # Load model
        model = SentenceTransformer(model_name)
        
        # Measure encoding time
        start_time = time.time()
        embeddings = model.encode(documents)
        encoding_time = time.time() - start_time
        
        # Measure search time
        start_time = time.time()
        query_embedding = model.encode([query])[0]
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:3]
        search_time = time.time() - start_time
        
        # Get dimensions
        dimensions = embeddings.shape[1]
        
        # Track results
        results.append({
            "model": model_name,
            "dimensions": dimensions,
            "encoding_time": encoding_time,
            "search_time": search_time,
            "top_results": [documents[i] for i in top_indices]
        })
    
    return results

benchmark_results = benchmark_models(
    "How does machine learning work?",
    documents,
    model_names
)

# Display results as a table
results_df = pd.DataFrame(benchmark_results)
print("\nModel Benchmark Results:")
print(results_df[["model", "dimensions", "encoding_time", "search_time"]])

# Display top result for each model
for result in benchmark_results:
    print(f"\nModel: {result['model']}")
    print("Top result:", result["top_results"][0])
```

**Advanced Implementation Pattern:**
- Document chunking optimization with embedding coherence analysis:
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import seaborn as sns

# Load embedding model
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Sample long document
long_document = """
[Long document text here...]
"""

# Function to analyze chunking strategies
def analyze_chunking_strategies(document, strategies, model):
    results = {}
    
    for name, config in strategies.items():
        print(f"Analyzing {name}...")
        chunk_size = config["chunk_size"]
        chunk_overlap = config["chunk_overlap"]
        
        # Create splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Split document
        chunks = splitter.split_text(document)
        
        # Create embeddings
        embeddings = model.encode(chunks)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Calculate statistics
        coherence_scores = []
        
        # Coherence: average similarity with adjacent chunks
        for i in range(len(chunks)):
            adjacent_indices = []
            if i > 0:
                adjacent_indices.append(i-1)  # Previous chunk
            if i < len(chunks) - 1:
                adjacent_indices.append(i+1)  # Next chunk
                
            if adjacent_indices:
                coherence = np.mean([similarity_matrix[i][j] for j in adjacent_indices])
                coherence_scores.append(coherence)
        
        # Calculate semantic drift
        if len(chunks) >= 2:
            first_last_similarity = similarity_matrix[0][-1]
        else:
            first_last_similarity = 1.0
        
        # Calculate clustering
        if len(chunks) >= 5:
            kmeans = KMeans(n_clusters=min(5, len(chunks)), random_state=42)
            clusters = kmeans.fit_predict(embeddings)
            cluster_distribution = np.bincount(clusters)
            cluster_evenness = np.std(cluster_distribution) / np.mean(cluster_distribution)
        else:
            cluster_evenness = 0
        
        # Store results
        results[name] = {
            "num_chunks": len(chunks),
            "avg_chunk_length": np.mean([len(chunk) for chunk in chunks]),
            "avg_coherence": np.mean(coherence_scores),
            "coherence_std": np.std(coherence_scores),
            "first_last_similarity": first_last_similarity,
            "cluster_evenness": cluster_evenness,
            "similarity_matrix": similarity_matrix,
            "chunks": chunks,
            "embeddings": embeddings
        }
    
    return results

# Define chunking strategies
strategies = {
    "small_chunks": {"chunk_size": 500, "chunk_overlap": 50},
    "medium_chunks": {"chunk_size": 1000, "chunk_overlap": 100},
    "large_chunks": {"chunk_size": 2000, "chunk_overlap": 200},
    "small_high_overlap": {"chunk_size": 500, "chunk_overlap": 200},
    "large_high_overlap": {"chunk_size": 2000, "chunk_overlap": 500}
}

# Analyze strategies
results = analyze_chunking_strategies(long_document, strategies, model)

# Visualize results
plt.figure(figsize=(12, 8))

# Bar chart for coherence
names = list(results.keys())
coherence_values = [results[name]["avg_coherence"] for name in names]
first_last_values = [results[name]["first_last_similarity"] for name in names]

x = np.arange(len(names))
width = 0.35

plt.bar(x - width/2, coherence_values, width, label='Avg Adjacent Coherence')
plt.bar(x + width/2, first_last_values, width, label='First-Last Similarity')

plt.ylabel('Similarity Score')
plt.title('Chunking Strategy Comparison')
plt.xticks(x, names, rotation=45)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("chunking_comparison.png")

# Visualize similarity matrices
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, name in enumerate(results.keys()):
    if i < len(axes):
        matrix = results[name]["similarity_matrix"]
        im = axes[i].imshow(matrix, cmap='viridis')
        axes[i].set_title(f"{name} - {results[name]['num_chunks']} chunks")
        fig.colorbar(im, ax=axes[i])

plt.tight_layout()
plt.savefig("similarity_matrices.png")

# Select optimal strategy
coherence_importance = 0.4
drift_importance = 0.3
cluster_importance = 0.3

scores = {}
for name, result in results.items():
    coherence_score = result["avg_coherence"] * coherence_importance
    drift_score = result["first_last_similarity"] * drift_importance
    cluster_score = (1 - result["cluster_evenness"]) * cluster_importance
    
    total_score = coherence_score + drift_score + cluster_score
    scores[name] = total_score

# Find best strategy
best_strategy = max(scores, key=scores.get)
print(f"\nBest chunking strategy: {best_strategy} (Score: {scores[best_strategy]:.4f})")
print(f"Number of chunks: {results[best_strategy]['num_chunks']}")
print(f"Average chunk length: {results[best_strategy]['avg_chunk_length']:.1f} characters")
print(f"Average coherence: {results[best_strategy]['avg_coherence']:.4f}")
```

## Synthetic data
**Core Concept:** Artificially generated data used for testing, training, and evaluating LLM systems.

**Detailed Explanation:**
- **Generation Approaches:**
  - Template-based generation
  - LLM-based generation (prompt engineering)
  - Adversarial examples creation
  - Programmatic manipulation of real data

- **Use Cases:**
  - Training data augmentation
  - Edge case testing
  - Privacy-preserving development
  - Evaluation dataset creation
  - Model behavior analysis

**Code Example:**
```python
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
import json
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define schemas for synthetic data
class CustomerQuery(BaseModel):
    query: str = Field(description="The customer's question or request")
    intent: str = Field(description="The primary intent of the query")
    entities: List[str] = Field(description="Named entities mentioned in the query")
    sentiment: str = Field(description="Sentiment of the query (positive, negative, neutral)")
    complexity: int = Field(description="Complexity level from 1 (simple) to 5 (complex)")

# Create a parser
parser = PydanticOutputParser(pydantic_object=CustomerQuery)

# Create LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.9)

# Define templates for different domains
domains = {
    "customer_support": {
        "template": """
        Generate a realistic customer support query about {product} related to {issue_type}.
        The query should have a {sentiment} sentiment and complexity level {complexity}.
        
        {format_instructions}
        """,
        "products": ["smartphone", "laptop", "smart TV", "wireless headphones", "fitness tracker"],
        "issue_types": ["technical problem", "billing issue", "return request", "feature question", "compatibility"]
    },
    "travel_booking": {
        "template": """
        Generate a realistic travel booking query about {destination} with {trip_type}.
        The query should have a {sentiment} sentiment and complexity level {complexity}.
        
        {format_instructions}
        """,
        "destinations": ["Europe", "Asia", "North America", "beach resorts", "mountain retreats"],
        "trip_types": ["family vacation", "business trip", "honeymoon", "solo adventure", "group tour"]
    },
    "healthcare": {
        "template": """
        Generate a realistic healthcare query about {condition} related to {concern_type}.
        The query should have a {sentiment} sentiment and complexity level {complexity}.
        
        {format_instructions}
        """,
        "conditions": ["diabetes", "hypertension", "allergies", "pregnancy", "mental health"],
        "concern_types": ["symptoms", "medication", "side effects", "treatment options", "lifestyle changes"]
    }
}

# Function to generate synthetic data
def generate_synthetic_queries(domain, num_samples=10):
    domain_config = domains[domain]
    template = domain_config["template"]
    
    prompt_template = PromptTemplate(
        template=template,
        input_variables=[k for k in domain_config.keys() if k != "template"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    synthetic_data = []
    
    for _ in range(num_samples):
        # Generate random parameters
        params = {}
        for key, values in domain_config.items():
            if key != "template":
                params[key] = random.choice(values)
        
        # Add sentiment and complexity
        params["sentiment"] = random.choice(["positive", "negative", "neutral"])
        params["complexity"] = random.randint(1, 5)
        
        # Generate query
        prompt = prompt_template.format(**params)
        response = llm.predict(prompt)
        
        try:
            parsed_query = parser.parse(response)
            synthetic_data.append(parsed_query.dict())
        except Exception as e:
            print(f"Error parsing response: {e}")
            print(f"Response: {response}")
    
    return synthetic_data

# Generate synthetic data for each domain
all_data = {}
for domain in domains.keys():
    print(f"Generating data for {domain}...")
    all_data[domain] = generate_synthetic_queries(domain, num_samples=20)

# Convert to DataFrame
all_records = []
for domain, records in all_data.items():
    for record in records:
        record["domain"] = domain
        all_records.append(record)

df = pd.DataFrame(all_records)

# Analyze data distribution
print(f"Generated {len(df)} synthetic queries")
print("\nDomain distribution:")
print(df["domain"].value_counts())

print("\nIntent distribution:")
print(df["intent"].value_counts().head(10))

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())

print("\nComplexity distribution:")
print(df["complexity"].value_counts())

# Visualize distributions
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.countplot(data=df, x="domain")
plt.title("Distribution by Domain")
plt.xticks(rotation=45)

plt.subplot(2, 2, 2)
sns.countplot(data=df, x="sentiment", hue="domain")
plt.title("Sentiment Distribution by Domain")

plt.subplot(2, 2, 3)
sns.countplot(data=df, x="complexity", hue="domain")
plt.title("Complexity Distribution by Domain")

plt.subplot(2, 2, 4)
entity_counts = df["entities"].apply(len)
sns.histplot(entity_counts, bins=range(0, max(entity_counts)+2))
plt.title("Number of Entities per Query")
plt.xlabel("Entity Count")

plt.tight_layout()
plt.savefig("synthetic_data_analysis.png")

# Function to create adversarial examples
def create_adversarial_examples(queries, difficulty="medium"):
    prompt_template = PromptTemplate(
        template="""
        Transform the following customer query into a more challenging version that would be difficult for an AI to handle correctly.
        
        Original Query: {query}
        
        Make it more {difficulty} by adding:
        - Ambiguity
        - Multiple intents
        - Spelling errors
        - Uncommon phrasing
        - Contradictory statements
        
        Return the modified query only, without any explanations.
        """,
        input_variables=["query", "difficulty"]
    )
    
    results = []
    for query_data in queries:
        prompt = prompt_template.format(query=query_data["query"], difficulty=difficulty)
        adversarial_query = llm.predict(prompt)
        
        results.append({
            "original_query": query_data["query"],
            "adversarial_query": adversarial_query,
            "original_intent": query_data["intent"],
            "original_sentiment": query_data["sentiment"],
            "domain": query_data.get("domain", "unknown")
        })
    
    return results

# Create adversarial examples
print("\nCreating adversarial examples...")
sample_queries = random.sample(all_records, 10)
adversarial_examples = create_adversarial_examples(sample_queries)

# Display original vs adversarial
print("\nOriginal vs Adversarial Examples:")
for i, example in enumerate(adversarial_examples):
    print(f"\nExample {i+1} ({example['domain']}):")
    print(f"Original: {example['original_query']}")
    print(f"Adversarial: {example['adversarial_query']}")

# Save synthetic data
output_file = "synthetic_queries.json"
with open(output_file, "w") as f:
    json.dump(all_records, f, indent=2)

print(f"\nSaved {len(all_records)} synthetic queries to {output_file}")
```

**Advanced Implementation Pattern:**
- Synthetic dataset generation for RAG evaluation:
```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import json
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define schemas for synthetic RAG evaluation data
class Document(BaseModel):
    id: str = Field(description="Unique identifier for the document")
    title: str = Field(description="Document title")
    content: str = Field(description="Document content")
    metadata: Dict[str, str] = Field(description="Additional document metadata")

class Question(BaseModel):
    id: str = Field(description="Unique identifier for the question")
    text: str = Field(description="The actual question")
    difficulty: str = Field(description="Difficulty level: easy, medium, or hard")
    type: str = Field(description="Question type (factoid, reasoning, etc.)")
    answer_source_ids: List[str] = Field(description="IDs of documents containing the answer")

class GroundTruthAnswer(BaseModel):
    id: str = Field(description="Unique identifier matching the question")
    text: str = Field(description="Model-generated ground truth answer")
    sources: List[str] = Field(description="Source document IDs used")

class EvaluationSet(BaseModel):
    name: str = Field(description="Name of the evaluation dataset")
    description: str = Field(description="Description of the dataset")
    domain: str = Field(description="Domain or topic area")
    documents: List[Document] = Field(description="Corpus of documents")
    questions: List[Question] = Field(description="Evaluation questions")
    ground_truth: List[GroundTruthAnswer] = Field(description="Ground truth answers")

# Create LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# Define domains for synthetic data
domains = [
    {
        "name": "healthcare",
        "description": "Medical and healthcare information",
        "topics": ["diabetes", "cardiovascular health", "mental health", "nutrition", "vaccine safety"]
    },
    {
        "name": "finance",
        "description": "Personal and business finance topics",
        "topics": ["investing", "retirement planning", "credit scores", "taxes", "small business finance"]
    },
    {
        "name": "technology",
        "description": "Computer science and technology concepts",
        "topics": ["machine learning", "cybersecurity", "cloud computing", "mobile technology", "software development"]
    }
]

# Generate synthetic document corpus
def generate_document_corpus(domain, num_documents=20):
    documents = []
    
    # Create document generation prompt
    doc_prompt = ChatPromptTemplate.from_template("""
    You are an expert in {domain}, particularly on the topic of {topic}.
    
    Write a detailed, factual document (300-500 words) about an aspect of {topic}.
    The document should contain specific facts, figures, and information that could be used to answer questions.
    
    Format your response as:
    Title: [Document title]
    
    [Document content with 3-5 paragraphs of detailed information]
    """)
    
    for i in range(num_documents):
        topic = random.choice(domain["topics"])
        
        response = llm.invoke(doc_prompt.format(domain=domain["name"], topic=topic))
        content = response.content
        
        # Extract title and content
        try:
            title_line = content.split("\n")[0]
            title = title_line.replace("Title:", "").strip()
            main_content = "\n".join(content.split("\n")[1:]).strip()
            
            # Create document
            document = Document(
                id=f"{domain['name']}_{i+1:03d}",
                title=title,
                content=main_content,
                metadata={
                    "domain": domain["name"],
                    "topic": topic,
                    "length": str(len(main_content))
                }
            )
            
            documents.append(document)
            print(f"Generated document {i+1}/{num_documents}: {title}")
            
        except Exception as e:
            print(f"Error processing document {i+1}: {e}")
    
    return documents

# Generate questions based on documents
def generate_questions(documents, num_questions=30):
    questions = []
    
    # Group documents by topic
    topics = {}
    for doc in documents:
        topic = doc.metadata["topic"]
        if topic not in topics:
            topics[topic] = []
        topics[topic].append(doc)
    
    # Create question generation prompt
    question_prompt = ChatPromptTemplate.from_template("""
    I'll provide you with information from 1-3 documents about {topic}. 
    Based on this information, generate {num_questions} diverse questions at {difficulty} difficulty level.
    
    The questions should require information from the documents to answer correctly.
    
    Documents:
    {documents}
    
    For each question, specify whether it's a factoid question (simple fact retrieval) or a reasoning question (requires synthesizing information).
    
    Format your response as a JSON array of objects, each with:
    - "question": The actual question text
    - "type": "factoid" or "reasoning"
    - "document_ids": Array of document IDs that contain information needed to answer this question
    
    Generate questions that vary in complexity and focus on different aspects of the documents.
    """)
    
    # Generate questions for each topic
    question_id = 1
    for topic, topic_docs in topics.items():
        for difficulty in ["easy", "medium", "hard"]:
            # Select 1-3 documents for context
            num_docs = min(random.randint(1, 3), len(topic_docs))
            selected_docs = random.sample(topic_docs, num_docs)
            
            # Format document text
            doc_text = "\n\n".join([
                f"Document ID: {doc.id}\nTitle: {doc.title}\n\n{doc.content[:500]}..."
                for doc in selected_docs
            ])
            
            # Generate questions
            questions_per_batch = 3
            response = llm.invoke(
                question_prompt.format(
                    topic=topic,
                    documents=doc_text,
                    difficulty=difficulty,
                    num_questions=questions_per_batch
                )
            )
            
            # Parse questions
            try:
                # Find JSON in response
                content = response.content
                json_start = content.find("[")
                json_end = content.rfind("]") + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    question_data = json.loads(json_str)
                    
                    for q_data in question_data:
                        question = Question(
                            id=f"q{question_id:03d}",
                            text=q_data["question"],
                            difficulty=difficulty,
                            type=q_data["type"],
                            answer_source_ids=q_data["document_ids"]
                        )
                        questions.append(question)
                        question_id += 1
                    
                    print(f"Generated {len(question_data)} {difficulty} questions for {topic}")
                else:
                    print(f"Could not find JSON in response for {topic}, {difficulty}")
                    
            except Exception as e:
                print(f"Error processing questions for {topic}, {difficulty}: {e}")
                print(f"Response: {response.content}")
    
    return questions

# Generate ground truth answers
def generate_ground_truth(documents, questions):
    ground_truth = []
    document_map = {doc.id: doc for doc in documents}
    
    # Create answer generation prompt
    answer_prompt = ChatPromptTemplate.from_template("""
    Answer the following question accurately based ONLY on the information provided in these documents.
    
    Question: {question}
    
    Documents:
    {documents}
    
    Provide a comprehensive answer that directly addresses the question, citing information from the documents.
    If the documents don't contain enough information to fully answer the question, state that explicitly.
    
    Answer:
    """)
    
    for question in questions:
        # Get relevant documents
        relevant_docs = [document_map[doc_id] for doc_id in question.answer_source_ids if doc_id in document_map]
        
        # Format document text
        doc_text = "\n\n".join([
            f"Document ID: {doc.id}\nTitle: {doc.title}\n\n{doc.content}"
            for doc in relevant_docs
        ])
        
        # Generate answer
        response = llm.invoke(
            answer_prompt.format(
                question=question.text,
                documents=doc_text
            )
        )
        
        # Create ground truth
        answer = GroundTruthAnswer(
            id=question.id,
            text=response.content.strip(),
            sources=question.answer_source_ids
        )
        
        ground_truth.append(answer)
        print(f"Generated answer for question {question.id}")
    
    return ground_truth

# Function to create evaluation set
def create_evaluation_set(domain_config, num_documents=15, num_questions=25):
    print(f"\nCreating evaluation set for {domain_config['name']}...")
    
    # Generate documents
    documents = generate_document_corpus(domain_config, num_documents)
    
    # Generate questions
    questions = generate_questions(documents, num_questions)
    
    # Generate ground truth answers
    ground_truth = generate_ground_truth(documents, questions)
    
    # Create evaluation set
    eval_set = EvaluationSet(
        name=f"{domain_config['name']}_eval",
        description=f"Evaluation dataset for {domain_config['name']} domain",
        domain=domain_config['name'],
        documents=documents,
        questions=questions,
        ground_truth=ground_truth
    )
    
    return eval_set

# Create evaluation sets
evaluation_sets = []
for domain in domains:
    eval_set = create_evaluation_set(domain)
    evaluation_sets.append(eval_set)
    
    # Save to file
    output# Comprehensive LLM Engineering Learning Notes

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
Classify the following text into one of these categories: "positive", "negative", or "neutral".

Think step-by-step about the sentiment expressed in the text.

Text: {text}

{format_instructions}
"""

# Create chat model
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Create chain
classification_chain = (
    ChatPromptTemplate.from_template(template)
    .partial(format_instructions=parser.get_format_instructions())
    | llm
    | parser
)

# Example usage
result = classification_chain.invoke({"text": "I absolutely loved the product, but the delivery took too long."})
print(f"Category: {result.category}")
print(f"Confidence: {result.confidence}")
print(f"Reasoning: {result.reasoning}")
```

**Advanced Implementation Pattern:**
- Multi-class classification with confidence thresholds:
```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional

# Define output schema for multi-class classification
class MultiClassPrediction(BaseModel):
    classes: List[str] = Field(description="List of applicable classes, ordered by confidence")
    confidences: List[float] = Field(description="Confidence scores for each class (0-1)")
    explanation: str = Field(description="Explanation for the classification")

class ClassificationResult(BaseModel):
    prediction: str = Field(description="The primary classification")
    confidence: float = Field(description="Confidence in primary classification (0-1)")
    secondary_classes: Optional[List[str]] = Field(description="Other applicable classes")
    explanation: str = Field(description="Reasoning behind the classification")

# Create parser
parser = PydanticOutputParser(pydantic_object=MultiClassPrediction)

# Create prompt
multi_class_prompt = ChatPromptTemplate.from_template("""
Analyze the following text and classify it into one or more of these categories:
- "technical_issue": Technical problems with products or services
- "billing_question": Questions about charges, refunds, or pricing
- "account_access": Problems accessing accounts or services
- "feature_request": Requests for new features or improvements
- "general_inquiry": General questions about products or services
- "complaint": Expressions of dissatisfaction

For each applicable category, provide a confidence score between 0 and 1.
Order the categories by confidence score (highest first).
Provide a brief explanation for your classification.

Text: {text}

{format_instructions}
""")

# Create LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Create classification chain
multi_class_chain = (
    multi_class_prompt
    .partial(format_instructions=parser.get_format_instructions())
    | llm
    | parser
)

# Create post-processing function with confidence threshold
def process_classification(result, confidence_threshold=0.6):
    # Get primary classification
    primary_class = result.classes[0] if result.classes else "unclassified"
    primary_confidence = result.confidences[0] if result.confidences else 0
    
    # Filter secondary classes based on threshold
    secondary_classes = [
        cls for cls, conf in zip(result.classes[1:], result.confidences[1:])
        if conf >= confidence_threshold
    ]
    
    return ClassificationResult(
        prediction=primary_class,
        confidence=primary_confidence,
        secondary_classes=secondary_classes if secondary_classes else None,
        explanation=result.explanation
    )

# Example usage
raw_result = multi_class_chain.invoke({"text": "I can't log into my account and I've been charged twice this month for the premium subscription."})
processed_result = process_classification(raw_result, confidence_threshold=0.6)

print(f"Primary classification: {processed_result.prediction} ({processed_result.confidence:.2f})")
if processed_result.secondary_classes:
    print(f"Secondary classifications: {', '.join(processed_result.secondary_classes)}")
print(f"Explanation: {processed_result.explanation}")
```

## Gradio & LangServe
**Core Concept:** Tools for deploying LLM applications with interactive UIs (Gradio) and API services (LangServe).

**Detailed Explanation:**
- **Gradio:**
  - Rapidly create web UIs for ML models
  - Supports text, image, audio, and video inputs
  - Built-in components like Chatbot, Slider, Dropdown
  - Automatic API generation
  - Easy sharing and collaboration

- **LangServe:**
  - Deploy LangChain chains as API endpoints
  - Built-in request validation
  - Streaming support
  - Automatic API documentation
  - Playground interface

**Code Example (Gradio):**
```python
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Set up LangChain components
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local("faiss_index", embeddings)
llm = ChatOpenAI(model="gpt-4", temperature=0.2)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# Define Gradio interface
def process_query(query, history):
    # Get answer from QA chain
    result = qa_chain({"query": query})
    answer = result["result"]
    
    # Get source documents
    source_docs = result.get("source_documents", [])
    sources = "\n\nSources:\n" + "\n".join([f"- {doc.metadata.get('source', 'Unknown')}" for doc in source_docs])
    
    # Return answer with sources
    return answer + sources

# Create Gradio interface
with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown("# Enterprise Knowledge Base")
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height=600)
            msg = gr.Textbox(placeholder="Ask a question about company policies...", scale=8)
            clear = gr.Button("Clear")
        
        with gr.Column(scale=1):
            gr.Markdown("### Settings")
            temperature = gr.Slider(0, 1, value=0.2, label="Temperature")
            k_documents = gr.Slider(1, 10, value=3, step=1, label="Number of documents")
    
    def update_retriever(k):
        qa_chain.retriever.search_kwargs["k"] = k
        return f"Updated to retrieve {k} documents"
    
    def update_temperature(temp):
        llm.temperature = temp
        return f"Updated temperature to {temp}"
    
    msg.submit(process_query, [msg, chatbot], [chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)
    k_documents.change(update_retriever, k_documents, None)
    temperature.change(update_temperature, temperature, None)

# Launch app
demo.launch()
```

**Code Example (LangServe):**
```python
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langserve import add_routes

# Create FastAPI app
app = FastAPI(
    title="Company Knowledge Base API",
    version="1.0",
    description="API for querying company knowledge base"
)

# Create model and chain
llm = ChatOpenAI(model="gpt-4", temperature=0.2)
template = """
You are a helpful assistant for {company_name}.
Answer the following question:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Create chain
chain = (
    {"company_name": RunnablePassthrough(), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Add routes to the app
add_routes(
    app,
    chain,
    path="/answer",
    input_type=dict,  # Input is a dictionary with company_name and question
    output_type=str,  # Output is a string
    config_keys=["company_name", "question"]  # Configuration keys
)

# Add documentation route
@app.get("/")
async def root():
    return {"message": "Welcome to the Company Knowledge Base API. Visit /docs for documentation."}

# Run with: uvicorn app:app --host 0.0.0.0 --port 8000
```

**Advanced Implementation Pattern:**
- Gradio file upload and processing with streaming:
```python
import gradio as gr
import tempfile
from langchain.document_loaders import PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# File processing functions
def process_file(file_path):
    # Determine file type
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".csv"):
        loader = CSVLoader(file_path)
    else:
        return "Unsupported file format"
    
    # Load documents
    documents = loader.load()
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    
    # Create embeddings and store in vector database
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(chunks, embeddings)
    
    # Return retriever
    return db.as_retriever()

# Streaming response generator
class GradioStreamingHandler(StreamingStdOutCallbackHandler):
    def __init__(self):
        self.tokens = []
        
    def on_llm_new_token(self, token, **kwargs):
        self.tokens.append(token)
        
    def get_response(self):
        return "".join(self.tokens)

# Initialize state
retriever = None
llm = ChatOpenAI(model="gpt-3.5-turbo", streaming=True, temperature=0.2)

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# Document Q&A System")
    
    with gr.Row():
        with gr.Column():
            file_upload = gr.File(label="Upload Document (PDF or CSV)")
            status = gr.Textbox(label="Status", value="No file uploaded")
            
            process_btn = gr.Button("Process Document")
            
            with gr.Accordion("Advanced Settings", open=False):
                model_dropdown = gr.Dropdown(
                    ["gpt-3.5-turbo", "gpt-4"], 
                    label="Model",
                    value="gpt-3.5-turbo"
                )
                temperature = gr.Slider(0, 1, value=0.2, label="Temperature")
        
        with gr.Column():
            chatbot = gr.Chatbot(height=600, label="Q&A")
            query = gr.Textbox(label="Question", placeholder="Ask about the document...")
            submit_btn = gr.Button("Submit")
            clear_btn = gr.Button("Clear Chat")
    
    # Event handlers
    def upload_file(file):
        return f"File uploaded: {file.name}. Click 'Process Document' to continue."
    
    def process_document(file):
        global retriever
        if file is None:
            return "Please upload a file first."
        
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        
        # Process file
        try:
            retriever = process_file(tmp_path)
            return f"Document processed successfully. Ready for questions."
        except Exception as e:
            return f"Error processing document: {str(e)}"
    
    def update_model(model_name):
        global llm
        llm = ChatOpenAI(model=model_name, streaming=True, temperature=float(temperature.value))
        return f"Model updated to {model_name}"
    
    def update_temperature(temp):
        global llm
        llm = ChatOpenAI(model=model_dropdown.value, streaming=True, temperature=float(temp))
        return f"Temperature updated to {temp}"
    
    def submit_question(question, history):
        global retriever
        if retriever is None:
            return history + [[question, "Please upload and process a document first."]]
        
        # Create streaming handler
        stream_handler = GradioStreamingHandler()
        
        # Create QA chain with streaming
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            callbacks=[stream_handler]
        )
        
        # Run query in a streaming fashion
        try:
            qa_chain({"query": question})
            response = stream_handler.get_response()
            return history + [[question, response]]
        except Exception as e:
            return history + [[question, f"Error: {str(e)}"]]
    
    # Set up event handlers
    file_upload.upload(upload_file, file_upload, status)
    process_btn.click(process_document, file_upload, status)
    model_dropdown.change(update_model, model_dropdown, None)
    temperature.change(update_temperature, temperature, None)
    
    submit_btn.click(submit_question, [query, chatbot], chatbot)
    query.submit(submit_question, [query, chatbot], chatbot)
    clear_btn.click(lambda: None, None, chatbot, queue=False)

# Launch app
demo.launch()
```

## JSON & Pydantic
**Core Concept:** Tools for structured data handling, validation, and type safety in Python applications.

**Detailed Explanation:**
- **JSON (JavaScript Object Notation):**
  - Lightweight data interchange format
  - Human-readable text format
  - Key-value pairs and arrays
  - Primary serialization format for APIs

- **Pydantic:**
  - Data validation and settings management
  - Runtime type checking
  - Schema generation
  - JSON serialization/deserialization
  - Integration with FastAPI

**Code Example:**
```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Union
import json
from datetime import datetime

# Define Pydantic models
class Author(BaseModel):
    name: str = Field(..., min_length=1)
    email: Optional[str] = None
    bio: Optional[str] = Field(None, max_length=500)

class Comment(BaseModel):
    text: str
    user_id: int
    created_at: datetime
    
    @validator('created_at', pre=True)
    def parse_datetime(cls, value):
        if isinstance(value, str):
            return datetime.fromisoformat(value.replace('Z', '+00:00'))
        return value

class Article(BaseModel):
    title: str = Field(..., min_length=5, max_length=100)
    content: str
    author: Author
    tags: List[str] = Field(default_factory=list)
    published: bool = False
    view_count: Optional[int] = 0
    comments: List[Comment] = Field(default_factory=list)
    metadata: Dict[str, Union[str, int, bool]] = Field(default_factory=dict)
    
    class Config:
        schema_extra = {
            "example": {
                "title": "Introduction to Pydantic",
                "content": "Pydantic is a data validation library...",
                "author": {
                    "name": "Jane Doe",
                    "email": "jane@example.com"
                },
                "tags": ["python", "validation", "tutorial"],
                "published": True
            }
        }

# Example usage
try:
    # Create from dictionary
    author_data = {
        "name": "John Smith",
        "email": "john@example.com",
        "bio": "Python developer and technical writer"
    }
    
    comment_data = [
        {"text": "Great article!", "user_id": 123, "created_at": "2023-09-15T14:30:00Z"},
        {"text": "Thanks for sharing", "user_id": 456, "created_at": "2023-09-16T09:15:00Z"}
    ]
    
    article_data = {
        "title": "Working with JSON and Pydantic",
        "content": "This article explains how to use JSON with Pydantic...",
        "author": author_data,
        "tags": ["python", "json", "pydantic"],
        "published": True,
        "view_count": 1250,
        "comments": comment_data,
        "metadata": {
            "featured": True,
            "category": "programming",
            "reading_time": 5
        }
    }
    
    # Parse and validate data
    article = Article.parse_obj(article_data)
    
    # Access validated data
    print(f"Article: {article.title} by {article.author.name}")
    print(f"Published: {article.published}")
    print(f"Tags: {', '.join(article.tags)}")
    print(f"Comments: {len(article.comments)}")
    
    # Serialize to JSON
    article_json = article.json(indent=2)
    print("\nJSON Output:")
    print(article_json)
    
    # Modify and validate
    article.view_count += 1
    article.tags.append("beginner")
    
    # Invalid data example
    try:
        invalid_article = Article(
            title="Too short",
            content="",
            author={"name": ""}
        )
    except Exception as e:
        print("\nValidation Error:")
        print(e)
        
except Exception as e:
    print(f"Error: {e}")
```

**Advanced Implementation Pattern:**
- Using Pydantic for LLM output parsing and validation:
```python
from pydantic import BaseModel, Field, validator, root_validator
from typing import List, Optional, Union, Dict
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
import re
from datetime import datetime

# Define complex nested schema
class ContactInfo(BaseModel):
    email: Optional[str] = None
    phone: Optional[str] = None
    
    @validator('email')
    def validate_email(cls, v):
        if v is None:
            return v
        if not re.match(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+, v):
            raise ValueError('Invalid email format')
        return v
    
    @validator('phone')
    def validate_phone(cls, v):
        if v is None:
            return v
        # Strip non-numeric characters
        phone_digits = re.sub(r'\D', '', v)
        if len(phone_digits) < 10:
            raise ValueError('Phone number must have at least 10 digits')
        return v

class Price(BaseModel):
    amount: float
    currency: str
    
    @validator('amount')
    def validate_amount(cls, v):
        if v < 0:
            raise ValueError('Price cannot be negative')
        return round(v, 2)  # Round to 2 decimal places
    
    @validator('currency')
    def validate_currency(cls, v):
        valid_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD']
        if v not in valid_currencies:
            raise ValueError(f'Currency must be one of {valid_currencies}')
        return v

class Product(BaseModel):
    id: str
    name: str = Field(..., min_length=3)
    description: str
    price: Price
    category: List[str]
    available: bool = True
    tags: List[str] = Field(default_factory=list)
    
    class Config:
        validate_assignment = True

class Customer(BaseModel):
    id: str
    name: str
    contact: ContactInfo
    preferences: Dict[str, Union[str, bool, int]] = Field(default_factory=dict)

class Order(BaseModel):
    id: str
    customer: Customer
    products: List[Product]
    order_date: datetime
    total_amount: float
    status: str = "pending"
    
    @root_validator
    def validate_total(cls, values):
        products = values.get('products', [])
        calculated_total = sum(p.price.amount for p in products)
        total_amount = values.get('total_amount')
        
        # Allow small differences due to floating point
        if abs(calculated_total - total_amount) > 0.01:
            values['total_amount'] = calculated_total
        
        return values
    
    @validator('status')
    def validate_status(cls, v):
        valid_statuses = ['pending', 'processing', 'shipped', 'delivered', 'cancelled']
        if v not in valid_statuses:
            raise ValueError(f'Status must be one of {valid_statuses}')
        return v

# Create parser
parser = PydanticOutputParser(pydantic_object=Order)

# Create LLM and prompt
llm = ChatOpenAI(temperature=0)
prompt = ChatPromptTemplate.from_template(
    """Given the following customer inquiry, extract the order information and format it according to the specification.
    
    Customer Inquiry:
    {inquiry}
    
    {format_instructions}
    """
)

# Create chain
chain = (
    prompt
    .partial(format_instructions=parser.get_format_instructions())
    | llm
    | parser
)

# Example usage
customer_inquiry = """
Hi, I'd like to place an order. My name is Sarah Johnson, and my email is sarah.j@example.com. 
I'd like to order a Premium Coffee Maker ($129.99) and a set of 4 Ceramic Coffee Mugs ($24.95).
Please deliver it as soon as possible.
"""

try:
    # Parse structured data from unstructured text
    result = chain.invoke({"inquiry": customer_inquiry})
    
    # Work with validated data
    print(f"Order #{result.id}")
    print(f"Customer: {result.customer.name}")
    print(f"Products:")
    for product in result.products:
        print(f"- {product.name}: ${product.price.amount} {product.price.currency}")
    print(f"Total: ${result.total_amount}")
    print(f"Status: {result.status}")
    
    # Update order status
    result.status = "processing"
    
    # Serialization
    order_json = result.json(indent=2)
    print("\nJSON representation:")
    print(order_json)
    
except Exception as e:
    print(f"Error: {e}")
```

## Context limits
**Core Concept:** Maximum token capacity of LLMs that impacts design decisions for handling long inputs.

**Detailed Explanation:**
- **Token Definition:** Subword units that models use to process text
- **Common Context Windows:**
  - GPT-3.5: 4K-16K tokens
  - GPT-4: 8K-32K tokens
  - Claude 3: Up to 200K tokens
  - Llama 3: 8K-128K tokens

- **Implications:**
  - Document chunking strategies
  - Retrieval design
  - Cost considerations
  - Memory management

**Code Example:**
```python
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

# Load tokenizer
encoding = tiktoken.encoding_for_model("gpt-4")

# Function to count tokens
def count_tokens(text):
    tokens = encoding.encode(text)
    return len(tokens)

# Example text
sample_text = """
[Long document text here...]
"""

# Count tokens in full text
token_count = count_tokens(sample_text)
print(f"Document contains {token_count} tokens")

# Create text splitter based on token count
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",  # OpenAI's encoding
    chunk_size=1000,  # Target tokens per chunk
    chunk_overlap=100,  # Overlap between chunks
    length_function=count_tokens,  # Our token counter
)

# Split text into chunks
chunks = text_splitter.split_text(sample_text)

# Print chunk information
for i, chunk in enumerate(chunks):
    chunk_tokens = count_tokens(chunk)
    print(f"Chunk {i+1}: {chunk_tokens} tokens")

# Function to process text within context limits
def process_within_limits(text, model_name="gpt-4", max_tokens=7000):
    # Count input tokens
    input_tokens = count_tokens(text)
    
    # Check if within limits
    if input_tokens <= max_tokens:
        # Process directly
        llm = ChatOpenAI(model=model_name)
        return llm.predict(text)
    else:
        # Split and process in chunks
        chunks = text_splitter.split_text(text)
        results = []
        
        for chunk in chunks:
            llm = ChatOpenAI(model=model_name)
            result = llm.predict(f"Process this text chunk: {chunk}")
            results.append(result)
        
        # Combine results
        combined = "\n\n".join(results)
        
        # Summarize if needed
        if count_tokens(combined) > max_tokens:
            llm = ChatOpenAI(model=model_name)
            return llm.predict(f"Summarize these results: {combined}")
        
        return combined
```

**Advanced Implementation Pattern:**
- Dynamic chunking based on semantic boundaries:
```python
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language
)
import tiktoken
import re
from langchain_openai import OpenAIEmbeddings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load tokenizer
encoding = tiktoken.encoding_for_model("gpt-4")

# Create embeddings model
embeddings_model = OpenAIEmbeddings()

# Helper functions
def count_tokens(text):
    return len(encoding.encode(text))

def get_embedding(text):
    return embeddings_model.embed_query(text)

def semantic_similarity(embed1, embed2):
    return cosine_similarity([embed1], [embed2])[0][0]

# Function to identify document type
def identify_document_type(text):
    # Check for code patterns
    code_patterns = [
        r'```python', r'```javascript', r'```java', r'```c\+\+',
        r'def\s+\w+\s*\(', r'function\s+\w+\s*\(', r'class\s+\w+\s*\{',
        r'import\s+\w+', r'from\s+\w+\s+import'
    ]
    
    for pattern in code_patterns:
        if re.search(pattern, text):
            return "code"
    
    # Check for legal document patterns
    legal_patterns = [
        r'WHEREAS', r'THEREFORE', r'Article \d+', r'Section \d+\.\d+',
        r'pursuant to', r'hereinafter', r'notwithstanding'
    ]
    
    for pattern in legal_patterns:
        if re.search(pattern, text):
            return "legal"
    
    # Check for academic patterns
    academic_patterns = [
        r'Abstract', r'Introduction', r'Methodology', r'References',
        r'et al\.', r'\(\d{4}\)', r'Fig\. \d+'
    ]
    
    for pattern in academic_patterns:
        if re.search(pattern, text):
            return "academic"
    
    # Default to general text
    return "general"

# Create specialized text splitters
def get_specialized_splitter(doc_type, chunk_size=1000, chunk_overlap=100):
    if doc_type == "code":
        return RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    elif doc_type == "legal":
        return RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", ";", ","],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    elif doc_type == "academic":
        return RecursiveCharacterTextSplitter(
            separators=["\n## ", "\n### ", "\n\n", "\n", ".", ";"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    else:
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

# Smart document chunking with semantic coherence
def smart_document_chunking(text, target_size=1000, min_size=200, max_size=1500):
    # Identify document type
    doc_type = identify_document_type(text)
    print(f"Document type: {doc_type}")
    
    # Get specialized splitter
    base_splitter = get_specialized_splitter(doc_type, chunk_size=max_size)
    
    # Get initial chunks
    initial_chunks = base_splitter.split_text(text)
    
    # Process chunks for semantic coherence
    final_chunks = []
    current_chunk = ""
    current_tokens = 0
    
    # Get embedding for first paragraph to serve as semantic anchor
    first_para = re.split(r'\n\n|\n', text)[0]
    anchor_embedding = get_embedding(first_para)
    
    for i, chunk in enumerate(initial_chunks):
        chunk_tokens = count_tokens(chunk)
        
        # If adding this chunk exceeds target but we have content, finalize current chunk
        if current_tokens + chunk_tokens > target_size and current_tokens >= min_size:
            final_chunks.append(current_chunk)
            current_chunk = chunk
            current_tokens = chunk_tokens
            
            # Create new semantic anchor based on this chunk
            anchor_embedding = get_embedding(chunk)
        else:
            # Check semantic similarity if we're about to add to an existing chunk
            if current_chunk:
                chunk_embedding = get_embedding(chunk)
                similarity = semantic_similarity(anchor_embedding, chunk_embedding)
                
                # If similarity is low, create a new chunk
                if similarity < 0.7:  # Threshold for semantic similarity
                    final_chunks.append(current_chunk)
                    current_chunk = chunk
                    current_tokens = chunk_tokens
                    anchor_embedding = chunk_embedding
                else:
                    # Otherwise append to current chunk
                    current_chunk += "\n\n" + chunk
                    current_tokens += chunk_tokens
            else:
                # First chunk in a new sequence
                current_chunk = chunk
                current_tokens = chunk_tokens
    
    # Add the last chunk if it has content
    if current_chunk:
        final_chunks.append(current_chunk)
    
    # Print chunk statistics
    chunk_sizes = [count_tokens(c) for c in final_chunks]
    print(f"Created {len(final_chunks)} chunks")
    print(f"Average chunk size: {sum(chunk_sizes)/len(chunk_sizes):.1f} tokens")
    print(f"Min chunk size: {min(chunk_sizes)} tokens")
    print(f"Max chunk size: {max(chunk_sizes)} tokens")
    
    return final_chunks

# Example usage
document = """
[Very long document text here...]
"""

# Process with smart chunking
chunks = smart_document_chunking(document)
```

## Bi-encoders and Cross-encoders
**Core Concept:** Two approaches for computing similarity between texts with different tradeoffs.

**Detailed Explanation:**
- **Bi-encoders:**
  - Encode query and documents separately
  - Fast similarity computation (dot product or cosine)
  - Easily scalable to large document collections
  - Used in first-stage retrieval

- **Cross-encoders:**
  - Encode query and document pairs together
  - More accurate relevance assessment
  - Computationally expensive
  - Used for re-ranking small candidate sets

**Code Example:**
```python
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import numpy as np
import time
from typing import List, Dict, Tuple

# Load bi-encoder model
bi_encoder = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

# Load cross-encoder model
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Sample corpus
corpus = [
    "Artificial intelligence (AI) is intelligence demonstrated by machines.",
    "Machine learning is a subset of AI focused on data-based learning.",
    "Natural language processing (NLP) enables machines to understand text.",
    "Computer vision allows machines to derive information from images.",
    "Neural networks are computing systems inspired by biological neural networks.",
    "Deep learning uses neural networks with many layers.",
    "Reinforcement learning trains algorithms using a reward system.",
    "Supervised learning uses labeled training data.",
    "Unsupervised learning finds patterns in unlabeled data.",
    "Transfer learning applies knowledge from one domain to another."
]

# Encode corpus using bi-encoder (done once, offline)
start_time = time.time()
corpus_embeddings = bi_encoder.encode(corpus, convert_to_tensor=True)
bi_encode_time = time.time() - start_time
print(f"Bi-encoder corpus encoding time: {bi_encode_time:.4f} seconds")

# Function to retrieve with bi-encoder
def retrieve_with_bi_encoder(query: str, top_k: int = 3) -> List[Dict]:
    # Encode query
    start_time = time.time()
    query_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    
    # Compute similarities
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    
    # Get top-k results
    top_results = np.argsort(-cos_scores.numpy())[:top_k]
    bi_encoder_time = time.time() - start_time
    
    results = []
    for idx in top_results:
        results.append({
            'corpus_id': idx,
            'text': corpus[idx],
            'score': float(cos_scores[idx])
        })
    
    return results, bi_encoder_time

# Function to re-rank with cross-encoder
def rerank_with_cross_encoder(query: str, candidates: List[Dict]) -> List[Dict]:
    # Prepare pairs for cross-encoder
    start_time = time.time()
    pairs = [(query, candidate['text']) for candidate in candidates]
    
    # Get scores
    scores = cross_encoder.predict(pairs)
    
    # Add scores to candidates
    for i in range(len(candidates)):
        candidates[i]['cross_score'] = float(scores[i])
    
    # Sort by cross-encoder score
    reranked = sorted(candidates, key=lambda x: x['cross_score'], reverse=True)
    cross_encoder_time = time.time() - start_time
    
    return reranked, cross_encoder_time

# Two-stage retrieval function
def two_stage_retrieval(query: str, first_stage_k: int = 5, final_k: int = 3) -> Tuple[List[Dict], Dict]:
    # Stage 1: Retrieve candidates with bi-encoder
    candidates, bi_time = retrieve_with_bi_encoder(query, first_stage_k)
    
    # Stage 2: Re-rank candidates with cross-encoder
    reranked, cross_time = rerank_with_cross_encoder(query, candidates)
    
    # Return top-k after re-ranking
    timing = {
        'bi_encoder': bi_time,
        'cross_encoder': cross_time,
        'total': bi_time + cross_time
    }
    
    return reranked[:final_k], timing

# Example usage
query = "How do machines learn from data?"

print("\nQuery:", query)

# Method 1: Bi-encoder only
bi_results, bi_time = retrieve_with_bi_encoder(query)
print("\nBi-encoder results:")
for i, result in enumerate(bi_results):
    print(f"{i+1}. {result['text']} (Score: {result['score']:.4f})")
print(f"Retrieval time: {bi_time:.4f} seconds")

# Method 2: Two-stage retrieval
two_stage_results, timing = two_stage_retrieval(query)
print("\nTwo-stage retrieval results:")
for i, result in enumerate(two_stage_results):
    print(f"{i+1}. {result['text']} (Bi-encoder: {result['score']:.4f}, Cross-encoder: {result['cross_score']:.4f})")
print(f"Total retrieval time: {timing['total']:.4f} seconds")
print(f"- Bi-encoder time: {timing['bi_encoder']:.4f} seconds")
print(f"- Cross-encoder time: {timing['cross_encoder']:.4f} seconds")
```

**Advanced Implementation Pattern:**
- Hybrid retrieval with ColBERT late-interaction model:
```python
import torch
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Optional
import time

# Simplified implementation inspired by ColBERT
class ColBERTRetriever:
    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.index = None
        self.documents = []
        
    def _encode(self, texts: List[str], is_query: bool = False) -> torch.Tensor:
        # Add [Q] token for queries
        if is_query:
            texts = ['[Q] ' + text for text in texts]
        
        # Tokenize
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Get token embeddings
        with torch.no_grad():
            outputs = self.model(**tokens)
            embeddings = outputs.last_hidden_state
        
        # Create attention mask for tokens
        mask = tokens.attention_mask.unsqueeze(-1)
        
        # Apply mask and normalize
        embeddings = embeddings * mask
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=2)
        
        return embeddings, mask
    
    def index_documents(self, documents: List[str]):
        self.documents = documents
        
        # Process documents in batches
        batch_size = 8
        all_embeddings = []
        all_masks = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            embeddings, masks = self._encode(batch)
            all_embeddings.append(embeddings)
            all_masks.append(masks)
        
        # Concatenate batches
        self.doc_embeddings = torch.cat(all_embeddings, dim=0)
        self.doc_masks = torch.cat(all_masks, dim=0)
        
        # Create flattened index for approximate search
        # This is a simplified version; production would use FAISS
        self.doc_tokens = []
        for i in range(len(documents)):
            for j in range(self.doc_embeddings.shape[1]):
                if self.doc_masks[i, j, 0] > 0:
                    self.doc_tokens.append((i, j))
        
        # Create FAISS index
        embed_dim = self.doc_embeddings.shape[2]
        flat_embeddings = []
        
        for i, j in self.doc_tokens:
            flat_embeddings.append(self.doc_embeddings[i, j].numpy())
        
        flat_embeddings = np.array(flat_embeddings).astype('float32')
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(embed_dim)
        self.index.add(flat_embeddings)
    
    def search(self, query: str, k: int = 3, max_docs: Optional[int] = None) -> List[Dict]:
        # Encode query
        query_embeddings, query_mask = self._encode([query], is_query=True)
        query_embeddings = query_embeddings[0]
        query_mask = query_mask[0]
        
        # Get query tokens
        query_tokens = []
        for j in range(query_embeddings.shape[0]):
            if query_mask[j, 0] > 0:
                query_tokens.append(query_embeddings[j].numpy())
        
        query_tokens = np.array(query_tokens).astype('float32')
        
        # Search for each query token
        scores, indices = self.index.search(query_tokens, k * 10)
        
        # Aggregate document scores
        doc_scores = {}
        
        for token_scores, token_indices in zip(scores, indices):
            for score, idx in zip(token_scores, token_indices):
                doc_idx, _ = self.doc_tokens[idx]
                doc_scores[doc_idx] = doc_scores.get(doc_idx, 0) + score
        
        # Sort and get top documents
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        if max_docs:
            sorted_docs = sorted_docs[:max_docs]
        
        # Format results
        results = []
        for doc_idx, score in sorted_docs:
            results.append({
                'corpus_id': doc_idx,
                'text': self.documents[doc_idx],
                'score': float(score)
            })
        
        return results

# Example usage
corpus = [
    "Artificial intelligence (AI) is intelligence demonstrated by machines.",
    "Machine learning is a subset of AI focused on data-based learning.",
    "Natural language processing (NLP) enables machines to understand text.",
    "Computer vision allows machines to derive information from images.",
    "Neural networks are computing systems inspired by biological neural networks.",
    "Deep learning uses neural networks with many layers.",
    "Reinforcement learning trains algorithms using a reward system.",
    "Supervised learning uses labeled training data.",
    "Unsupervised learning finds patterns in unlabeled data.",
    "Transfer learning applies knowledge from one domain to another."
]

# Create and index with ColBERT
print("Indexing documents with ColBERT...")
colbert = ColBERTRetriever()
colbert.index_documents(corpus)

# Search
query = "How do machines learn from data?"
print(f"\nQuery: {query}")

start_time = time.time()
results = colbert.search(query, k=3)
search_time = time.time() - start_time

print("\nColBERT results:")
for i, result in enumerate(results):
    print(f"{i+1}. {result['text']} (Score: {result['score']:.4f})")
print(f"Retrieval time: {search_time:.4f} seconds")
```

## Knowledge bases
**Core Concept:** Structured repositories of information that enhance LLM responses with verified facts and relationships.

**Detailed Explanation:**
- **Types of Knowledge Bases:**
  - **Knowledge Graphs:** Entity-relationship structures
  - **Relational Databases:** Traditional SQL databases
  - **Document Databases:** Unstructured or semi-structured documents
  - **Vector Databases:** Semantic embeddings for similarity search

- **Integration Approaches:**
  - Direct querying based on question analysis
  - RAG with structured knowledge retrieval
  - Entity linking between text and knowledge base
  - Hybrid search combining multiple sources

**Code Example:**
```python
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import sqlite3
import pandas as pd
from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional

# Class for SQLite knowledge base
class SQLiteKnowledgeBase:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
    
    def execute_query(self, query: str, params: tuple = ()) -> List[tuple]:
        self.cursor.execute(query, params)
        return self.cursor.fetchall()
    
    def get_schema(self) -> Dict[str, List[str]]:
        # Get all tables
        tables = self.execute_query("SELECT name FROM sqlite_master WHERE type='table';")
        
        schema = {}
        for table in tables:
            table_name = table[0]
            # Get columns for each table
            columns = self.execute_query(f"PRAGMA table_info({table_name});")
            column_names = [col[1] for col in columns]
            schema[table_name] = column_names
        
        return schema
    
    def search_by_entity(self, entity: str, limit: int = 5) -> List[Dict[str, Any]]:
        results = []
        schema = self.get_schema()
        
        for table_name, columns in schema.items():
            # Search in text columns
            for column in columns:
                try:
                    query = f"SELECT * FROM {table_name} WHERE {column} LIKE ? LIMIT {limit}"
                    rows = self.execute_query(query, (f"%{entity}%",))
                    
                    if rows:
                        for row in rows:
                            result = {
                                "table": table_name,
                                "data": dict(zip(columns, row))
                            }
                            results.append(result)
                except:
                    # Skip columns that can't be searched (e.g., binary data)
                    pass
        
        return results
    
    def natural_language_query(self, question: str, llm) -> str:
        # Get schema information
        schema = self.get_schema()
        schema_str = ""
        for table, columns in schema.items():
            schema_str += f"Table: {table}\nColumns: {', '.join(columns)}\n\n"
        
        # Create SQL generation prompt
        prompt = PromptTemplate(
            input_variables=["schema", "question"],
            template="""
            You are an expert SQL query generator. Given the following database schema and natural language question,
            generate a SQLite query that would answer the question.
            
            Database Schema:
            {schema}
            
            Question: {question}
            
            SQLite Query:
            """
        )
        
        # Generate SQL query
        sql_chain = LLMChain(llm=llm, prompt=prompt)
        sql_query = sql_chain.run(schema=schema_str, question=question)
        
        # Clean up query
        sql_query = sql_query.strip().replace(";", "").strip()
        
        try:
            # Execute query
            results = self.execute_query(sql_query)
            
            # Format results
            if results:
                df = pd.DataFrame(results)
                return f"SQL Query: {sql_query}\n\nResults:\n{df.to_string()}"
            else:
                return f"SQL Query: {sql_query}\n\nNo results found."
        except Exception as e:
            return f"Error executing query: {str(e)}\n\nGenerated query: {sql_query}"
    
    def close(self):
        self.conn.close()

# Class for Neo4j knowledge graph
class Neo4jKnowledgeGraph:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def execute_query(self, query: str, params: dict = None) -> List[Dict]:
        with self.driver.session() as session:
            result = session.run(query, params)
            return [record.data() for record in result]
    
    def get_schema(self) -> Dict:
        # Get node labels
        node_query = "CALL db.labels()"
        node_labels = self.execute_query(node_query)
        
        # Get relationship types
        rel_query = "CALL db.relationshipTypes()"
        rel_types = self.execute_query(rel_query)
        
        # Get property keys
        prop_query = "CALL db.propertyKeys()"
        properties = self.execute_query(prop_query)
        
        return {
            "node_labels": [label["label"] for label in node_labels],
            "relationship_types": [rel["relationshipType"] for rel in rel_types],
            "property_keys": [prop["propertyKey"] for prop in properties]
        }
    
    def search_entity(self, entity: str, limit: int = 5) -> List[Dict]:
        query = """
        MATCH (n)
        WHERE n.name CONTAINS $entity OR n.description CONTAINS $entity
        RETURN n
        LIMIT $limit
        """
        return self.execute_query(query, {"entity": entity, "limit": limit})
    
    def get_relationships(self, entity: str, limit: int = 10) -> List[Dict]:
        query = """
        MATCH (n)-[r]-(m)
        WHERE n.name CONTAINS $entity
        RETURN n, r, m
        LIMIT $limit
        """
        return self.execute_query(query, {"entity": entity, "limit": limit})
    
    def natural_language_query(self, question: str, llm) -> str:
        # Get schema information
        schema = self.get_schema()
        schema_str = "Node labels: " + ", ".join(schema["node_labels"]) + "\n"
        schema_str += "Relationship types: " + ", ".join(schema["relationship_types"]) + "\n"
        schema_str += "Property keys: " + ", ".join(schema["property_keys"])
        
        # Create Cypher generation prompt
        prompt = PromptTemplate(
            input_variables=["schema", "question"],
            template="""
            You are an expert Cypher query generator. Given the following Neo4j graph schema and natural language question,
            generate a Cypher query that would answer the question.
            
            Graph Schema:
            {schema}
            
            Question: {question}
            
            Cypher Query:
            """
        )
        
        # Generate Cypher query
        cypher_chain = LLMChain(llm=llm, prompt=prompt)
        cypher_query = cypher_chain.run(schema=schema_str, question=question)
        
        # Clean up query
        cypher_query = cypher_query.strip()
        
        try:
            # Execute query
            results = self.execute_query(cypher_query)
            
            # Format results
            if results:
                return f"Cypher Query: {cypher_query}\n\nResults:\n{results}"
            else:
                return f"Cypher Query: {cypher_query}\n\nNo results found."
        except Exception as e:
            return f"Error executing query: {str(e)}\n\nGenerated query: {cypher_query}"
    
    def close(self):
        self.driver.close()

# Example usage
def hybrid_knowledge_search(question: str, sql_kb: Optional[SQLiteKnowledgeBase] = None, 
                           graph_kb: Optional[Neo4jKnowledgeGraph] = None, llm = None):
    if not llm:
        llm = OpenAI(temperature=0)
    
    # Question analysis prompt
    prompt = PromptTemplate(
        input_variables=["question"],
        template="""
        Analyze the following question and determine which knowledge source would be most appropriate:
        1. Relational database - for structured data, aggregations, counts, specific lookups
        2. Knowledge graph - for relationships, connections, paths between entities
        3. Both - if the question requires data from both sources
        
        Question: {question}
        
        Answer with only "relational", "graph", or "both".
        """
    )
    
    # Determine which knowledge source to use
    source_chain = LLMChain(llm=llm, prompt=prompt)
    source = source_chain.run(question=question).strip().lower()
    
    results = {}
    
    # Query appropriate sources
    if source in ["relational", "both"] and sql_kb:
        results["relational"] = sql_kb.natural_language_query(question, llm)
    
    if source in ["graph", "both"] and graph_kb:
        results["graph"] = graph_kb.natural_language_query(question, llm)
    
    # Integration prompt
    if len(results) > 1:
        integration_prompt = PromptTemplate(
            input_variables=["question", "relational_results", "graph_results"],
            template="""
            You need to integrate information from multiple knowledge sources to answer this question:
            
            Question: {question}
            
            Relational Database Results:
            {relational_results}
            
            Knowledge Graph Results:
            {graph_results}
            
            Provide a comprehensive answer combining all relevant information:
            """
        )
        
        integration_chain = LLMChain(llm=llm, prompt=integration_prompt)
        final_answer = integration_chain.run(
            question=question,
            relational_results=results.get("relational", "No results"),
            graph_results=results.get("graph", "No results")
        )
        
        return {
            "answer": final_answer,
            "sources": results
        }
    elif results:
        # Just return the single result
        key = list(results.keys())[0]
        return {
            "answer": f"Based on the {key} database: " + results[key],
            "sources": results
        }
    else:
        return {
            "answer": "No knowledge sources available to answer this question.",
            "sources": {}
        }
```

**Advanced Implementation Pattern:**
- Knowledge graph-augmented RAG:
```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
from typing import List, Dict, Any, Optional
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from neo4j import GraphDatabase

# Neo4j knowledge graph helper
class KnowledgeGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def query(self, cypher_query, parameters=None):
        with self.driver.session() as session:
            result = session.run(cypher_query, parameters or {})
            return [record.data() for record in result]
    
    def get_entity_info(self, entity_name):
        # Query to get entity details and relationships
        query = """
        MATCH (n) 
        WHERE toLower(n.name) CONTAINS toLower($name) 
        OPTIONAL MATCH (n)-[r]->(m)
        RETURN n AS entity, 
               collect({relationship: type(r), target: m.name, properties: properties(r)}) AS relationships,
               labels(n) AS types
        LIMIT 5
        """
        return self.query(query, {"name": entity_name})
    
    def get_entity_neighbors(self, entity_name, max_hops=2):
        # Query to get neighborhood subgraph
        query = f"""
        MATCH path = (n)-[*1..{max_hops}]-(m)
        WHERE toLower(n.name) CONTAINS toLower($name)
        RETURN path
        LIMIT 50
        """
        return self.query(query, {"name": entity_name})
    
    def visualize_subgraph(self, entity_name, max_hops=1):
        # Get subgraph data
        result = self.get_entity_neighbors(entity_name, max_hops)
        
        # Create NetworkX graph
        G = nx.DiGraph()
        
        # Process paths and add nodes/edges
        for record in result:
            path = record["path"]
            nodes = path.nodes
            relationships = path.relationships
            
            # Add nodes
            for node in nodes:
                node_properties = dict(node)
                node_id = node.id
                node_name = node_properties.get("name", f"Node_{node_id}")
                G.add_node(node_id, name=node_name, properties=node_properties)
            
            # Add edges
            for rel in relationships:
                start_node = rel.start_node.id
                end_node = rel.end_node.id
                rel_type = rel.type
                G.add_edge(start_node, end_node, type=rel_type)
        
        # Visualize
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=False, node_color="skyblue", node_size=700, arrows=True)
        
        # Add labels
        labels = {node: G.nodes[node]["name"] for node in G.nodes}
        nx.draw_networkx_labels(G, pos, labels, font_size=10)
        
        # Add edge labels
        edge_labels = {(u, v): G.edges[u, v]["type"] for u, v in G.edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        plt.title(f"Knowledge Graph around '{entity_name}'")
        plt.axis("off")
        return plt

# Entity extraction and linking
def extract_entities(text, llm):
    prompt = ChatPromptTemplate.from_template("""
    Extract all meaningful entities from the following text. Focus on specific named entities, 
    technical terms, concepts, and important objects. Return them as a comma-separated list.
    
    Text: {text}
    
    Entities:
    """)
    
    chain = prompt | llm | (lambda x: x.content.split(","))
    entities = chain.invoke({"text": text})
    
    # Clean entities
    cleaned_entities = [entity.strip() for entity in entities if entity.strip()]
    return cleaned_entities

# Get knowledge graph context for entities
def get_kg_context(entities, kg, max_entities=3):
    all_entity_info = []
    
    for entity in entities[:max_entities]:
        entity_data = kg.get_entity_info(entity)
        if entity_data:
            all_entity_info.append({
                "entity": entity,
                "data": entity_data
            })
    
    # Format knowledge graph context
    if not all_entity_info:
        return "No relevant knowledge graph information found."
    
    formatted_context = []
    
    for item in all_entity_info:
        entity = item["entity"]
        entity_data = item["data"]
        
        for data in entity_data:
            entity_node = data["entity"]
            entity_name = entity_node.get("name", entity)
            entity_types = ", ".join(data["types"])
            relationships = data["relationships"]
            
            entity_info = f"Entity: {entity_name} (Types: {entity_types})\n"
            
            # Add properties
            properties = [f"- {key}: {value}" for key, value in entity_node.items() if key != "name"]
            if properties:
                entity_info += "Properties:\n" + "\n".join(properties) + "\n"
            
            # Add relationships
            if relationships:
                entity_info += "Relationships:\n"
                for rel in relationships:
                    if rel["target"]:
                        entity_info += f"- {rel['relationship']} -> {rel['target']}\n"
            
            formatted_context.append(entity_info)
    
    return "\n\n".join(formatted_context)

# RAG with knowledge graph augmentation
def kg_augmented_rag(query, document_store, kg, llm):
    # Extract entities from query
    entities = extract_entities(query, llm)
    
    # Get document context
    retriever = document_store.as_retriever(search_kwargs={"k": 3})
    
    # Get knowledge graph context
    kg_context = get_kg_context(entities, kg)
    
    # Create retrieval chain with knowledge graph context
    retrieval_chain = RunnableParallel({
        "documents": retriever,
        "kg_context": lambda x: kg_context,
        "question": RunnablePassthrough()
    })
    
    # Create prompt
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based on both the document context and knowledge graph information.
    
    Document Context:
    {documents}
    
    Knowledge Graph Context:
    {kg_context}
    
    Question: {question}
    
    Answer:
    """)
    
    # Create chain
    chain = (
        retrieval_chain
        | prompt
        | llm
        | (lambda x: x.content)
    )
    
    # Run chain
    result = chain.invoke(query)
    return {
        "answer": result,
        "extracted_entities": entities,
        "kg_context": kg_context
    }

# Example usage
# uri = "neo4j://localhost:7687"
# user = "neo4j"
# password = "password"
# 
# kg = KnowledgeGraph(uri, user, password)
# 
# embeddings = OpenAIEmbeddings()
# documents = [...] # Your documents
# document_store = Chroma.from_documents(documents, embeddings)
# 
# llm = ChatOpenAI(model="gpt-4")
# 
# result = kg_augmented_rag(
#     "What are the relationships between carbon emissions and global warming?",
#     document_store,
