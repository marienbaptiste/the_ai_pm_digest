export const lessons = {

  // ─────────────────────────────────────────────
  // L01 — Embeddings: From Words to Vectors
  // ─────────────────────────────────────────────
  l01: {
    title: 'Embeddings — From Words to Vectors',
    content: `
<h2>Why Embeddings Are the Foundation of Modern AI</h2>
<p>
  Before a machine can reason about text, images, or any data, it needs a numerical representation.
  An <span class="term" data-term="embedding-model">embedding</span> is a learned mapping from
  discrete, high-dimensional input (like words or documents) to dense, continuous vectors in a
  lower-dimensional space. These vectors capture <em>semantic meaning</em> — similar concepts
  land near each other, and the geometric relationships between vectors encode meaningful analogies.
</p>
<p>
  Embeddings are not just a preprocessing step — they are the core representation that determines
  what an AI system can and cannot do. As an AI PM, understanding embeddings will help you make
  critical decisions about search quality, retrieval performance, and the entire
  <span class="term" data-term="rag">RAG</span> pipeline.
</p>

<h2>From One-Hot to Dense Vectors: The Evolution</h2>
<p>
  Early NLP represented words as <strong>one-hot vectors</strong>: a vocabulary of 50,000 words
  meant each word was a 50,000-dimensional vector with a single 1 and 49,999 zeros. This
  representation has two fatal flaws:
</p>
<ul>
  <li><strong>No semantic similarity:</strong> "king" and "queen" are as distant as "king" and
      "refrigerator" — every pair of words is equally far apart.</li>
  <li><strong>Dimensionality explosion:</strong> The vector size scales linearly with vocabulary size,
      making computation intractable.</li>
</ul>
<p>
  The breakthrough came with <strong>distributed representations</strong> — the idea that meaning
  can be captured by patterns of activation across many dimensions, not by a single dimension.
</p>

<h3>Word2Vec (Mikolov et al., 2013)</h3>
<p>
  Word2Vec trained a shallow neural network to predict a word from its context (CBOW) or predict
  context from a word (Skip-gram). The hidden layer weights became the word embeddings — typically
  100-300 dimensions.
</p>
<p>
  The famous result: <code>vector("king") - vector("man") + vector("woman") ≈ vector("queen")</code>.
  This showed that embedding spaces encode <em>relational structure</em>, not just similarity.
</p>

<h3>GloVe (Pennington et al., 2014)</h3>
<p>
  GloVe (Global Vectors) took a different approach: instead of prediction, it factorized the
  word-word co-occurrence matrix. The result was similar quality embeddings with a more principled
  mathematical framework.
</p>

<div class="interactive" data-interactive="embedding-projector"></div>

<h3>Contextual Embeddings (ELMo, BERT, GPT)</h3>
<p>
  Static embeddings (Word2Vec, GloVe) assign one vector per word regardless of context. The word
  "bank" gets the same embedding whether it refers to a financial institution or a river bank.
  <strong>Contextual embeddings</strong> changed this:
</p>
<ul>
  <li><strong>ELMo (2018):</strong> Used bidirectional LSTMs to produce context-dependent word vectors.</li>
  <li><strong>BERT (2019):</strong> Transformer encoder that produces deeply contextualized token
      embeddings. The embedding for "bank" in "river bank" differs from "bank account."</li>
  <li><strong>Modern embedding models:</strong> Sentence-level models like Sentence-BERT, E5, and
      OpenAI's text-embedding-ada-002 produce a single vector for an entire passage, optimized
      for <span class="term" data-term="semantic-search">semantic search</span>.</li>
</ul>

<div class="key-concept">
  <strong>Key Concept:</strong> The shift from word-level to sentence/passage-level embeddings was
  critical for RAG. In a retrieval system, you need to compare the meaning of a user's question
  against stored document chunks. Word embeddings require aggregation (averaging, etc.) which loses
  information. Passage embedding models produce a single vector that captures the meaning of an
  entire paragraph, enabling direct comparison via
  <span class="term" data-term="cosine-similarity">cosine similarity</span>.
</div>

<h2>How Modern Embedding Models Are Trained</h2>
<p>
  Modern embedding models like E5, BGE, and Cohere's embed models are typically trained in stages:
</p>
<ol>
  <li><strong>Pretraining:</strong> A Transformer encoder (often BERT-like) is pretrained on large
      text corpora with masked language modeling.</li>
  <li><strong>Contrastive fine-tuning:</strong> The model is fine-tuned on (query, positive document,
      negative document) triplets using a contrastive loss:
      <code>L = -log(exp(sim(q, d+)/τ) / Σ exp(sim(q, d-)/τ))</code>
      where <code>sim</code> is cosine similarity and <code>τ</code> is a temperature parameter.</li>
  <li><strong>Hard negative mining:</strong> The most informative negatives are documents that are
      superficially similar but semantically different from the query. These "hard negatives" force
      the model to learn fine-grained distinctions.</li>
</ol>

<div class="example-box">
  <h4>Example</h4>
  <p>Consider training an embedding model for a medical knowledge base. A query might be "symptoms of
  type 2 diabetes." A positive document describes insulin resistance and hyperglycemia. A simple
  negative might be about car maintenance. A <em>hard</em> negative might be about type 1 diabetes —
  semantically related but different enough that the retrieval system must distinguish them. Hard
  negatives are what make embedding models useful in practice.</p>
</div>

<h2>Embedding Dimensions and Trade-offs</h2>
<p>
  Embedding dimensionality directly affects system performance:
</p>
<table>
  <thead>
    <tr><th>Model</th><th>Dimensions</th><th>Max Tokens</th><th>Use Case</th></tr>
  </thead>
  <tbody>
    <tr><td>OpenAI text-embedding-3-small</td><td>1536</td><td>8191</td><td>General purpose, cost-effective</td></tr>
    <tr><td>OpenAI text-embedding-3-large</td><td>3072</td><td>8191</td><td>High-quality retrieval</td></tr>
    <tr><td>Cohere embed-v3</td><td>1024</td><td>512</td><td>Multilingual, efficient</td></tr>
    <tr><td>BGE-large-en</td><td>1024</td><td>512</td><td>Open-source, strong English</td></tr>
    <tr><td>E5-mistral-7b-instruct</td><td>4096</td><td>32768</td><td>LLM-based, instruction-following</td></tr>
  </tbody>
</table>

<div class="pro-tip">
  <strong>PM Perspective:</strong> Higher dimensions generally capture more nuance but cost more to
  store and search. For a system with 10 million document chunks, each 1536-dimensional embedding
  (float32) takes 6KB. Total: ~60GB of embeddings alone. Doubling dimensions doubles storage and
  roughly doubles search time. When scoping a RAG product, get your engineering team to model the
  storage and latency implications of embedding dimensionality early.
</div>

<h2>Similarity Metrics</h2>
<p>
  Given two embedding vectors, how do you measure their similarity?
</p>
<ul>
  <li><strong><span class="term" data-term="cosine-similarity">Cosine similarity</span>:</strong>
      <code>cos(a, b) = (a · b) / (||a|| · ||b||)</code>. Measures the angle between vectors,
      ignoring magnitude. Range: [-1, 1]. Most common for text embeddings.</li>
  <li><strong>Dot product:</strong> <code>a · b</code>. If embeddings are normalized (unit length),
      this equals cosine similarity. Computationally cheaper.</li>
  <li><strong>Euclidean distance:</strong> <code>||a - b||</code>. Measures straight-line distance.
      Sensitive to magnitude. Less common for text but used in some image embeddings.</li>
</ul>

<div class="warning">
  <strong>Common Misconception:</strong> Many developers assume all embedding models produce
  normalized vectors and use dot product interchangeably with cosine similarity. This is not always
  true — some models produce unnormalized embeddings where magnitude carries information (e.g.,
  confidence or specificity). Always check the embedding model's documentation and normalize
  explicitly if your vector database uses dot product distance.
</div>

<h2>Chunking: Preparing Documents for Embedding</h2>
<p>
  Embedding models have a maximum input length (typically 256-8192 tokens). Most documents exceed
  this, so they must be split into <span class="term" data-term="chunking">chunks</span> before
  embedding. Chunking strategy profoundly affects retrieval quality:
</p>
<ul>
  <li><strong>Fixed-size chunking:</strong> Split every N tokens. Simple but can break mid-sentence
      or mid-concept.</li>
  <li><strong>Sentence-level chunking:</strong> Split on sentence boundaries. Preserves grammatical
      units but chunks may be too short for context.</li>
  <li><strong>Semantic chunking:</strong> Use embedding similarity between adjacent sentences to
      detect topic boundaries. Groups related content together.</li>
  <li><strong>Recursive/hierarchical chunking:</strong> Try to split on headers, then paragraphs,
      then sentences — respecting document structure.</li>
  <li><strong>Overlapping windows:</strong> Include N tokens of overlap between adjacent chunks to
      preserve context at boundaries.</li>
</ul>

<div class="key-concept">
  <strong>Key Concept:</strong> Chunk size creates a precision-recall trade-off. Small chunks
  (100-200 tokens) are more precise — each chunk covers one idea, so retrieval is more targeted.
  Large chunks (500-1000 tokens) provide more context — the LLM gets a richer passage to reason
  over, but retrieval may pull in irrelevant information. Most production systems settle on
  300-500 tokens with 50-100 token overlap as a starting point, then tune based on evaluation.
</div>

<div class="pro-tip">
  <strong>PM Perspective:</strong> Chunking is one of those "boring" infrastructure decisions that
  has an outsized impact on user-perceived quality. If your RAG system retrieves the wrong chunks,
  the LLM will confidently generate wrong answers. Invest in chunking experimentation early —
  run retrieval evaluations with different chunk sizes and overlaps on representative queries
  before committing to a production configuration.
</div>
`,
    quiz: {
      questions: [
        {
          question: 'Your team is building a RAG system over 5 million technical documents. The engineering lead proposes using 3072-dimensional embeddings for maximum quality. What concern should you raise as a PM?',
          type: 'mc',
          options: [
            'Higher-dimensional embeddings always produce worse results on technical content due to the curse of dimensionality',
            'Storage and search latency costs double — 5M docs at 3072 dims requires ~60GB vs ~30GB at 1536 dims, and ANN search time scales proportionally. Verify the quality gain justifies infrastructure costs.',
            '3072-dimensional embeddings cannot represent technical terminology accurately compared to lower-dimensional representations',
            'Embedding models with higher dimensions require GPU-accelerated inference hardware, which is prohibitively expensive at scale'
          ],
          correct: 1,
          explanation: 'Higher dimensionality means more storage (each vector is 12KB at 3072 dims vs. 6KB at 1536), more memory for the vector index, and slower search. For 5M documents, this is the difference between ~30GB and ~60GB. The PM should ask for benchmarks showing the quality improvement (e.g., recall@10 on a representative query set) before accepting the additional infrastructure cost. The answer may well be worth it, but it should be a data-driven decision.',
          difficulty: 'applied',
          expertNote: 'In practice, many teams use Matryoshka embeddings (where the first N dimensions can be used independently) or quantization (reducing from float32 to int8) to manage this trade-off. OpenAI\'s text-embedding-3 models support dimension reduction natively.'
        },
        {
          question: 'What was the key limitation of Word2Vec that contextual embedding models like BERT addressed?',
          type: 'mc',
          options: [
            'Word2Vec could not process words exceeding ten characters in length',
            'Word2Vec assigned a single fixed vector per word regardless of context, making polysemous words indistinguishable',
            'Word2Vec required labeled training data while BERT uses unsupervised learning',
            'Word2Vec embeddings were too high-dimensional for practical applications'
          ],
          correct: 1,
          explanation: 'Word2Vec produces one static embedding per word. The word "bank" always gets the same vector whether it means a financial institution or a riverbank. BERT and other contextual models produce different embeddings for the same word depending on its surrounding context, capturing the actual meaning in each usage.',
          difficulty: 'foundational',
          expertNote: 'For RAG systems, the contextual nature of modern embeddings is critical. A query about "Java island" and "Java programming" should retrieve different documents, and contextual embeddings enable this distinction.'
        },
        {
          question: 'You are evaluating two chunking strategies for a customer support RAG system. Strategy A uses 150-token chunks with no overlap. Strategy B uses 400-token chunks with 75-token overlap. Users typically ask short, specific questions. Which strategy should you test first, and why?',
          type: 'mc',
          options: [
            'Strategy A — smaller chunks provide precision for specific queries, though adding overlap prevents boundary context loss',
            'Strategy B — larger chunks always produce superior RAG quality across all query types',
            'Neither — chunk size does not significantly affect retrieval quality in practice',
            'Strategy A — zero overlap is optimal because it prevents duplicate information retrieval'
          ],
          correct: 0,
          explanation: 'For short, specific questions, smaller chunks tend to perform better because each retrieved chunk is more focused on a single topic, reducing noise. However, the lack of overlap in Strategy A risks splitting relevant information across chunk boundaries. The best starting point would be Strategy A\'s size with some overlap (e.g., 150 tokens with 30-token overlap). Strategy B\'s larger chunks may retrieve too much irrelevant context for precise queries.',
          difficulty: 'applied',
          expertNote: 'The optimal chunking strategy is highly domain-dependent. Customer support content (short, factual articles) benefits from smaller chunks. Legal documents (where a single paragraph may contain crucial context for a clause) may need larger chunks. Always evaluate empirically on representative queries.'
        },
        {
          question: 'Which of the following correctly describe the role of hard negatives in embedding model training? Select all that apply.',
          type: 'multi',
          options: [
            'Hard negatives are documents that are superficially similar to the query but semantically different',
            'They force the model to learn fine-grained semantic distinctions rather than surface-level pattern matching',
            'Hard negatives are always manually curated by human annotators',
            'Using hard negatives during training significantly improves retrieval precision on ambiguous queries'
          ],
          correct: [0, 1, 3],
          explanation: 'Hard negatives are retrieved documents that score high on surface similarity (e.g., sharing many keywords with the query) but are not relevant answers. Training with these examples forces the model to distinguish genuine semantic similarity from keyword overlap. Hard negatives are typically mined automatically (using a weaker retrieval model to find near-misses), not manually curated.',
          difficulty: 'foundational',
          expertNote: 'The quality of hard negatives is one of the most important factors in embedding model quality. State-of-the-art training pipelines use iterative hard negative mining — train the model, use it to find hard negatives, retrain, repeat. This is why training a competitive embedding model is significantly more involved than simply fine-tuning BERT.'
        },
        {
          question: 'A developer on your team proposes using Euclidean distance instead of cosine similarity for your text retrieval system, arguing it captures magnitude information. Under what circumstances would this be appropriate?',
          type: 'mc',
          options: [
            'Always — Euclidean distance is strictly superior for all text retrieval tasks',
            'When the embedding model is specifically trained with Euclidean distance and magnitude encodes meaningful signals like specificity',
            'Never — Euclidean distance fundamentally cannot be used with text embeddings',
            'Only when documents are very short, typically fewer than 50 tokens'
          ],
          correct: 1,
          explanation: 'Cosine similarity is standard for text embeddings because most models are trained with cosine-based losses, and direction (not magnitude) captures semantic meaning. However, if a model is specifically trained with a Euclidean loss and the magnitude of vectors encodes useful information (like specificity), Euclidean distance can be appropriate. The key is matching the distance metric to the training objective of the embedding model.',
          difficulty: 'expert',
          expertNote: 'In practice, most modern embedding models normalize their outputs, making cosine similarity and dot product equivalent. The choice of metric matters most for legacy or custom-trained models. Always test both metrics on a held-out evaluation set before deciding.'
        }
      ]
    }
  },

  // ─────────────────────────────────────────────
  // L02 — Vector Databases
  // ─────────────────────────────────────────────
  l02: {
    title: 'Vector Databases — FAISS, Pinecone, Chroma',
    content: `
<h2>Why Vector Databases Exist</h2>
<p>
  Once you have millions of document embeddings, you need infrastructure to store them and
  efficiently find the nearest neighbors to a query embedding. Traditional databases (PostgreSQL,
  MongoDB) are optimized for exact matches and range queries — they cannot efficiently answer
  "find the 10 vectors most similar to this query vector in a 1536-dimensional space."
</p>
<p>
  <span class="term" data-term="vector-database">Vector databases</span> are purpose-built for
  this problem. They use specialized indexing algorithms that trade a small amount of accuracy
  for dramatic speedups, enabling sub-millisecond similarity search across billions of vectors.
</p>

<h2>The Core Problem: Approximate Nearest Neighbor (ANN) Search</h2>
<p>
  Exact nearest neighbor search in high-dimensional spaces requires comparing the query against
  every stored vector — <code>O(n)</code> time. For 10 million vectors at 1536 dimensions, this
  means 10 million dot products per query. At scale, this is intractable.
</p>
<p>
  <strong>Approximate Nearest Neighbor (ANN)</strong> algorithms sacrifice perfect recall for speed,
  typically achieving 95-99% recall (i.e., finding 95-99% of the true nearest neighbors) while
  being 100-1000x faster than brute force.
</p>

<h3>Key ANN Algorithms</h3>

<h4>1. IVF (Inverted File Index)</h4>
<p>
  Clusters vectors using k-means. At query time, only searches the nearest clusters rather than
  all vectors. Controlled by <code>nprobe</code> — how many clusters to search. Higher nprobe
  = better recall but slower.
</p>

<h4>2. HNSW (Hierarchical Navigable Small World)</h4>
<p>
  Builds a multi-layer graph where each vector is connected to its approximate nearest neighbors.
  Query starts at a coarse top layer and navigates down through increasingly fine-grained layers.
  Generally achieves the best recall-speed trade-off and is the most popular algorithm in production.
</p>

<h4>3. PQ (Product Quantization)</h4>
<p>
  Compresses vectors by splitting each into sub-vectors and quantizing each sub-vector independently.
  A 1536-dim float32 vector (6KB) can be compressed to ~96 bytes — a 64x reduction. This enables
  indexing billions of vectors in memory. Some quality loss from compression.
</p>

<div class="key-concept">
  <strong>Key Concept:</strong> Most production systems combine these techniques — e.g., IVF-PQ
  (cluster, then compress) or HNSW with scalar quantization. The choice depends on your scale:
  at 1 million vectors, brute force or simple IVF may suffice. At 1 billion vectors, you need
  HNSW with quantization and potentially sharding across multiple machines.
</div>

<h2>FAISS: Facebook AI Similarity Search</h2>
<p>
  <span class="term" data-term="faiss">FAISS</span>, developed by Meta AI Research, is the most
  widely used open-source library for vector similarity search. It is a <em>library</em>, not a
  database — it runs in-process and provides indexing algorithms without persistence, replication,
  or API layers.
</p>
<p>
  Key characteristics:
</p>
<ul>
  <li><strong>GPU acceleration:</strong> FAISS provides CUDA implementations of most algorithms,
      enabling search over 1 billion vectors in milliseconds on a single GPU.</li>
  <li><strong>Index types:</strong> Flat (brute force), IVF, HNSW, PQ, and combinations like
      IVF-PQ and IVF-HNSW.</li>
  <li><strong>Python and C++ APIs:</strong> Low-level control over index construction and search.</li>
  <li><strong>No built-in persistence:</strong> You must serialize indexes to disk manually. No
      built-in replication or distributed search.</li>
</ul>

<div class="example-box">
  <h4>Example</h4>
  <p>A typical FAISS pipeline for a RAG system with 1 million chunks:
  (1) Embed all chunks using your embedding model → 1M vectors of dimension 1536.
  (2) Build an IVF-PQ index with 1024 centroids and 64 sub-quantizers.
  (3) At query time, embed the user's question, search with nprobe=32, retrieve top-10 chunks.
  This setup uses ~300MB of memory and returns results in under 5ms on CPU.
  FAISS is ideal when you want maximum control and are willing to manage infrastructure yourself.</p>
</div>

<h2>Pinecone: Managed Vector Database</h2>
<p>
  Pinecone is a fully managed, cloud-hosted vector database. It abstracts away the complexity of
  index building, sharding, and infrastructure management.
</p>
<ul>
  <li><strong>Serverless and pod-based tiers:</strong> Serverless scales automatically; pod-based
      gives dedicated compute.</li>
  <li><strong>Metadata filtering:</strong> Attach key-value metadata to vectors and filter during
      search (e.g., "find similar documents from the last 30 days").</li>
  <li><strong>Namespaces:</strong> Logical partitioning within an index for multi-tenant applications.</li>
  <li><strong>Hybrid search:</strong> Combines dense vector search with sparse keyword matching.</li>
  <li><strong>Managed scaling:</strong> Handles sharding, replication, and load balancing automatically.</li>
</ul>

<div class="pro-tip">
  <strong>PM Perspective:</strong> Pinecone's value proposition is operational simplicity. Your team
  does not need to manage FAISS indices, handle serialization, or build distributed search infrastructure.
  The trade-off is cost and vendor lock-in. For a startup or early-stage product, Pinecone's managed
  service can accelerate time-to-market by weeks. For a large-scale system at DeepMind processing
  billions of vectors, the economics may favor self-hosted FAISS or a custom solution.
</div>

<h2>Chroma: Developer-Friendly Open Source</h2>
<p>
  Chroma positions itself as the "AI-native open-source embedding database." It is designed for
  rapid prototyping and integrates deeply with LangChain and LlamaIndex.
</p>
<ul>
  <li><strong>In-process or client-server:</strong> Can run embedded in your Python process (no
      separate server) or as a standalone service.</li>
  <li><strong>Built-in embedding:</strong> Can automatically embed documents using integrated
      embedding functions (Sentence Transformers, OpenAI, etc.).</li>
  <li><strong>Simple API:</strong> <code>collection.add(documents=[...], ids=[...])</code> and
      <code>collection.query(query_texts=["..."], n_results=10)</code>.</li>
  <li><strong>Persistence:</strong> SQLite-based storage for durability.</li>
</ul>

<h2>Comparison Matrix</h2>
<table>
  <thead>
    <tr><th>Feature</th><th>FAISS</th><th>Pinecone</th><th>Chroma</th><th>Weaviate</th><th>Qdrant</th></tr>
  </thead>
  <tbody>
    <tr><td>Type</td><td>Library</td><td>Managed SaaS</td><td>Embedded/Server</td><td>Self-hosted/Cloud</td><td>Self-hosted/Cloud</td></tr>
    <tr><td>Scale</td><td>Billions</td><td>Billions</td><td>Millions</td><td>Billions</td><td>Billions</td></tr>
    <tr><td>GPU support</td><td>Yes (native)</td><td>N/A (managed)</td><td>No</td><td>No</td><td>No</td></tr>
    <tr><td>Metadata filtering</td><td>Manual</td><td>Built-in</td><td>Built-in</td><td>Built-in</td><td>Built-in</td></tr>
    <tr><td>Hybrid search</td><td>Manual</td><td>Built-in</td><td>Limited</td><td>Built-in</td><td>Built-in</td></tr>
    <tr><td>Open source</td><td>Yes (MIT)</td><td>No</td><td>Yes (Apache 2)</td><td>Yes (BSD)</td><td>Yes (Apache 2)</td></tr>
    <tr><td>Best for</td><td>Max perf, research</td><td>Managed production</td><td>Prototyping</td><td>Knowledge graphs</td><td>Production self-hosted</td></tr>
  </tbody>
</table>

<h2>Beyond Vector Search: Metadata, Filtering, and Hybrid Approaches</h2>
<p>
  In production RAG systems, pure vector similarity search is rarely sufficient. You almost always
  need to combine semantic search with structured filtering:
</p>
<ul>
  <li><strong>Metadata filtering:</strong> "Find similar documents, but only from the legal department
      and published after January 2024." This requires the vector database to support pre- or
      post-filtering on metadata fields.</li>
  <li><strong>Hybrid search:</strong> Combine dense embeddings (semantic) with sparse representations
      like BM25 (keyword). A query for "GDPR Article 17" needs exact keyword matching (sparse) as
      much as semantic understanding (dense). More on this in Lesson 4.</li>
  <li><strong>Multi-tenancy:</strong> In a SaaS product, each customer's data must be isolated.
      Vector databases handle this via namespaces, collections, or access controls.</li>
</ul>

<div class="warning">
  <strong>Common Misconception:</strong> People often assume that vector search replaces traditional
  search entirely. In reality, vector search struggles with exact matches (product IDs, legal
  citations, code identifiers) and performs best for fuzzy, semantic queries. Production systems
  almost always combine vector search with keyword search — the question is how to merge the rankings.
</div>

<h2>Operational Considerations for PMs</h2>
<p>
  Choosing a vector database is as much an operational decision as a technical one:
</p>
<ul>
  <li><strong>Index build time:</strong> How long does it take to index 10 million new vectors?
      This affects your data freshness SLA. HNSW builds can take hours; IVF builds are faster.</li>
  <li><strong>Update latency:</strong> How quickly can you add, update, or delete vectors? Some
      indexes must be rebuilt entirely; others support incremental updates.</li>
  <li><strong>Memory vs. disk:</strong> HNSW requires the full graph in memory. PQ indices fit in
      less memory. Disk-based indices (DiskANN) trade speed for cost at extreme scale.</li>
  <li><strong>Cost modeling:</strong> For cloud-hosted solutions, model your costs carefully.
      Pinecone charges per vector stored and per query. At high query volumes on large indices,
      costs can grow rapidly.</li>
</ul>

<div class="pro-tip">
  <strong>PM Perspective:</strong> The vector database is often the most visible infrastructure cost
  in a RAG system — more visible than embedding model inference or LLM calls because it scales
  directly with data volume. Build a simple cost model: (vectors × dimensions × bytes) for storage,
  (queries/second × p99 latency) for compute. Present this to stakeholders alongside your feature
  roadmap so infrastructure costs are never a surprise.
</div>
`,
    quiz: {
      questions: [
        {
          question: 'Your RAG system currently uses brute-force FAISS search over 500,000 vectors and returns results in 50ms. Your roadmap calls for scaling to 50 million vectors. Without changing the search algorithm, what approximate latency should you expect, and what should you recommend?',
          type: 'scenario',
          options: [
            'Latency will remain at 50ms because FAISS automatically optimizes at scale',
            'Latency will increase to ~5 seconds (100x data = 100x time for brute force). You should recommend switching to an ANN index like HNSW or IVF-PQ.',
            'Latency will increase to ~500ms, which is acceptable for most applications',
            'You should recommend switching to a keyword-based search system instead'
          ],
          correct: 1,
          explanation: 'Brute-force search is O(n) — scaling data by 100x scales latency by ~100x. At 5 seconds per query, the system is unusable for interactive applications. ANN algorithms like HNSW or IVF-PQ provide sub-linear search time, maintaining millisecond latency even at 50M vectors. The PM should recommend this architectural change well before hitting the scaling wall, ideally when the roadmap is set.',
          difficulty: 'applied',
          expertNote: 'A common anti-pattern is building with brute force during prototyping and discovering the scaling problem too late. Good PM practice: always ask "what happens at 10x and 100x our current data volume?" when reviewing system architecture.'
        },
        {
          question: 'What is the primary trade-off when using Product Quantization (PQ) to compress vectors?',
          type: 'mc',
          options: [
            'PQ increases search speed but makes it impossible to add new vectors',
            'PQ reduces memory usage dramatically (often 64x) but introduces some loss in search accuracy due to vector compression',
            'PQ improves search accuracy but increases memory usage',
            'PQ is only compatible with GPU-based search and cannot run on CPU'
          ],
          correct: 1,
          explanation: 'Product Quantization compresses vectors by splitting them into sub-vectors and quantizing each independently. A 1536-dim float32 vector (6KB) can be compressed to ~96 bytes. This enables fitting billions of vectors in memory but introduces quantization error — the compressed vector is an approximation of the original, leading to some search accuracy loss (typically 1-5% recall reduction).',
          difficulty: 'foundational',
          expertNote: 'The key insight is that PQ error is bounded and controllable. More sub-quantizers = less compression but less error. Teams should benchmark PQ recall against their specific data distribution to find the right compression level.'
        },
        {
          question: 'You are deciding between Pinecone (managed) and self-hosted Qdrant for a production RAG system at a large enterprise. Which factors should you weigh? Select all that apply.',
          type: 'multi',
          options: [
            'Pinecone eliminates operational burden but introduces vendor lock-in and ongoing per-vector costs',
            'Self-hosted Qdrant gives full control over data residency (important for compliance), but requires DevOps expertise',
            'Pinecone always provides better search quality than open-source alternatives',
            'At high scale (billions of vectors, thousands of QPS), self-hosted solutions often have better economics',
            'Managed solutions like Pinecone can accelerate time-to-market for early-stage products'
          ],
          correct: [0, 1, 3, 4],
          explanation: 'The managed vs. self-hosted decision involves trade-offs across operational burden, vendor lock-in, data residency compliance, cost economics at scale, and time-to-market. Pinecone does not inherently provide better search quality — both use similar underlying algorithms (HNSW, etc.). The choice is primarily about operational and business considerations, not search quality.',
          difficulty: 'applied',
          expertNote: 'For a DeepMind PM, data residency and control are especially important. Working with sensitive or proprietary data often makes managed third-party solutions untenable. This is why Google built its own vector search infrastructure (Vertex AI Vector Search / ScaNN) rather than relying on third parties.'
        },
        {
          question: 'Why is HNSW (Hierarchical Navigable Small World) generally preferred over IVF for production vector search?',
          type: 'mc',
          options: [
            'HNSW requires less memory than IVF',
            'HNSW provides better recall at equivalent query latency because its graph navigation more efficiently explores the vector space compared to IVF\'s cluster-based approach',
            'HNSW supports metadata filtering while IVF does not',
            'HNSW is the only algorithm that works with high-dimensional vectors'
          ],
          correct: 1,
          explanation: 'HNSW builds a multi-layer proximity graph that enables efficient navigation from coarse to fine-grained similarity. This graph structure typically achieves 95-99% recall in microseconds, outperforming IVF at equivalent latency. The trade-off is that HNSW requires more memory (the full graph must be in RAM) and has slower index build times. IVF is preferred when memory is the binding constraint.',
          difficulty: 'foundational',
          expertNote: 'The HNSW vs. IVF choice often comes down to operational constraints: HNSW wins on query performance, IVF wins on memory efficiency and build speed. DiskANN (Microsoft) offers a third option — near-HNSW performance with disk-based storage — for extreme-scale use cases.'
        },
        {
          question: 'A product manager at a SaaS company notices that their vector database costs have tripled in the last quarter despite only a 50% increase in data volume. What is the most likely cause?',
          type: 'mc',
          options: [
            'The embedding model was changed to a higher-dimensional version, increasing storage per vector',
            'Query volume increased significantly, and the managed database charges per query in addition to per vector stored',
            'Vector databases become exponentially more expensive as data volume increases',
            'The data was re-indexed, which always triples costs'
          ],
          correct: 1,
          explanation: 'Managed vector database pricing typically has two components: storage (per vector) and compute (per query). A 50% data increase explains only part of the cost growth. If query volume grew disproportionately (e.g., due to a product launch or new feature that generates more retrieval calls), the per-query charges can dominate. This is a common surprise for PMs who model only storage costs.',
          difficulty: 'applied',
          expertNote: 'This scenario is extremely common and highlights why cost modeling must include both storage and compute dimensions. Some teams mitigate this with caching layers (cache frequent query results), query deduplication, or switching to a self-hosted solution when query volume crosses a cost threshold.'
        }
      ]
    }
  },

  // ─────────────────────────────────────────────
  // L03 — RAG Architecture End to End
  // ─────────────────────────────────────────────
  l03: {
    title: 'RAG Architecture — End to End',
    content: `
<h2>What Is RAG and Why Does It Matter?</h2>
<p>
  <span class="term" data-term="rag">Retrieval-Augmented Generation (RAG)</span> is an architecture
  pattern that combines a <span class="term" data-term="retrieval">retrieval</span> system with a
  large language model. Instead of relying solely on the LLM's parametric knowledge (what it
  memorized during training), RAG retrieves relevant documents from an external knowledge base
  and includes them in the LLM's context window. The LLM then generates an answer
  <span class="term" data-term="grounding">grounded</span> in the retrieved evidence.
</p>
<p>
  RAG solves several critical problems with standalone LLMs:
</p>
<ul>
  <li><strong>Knowledge cutoff:</strong> LLMs only know what was in their training data. RAG gives
      them access to up-to-date information.</li>
  <li><strong>Hallucination reduction:</strong> By providing source documents, RAG gives the LLM
      evidence to base answers on, reducing confabulation.</li>
  <li><strong>Domain specialization:</strong> Instead of fine-tuning an LLM on domain-specific data
      (expensive, slow), RAG lets you plug in any knowledge base.</li>
  <li><strong>Auditability:</strong> RAG can cite sources, enabling users to verify answers — a
      critical requirement for enterprise and regulated industries.</li>
</ul>

<div class="key-concept">
  <strong>Key Concept:</strong> RAG is not just a technique — it is the dominant paradigm for building
  knowledge-grounded AI applications. ChatGPT with browsing, Google's AI Overviews, Perplexity,
  enterprise copilots — all are RAG systems at their core. Understanding RAG architecture end-to-end
  is arguably the most practically important skill for an AI PM today.
</div>

<h2>The RAG Pipeline: Three Phases</h2>
<p>
  A complete RAG system has three phases: <strong>Ingestion</strong>, <strong>Retrieval</strong>,
  and <strong>Generation</strong>.
</p>

<h3>Phase 1: Ingestion (Offline)</h3>
<p>
  The ingestion pipeline prepares your knowledge base for retrieval:
</p>
<ol>
  <li><strong>Data collection:</strong> Gather source documents — PDFs, web pages, databases, APIs,
      internal wikis, support tickets, etc.</li>
  <li><strong>Parsing & extraction:</strong> Convert raw documents into clean text. Handle HTML
      stripping, PDF extraction (OCR if scanned), table parsing, metadata extraction.</li>
  <li><strong><span class="term" data-term="chunking">Chunking</span>:</strong> Split documents into
      retrievable units (300-500 tokens typical). Add overlapping windows to preserve context at
      chunk boundaries.</li>
  <li><strong>Embedding:</strong> Pass each chunk through an <span class="term" data-term="embedding-model">embedding model</span>
      to produce a dense vector representation.</li>
  <li><strong>Indexing:</strong> Store vectors and associated metadata in a
      <span class="term" data-term="vector-database">vector database</span>. Build ANN index for
      fast retrieval.</li>
</ol>

<div class="example-box">
  <h4>Example</h4>
  <p>Ingestion pipeline for a customer support RAG system:
  (1) Crawl 50,000 help center articles daily.
  (2) Parse HTML, extract article body, title, product area, last-modified date.
  (3) Chunk each article into 400-token segments with 100-token overlap. Store article title and
  URL as metadata on each chunk.
  (4) Embed using text-embedding-3-small (1536 dims).
  (5) Upsert into Pinecone with metadata filters for product area and date.
  Total: ~200,000 chunks, ~1.2GB of embeddings, refreshed daily.</p>
</div>

<div class="warning">
  <strong>Common Misconception:</strong> Many teams treat ingestion as a one-time setup. In reality,
  source data changes continuously — articles are updated, products launch, policies change. A
  production RAG system needs a continuous ingestion pipeline with incremental updates, change
  detection, and stale data cleanup. Failing to maintain data freshness is the most common cause
  of RAG quality degradation over time.
</div>

<h3>Phase 2: Retrieval (Online)</h3>
<p>
  When a user asks a question, the retrieval phase finds the most relevant chunks:
</p>
<ol>
  <li><strong>Query processing:</strong> Optionally rewrite or expand the user's query for better
      retrieval (see Advanced RAG lesson).</li>
  <li><strong>Embedding:</strong> Encode the query using the <em>same</em> embedding model used
      during ingestion.</li>
  <li><strong>Vector search:</strong> Find the top-k most similar chunks via ANN search in the
      vector database. Typically k = 5-20.</li>
  <li><strong>Filtering:</strong> Apply metadata filters (date range, document type, access
      permissions) either pre- or post-search.</li>
  <li><strong>Reranking (optional):</strong> Use a cross-encoder model to rescore the retrieved
      chunks against the query for higher precision. More on this in the next lesson.</li>
</ol>

<div class="key-concept">
  <strong>Key Concept:</strong> Retrieval quality is the single biggest determinant of RAG answer
  quality. If the right documents are not retrieved, no amount of LLM sophistication can produce a
  correct answer. The phrase "garbage in, garbage out" applies forcefully — invest at least as much
  effort in retrieval quality as in prompt engineering.
</div>

<h3>Phase 3: Generation (Online)</h3>
<p>
  The retrieved chunks are combined with the user's question and a system prompt, then sent to an LLM:
</p>
<ol>
  <li><strong>Context assembly:</strong> Arrange retrieved chunks into the LLM's prompt. Order
      matters — some models attend more strongly to the beginning and end of the context.</li>
  <li><strong>System prompt:</strong> Instruct the LLM to answer based only on the provided context,
      cite sources, and say "I don't know" when the context is insufficient.</li>
  <li><strong>Generation:</strong> The LLM produces an answer grounded in the retrieved evidence.</li>
  <li><strong>Post-processing:</strong> Extract citations, verify format, apply safety filters.</li>
</ol>

<div class="example-box">
  <h4>Example</h4>
  <p>A simplified RAG prompt template:</p>
  <p><code>
  System: You are a helpful assistant. Answer the user's question based ONLY on the following context.
  If the context does not contain enough information, say "I don't have enough information to answer."
  Always cite which source(s) you used.
  <br/><br/>
  Context:<br/>
  [Source 1: help-article-123] Chunk text here...<br/>
  [Source 2: help-article-456] Chunk text here...<br/>
  [Source 3: help-article-789] Chunk text here...<br/><br/>
  User: How do I reset my password?
  </code></p>
</div>

<h2>The "Lost in the Middle" Problem</h2>
<p>
  Research by Liu et al. (2023) demonstrated that LLMs struggle to use information placed in the
  middle of long contexts. When relevant information is at the beginning or end, models use it
  effectively. When it is buried in the middle among irrelevant chunks, performance degrades
  significantly.
</p>
<p>
  Practical mitigations:
</p>
<ul>
  <li>Place the most relevant chunks first and last in the context.</li>
  <li>Retrieve fewer, higher-quality chunks rather than many mediocre ones.</li>
  <li>Use reranking to ensure the most relevant chunks are surfaced.</li>
  <li>Consider summarizing or compressing retrieved chunks before insertion.</li>
</ul>

<div class="pro-tip">
  <strong>PM Perspective:</strong> The "lost in the middle" effect means that retrieval precision
  (fraction of retrieved chunks that are relevant) matters as much as recall (fraction of relevant
  chunks that are retrieved). Retrieving 20 chunks where only 3 are relevant dilutes the signal.
  Retrieving 5 highly relevant chunks gives the LLM a cleaner signal. This is why reranking and
  careful top-k tuning are so impactful.
</div>

<h2>Evaluation: Measuring RAG Quality</h2>
<p>
  RAG systems require evaluation at two levels:
</p>

<h3>Retrieval Evaluation</h3>
<ul>
  <li><strong>Recall@k:</strong> What fraction of relevant documents appear in the top k results?</li>
  <li><strong>Precision@k:</strong> What fraction of the top k results are relevant?</li>
  <li><strong>MRR (Mean Reciprocal Rank):</strong> How high does the first relevant result appear?</li>
  <li><strong>NDCG (Normalized Discounted Cumulative Gain):</strong> Do relevant results appear in
      the right order?</li>
</ul>

<h3>End-to-End (Answer) Evaluation</h3>
<ul>
  <li><strong>Faithfulness:</strong> Is the answer supported by the retrieved context? (Measures
      hallucination.)</li>
  <li><strong>Answer relevance:</strong> Does the answer address the user's question?</li>
  <li><strong>Context relevance:</strong> Are the retrieved documents actually relevant to the question?</li>
  <li><strong>Correctness:</strong> Is the answer factually correct (compared to ground truth)?</li>
</ul>

<div class="key-concept">
  <strong>Key Concept:</strong> Frameworks like RAGAS and TruLens provide automated RAG evaluation
  using LLM-as-judge approaches. They compute faithfulness, relevance, and correctness scores
  without human annotation. While not perfect, they enable continuous monitoring of RAG quality
  in production — essential for catching regressions when data or models change.
</div>

<h2>Common Failure Modes</h2>
<p>
  Understanding how RAG fails helps you build more robust systems:
</p>
<table>
  <thead>
    <tr><th>Failure Mode</th><th>Symptom</th><th>Root Cause</th><th>Mitigation</th></tr>
  </thead>
  <tbody>
    <tr>
      <td>Retrieval miss</td>
      <td>Answer is wrong or "I don't know" when answer exists in knowledge base</td>
      <td>Query-document embedding mismatch; poor chunking</td>
      <td>Query expansion, hybrid search, better chunking</td>
    </tr>
    <tr>
      <td>Wrong context</td>
      <td>Answer is confidently wrong, citing irrelevant documents</td>
      <td>Retrieved chunks are semantically similar but not relevant</td>
      <td>Reranking, metadata filtering, hard negative mining in embeddings</td>
    </tr>
    <tr>
      <td>Context ignored</td>
      <td>Answer comes from LLM's parametric knowledge, not the context</td>
      <td>LLM's pretraining knowledge overrides retrieved context</td>
      <td>Stronger system prompt, lower temperature, instruction-tuned models</td>
    </tr>
    <tr>
      <td>Stale data</td>
      <td>Answer is outdated despite knowledge base being current</td>
      <td>Ingestion pipeline lagging; old chunks not replaced</td>
      <td>Continuous ingestion, TTL on chunks, version tracking</td>
    </tr>
    <tr>
      <td>Citation error</td>
      <td>Answer is correct but citations point to wrong sources</td>
      <td>LLM confuses which chunk provided which information</td>
      <td>Clearer source labeling in prompt, fewer context chunks</td>
    </tr>
  </tbody>
</table>

<div class="pro-tip">
  <strong>PM Perspective:</strong> Build a taxonomy of failure modes for your specific RAG application
  and track the frequency of each. This gives you a clear signal for where to invest engineering
  effort. If 60% of failures are retrieval misses, focus on retrieval quality. If 30% are stale data,
  invest in ingestion freshness. Data-driven prioritization beats gut feel when debugging RAG quality.
</div>

<h2>RAG vs. Fine-Tuning vs. Long Context</h2>
<p>
  RAG is one of three approaches to giving an LLM domain-specific knowledge. Understanding when
  to use each is a critical PM skill:
</p>
<table>
  <thead>
    <tr><th>Approach</th><th>Best For</th><th>Limitations</th></tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>RAG</strong></td>
      <td>Dynamic knowledge, factual accuracy, citation needs, large knowledge bases</td>
      <td>Retrieval quality bottleneck, latency from retrieval step, complex pipeline</td>
    </tr>
    <tr>
      <td><strong>Fine-tuning</strong></td>
      <td>Teaching style, format, domain-specific reasoning patterns</td>
      <td>Does not reliably inject facts, expensive to update, risk of catastrophic forgetting</td>
    </tr>
    <tr>
      <td><strong>Long context</strong></td>
      <td>Small knowledge bases (< 200K tokens), single-session analysis</td>
      <td>Expensive per-query (all tokens processed every time), "lost in the middle" at scale</td>
    </tr>
  </tbody>
</table>

<div class="warning">
  <strong>Common Misconception:</strong> Fine-tuning is NOT a reliable way to inject new facts into
  an LLM. When you fine-tune a model on company documentation, the model learns the <em>style</em>
  of the documentation but does not reliably memorize specific facts. For factual accuracy, RAG is
  the correct approach. Fine-tuning and RAG are complementary, not competing: fine-tune for style,
  RAG for facts.
</div>
`,
    quiz: {
      questions: [
        {
          question: 'You are building an internal AI assistant for DeepMind researchers that must answer questions about both published papers (public) and internal research memos (private, frequently updated). Which architecture best fits these requirements?',
          type: 'scenario',
          options: [
            'Fine-tune the LLM on all internal memos so it memorizes the information',
            'Use long context — paste all relevant memos into the prompt for every query',
            'Build a RAG system with continuous ingestion from both public paper repositories and internal document stores, with access control metadata to enforce permissions',
            'Use a general-purpose LLM without RAG — frontier models already know most published research'
          ],
          correct: 2,
          explanation: 'RAG is ideal here because: (1) internal memos are frequently updated, requiring dynamic knowledge access; (2) access control metadata ensures researchers only retrieve documents they have permission to see; (3) the knowledge base is too large for long context; (4) fine-tuning cannot reliably memorize specific research findings and would need retraining whenever memos are updated. Continuous ingestion keeps the system current.',
          difficulty: 'applied',
          expertNote: 'Access control in RAG is a frequently overlooked requirement. The retrieval step must respect document-level permissions — otherwise a junior researcher could ask a question and receive context from classified project documents. This requires metadata filtering at query time, not just at index time.'
        },
        {
          question: 'What is the "lost in the middle" problem and why does it matter for RAG system design?',
          type: 'mc',
          options: [
            'Chunks in the middle of a document are harder to embed accurately',
            'LLMs struggle to use information placed in the middle of long contexts, preferentially attending to the beginning and end — so chunk ordering and retrieval precision directly affect answer quality',
            'Vector databases lose accuracy for vectors stored in the middle of large indices',
            'The middle steps of the RAG pipeline (embedding and indexing) are the most error-prone'
          ],
          correct: 1,
          explanation: 'Liu et al. (2023) showed that when relevant information is placed in the middle of a long context surrounded by irrelevant chunks, LLM performance degrades significantly. This means retrieval precision (not just recall) matters enormously — retrieving 20 chunks with 3 relevant ones dilutes the signal. Fewer, higher-quality chunks placed strategically produce better answers.',
          difficulty: 'foundational',
          expertNote: 'This finding motivated several architectural innovations: (1) placing the most relevant chunks at the beginning and end of the context, (2) using reranking to improve precision before generation, and (3) context compression techniques that summarize retrieved passages.'
        },
        {
          question: 'A stakeholder asks why you chose RAG over fine-tuning for your enterprise knowledge assistant. Which of the following are valid reasons? Select all that apply.',
          type: 'multi',
          options: [
            'RAG enables the system to cite sources, providing auditability required by compliance',
            'The knowledge base changes weekly, and fine-tuning would require expensive retraining each time',
            'RAG always produces higher quality answers than fine-tuned models on all tasks',
            'Fine-tuning does not reliably inject specific facts — it primarily teaches style and reasoning patterns',
            'RAG allows you to update knowledge without retraining the base model'
          ],
          correct: [0, 1, 3, 4],
          explanation: 'RAG provides citation/auditability, handles frequently changing knowledge without retraining, leverages external knowledge without parametric memorization, and allows knowledge updates without model changes. However, RAG does not always produce higher quality answers — for tasks that require deep domain-specific reasoning patterns (not just fact retrieval), fine-tuning can be superior. The two approaches are complementary.',
          difficulty: 'applied',
          expertNote: 'The most sophisticated enterprise systems combine RAG and fine-tuning: the model is fine-tuned to follow a specific response format, citation style, and reasoning pattern, while RAG provides the factual grounding. This separation of concerns produces the best results.'
        },
        {
          question: 'Your RAG system has high retrieval recall (85% of relevant documents are found) but users report that answers are often wrong. Looking at the failure analysis, most errors involve the LLM citing retrieved documents that are topically related but do not actually answer the question. What is the most effective intervention?',
          type: 'mc',
          options: [
            'Increase retrieval top-k from 5 to 20 to include more potentially relevant documents',
            'Add a reranking step using a cross-encoder to improve precision — ensuring the chunks that reach the LLM are truly relevant, not just topically similar',
            'Switch to a more powerful LLM that can better distinguish relevant from irrelevant context',
            'Increase the embedding model dimensionality for finer-grained similarity'
          ],
          correct: 1,
          explanation: 'The failure pattern described — high recall but low precision — is exactly what reranking solves. A cross-encoder reranker jointly encodes the query and each candidate document, producing a more accurate relevance score than the bi-encoder\'s cosine similarity. This filters out topically related but non-relevant chunks before they reach the LLM. Increasing top-k would worsen the precision problem. A more powerful LLM might help but does not address the root cause.',
          difficulty: 'expert',
          expertNote: 'The bi-encoder (embedding model) vs. cross-encoder (reranker) distinction is fundamental. Bi-encoders are fast but approximate — they encode query and document independently. Cross-encoders jointly attend to both, capturing fine-grained relevance. The standard pattern is: bi-encoder for fast retrieval of 50-100 candidates, then cross-encoder reranking to select the top 5-10.'
        },
        {
          question: 'You notice that your RAG system gives correct answers from its knowledge base but sometimes contradicts retrieved context by relying on the LLM\'s pre-training knowledge instead. What is this failure mode called, and what is the standard mitigation?',
          type: 'mc',
          options: [
            'Hallucination — mitigated by increasing the temperature parameter',
            'Context override / parametric vs. contextual conflict — mitigated by explicit system prompts instructing the model to prioritize retrieved context, lower temperature, and using instruction-tuned models',
            'Retrieval miss — mitigated by increasing top-k retrieval',
            'Embedding drift — mitigated by retraining the embedding model'
          ],
          correct: 1,
          explanation: 'When the LLM\'s parametric knowledge (from pretraining) conflicts with retrieved context, some models default to their pretraining. This is mitigated by: (1) explicit system prompts ("Answer ONLY based on the provided context"), (2) lower temperature to reduce creative elaboration, (3) using instruction-tuned models that are better at following grounding instructions, and (4) providing clear source labels to help the model distinguish context from its own knowledge.',
          difficulty: 'applied',
          expertNote: 'This is an active research area. Some models (like certain Gemini configurations) have been specifically trained to prioritize retrieved context over parametric knowledge when instructed to do so. Testing for this failure mode should be part of your model selection process for RAG applications.'
        }
      ]
    }
  },

  // ─────────────────────────────────────────────
  // L04 — Advanced RAG
  // ─────────────────────────────────────────────
  l04: {
    title: 'Advanced RAG — Reranking, Hybrid Search, Agentic RAG',
    content: `
<h2>Beyond Basic RAG: Why Advanced Techniques Matter</h2>
<p>
  A basic RAG pipeline — embed, retrieve, generate — works surprisingly well for simple use cases.
  But production systems face challenges that basic RAG cannot handle:
</p>
<ul>
  <li>Queries that need both semantic understanding AND exact keyword matching</li>
  <li>Multi-step questions requiring reasoning across multiple document sources</li>
  <li>Source data that changes format, structure, or availability over time</li>
  <li>Systems that must degrade gracefully when retrieval fails or returns stale data</li>
</ul>
<p>
  This lesson covers the advanced techniques that separate prototype RAG from production-grade
  retrieval systems. These are the techniques that make the difference between a demo that impresses
  stakeholders and a system that reliably serves millions of users.
</p>

<h2>Reranking: The Precision Layer</h2>
<p>
  <span class="term" data-term="reranking">Reranking</span> adds a second scoring pass after
  initial retrieval. The bi-encoder (embedding model) is fast but approximate — it encodes query
  and document independently and compares via cosine similarity. A <strong>cross-encoder</strong>
  reranker jointly processes the query and each candidate document, producing a much more accurate
  relevance score.
</p>
<p>
  Architecture:
</p>
<ol>
  <li>Bi-encoder retrieves top 50-100 candidates (fast, ~1ms).</li>
  <li>Cross-encoder scores each candidate against the query (slower, ~50-200ms for 100 candidates).</li>
  <li>Top 5-10 reranked results are passed to the LLM.</li>
</ol>

<div class="key-concept">
  <strong>Key Concept:</strong> The bi-encoder vs. cross-encoder distinction is fundamental to
  understanding retrieval quality. A bi-encoder encodes the query and document separately — like
  summarizing each in a few numbers and comparing the summaries. A cross-encoder reads the query
  and document together, attending from every query token to every document token. This joint
  attention captures nuances that independent encoding misses (negation, conditionals, specific
  requirements).
</div>

<p>
  Popular reranking models:
</p>
<table>
  <thead>
    <tr><th>Model</th><th>Provider</th><th>Notes</th></tr>
  </thead>
  <tbody>
    <tr><td>Cohere Rerank v3</td><td>Cohere</td><td>API-based, strong multilingual support</td></tr>
    <tr><td>bge-reranker-v2-m3</td><td>BAAI (open source)</td><td>Multilingual, self-hostable</td></tr>
    <tr><td>cross-encoder/ms-marco-MiniLM-L-12-v2</td><td>Sentence Transformers</td><td>Lightweight, fast, good for prototyping</td></tr>
    <tr><td>RankLLaMA / RankGPT</td><td>Research</td><td>Use LLMs as rerankers via listwise prompting</td></tr>
  </tbody>
</table>

<div class="pro-tip">
  <strong>PM Perspective:</strong> Reranking typically improves end-to-end RAG quality by 5-15%
  (measured by answer correctness). The cost is 50-200ms of additional latency. For most products,
  this is an excellent trade-off — the quality improvement is immediately noticeable to users, and
  the latency is acceptable for conversational interfaces (where users expect a brief "thinking"
  period). Prioritize implementing reranking before investing in more exotic retrieval improvements.
</div>

<h2>Hybrid Search: Best of Both Worlds</h2>
<p>
  <span class="term" data-term="hybrid-search">Hybrid search</span> combines dense vector search
  (semantic) with sparse retrieval (keyword-based) to get the best of both approaches.
</p>
<p>
  <strong>Dense retrieval (embeddings):</strong> Captures semantic meaning. "How to reset credentials"
  matches "password recovery steps" even though they share no keywords.
</p>
<p>
  <strong>Sparse retrieval (BM25, TF-IDF):</strong> Excels at exact matching. "GDPR Article 17"
  retrieves documents containing that exact phrase, which dense retrieval might miss if the
  embedding does not capture legal citation formats.
</p>

<div class="warning">
  <strong>Common Misconception:</strong> Many teams assume dense vector search supersedes keyword
  search entirely. In practice, dense retrieval fails on:
  (1) exact identifiers ("error code XJ-4023"),
  (2) rare or domain-specific terms not well-represented in the embedding model's training data,
  (3) queries where the user knows exactly what they want to find (navigational queries).
  Hybrid search combines both, covering each method's blind spots.
</div>

<p>
  Fusion strategies for combining dense and sparse results:
</p>
<ul>
  <li><strong>Reciprocal Rank Fusion (RRF):</strong> Score each document by
      <code>1 / (k + rank)</code> across both retrieval lists, then sum. Simple, effective, and
      does not require score calibration.</li>
  <li><strong>Linear combination:</strong> <code>score = &alpha; &middot; dense_score + (1-&alpha;) &middot; sparse_score</code>.
      Requires normalizing scores to the same scale. The weight <code>&alpha;</code> is tuned on
      evaluation data.</li>
  <li><strong>Learned fusion:</strong> Train a model to optimally combine signals from multiple
      retrieval sources. Most complex but highest quality.</li>
</ul>

<div class="example-box">
  <h4>Example</h4>
  <p>A legal research RAG system uses hybrid search:
  Dense retrieval finds documents about privacy rights and data protection (semantic match).
  Sparse retrieval finds documents specifically mentioning "GDPR Article 17" or "right to erasure"
  (exact keyword match).
  RRF fusion combines both result lists, ensuring the final top-10 includes both semantically
  relevant legal analysis AND documents containing the exact cited provisions.</p>
</div>

<h2>Query Transformation: Improving What You Search For</h2>
<p>
  Users rarely phrase queries in ways that perfectly match document content. Query transformation
  techniques bridge this gap:
</p>
<ul>
  <li><strong>Query rewriting:</strong> Use an LLM to rephrase the user's question for better
      retrieval. "Why won't my code compile?" → "common compilation errors and fixes."</li>
  <li><strong>HyDE (Hypothetical Document Embeddings):</strong> Generate a hypothetical answer to
      the query, embed the answer, and use that embedding for retrieval. The intuition: the
      hypothetical answer is more likely to be semantically similar to real documents than the
      short query.</li>
  <li><strong>Multi-query:</strong> Generate multiple variations of the query, retrieve for each,
      and merge results. Increases recall by covering different phrasings.</li>
  <li><strong>Step-back prompting:</strong> For specific questions, generate a broader version:
      "What is the learning rate for ResNet-50 on ImageNet?" → "How are ResNet models trained
      on ImageNet?" The broader query is more likely to match available documents.</li>
</ul>

<h2>Agentic RAG: Multi-Step Retrieval and Reasoning</h2>
<p>
  Basic RAG performs a single retrieval step. <strong>Agentic RAG</strong> gives the LLM the ability
  to make multiple retrieval calls, reason about intermediate results, and adaptively decide what
  to search for next:
</p>
<ul>
  <li><strong>Multi-hop reasoning:</strong> "What is the market cap of the company that acquired
      DeepMind?" Step 1: Retrieve "Google acquired DeepMind." Step 2: Retrieve "Alphabet/Google
      market cap."</li>
  <li><strong>Self-RAG:</strong> The model generates an answer, then decides whether it needs
      additional retrieval. It can issue retrieval calls mid-generation when it encounters
      uncertainty.</li>
  <li><strong>Tool-augmented RAG:</strong> The LLM has access to multiple retrieval tools — vector
      search, SQL database, API calls, web search — and decides which to use based on the query.</li>
  <li><strong>Corrective RAG (CRAG):</strong> After retrieval, evaluate whether the retrieved
      documents are relevant. If not, try alternative search strategies or fall back to the
      LLM's parametric knowledge with appropriate caveats.</li>
</ul>

<div class="key-concept">
  <strong>Key Concept:</strong> Agentic RAG transforms the LLM from a passive consumer of retrieved
  context into an active researcher that plans retrieval strategies, evaluates results, and iterates.
  This is the direction the field is moving — from single-turn retrieval to multi-turn research
  agents. Products like Perplexity and Google's AI Overviews already implement forms of agentic RAG.
</div>

<h2>Schema Drift and Source Resilience</h2>
<p>
  One of the most underappreciated challenges in production RAG is
  <span class="term" data-term="schema-drift">schema drift</span> — when the structure, format, or
  accessibility of your source data changes over time. This is especially common when retrieving
  from external sources:
</p>
<ul>
  <li><strong>API changes:</strong> A data source changes its API schema — field names change,
      endpoints are deprecated, response formats are restructured. Your parser breaks silently,
      producing malformed chunks.</li>
  <li><strong>DOM/template changes:</strong> If you scrape web content, the source website may
      redesign its layout. A CSS selector that found article bodies now returns navigation menus.
      Instagram, for example, has changed its DOM structure numerous times, breaking scrapers
      built on specific selectors.</li>
  <li><strong>Data format migration:</strong> A source migrates from XML to JSON, or from
      REST to GraphQL. Your ingestion pipeline must handle the transition without data loss.</li>
  <li><strong>Content structure changes:</strong> An internal wiki changes its template for
      product documentation. Sections are renamed, tables are reformatted, metadata fields are
      added or removed.</li>
</ul>

<div class="warning">
  <strong>Common Misconception:</strong> Teams often build RAG ingestion pipelines assuming source
  data structures are stable. In practice, external sources change frequently and without notice.
  Even internal data sources evolve as teams adopt new tools and formats. A parser that works
  today may silently fail in three months, causing your RAG system to serve stale or corrupted
  information without any alert.
</div>

<h3>Building Resilient Parsers</h3>
<p>
  To survive schema drift, build your ingestion parsers with resilience in mind:
</p>
<ul>
  <li><strong>Semantic parsing over structural parsing:</strong> Instead of relying on specific CSS
      selectors or JSON paths, use LLM-based extraction that understands the <em>meaning</em> of
      content regardless of format. Example: "Extract the article title and body" works even if
      the HTML structure changes.</li>
  <li><strong>Fallback chains:</strong> Try multiple parsing strategies in priority order. If the
      primary parser fails (e.g., specific CSS selector), fall back to a broader parser (e.g.,
      largest text block), then to a raw text extraction.</li>
  <li><strong>Schema versioning:</strong> Maintain versioned parser configurations. When a source
      changes, add a new version rather than modifying the old one. This enables rollback and
      A/B testing of parsing strategies.</li>
  <li><strong>Content validation:</strong> After parsing, validate extracted content against
      expected patterns: minimum content length, presence of expected fields, absence of
      navigation/boilerplate text. Flag anomalies for human review.</li>
</ul>

<div class="example-box">
  <h4>Example</h4>
  <p>A production RAG system monitors a set of product documentation sites. When the ingestion
  pipeline detects that parsed content length drops by more than 50% for a source, or that the
  expected metadata fields are missing, it:
  (1) Alerts the on-call engineer.
  (2) Falls back to the last known good version of the parsed content.
  (3) Logs the raw HTML for debugging.
  (4) Marks affected chunks as "stale" in the vector database metadata.
  This ensures users never receive answers based on corrupted data while the team fixes the parser.</p>
</div>

<h2>Monitoring and Freshness: Good Habits for Evolving Sources</h2>
<p>
  Production RAG systems need continuous monitoring to catch quality degradation:
</p>
<ul>
  <li><strong>Source health monitoring:</strong> Track parse success rate, content length distribution,
      and field completeness for each data source. Alert on anomalies.</li>
  <li><strong>Freshness SLAs:</strong> Define how stale data can be before it degrades user experience.
      A support article should be updated within hours of a product change. A research paper
      citation can be weeks old.</li>
  <li><strong>Version tracking:</strong> Store the source document version, last-modified timestamp,
      and ingestion timestamp as metadata on each chunk. This enables queries like "show me only
      information from the last 30 days."</li>
  <li><strong>Change detection:</strong> Use content hashing or diff-based detection to identify
      when source documents change. Only re-embed changed chunks to save compute.</li>
  <li><strong>Automated regression testing:</strong> Maintain a set of golden queries with known
      correct answers. Run these daily against the RAG system to detect quality regressions
      caused by data changes.</li>
</ul>

<div class="pro-tip">
  <strong>PM Perspective:</strong> Schema drift is the silent killer of RAG systems. Your system
  might work perfectly at launch and gradually degrade over months as sources change. Invest in
  monitoring from day one. A dashboard showing parse success rates, data freshness, and retrieval
  quality metrics for each source is as important as the feature itself. The first sign of trouble
  is usually a spike in "I don't know" responses or a drop in user satisfaction — by then, the
  damage is done.
</div>

<h2>Graceful Degradation: When Retrieval Fails</h2>
<p>
  Every retrieval system will occasionally fail — sources go down, parsers break, indices become
  stale. A production RAG system must handle these failures gracefully through
  <span class="term" data-term="graceful-degradation">graceful degradation</span>:
</p>
<ul>
  <li><strong>Confidence scoring:</strong> Assign confidence scores to retrieved chunks based on
      similarity, source freshness, and source reliability. If no chunk exceeds a confidence
      threshold, explicitly tell the user: "I found some related information but I'm not confident
      it answers your question."</li>
  <li><strong>Tiered fallback:</strong>
    <ol>
      <li><strong>Tier 1:</strong> Primary retrieval returns high-confidence results → generate normally.</li>
      <li><strong>Tier 2:</strong> Primary retrieval returns low-confidence results → generate with
          caveats ("Based on limited information...") and suggest the user verify.</li>
      <li><strong>Tier 3:</strong> Primary retrieval fails entirely → fall back to a cached version
          of the knowledge base, clearly marking responses as "based on information from [date]."</li>
      <li><strong>Tier 4:</strong> All retrieval fails → use the LLM's parametric knowledge with
          a clear disclaimer that no external sources were consulted, or gracefully decline to answer.</li>
    </ol>
  </li>
  <li><strong>Staleness indicators:</strong> If retrieved content was last updated more than N days
      ago (configurable per source), add a warning: "This information was last updated on [date]
      and may be outdated."</li>
  <li><strong>Circuit breakers:</strong> If a data source has been failing consistently, stop
      attempting retrieval from it (to avoid latency) and surface only results from healthy sources.</li>
</ul>

<div class="key-concept">
  <strong>Key Concept:</strong> Graceful degradation is not optional in production RAG — it is a
  core requirement. Users trust AI systems less when they give confidently wrong answers than when
  they honestly say "I'm not sure." Design your system to be transparent about its confidence level
  and the freshness of its information. This builds trust and prevents the worst failure mode: users
  acting on incorrect information.
</div>

<h2>Advanced Retrieval Patterns</h2>

<h3>Parent-Child Retrieval</h3>
<p>
  Retrieve small, precise chunks (children) but pass the surrounding larger context (parent) to
  the LLM. This gives you precise retrieval with rich generation context. Implemented by storing
  both fine-grained and coarse-grained chunks and linking them via document IDs.
</p>

<h3>Contextual Retrieval (Anthropic)</h3>
<p>
  Before embedding a chunk, prepend a short context statement that situates the chunk within its
  source document: "This chunk is from the employee benefits section of the 2024 HR policy manual.
  It covers dental insurance options." This added context helps the embedding model represent the
  chunk more accurately.
</p>

<h3>Colbert / Late Interaction</h3>
<p>
  Instead of compressing a document into a single embedding vector, Colbert stores one embedding
  per token. At query time, it performs a MaxSim operation — computing similarity between each
  query token and each document token, then summing the maximums. This preserves more fine-grained
  information than a single vector at the cost of higher storage.
</p>

<div class="pro-tip">
  <strong>PM Perspective:</strong> The RAG landscape is evolving rapidly. New techniques appear
  monthly. Your job as a PM is not to know every technique, but to understand the failure modes
  of your current system and match techniques to those specific problems. If your system struggles
  with precision → try reranking. If it misses keyword queries → add hybrid search. If it fails on
  multi-step questions → explore agentic RAG. Always diagnose before prescribing.
</div>
`,
    quiz: {
      questions: [
        {
          question: 'Your production RAG system for a healthcare company pulls information from several medical databases. One morning, you notice that answers about drug interactions have become unreliable. Investigation reveals that one database changed its API response format two weeks ago, and your parser has been silently extracting incorrect data. Which combination of practices would have prevented this?',
          type: 'scenario',
          options: [
            'Using a more powerful LLM that can handle incorrect context better',
            'Content validation after parsing (checking for expected fields and reasonable content length), source health monitoring dashboards with anomaly alerts, and automated regression testing with golden queries that would have caught the quality drop',
            'Switching to a different database vendor with more stable APIs',
            'Implementing hybrid search so that keyword matching compensates for corrupted vector data'
          ],
          correct: 1,
          explanation: 'This is a classic schema drift failure. Content validation would have caught the malformed data at ingestion time (missing or incorrect fields). Source health monitoring would have flagged the anomaly in parsed content statistics. Automated regression testing with golden queries (e.g., "What drugs interact with warfarin?") would have detected the quality drop within a day, not two weeks. Together, these practices form a defense-in-depth against schema drift.',
          difficulty: 'expert',
          expertNote: 'In healthcare RAG, this failure mode is not just a quality issue — it is a safety issue. Incorrect drug interaction information could harm patients. This is why regulated industries require both monitoring AND human review processes for AI-generated medical information.'
        },
        {
          question: 'What is the primary advantage of Reciprocal Rank Fusion (RRF) over linear score combination for hybrid search?',
          type: 'mc',
          options: [
            'RRF always produces more relevant results than linear combination',
            'RRF does not require score calibration between retrieval systems — it uses only rank positions, which are comparable across any scoring method',
            'RRF is computationally cheaper than linear combination',
            'RRF works only with dense retrieval, making it simpler to implement'
          ],
          correct: 1,
          explanation: 'Dense retrieval (cosine similarity: 0-1) and sparse retrieval (BM25: unbounded positive scores) produce scores on completely different scales. Linear combination requires normalizing these to comparable ranges, which is tricky and sensitive to the normalization method. RRF sidesteps this entirely by using only rank positions (1st, 2nd, 3rd...), which are naturally comparable regardless of the underlying scoring method.',
          difficulty: 'applied',
          expertNote: 'RRF uses the formula score(d) = sum(1 / (k + rank_i(d))) across retrieval systems, where k is typically 60. This is remarkably effective despite its simplicity. In practice, RRF matches or outperforms carefully tuned linear combination for most use cases.'
        },
        {
          question: 'Your RAG system scrapes product documentation from a partner company\'s website. The partner redesigns their site, breaking your HTML parser. Which resilient parsing strategy would best protect against this type of failure?',
          type: 'mc',
          options: [
            'Hard-coding new CSS selectors every time the site changes',
            'Using LLM-based semantic extraction that identifies content by meaning rather than DOM structure, with fallback chains and content validation',
            'Switching from web scraping to manual data entry for reliability',
            'Increasing the scraping frequency to detect changes faster'
          ],
          correct: 1,
          explanation: 'LLM-based semantic extraction (e.g., "Extract the product description and specifications from this page") is resilient to DOM changes because it understands content meaning, not structure. Fallback chains (try specific parser → broad parser → raw text) provide safety nets. Content validation catches cases where all parsers fail. Hard-coding selectors is fragile by definition. Manual entry does not scale. Faster scraping detects problems sooner but does not prevent them.',
          difficulty: 'applied',
          expertNote: 'This is an increasingly common pattern: using LLMs as robust parsers in the ingestion pipeline itself. The cost of LLM-based parsing is higher per document but the reduced maintenance burden often makes it economically favorable over time, especially for sources with frequent template changes.'
        },
        {
          question: 'Which of the following are valid components of a graceful degradation strategy for RAG? Select all that apply.',
          type: 'multi',
          options: [
            'Confidence thresholds that trigger different response behaviors based on retrieval quality',
            'Staleness warnings when retrieved content exceeds a freshness SLA',
            'Automatically generating fabricated sources to fill gaps when retrieval fails',
            'Circuit breakers that stop querying consistently failing data sources',
            'Tiered fallback from primary retrieval to cached data to parametric knowledge with appropriate disclaimers'
          ],
          correct: [0, 1, 3, 4],
          explanation: 'Graceful degradation includes confidence-based response tiers, staleness warnings, circuit breakers for failing sources, and explicit fallback strategies with transparency. Generating fabricated sources is the opposite of graceful degradation — it is deceptive and would destroy user trust. The core principle is: be transparent about limitations rather than masking them with fake confidence.',
          difficulty: 'foundational',
          expertNote: 'Graceful degradation is a borrowed concept from distributed systems engineering (where it refers to maintaining partial functionality when components fail). Applying it to RAG means treating retrieval as an inherently unreliable component and designing the system to behave predictably when it degrades.'
        },
        {
          question: 'A DeepMind PM is evaluating whether to implement Agentic RAG (multi-step retrieval with LLM reasoning) for a complex research assistant. What is the primary trade-off compared to single-turn RAG?',
          type: 'mc',
          options: [
            'Agentic RAG always produces worse results because multiple retrieval steps introduce more noise',
            'Agentic RAG can answer complex multi-hop questions that single-turn RAG cannot, but significantly increases latency, cost, and unpredictability since the LLM controls the retrieval strategy',
            'Agentic RAG eliminates the need for a vector database',
            'Agentic RAG is only useful for code generation tasks, not general knowledge'
          ],
          correct: 1,
          explanation: 'Agentic RAG enables multi-hop reasoning (e.g., finding that Google acquired DeepMind, then looking up Google\'s market cap) that single-turn RAG cannot perform. However, each additional retrieval step adds latency (typically 1-3 seconds per step) and cost (embedding + vector search + LLM reasoning per step). The LLM\'s retrieval decisions can be unpredictable — it might pursue irrelevant tangents or loop. Production agentic RAG requires careful guardrails: step limits, relevance checks between steps, and timeout budgets.',
          difficulty: 'expert',
          expertNote: 'The unpredictability of agentic RAG is its biggest production challenge. Unlike single-turn RAG where latency and cost are bounded, agentic RAG can take 2-10+ retrieval steps depending on query complexity. PMs must define SLAs carefully: maximum steps per query, maximum total latency, and fallback behavior when the agent exceeds its budget.'
        },
        {
          question: 'Your team is building a RAG system that retrieves from an internal wiki. Documents range from 50 to 10,000 words. A colleague suggests using parent-child retrieval. What problem does this solve?',
          type: 'mc',
          options: [
            'It eliminates the need for embeddings by using keyword search on parent documents',
            'It resolves the tension between small chunks (precise retrieval) and large chunks (rich LLM context) by retrieving on small chunks but passing their parent context to the LLM',
            'It automatically splits documents into optimal chunk sizes',
            'It enables multi-lingual retrieval by translating parent documents'
          ],
          correct: 1,
          explanation: 'Parent-child retrieval addresses a fundamental chunk size dilemma: small chunks (100-200 tokens) give precise retrieval scores (the chunk is about exactly one thing), but provide insufficient context for the LLM to generate a good answer. Large chunks (500-1000 tokens) give the LLM more context but dilute retrieval precision. Parent-child retrieval retrieves on small child chunks for precision, then expands to the parent section for generation context — getting the best of both.',
          difficulty: 'applied',
          expertNote: 'This is one of the highest-impact RAG improvements for long-document use cases. Implementation requires storing both chunk levels and maintaining parent-child relationships via metadata. LlamaIndex and LangChain both provide built-in support for this pattern.'
        }
      ]
    }
  }
};
