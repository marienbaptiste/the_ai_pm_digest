export const lessons = {

  // ─────────────────────────────────────────────
  // L01 — The Attention Revolution
  // ─────────────────────────────────────────────
  l01: {
    title: 'The Attention Revolution — Why Attention Is All You Need',
    content: `
<h2>The Problem with Sequential Processing</h2>
<p>Before <span class="term" data-term="transformer">Transformers</span> arrived, the dominant architectures for sequence-to-sequence tasks — machine translation, text summarization, question answering — were Recurrent Neural Networks (RNNs), Long Short-Term Memory networks (LSTMs), and Gated Recurrent Units (GRUs). These models processed input one <span class="term" data-term="token">token</span> at a time, maintaining a hidden state that was passed from one time step to the next. This sequential bottleneck created two critical problems that limited the field for nearly a decade.</p>

<p>First, <strong>information loss over long sequences</strong>. Even with gating mechanisms like those in LSTMs, the hidden state had a fixed capacity. When translating a 100-word paragraph, by the time the model reached the end of the sentence, much of the information from the beginning had been compressed, overwritten, or lost. This manifested as poor performance on long-range dependencies — for instance, maintaining subject-verb agreement across many intervening clauses.</p>

<p>Second, <strong>inability to parallelize</strong>. Because each time step depended on the output of the previous step, RNN training could not be parallelized across the sequence dimension. A sentence of length 50 required 50 sequential forward passes. On modern GPU hardware designed for massive parallelism, this was catastrophically inefficient. Training on large datasets took weeks or months, and scaling up was practically infeasible.</p>

<div class="key-concept"><strong>Key Concept:</strong> The fundamental limitation of RNNs was not their expressiveness but their sequential nature. The hidden state created a bottleneck that degraded both the quality of long-range representations and the speed of training. The Transformer solved both problems simultaneously by replacing recurrence with attention.</div>

<h2>The Birth of Attention (Before Transformers)</h2>
<p>The attention mechanism actually predates the Transformer. In 2014, Bahdanau, Cho, and Bengio introduced <span class="term" data-term="attention">attention</span> in the context of neural machine translation. Their insight was elegant: instead of forcing the decoder to rely solely on the final hidden state of the encoder, allow the decoder to "look back" at all encoder hidden states and compute a weighted sum, focusing on the parts most relevant to the current decoding step.</p>

<p>This additive attention mechanism dramatically improved translation quality, especially on longer sentences. The decoder could now attend to the word "cat" in the source sentence when generating "chat" in French, even if they were far apart in the sequence. However, this early form of attention was still used <em>on top of</em> an RNN — the underlying sequential processing remained.</p>

<div class="example-box"><h4>Example</h4>
<p>Consider translating "The cat sat on the mat" to French. In a vanilla RNN encoder-decoder, by the time the decoder generates the fifth French word, the hidden state has been updated many times, and early information may be degraded. With Bahdanau attention, the decoder can directly attend to the encoder's representation of "cat" at any step, preserving the critical association.</p>
</div>

<h2>"Attention Is All You Need" — The 2017 Revolution</h2>
<p>In June 2017, Vaswani et al. at Google Brain published what would become one of the most influential papers in AI history: <em>"Attention Is All You Need."</em> The core claim was radical — you could build a state-of-the-art sequence-to-sequence model using <strong>only</strong> attention mechanisms, completely dispensing with recurrence and convolutions.</p>

<p>The paper introduced the <span class="term" data-term="transformer">Transformer</span> architecture, which consisted of stacked layers of <span class="term" data-term="self-attention">self-attention</span> and feed-forward networks. The key innovations were:</p>

<table>
<thead>
<tr><th>Innovation</th><th>What It Solved</th><th>How</th></tr>
</thead>
<tbody>
<tr><td>Self-Attention</td><td>Long-range dependencies</td><td>Every token attends to every other token directly, regardless of distance</td></tr>
<tr><td><span class="term" data-term="multi-head-attention">Multi-Head Attention</span></td><td>Diverse relationship types</td><td>Multiple parallel attention heads capture different types of relationships (syntactic, semantic, positional)</td></tr>
<tr><td><span class="term" data-term="positional-encoding">Positional Encoding</span></td><td>Sequence order information</td><td>Sinusoidal functions inject position information since attention itself is permutation-invariant</td></tr>
<tr><td>Full Parallelism</td><td>Training speed</td><td>All tokens processed simultaneously, enabling massive GPU parallelism</td></tr>
</tbody>
</table>

<p>The results were stunning. On the WMT 2014 English-to-German translation benchmark, the Transformer achieved a new state-of-the-art BLEU score of 28.4 — surpassing all previous models — while training in a fraction of the time. The English-to-French model trained in just 3.5 days on 8 GPUs, compared to weeks for equivalent RNN-based models.</p>

<div class="warning"><strong>Common Misconception:</strong> The Transformer did not simply "add attention to existing architectures." It eliminated the RNN entirely and showed that attention alone — combined with simple feed-forward layers and normalization — was sufficient for state-of-the-art sequence modeling. This was a paradigm shift, not an incremental improvement.</div>

<h2>Why the Transformer Scaled Where RNNs Could Not</h2>
<p>The Transformer's ability to parallelize across the sequence dimension was transformative for scaling. Consider the computational profile:</p>

<p>In an RNN, processing a sequence of length <code>n</code> requires <code>O(n)</code> sequential operations because each step depends on the previous hidden state. Even if each operation is fast, you cannot do them simultaneously. In a Transformer, self-attention computes all pairwise interactions in <code>O(1)</code> sequential steps (though with <code>O(n²)</code> total compute for the attention matrix). This means the entire self-attention computation can be parallelized across all <code>n</code> positions simultaneously on a GPU.</p>

<p>This parallelism unlocked a new regime of scaling. Researchers could now train on far larger datasets, use far larger models, and iterate far faster. Within two years, this would lead to GPT-2, BERT, and eventually the large language model revolution.</p>

<div class="pro-tip"><strong>PM Perspective:</strong> As a PM at DeepMind, understanding the Transformer's parallelism is critical for capacity planning and cost estimation. The <code>O(n²)</code> complexity of attention means that doubling the <span class="term" data-term="context-window">context window</span> quadruples the attention compute. This directly impacts serving costs, latency SLAs, and the feasibility of long-context features. When a stakeholder asks "why can't we just make the context window 10x longer?" — this is the answer.</div>

<h2>The Transformer's Impact Beyond NLP</h2>
<p>While the Transformer was designed for machine translation, its impact quickly spread far beyond NLP. The architecture's generality — it operates on sequences of vectors — meant it could be applied to any domain where data can be tokenized into sequences:</p>

<ul>
<li><strong>Computer Vision:</strong> Vision Transformers (ViT) split images into patches, treat each patch as a token, and apply standard Transformer layers. This approach matched or exceeded CNNs on image classification.</li>
<li><strong>Protein Folding:</strong> DeepMind's AlphaFold2 uses attention mechanisms to model pairwise residue interactions in protein sequences, achieving breakthrough accuracy in structure prediction.</li>
<li><strong>Audio & Speech:</strong> Whisper (OpenAI) and other speech models use Transformer architectures for audio transcription and generation.</li>
<li><strong>Code Generation:</strong> Models like Codex and AlphaCode apply Transformer architectures to source code, treating code as a sequence of tokens.</li>
<li><strong>Multimodal:</strong> Models like Gemini and GPT-4V process interleaved text, image, audio, and video tokens through unified Transformer backbones.</li>
</ul>

<div class="key-concept"><strong>Key Concept:</strong> The Transformer is not just an NLP architecture — it is a general-purpose sequence processor. Its dominance across vision, protein science, robotics, and code demonstrates that self-attention is a fundamental computational primitive, not a domain-specific trick. This universality is what makes it the backbone of modern AI.</div>

<h2>The Positional Encoding Problem and RoPE</h2>
<p>Self-attention is <strong>permutation-invariant</strong> — it treats tokens as an unordered set, not a sequence. "Dog bites man" and "Man bites dog" would produce identical attention scores without positional information. The original Transformer solved this with <span class="term" data-term="positional-encoding">sinusoidal positional encodings</span>: deterministic sine and cosine functions that inject position information by adding a position-dependent vector to each token embedding.</p>

<p>But sinusoidal encodings had limitations. As the field pushed toward longer contexts (from 512 tokens in BERT to 128K+ in modern models), the original scheme struggled to generalise to sequence lengths beyond what was seen during training. This sparked an evolution of positional encoding methods that is one of the most consequential developments in post-2017 Transformer research:</p>

<table>
<thead>
<tr><th>Method</th><th>Mechanism</th><th>Length Generalisation</th><th>Used In</th></tr>
</thead>
<tbody>
<tr><td><strong>Sinusoidal</strong></td><td>Add fixed sin/cos vectors to input embeddings</td><td>Moderate — degrades beyond training length</td><td>Original Transformer (2017)</td></tr>
<tr><td><strong>Learned Positions</strong></td><td>Learnable embedding per position slot</td><td>Poor — hard limit at max trained length</td><td>BERT, GPT-2</td></tr>
<tr><td><strong>RoPE</strong> (Rotary Position Embedding)</td><td>Rotates Q and K vectors by position-dependent angles; relative distance encoded in dot product</td><td>Good — extendable via NTK-aware scaling, YaRN</td><td>LLaMA, Gemini, Mistral, Qwen</td></tr>
<tr><td><strong>ALiBi</strong></td><td>Adds linear bias to attention scores based on token distance</td><td>Excellent — no learned parameters, zero-shot extrapolation</td><td>BLOOM, MPT</td></tr>
</tbody>
</table>

<p><strong>RoPE</strong> (Su et al., 2021) has emerged as the dominant choice for modern LLMs. The core insight is elegant: instead of adding positional information to the token embeddings, RoPE applies a rotation matrix to the Query and Key vectors before the dot product. The rotation angle depends on the token's position in the sequence. Because the dot product of two rotated vectors naturally encodes the <em>relative</em> angular difference between them, the attention score between two tokens automatically reflects their relative distance — without learning any extra parameters. This means "how far apart are these tokens?" is baked into the geometry of the attention computation itself.</p>

<p>Critically, RoPE-based models can be extended to longer contexts after training through interpolation techniques like <strong>NTK-aware scaling</strong> (which adjusts the frequency base of the rotations) and <strong>YaRN</strong> (Yet another RoPE extensioN). This capability is what allows models trained on 8K contexts to be extended to 128K+ — a feature directly impacting product capabilities like long-document analysis and extended conversations in Gemini.</p>

<div class="pro-tip"><strong>PM Perspective:</strong> The choice of positional encoding has direct product impact. When a customer asks "Can Gemini analyse my entire codebase in a single prompt?", the answer depends on whether the model's positional encoding supports that context length — and at what quality. RoPE scaling enables extension, but quality degrades at extreme lengths. A PM must insist on "needle in a haystack" evaluations at the target length before committing to context-window claims in product marketing.</div>

<h2>From One Paper to Industry Transformation</h2>
<p>The trajectory from the 2017 paper to today's multi-billion-dollar AI industry is remarkably direct. BERT (2018) showed that Transformer encoders could be pre-trained for powerful representations. GPT-2 (2019) showed that Transformer decoders could generate coherent text. GPT-3 (2020) showed that scaling these models to 175 billion parameters unlocked surprising emergent capabilities like <span class="term" data-term="few-shot">few-shot</span> learning. By 2023, Transformer-based <span class="term" data-term="llm">LLMs</span> were powering consumer products serving hundreds of millions of users.</p>

<p>Every major AI lab — Google DeepMind, OpenAI, Anthropic, Meta AI, Mistral — uses Transformer variants as the foundation of their flagship models. Understanding this architecture is not optional for anyone working in AI — it is foundational literacy.</p>
`,
    quiz: {
      questions: [
        {
          question: 'You are a PM evaluating a proposal to replace an LSTM-based text classification pipeline with a Transformer model. The dataset consists of legal documents averaging 5,000 tokens. Your ML lead says the Transformer will be "strictly better." What is the most important technical nuance you should push back on?',
          type: 'mc',
          options: [
            'Transformers cannot process text longer than 512 tokens under any circumstances',
            'The O(n²) attention complexity means 5,000-token documents will be very expensive to process, and you need to discuss context-window strategies',
            'LSTMs are always more accurate than Transformers on long documents because they have explicit memory',
            'Transformers require more training data than LSTMs to achieve comparable accuracy'
          ],
          correct: 1,
          explanation: 'Self-attention has O(n²) complexity with respect to sequence length. At 5,000 tokens, the attention matrix has 25 million entries per layer per head, making compute and memory significant concerns. A good PM would push for discussion of efficient attention variants, chunking strategies, or hierarchical approaches before committing to the migration.',
          difficulty: 'applied',
          expertNote: 'A world-class AI PM would also consider whether the task even requires full bidirectional attention over the entire document, or if a sparse/local attention pattern could suffice — and would know about methods like Longformer or BigBird that address this.'
        },
        {
          question: 'Which of the following were genuine limitations of RNN-based architectures that the Transformer addressed? Select all that apply.',
          type: 'multi',
          options: [
            'Sequential processing prevented GPU parallelism across the time dimension',
            'Information degradation over long sequences due to the hidden-state bottleneck',
            'RNNs could not model any form of dependencies between tokens',
            'RNNs could not be trained with backpropagation',
            'Difficulty scaling to large datasets due to training speed constraints'
          ],
          correct: [0, 1, 4],
          explanation: 'RNNs could model dependencies (option C is false) and were trained with backpropagation through time (option D is false). The real limitations were the sequential bottleneck preventing parallelism, information degradation in the hidden state over long sequences, and the resulting inability to scale training to massive datasets efficiently.',
          difficulty: 'foundational',
          expertNote: 'An expert would also note that vanishing/exploding gradients in RNNs (partially addressed by LSTMs) compounded the scaling problem, making reliable training of very deep/long RNN models fragile compared to Transformers with residual connections and layer normalization.'
        },
        {
          question: 'A researcher at DeepMind proposes using a Transformer to model protein-protein interactions by treating amino acid residues as tokens. A skeptical colleague argues that "Transformers are a language model architecture and cannot work outside NLP." How would you evaluate this claim?',
          type: 'scenario',
          options: null,
          correct: 'The claim is incorrect. The Transformer is a general-purpose sequence processor — it operates on sequences of vectors, not inherently on language. Any data that can be tokenized into a sequence of embeddings can be processed by a Transformer. AlphaFold2 already demonstrated this for protein structure prediction. The key insight is that self-attention captures pairwise relationships between elements, which is useful in many domains beyond NLP. The PM should support cross-domain exploration while ensuring the team has domain expertise to design appropriate tokenization and training schemes.',
          explanation: 'Transformers operate on abstract sequences of vectors. The self-attention mechanism computes pairwise interactions between any elements in a sequence, making it domain-agnostic. Success in vision (ViT), protein science (AlphaFold2), and audio (Whisper) confirms this generality.',
          difficulty: 'applied',
          expertNote: 'A top AI PM would recognize that the real challenge in cross-domain Transformer application is not the architecture itself but the design of appropriate tokenization, positional encoding, and training objectives for the new domain.'
        },
        {
          question: 'The original "Attention Is All You Need" paper achieved state-of-the-art results on machine translation. What was the most significant practical advantage of the Transformer over previous SOTA models beyond raw accuracy?',
          type: 'mc',
          options: [
            'It required no pre-training data and could work zero-shot',
            'It dramatically reduced training time by enabling full parallelization across the sequence dimension',
            'It eliminated the need for tokenization entirely',
            'It used fewer parameters than any prior model'
          ],
          correct: 1,
          explanation: 'The Transformer trained in 3.5 days on 8 GPUs for English-to-French translation, compared to weeks for comparable RNN-based models. This speedup came from eliminating sequential dependencies, allowing all positions to be processed in parallel. This efficiency gain was arguably even more impactful than the accuracy improvement, because it unlocked the scaling regime that led to modern LLMs.',
          difficulty: 'foundational',
          expertNote: 'A world-class PM understands that training efficiency is not just a research convenience — it directly determines the iteration speed of the product team, the cost of experimentation, and the feasibility of scaling to larger models and datasets.'
        },
        {
          question: 'Your team is building a real-time document analysis feature for Gemini. Users will submit documents of highly variable lengths — some 200 tokens, others 50,000 tokens. Given the Transformer\'s computational profile, which product strategy best addresses this?',
          type: 'mc',
          options: [
            'Set a universal context window of 50,000 tokens for all requests to ensure no document is truncated',
            'Implement tiered processing: use the actual document length to allocate compute dynamically, with efficient attention variants for long documents and standard attention for short ones',
            'Limit all documents to 2,000 tokens to keep costs predictable',
            'Use an RNN-based model for long documents and a Transformer for short ones'
          ],
          correct: 1,
          explanation: 'Dynamic compute allocation is the correct product strategy. Using a fixed 50,000-token window wastes enormous compute on short documents (O(n²) means padding is costly). Truncating to 2,000 tokens destroys the value proposition for long-document users. Switching architectures creates maintenance burden and inconsistent quality. Tiered processing with efficient attention variants (sparse attention, sliding window) for long documents is the standard industry approach.',
          difficulty: 'expert',
          expertNote: 'At DeepMind, this is exactly the kind of tradeoff PMs navigate daily with Gemini. The PM should also consider whether to expose context-window tiers as a pricing dimension in the API, and should understand techniques like KV-cache optimization that reduce serving costs for long-context inference.'
        }
      ]
    }
  },

  // ─────────────────────────────────────────────
  // L02 — Self-Attention: Q, K, V Matrices
  // ─────────────────────────────────────────────
  l02: {
    title: 'Self-Attention — Q, K, V Matrices Step by Step',
    content: `
<h2>The Intuition Behind Self-Attention</h2>
<p><span class="term" data-term="self-attention">Self-attention</span> is the core mechanism that gives <span class="term" data-term="transformer">Transformers</span> their power. The fundamental idea is deceptively simple: for each element in a sequence, compute how much "attention" it should pay to every other element, then use those attention weights to create a new, context-enriched representation of each element.</p>

<p>Consider the sentence: "The animal didn't cross the street because <strong>it</strong> was too tired." What does "it" refer to? A human reader effortlessly resolves this — "it" refers to "the animal" (because animals get tired, streets do not). Self-attention provides the mechanism for a model to learn exactly this kind of contextual resolution. When processing the token "it," the self-attention mechanism can assign a high attention weight to "animal," effectively building a representation of "it" that encodes the information "this pronoun refers to the animal."</p>

<div class="key-concept"><strong>Key Concept:</strong> Self-attention allows every token in a sequence to directly interact with every other token, regardless of their distance. This creates representations that are contextualized — the representation of a word changes depending on its surrounding context. The word "bank" gets a different representation in "river bank" vs. "bank account" because it attends to different context tokens.</div>

<h2>The Query-Key-Value Framework</h2>
<p>The mathematical machinery of self-attention is built on three learned linear projections called <span class="term" data-term="query-key-value">Query (Q), Key (K), and Value (V)</span>. Understanding these is essential for any AI PM who needs to discuss model architecture with engineering teams.</p>

<p>The analogy that works best: imagine a library search system.</p>
<ul>
<li><strong>Query (Q):</strong> "What am I looking for?" — Each token generates a query vector that represents what information it needs from the rest of the sequence.</li>
<li><strong>Key (K):</strong> "What do I contain?" — Each token generates a key vector that advertises what information it has to offer.</li>
<li><strong>Value (V):</strong> "Here is my actual content." — Each token generates a value vector containing the information that will be retrieved if the query matches the key.</li>
</ul>

<p>The attention score between two tokens is the dot product of one token's Query with another token's Key. High dot products mean the query and key are "aligned" — the information being sought matches the information being offered. These scores determine how much of each token's Value vector gets incorporated into the output.</p>

<div class="example-box"><h4>Example</h4>
<p>For the sentence "The cat sat on the mat":</p>
<ul>
<li>When processing "sat", its Query might encode "I need to know WHO is doing the sitting."</li>
<li>"cat" generates a Key that encodes "I am an agent/subject."</li>
<li>The dot product <code>Q("sat") · K("cat")</code> is high, meaning "sat" will attend strongly to "cat."</li>
<li>The Value vector of "cat" — containing rich semantic information about this particular cat in context — then flows into the updated representation of "sat."</li>
</ul>
<p>The result: the output representation of "sat" now encodes not just the concept of sitting, but <em>who</em> is sitting. This is contextual representation in action.</p>
</div>

<h2>The Mathematics Step by Step</h2>
<p>Let's walk through the exact computation. Given an input sequence of <code>n</code> <span class="term" data-term="token">tokens</span>, each represented as a <code>d</code>-dimensional <span class="term" data-term="embedding">embedding</span> vector, we stack them into a matrix <code>X</code> of shape <code>(n × d)</code>.</p>

<p><strong>Step 1: Linear Projections</strong></p>
<p>We learn three weight matrices: <code>W_Q</code>, <code>W_K</code>, <code>W_V</code>, each of shape <code>(d × d_k)</code> where <code>d_k</code> is the dimensionality of the queries and keys (often <code>d_k = d / h</code> where <code>h</code> is the number of attention heads).</p>
<p><code>Q = X · W_Q</code> &nbsp;&nbsp; (shape: n × d_k)</p>
<p><code>K = X · W_K</code> &nbsp;&nbsp; (shape: n × d_k)</p>
<p><code>V = X · W_V</code> &nbsp;&nbsp; (shape: n × d_v)</p>

<p><strong>Step 2: Compute Attention Scores</strong></p>
<p>The raw attention scores are computed as the dot product of Q and the transpose of K:</p>
<p><code>Scores = Q · K^T</code> &nbsp;&nbsp; (shape: n × n)</p>
<p>This produces an <code>n × n</code> matrix where entry <code>(i, j)</code> represents how much token <code>i</code> wants to attend to token <code>j</code>.</p>

<p><strong>Step 3: Scale</strong></p>
<p>The scores are divided by <code>√d_k</code> to prevent the dot products from growing too large in magnitude, which would cause the softmax to saturate into near-one-hot distributions:</p>
<p><code>Scaled_Scores = Scores / √d_k</code></p>

<div class="warning"><strong>Common Misconception:</strong> The scaling factor <code>√d_k</code> is not arbitrary. As the dimensionality <code>d_k</code> grows, the variance of dot products grows proportionally. Without scaling, the softmax would produce extremely peaked distributions, causing the model to attend to only one token and ignore all others. This is equivalent to losing the ability to combine information from multiple sources — defeating the purpose of attention.</div>

<p><strong>Step 4: Softmax</strong></p>
<p>Apply softmax row-wise to convert scores into probability distributions:</p>
<p><code>Attention_Weights = softmax(Scaled_Scores)</code> &nbsp;&nbsp; (shape: n × n)</p>
<p>Each row sums to 1, representing the attention distribution for that token over all other tokens.</p>

<p><strong>Step 5: Weighted Sum of Values</strong></p>
<p>Finally, multiply the attention weights by the Value matrix:</p>
<p><code>Output = Attention_Weights · V</code> &nbsp;&nbsp; (shape: n × d_v)</p>
<p>Each output row is a weighted combination of all Value vectors, where the weights are determined by the attention scores.</p>

<p>Putting it all together, the famous formula:</p>
<p><code>Attention(Q, K, V) = softmax(Q · K^T / √d_k) · V</code></p>

<div class="key-concept"><strong>Key Concept:</strong> The entire self-attention computation is a series of matrix multiplications and a softmax. There are no sequential dependencies — every token's output can be computed in parallel. This is why Transformers are so efficient on modern GPUs, which are optimized for large matrix operations.</div>

<h2>What the Attention Matrix Reveals</h2>
<p>The <code>n × n</code> attention matrix is one of the most useful tools for interpreting Transformer behavior. Each row shows the attention distribution for one token — which other tokens it's "looking at." Researchers and engineers visualize these matrices as heatmaps to understand what the model has learned.</p>

<p>Common patterns that emerge in trained models:</p>
<ul>
<li><strong>Diagonal patterns:</strong> Tokens attending to themselves or adjacent tokens (local/positional attention).</li>
<li><strong>Columnar patterns:</strong> Many tokens attending to a specific anchor token like a period, comma, or [CLS] token.</li>
<li><strong>Coreference patterns:</strong> Pronouns strongly attending to their antecedents ("it" → "animal").</li>
<li><strong>Syntactic patterns:</strong> Verbs attending to their subjects, adjectives attending to the nouns they modify.</li>
</ul>

<div class="pro-tip"><strong>PM Perspective:</strong> Attention visualizations are a powerful tool for debugging and building trust with stakeholders. When a model makes an incorrect prediction, showing that it attended to the wrong part of the input provides an intuitive explanation — far more accessible than "the loss function had a high value." As a PM, you should know that attention-based explanations are approximate (they show correlation, not necessarily causation), but they are the most interpretable window into Transformer behavior and invaluable for responsible AI reviews.</div>

<div class="interactive" data-interactive="attention-viz"></div>

<h2>Causal (Masked) Self-Attention</h2>
<p>In autoregressive models like <span class="term" data-term="gpt">GPT</span>, there is a critical constraint: when predicting token <code>t+1</code>, the model must not "see" tokens at positions <code>t+1, t+2, ...</code> because those are the tokens it's trying to predict. This is enforced through <strong>causal masking</strong> (also called "masked self-attention").</p>

<p>Before the softmax step, a mask is applied that sets the upper-triangular entries of the score matrix to <code>-∞</code>. After softmax, these become 0, ensuring that each token can only attend to itself and previous tokens. This creates the autoregressive property: the model generates text left-to-right, and each position only has access to its left context.</p>

<table>
<thead>
<tr><th>Attention Type</th><th>Masking</th><th>Use Case</th><th>Example Models</th></tr>
</thead>
<tbody>
<tr><td>Bidirectional (Full)</td><td>None — all tokens see all tokens</td><td>Understanding/encoding tasks</td><td>BERT, encoders in T5</td></tr>
<tr><td>Causal (Masked)</td><td>Upper triangle masked — tokens see only left context</td><td>Text generation / autoregressive</td><td>GPT, Gemini, LLaMA</td></tr>
<tr><td>Cross-Attention</td><td>Varies — decoder attends to encoder outputs</td><td>Encoder-decoder translation</td><td>Original Transformer, T5, Whisper</td></tr>
</tbody>
</table>

<h2>The Computational Cost of Self-Attention</h2>
<p>Self-attention's <code>O(n²)</code> complexity is both its strength and its limitation. The <code>n × n</code> attention matrix means that both memory and compute scale quadratically with sequence length. For a sequence of 1,000 tokens, that's 1 million attention scores per head per layer. For 100,000 tokens, it's 10 billion.</p>

<p>This has driven a rich area of research into efficient attention variants:</p>
<ul>
<li><strong>Sparse Attention:</strong> Only compute attention for a subset of token pairs (e.g., local windows + global tokens).</li>
<li><strong>Linear Attention:</strong> Reformulate the computation to avoid materializing the full n × n matrix.</li>
<li><strong>Flash Attention:</strong> Hardware-aware implementations that reduce memory I/O by fusing operations, achieving significant wall-clock speedups without approximation.</li>
<li><strong>Sliding Window:</strong> Each token only attends to a fixed-size local window, with information propagating through multiple layers.</li>
</ul>

<div class="pro-tip"><strong>PM Perspective:</strong> When scoping a product feature that involves long documents or conversations, the PM must understand that context length is not a free parameter. Doubling the context window roughly quadruples the serving cost for that request. This directly impacts pricing tiers, rate limits, and latency targets. A PM who understands Q, K, V can have a meaningful conversation with engineers about whether a particular use case truly needs 100K context or whether RAG or summarization could achieve the same user outcome at a fraction of the cost.</div>
`,
    quiz: {
      questions: [
        {
          question: 'In the self-attention formula Attention(Q, K, V) = softmax(QK^T / √d_k) · V, what would happen if the scaling factor √d_k were removed?',
          type: 'mc',
          options: [
            'The model would be unable to learn any attention patterns',
            'The softmax outputs would become nearly one-hot, causing the model to attend to only a single token per position and losing the ability to aggregate information from multiple sources',
            'The model would attend equally to all tokens regardless of content',
            'Training would become faster due to simplified computation'
          ],
          correct: 1,
          explanation: 'Without scaling, as d_k grows, the dot products grow in variance, causing the softmax inputs to have large magnitudes. Softmax with large inputs produces near-one-hot distributions (one value close to 1, rest close to 0). This means each token would attend almost exclusively to one other token, losing the crucial ability to blend information from multiple context positions.',
          difficulty: 'applied',
          expertNote: 'This is also related to the "entropy collapse" problem in attention. A world-class PM should understand that attention temperature tuning and scaling are actively researched areas, especially for very deep models where attention patterns can degenerate in later layers.'
        },
        {
          question: 'Your team is debating whether to use bidirectional or causal (masked) attention for a new document understanding API. The API needs to classify document types, extract key entities, and generate summaries. Which attention strategy is most appropriate and why?',
          type: 'scenario',
          options: null,
          correct: 'This use case has both understanding tasks (classification, entity extraction) and generation tasks (summarization). The optimal approach is to use a model with bidirectional attention for the encoding/understanding phases and causal attention for generation — essentially an encoder-decoder architecture or a decoder-only model used with careful prompting. A purely causal model can handle all tasks but may sacrifice encoding quality since each token cannot attend to future context. A PM should weigh the engineering complexity of maintaining two attention modes vs. the quality gains, and consider whether a strong decoder-only model with sufficient scale can handle all three tasks well enough through prompting.',
          explanation: 'Classification and extraction benefit from bidirectional context (seeing the whole document), while summarization requires autoregressive generation (causal masking). An encoder-decoder setup naturally handles this, while a decoder-only model can approximate it by first processing the document as a "prompt" and then generating. The PM must balance architectural complexity against task quality.',
          difficulty: 'expert',
          expertNote: 'At DeepMind, Gemini uses a decoder-only architecture that handles both understanding and generation. The PM should know that modern decoder-only models at sufficient scale can rival encoder-decoder models on understanding tasks, simplifying the serving infrastructure.'
        },
        {
          question: 'A junior engineer on your team says: "The Query, Key, and Value matrices in self-attention are like a database lookup — the Query searches for matching Keys, and returns the corresponding Values." How would you evaluate this analogy?',
          type: 'mc',
          options: [
            'It is completely wrong — Q, K, V have no relationship to database concepts',
            'It is a useful analogy with one critical caveat: unlike a database lookup that returns one exact match, attention returns a weighted combination of ALL values, with weights determined by query-key similarity',
            'It is perfect and needs no caveats',
            'It is misleading because Queries and Keys are the same thing in self-attention'
          ],
          correct: 1,
          explanation: 'The database analogy captures the core idea well: Q represents "what I\'m looking for," K represents "what I have to offer," and V represents "the content to retrieve." But the critical difference is that attention performs soft retrieval — it returns a blended mixture of all values, not a single exact match. This blending is what makes attention so powerful for capturing nuanced contextual relationships.',
          difficulty: 'foundational',
          expertNote: 'An expert PM would also note that Q, K, and V are all derived from the same input through different learned projections in self-attention, whereas in cross-attention, Q comes from one sequence and K, V come from another — a distinction important for encoder-decoder architectures.'
        },
        {
          question: 'Which of the following are valid approaches to reduce the computational cost of self-attention for long sequences? Select all that apply.',
          type: 'multi',
          options: [
            'Flash Attention — hardware-aware fused kernels that reduce memory I/O without approximation',
            'Sparse attention patterns like sliding windows or global+local tokens',
            'Simply reducing the model\'s hidden dimension d to reduce d_k',
            'Using linear attention reformulations that avoid materializing the n×n matrix',
            'Removing the softmax to make attention purely linear'
          ],
          correct: [0, 1, 3],
          explanation: 'Flash Attention, sparse attention, and linear attention are all established techniques for reducing attention cost. Reducing d_k (option C) would reduce compute per attention head but would also reduce the model\'s representational capacity and is not primarily an efficiency technique for long sequences. Removing softmax entirely (option E) would fundamentally change the attention mechanism\'s properties and is not a standard approach.',
          difficulty: 'applied',
          expertNote: 'A top PM should know that Flash Attention specifically is critical to production systems — it provides 2-4x wall-clock speedups with no quality loss, making it a free lunch for serving efficiency. Understanding the difference between exact and approximate attention methods matters for quality guarantees in production.'
        },
        {
          question: 'You are reviewing attention visualization heatmaps from a model that frequently misclassifies sarcastic product reviews as positive. What pattern in the attention maps would most likely explain this failure?',
          type: 'mc',
          options: [
            'The attention maps show uniform attention across all tokens',
            'The model strongly attends to positive sentiment words (e.g., "great," "love") while failing to attend to negation cues and contextual markers of sarcasm (e.g., "oh sure," "as if," "yeah right")',
            'The attention maps show perfect coreference resolution',
            'The model attends exclusively to punctuation marks'
          ],
          correct: 1,
          explanation: 'Sarcasm detection requires attending to subtle contextual cues that invert the surface sentiment. If attention maps show the model fixating on positive words while ignoring sarcastic markers, this explains the misclassification. The model is essentially doing shallow keyword matching rather than understanding the pragmatic context. This insight could guide targeted data augmentation or fine-tuning strategies.',
          difficulty: 'applied',
          expertNote: 'A PM conducting a responsible AI review would use this attention analysis to build a case for targeted improvement — e.g., curating a sarcasm-focused evaluation dataset, and tracking the attention pattern shift after fine-tuning as evidence of genuine improvement rather than just metric gaming.'
        }
      ]
    }
  },

  // ─────────────────────────────────────────────
  // L03 — Multi-Head Attention & Positional Encoding
  // ─────────────────────────────────────────────
  l03: {
    title: 'Multi-Head Attention & Positional Encoding',
    content: `
<h2>Why One Attention Head Is Not Enough</h2>
<p>A single <span class="term" data-term="self-attention">self-attention</span> head computes one set of attention weights — one "perspective" on how tokens relate to each other. But language (and data in general) has many simultaneous types of relationships. In the sentence "The lawyer who represented the defendant argued the case brilliantly," a single attention head would have to simultaneously capture syntactic structure (who → lawyer), semantic role (argued → lawyer as agent), and coreference (the case → the defendant's case). Asking one set of Q, K, V projections to capture all these diverse relationships is asking too much.</p>

<p><span class="term" data-term="multi-head-attention">Multi-head attention</span> solves this by running multiple attention heads in parallel, each with its own learned <code>W_Q</code>, <code>W_K</code>, <code>W_V</code> projection matrices. Each head can specialize in capturing a different type of relationship. The outputs of all heads are then concatenated and linearly projected to produce the final output.</p>

<div class="key-concept"><strong>Key Concept:</strong> Multi-head attention is the Transformer's way of jointly attending to information from different representation subspaces at different positions. Each head operates on a lower-dimensional projection of the input (typically d_model / h, where h is the number of heads), keeping the total compute roughly constant compared to a single full-dimension head while dramatically increasing representational richness.</div>

<h2>The Multi-Head Attention Mechanism</h2>
<p>Formally, given <code>h</code> attention heads, the computation proceeds as follows:</p>

<p><strong>For each head <code>i</code> (from 1 to h):</strong></p>
<p><code>head_i = Attention(X · W_Q^i, X · W_K^i, X · W_V^i)</code></p>

<p>where <code>W_Q^i</code>, <code>W_K^i</code>, <code>W_V^i</code> are the projection matrices for head <code>i</code>, each of shape <code>(d_model × d_k)</code> with <code>d_k = d_model / h</code>.</p>

<p><strong>Concatenate all head outputs:</strong></p>
<p><code>MultiHead(X) = Concat(head_1, head_2, ..., head_h) · W_O</code></p>

<p>where <code>W_O</code> is a final output projection matrix of shape <code>(h · d_k × d_model)</code> that maps the concatenated heads back to the model's dimensionality.</p>

<div class="example-box"><h4>Example</h4>
<p>In the original Transformer, <code>d_model = 512</code> and <code>h = 8</code> heads. Each head operates on <code>d_k = 512 / 8 = 64</code> dimensions. The total compute for 8 heads of 64 dimensions is comparable to a single head of 512 dimensions, but the model gets 8 independent "views" of the token relationships.</p>
<p>In modern large models, the numbers are much larger. GPT-3 uses <code>d_model = 12,288</code> with <code>96</code> attention heads, giving <code>d_k = 128</code> per head. Gemini Ultra reportedly uses even larger configurations.</p>
</div>

<h2>What Different Heads Learn</h2>
<p>Research on interpreting attention heads has revealed fascinating specialization patterns. In trained Transformer models, different heads reliably learn to capture different linguistic phenomena:</p>

<table>
<thead>
<tr><th>Head Type</th><th>What It Captures</th><th>Example Pattern</th></tr>
</thead>
<tbody>
<tr><td>Positional heads</td><td>Relative position / adjacency</td><td>Each token attends strongly to the token immediately before or after it</td></tr>
<tr><td>Syntactic heads</td><td>Grammatical dependencies</td><td>Verbs attend to their subjects; determiners attend to their nouns</td></tr>
<tr><td>Coreference heads</td><td>Pronoun resolution</td><td>"it", "they", "she" attend strongly to their antecedents</td></tr>
<tr><td>Rare token heads</td><td>Infrequent / important tokens</td><td>High attention to named entities, technical terms, numbers</td></tr>
<tr><td>Separator heads</td><td>Structural markers</td><td>Broad attention to punctuation, paragraph breaks, special tokens</td></tr>
</tbody>
</table>

<p>This specialization is not explicitly programmed — it emerges naturally from training. Each head's projection matrices evolve to create Q, K, V spaces that are most useful for the relationships that head has learned to detect.</p>

<div class="warning"><strong>Common Misconception:</strong> It is tempting to think that every attention head is equally important and captures a unique, essential pattern. In practice, research has shown that many heads in trained Transformers are redundant — some can be pruned (removed) with minimal impact on performance. This finding has important implications for model compression and efficient inference. Not all heads earn their compute budget.</div>

<h2>Grouped-Query Attention (GQA) and Multi-Query Attention (MQA)</h2>
<p>A significant optimization in modern <span class="term" data-term="llm">LLMs</span> is reducing the memory cost of the KV cache during inference. In standard multi-head attention, each head has its own K and V projections, which must be cached for every token in the sequence during autoregressive generation. For a model with 96 heads and a 100K <span class="term" data-term="context-window">context window</span>, this KV cache can consume tens of gigabytes of GPU memory.</p>

<p><strong>Multi-Query Attention (MQA):</strong> All attention heads share a single K and V projection, while each head has its own Q projection. This reduces the KV cache by a factor of <code>h</code> (number of heads), dramatically reducing memory usage during inference.</p>

<p><strong>Grouped-Query Attention (GQA):</strong> A compromise between standard multi-head and multi-query attention. Heads are divided into groups, and each group shares K and V projections. With <code>g</code> groups, the KV cache is reduced by a factor of <code>h/g</code>. GQA is used in models like LLaMA 2 and Gemini.</p>

<div class="pro-tip"><strong>PM Perspective:</strong> GQA is a perfect example of a technical optimization with direct product impact. Reducing the KV cache memory by 4-8x means you can serve longer context windows on the same hardware, or serve more concurrent users. When planning capacity for a Gemini API launch, a PM must understand that the choice between MHA, GQA, and MQA directly affects how many users can be served per GPU, which determines infrastructure cost, pricing, and rate limits.</div>

<h2>The Positional Encoding Problem</h2>
<p>Self-attention is fundamentally <strong>permutation-invariant</strong>. If you shuffle the order of tokens in a sequence but keep the same set of tokens, the attention scores between any two tokens remain identical. The operation <code>Q · K^T</code> only cares about the content of the Q and K vectors, not their positions.</p>

<p>But word order is crucial for language! "Dog bites man" and "Man bites dog" contain the same tokens but have entirely different meanings. Without some mechanism to inject position information, a Transformer cannot distinguish between these two sentences.</p>

<p><span class="term" data-term="positional-encoding">Positional encoding</span> solves this by adding position-dependent signals to the input <span class="term" data-term="embedding">embeddings</span> before they enter the Transformer layers.</p>

<h2>Sinusoidal Positional Encoding (Original Transformer)</h2>
<p>The original Transformer used deterministic sinusoidal functions to generate positional encodings:</p>

<p><code>PE(pos, 2i) = sin(pos / 10000^(2i/d_model))</code></p>
<p><code>PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))</code></p>

<p>where <code>pos</code> is the position index and <code>i</code> is the dimension index. This creates a unique encoding for each position, with several desirable properties:</p>

<ul>
<li><strong>Unique representation:</strong> Each position gets a distinct encoding vector.</li>
<li><strong>Bounded magnitude:</strong> Sine and cosine values are always in [-1, 1], preventing positional signals from dominating content signals.</li>
<li><strong>Relative position encoding:</strong> For any fixed offset <code>k</code>, <code>PE(pos + k)</code> can be represented as a linear function of <code>PE(pos)</code>, enabling the model to learn attention patterns based on relative distance.</li>
<li><strong>Extrapolation potential:</strong> Since the functions are continuous, the encoding can theoretically generalize to positions longer than those seen during training.</li>
</ul>

<h2>Learned and Rotary Positional Encodings</h2>
<p>Modern models have moved beyond sinusoidal encodings:</p>

<p><strong>Learned Positional Embeddings:</strong> Used in BERT and GPT-2. A learnable embedding vector is assigned to each position (up to a maximum sequence length). These are added to the token embeddings. The downside: the model cannot generalize to positions beyond the maximum length seen during training.</p>

<p><strong>Rotary Position Embedding (RoPE):</strong> Used in LLaMA, Gemini, and many modern LLMs. Instead of adding positional information to the input, RoPE applies a rotation to the Q and K vectors based on their position. The rotation angle depends on the position, so the dot product <code>Q · K</code> naturally encodes relative position information. RoPE has excellent properties:</p>

<ul>
<li>Naturally encodes relative positions through the angle between rotated vectors</li>
<li>The attention score between two tokens depends on their relative distance, not absolute positions</li>
<li>Can be extended to longer sequences through interpolation techniques (e.g., NTK-aware scaling, YaRN)</li>
</ul>

<p><strong>ALiBi (Attention with Linear Biases):</strong> Instead of modifying embeddings, ALiBi adds a linear bias to the attention scores based on the distance between tokens. Closer tokens get higher scores, with the slope varying per head. This is extremely simple to implement and generalizes well to longer sequences.</p>

<table>
<thead>
<tr><th>Encoding Method</th><th>Added To</th><th>Length Generalization</th><th>Used In</th></tr>
</thead>
<tbody>
<tr><td>Sinusoidal</td><td>Input embeddings</td><td>Moderate</td><td>Original Transformer</td></tr>
<tr><td>Learned</td><td>Input embeddings</td><td>Poor (fixed max)</td><td>BERT, GPT-2</td></tr>
<tr><td>RoPE</td><td>Q and K vectors</td><td>Good (with scaling)</td><td>LLaMA, Gemini, Mistral</td></tr>
<tr><td>ALiBi</td><td>Attention scores</td><td>Excellent</td><td>BLOOM, MPT</td></tr>
</tbody>
</table>

<div class="key-concept"><strong>Key Concept:</strong> Positional encoding is not a solved problem — it remains an active area of research, especially for enabling models to generalize to sequence lengths far beyond their training data. The choice of positional encoding method directly impacts a model's ability to handle long documents, multi-turn conversations, and other long-context applications that users increasingly demand.</div>

<div class="pro-tip"><strong>PM Perspective:</strong> When a user requests a feature involving very long contexts — like analyzing an entire codebase or processing a book-length document — the PM should understand that the model's positional encoding is often the binding constraint, not just the attention compute. A model trained with a 8K context window may not generalize well to 128K, even if you have the compute budget for the attention. Understanding RoPE scaling and its limitations helps a PM set realistic expectations and plan appropriate evaluation before launch.</div>
`,
    quiz: {
      questions: [
        {
          question: 'Your team is deciding between standard multi-head attention (MHA) and grouped-query attention (GQA) for a new model that will be served via API. The API must support 128K context windows. What is the primary trade-off the PM should understand?',
          type: 'mc',
          options: [
            'GQA is always worse in quality — the choice is purely about cost savings',
            'GQA reduces the KV cache memory by sharing K and V across head groups, enabling longer context windows and more concurrent users per GPU, with minimal quality degradation in practice',
            'GQA and MHA have identical memory profiles — the difference is only in training speed',
            'GQA cannot be used with causal masking, limiting it to encoder-only models'
          ],
          correct: 1,
          explanation: 'GQA shares K and V projections across groups of heads, reducing the KV cache by a factor equal to (number of heads / number of groups). For a 128K context window, this can mean the difference between fitting the model on available hardware or not. Empirically, GQA with 8 groups shows minimal quality loss compared to full MHA, making it the dominant choice for production LLMs that need long context support.',
          difficulty: 'applied',
          expertNote: 'At DeepMind, the KV cache is one of the primary constraints on serving Gemini at scale. A PM involved in capacity planning should be able to calculate approximate KV cache sizes: (2 × num_layers × num_kv_heads × d_head × seq_len × bytes_per_element) to understand the hardware implications of context window decisions.'
        },
        {
          question: 'A Transformer model trained with a maximum context length of 4,096 tokens using learned positional embeddings is now being asked to process 16,384-token documents. What will most likely happen, and what should the PM recommend?',
          type: 'scenario',
          options: null,
          correct: 'The model will fail or produce degraded outputs on tokens beyond position 4,096 because learned positional embeddings have no representation for positions beyond the training maximum. The PM should recommend either: (1) Re-training or fine-tuning with extended context using a position encoding method that supports length extrapolation (like RoPE with NTK scaling or ALiBi), (2) Using a chunking/sliding-window strategy to process the document in segments, or (3) Using a RAG-based approach to retrieve relevant sections rather than processing the full document. The PM should evaluate these options based on quality requirements, latency budget, and engineering cost.',
          explanation: 'Learned positional embeddings are fixed-size lookup tables — position 16,384 simply has no embedding, causing either an error or garbage outputs. This is a fundamental limitation of learned position encodings and one reason modern models prefer RoPE or ALiBi, which have better length generalization properties.',
          difficulty: 'applied',
          expertNote: 'A world-class PM would also know that even RoPE-based models degrade beyond their training length without explicit context extension techniques, and would insist on rigorous evaluation at the target length rather than assuming generalization.'
        },
        {
          question: 'Which of the following statements about multi-head attention are true? Select all that apply.',
          type: 'multi',
          options: [
            'Different heads can learn to capture different linguistic phenomena (syntax, coreference, etc.)',
            'The total compute of multi-head attention is roughly equivalent to a single head with full dimensionality',
            'Every attention head in a trained model captures a unique, essential pattern that cannot be pruned',
            'The outputs of all heads are concatenated and then linearly projected',
            'Multi-head attention requires sequential processing — one head at a time'
          ],
          correct: [0, 1, 3],
          explanation: 'Heads do specialize in different patterns (A), the total compute is comparable to a single full-dimension head (B), and outputs are concatenated then projected (D). However, research shows many heads are redundant and can be pruned (C is false), and all heads compute in parallel, not sequentially (E is false).',
          difficulty: 'foundational',
          expertNote: 'Head pruning research (e.g., Michel et al. 2019, "Are Sixteen Heads Really Better than One?") showed that in many layers, removing most heads barely affects performance. This finding drives practical model compression techniques important for serving efficiency.'
        },
        {
          question: 'Rotary Position Embedding (RoPE) has become the dominant positional encoding in modern LLMs like LLaMA and Gemini. What property of RoPE makes it particularly well-suited for language models compared to sinusoidal or learned encodings?',
          type: 'mc',
          options: [
            'RoPE requires fewer parameters than any other method',
            'RoPE naturally encodes relative position information in the attention score through rotation of Q and K vectors, and can be extended to longer contexts through interpolation techniques',
            'RoPE completely eliminates the need for positional information',
            'RoPE works only with English text, which is why it was designed for English-centric models'
          ],
          correct: 1,
          explanation: 'RoPE applies position-dependent rotations to Q and K vectors such that the dot product QK naturally reflects the relative distance between tokens. This relative position encoding is linguistically natural (we care about how far apart words are, not their absolute positions). Additionally, RoPE can be extended beyond training length through interpolation methods like NTK-aware scaling and YaRN, enabling long-context applications.',
          difficulty: 'applied',
          expertNote: 'The PM should understand that RoPE extension techniques (like the scaling used to extend LLaMA from 4K to 128K context) require careful evaluation — the model\'s quality at extended lengths is not guaranteed and often degrades on tasks requiring precise long-range retrieval ("needle in a haystack" tests).'
        },
        {
          question: 'You are planning the next Gemini API pricing tier. Engineering tells you that switching from 8-group GQA to 4-group GQA (fewer groups = more sharing = less KV cache) would reduce serving cost by 30% but might reduce quality on multi-step reasoning benchmarks by 1-2%. How should you approach this decision?',
          type: 'mc',
          options: [
            'Always prioritize cost reduction — 1-2% quality loss is negligible',
            'Never compromise quality — maintain 8-group GQA regardless of cost',
            'Run rigorous evaluations on the specific use cases your API customers care about, quantify the quality impact on those tasks, and make the decision based on whether the quality-cost tradeoff aligns with your product positioning and customer expectations',
            'Let the ML team decide independently since this is a purely technical choice'
          ],
          correct: 2,
          explanation: 'This is a classic PM tradeoff decision. A 30% cost reduction could translate to lower prices or better margins, but quality regressions on reasoning tasks could hurt enterprise customers who depend on multi-step analysis. The PM should drive a rigorous evaluation on task-specific benchmarks, consult with key customers or customer-facing teams, and make a data-informed decision aligned with the product strategy.',
          difficulty: 'expert',
          expertNote: 'A top AI PM would also consider offering both configurations as different API tiers — a cost-optimized tier and a quality-optimized tier — and would design the pricing and documentation to help customers self-select the appropriate tier for their use case.'
        }
      ]
    }
  },

  // ─────────────────────────────────────────────
  // L04 — Encoder-Decoder vs Decoder-Only Architectures
  // ─────────────────────────────────────────────
  l04: {
    title: 'Encoder-Decoder vs Decoder-Only Architectures',
    content: `
<h2>The Original Transformer: An Encoder-Decoder Architecture</h2>
<p>The <span class="term" data-term="transformer">Transformer</span> introduced in "Attention Is All You Need" was an <span class="term" data-term="encoder">encoder</span>-<span class="term" data-term="decoder">decoder</span> architecture, designed specifically for sequence-to-sequence tasks like machine translation. Understanding the full encoder-decoder design is essential for appreciating why modern architectures have diverged into different variants — and why <span class="term" data-term="decoder">decoder-only</span> models have come to dominate.</p>

<p>The original architecture consists of two halves:</p>

<p><strong>The Encoder</strong> processes the entire input sequence with <em>bidirectional</em> <span class="term" data-term="self-attention">self-attention</span>. Every input token can attend to every other input token, building rich contextual representations. The encoder stack (6 layers in the original paper) transforms the raw token embeddings into deep, context-aware representations.</p>

<p><strong>The Decoder</strong> generates the output sequence one token at a time using <em>causal</em> (masked) self-attention. Each decoder layer also includes a <strong>cross-attention</strong> sublayer, where the decoder's queries attend to the encoder's output representations. This is how information flows from the input to the output.</p>

<div class="key-concept"><strong>Key Concept:</strong> Cross-attention is the bridge between encoder and decoder. In cross-attention, the Queries come from the decoder (representing what the decoder needs to know right now), while the Keys and Values come from the encoder (representing the input information available). This allows the decoder to selectively focus on relevant parts of the input at each generation step.</div>

<h2>The Three Architectural Paradigms</h2>
<p>After the original Transformer, the field diverged into three major architectural paradigms, each suited to different tasks:</p>

<table>
<thead>
<tr><th>Architecture</th><th>Attention Type</th><th>Strengths</th><th>Key Models</th></tr>
</thead>
<tbody>
<tr><td><strong>Encoder-Only</strong></td><td>Bidirectional self-attention</td><td>Understanding, classification, embeddings</td><td>BERT, RoBERTa, DeBERTa</td></tr>
<tr><td><strong>Encoder-Decoder</strong></td><td>Bidirectional (encoder) + causal + cross-attention (decoder)</td><td>Translation, summarization, structured generation</td><td>T5, BART, mBART, Whisper</td></tr>
<tr><td><strong>Decoder-Only</strong></td><td>Causal (masked) self-attention</td><td>Text generation, general-purpose reasoning</td><td><span class="term" data-term="gpt">GPT</span> series, <span class="term" data-term="llm">LLaMA</span>, Gemini, Claude, Mistral</td></tr>
</tbody>
</table>

<h2>Encoder-Only Models: BERT and Understanding</h2>
<p>BERT (Bidirectional Encoder Representations from Transformers), released by Google in 2018, demonstrated the power of <span class="term" data-term="pre-training">pre-training</span> a Transformer encoder on massive text data. The key innovation was the training objective: <strong>Masked Language Modeling (MLM)</strong>. Random tokens in the input are replaced with a [MASK] token, and the model must predict the original token using bidirectional context (both left and right).</p>

<p>This bidirectionality gives encoder-only models a significant advantage for understanding tasks. When classifying the sentiment of a review, the representation of each word benefits from knowing what comes both before and after it. BERT and its successors dominated NLU benchmarks for several years.</p>

<div class="example-box"><h4>Example</h4>
<p><strong>Why bidirectional context matters for understanding:</strong></p>
<p>Consider: "The movie was not good, but the acting was [MASK]."</p>
<p>A left-to-right model would only know "The movie was not good, but the acting was" — the word "but" suggests a contrast, but the model has limited information. A bidirectional model also sees the period after [MASK], confirming the sentence ends there, and can use the contrastive structure ("not good, but...") to strongly predict a positive word like "excellent" or "superb."</p>
</div>

<p>However, encoder-only models have a fundamental limitation: they cannot naturally <em>generate</em> text. They produce fixed-length representations, not sequential outputs. To use BERT for generation, you need workarounds that are slow and awkward. This limitation became increasingly problematic as the field shifted toward generation-centric applications.</p>

<h2>Encoder-Decoder Models: T5 and Versatility</h2>
<p>Google's T5 (Text-to-Text Transfer Transformer, 2020) took a different approach. By framing every NLP task as a text-to-text problem — where both input and output are text strings — T5 showed that a single encoder-decoder model could handle classification, translation, summarization, and question answering simultaneously.</p>

<p>For classification, the input might be "classify: This movie is terrible" and the output "negative." For translation, "translate English to French: Hello world" → "Bonjour le monde." This unified framing was elegant and demonstrated that the encoder-decoder architecture was genuinely versatile.</p>

<p>Encoder-decoder models maintain an advantage in tasks where the input and output are clearly distinct and the output is shorter than the input (like summarization or translation), because the encoder can build a rich bidirectional representation of the full input, which the decoder then consumes. The decoder never needs to "waste" capacity re-encoding the input.</p>

<h2>Decoder-Only Models: The GPT Paradigm</h2>
<p>The decoder-only architecture, pioneered by OpenAI's GPT series, uses only the decoder half of the Transformer with causal masking. The model is trained with a single, simple objective: predict the next <span class="term" data-term="token">token</span>. Given a sequence of tokens, predict what comes next. This <span class="term" data-term="pre-training">next-token prediction</span> objective turns out to be extraordinarily powerful when applied at scale.</p>

<p>In a decoder-only model, there is no separate encoder. The input (prompt) and output (completion) are treated as a single continuous sequence. The model processes the prompt tokens with causal attention, then generates new tokens one at a time, each conditioned on all previous tokens (both prompt and already-generated tokens).</p>

<div class="key-concept"><strong>Key Concept:</strong> The decoder-only model handles "understanding" and "generation" in the same architecture and with the same mechanism. The prompt is "understood" through causal self-attention as it's processed, and generation continues seamlessly from where the prompt ends. This simplicity — one architecture, one training objective, one serving path — is a major engineering advantage.</div>

<h2>Why Decoder-Only Won (For Now)</h2>
<p>By 2023, virtually all frontier <span class="term" data-term="llm">LLMs</span> — GPT-4, Gemini, Claude, LLaMA, Mistral — had converged on the decoder-only architecture. This convergence was driven by several factors:</p>

<p><strong>1. Scaling simplicity.</strong> A decoder-only model has one set of weights, one attention pattern, and one training objective. This simplicity makes it easier to scale to hundreds of billions of parameters, distribute across thousands of GPUs, and optimize serving infrastructure.</p>

<p><strong>2. Emergent general-purpose capability.</strong> At sufficient scale, decoder-only models trained on next-token prediction exhibit remarkable <span class="term" data-term="emergent-abilities">emergent abilities</span>: <span class="term" data-term="in-context-learning">in-context learning</span>, <span class="term" data-term="chain-of-thought">chain-of-thought</span> reasoning, code generation, and more. The same model handles translation, summarization, QA, and creative writing through prompting — without architectural changes.</p>

<p><strong>3. KV cache efficiency.</strong> During autoregressive generation, decoder-only models can cache the Key and Value computations for all previous tokens (the KV cache). Each new token only requires computing Q, K, V for that single token and attending to the cached K, V. With an encoder-decoder model, you must maintain both the encoder's output representations and the decoder's KV cache, complicating memory management.</p>

<p><strong>4. Unified serving infrastructure.</strong> A decoder-only model has one forward pass path, making it simpler to optimize, batch, and deploy compared to encoder-decoder models that require managing two different forward passes.</p>

<div class="warning"><strong>Common Misconception:</strong> "Decoder-only models can't understand input as well as encoder-decoder models because they only have causal attention." This is misleading. While it's true that a decoder-only model cannot use future context when processing a prompt token, empirical evidence at scale shows that sufficiently large decoder-only models match or exceed encoder-decoder models on understanding benchmarks. The causal constraint is compensated for by the enormous model capacity and the implicit bidirectional reasoning that emerges in deep causal models.</div>

<h2>When Encoder-Decoder Still Makes Sense</h2>
<p>Despite the decoder-only dominance, encoder-decoder architectures remain the better choice in specific scenarios:</p>

<ul>
<li><strong>Machine translation:</strong> When input and output are clearly separated (source and target language), the encoder can build a rich bidirectional representation of the source that the decoder consults via cross-attention. Models like NLLB (No Language Left Behind) use encoder-decoder for this reason.</li>
<li><strong>Speech recognition:</strong> Whisper (OpenAI) uses an encoder-decoder Transformer. The encoder processes the audio spectrogram bidirectionally, and the decoder generates the text transcript.</li>
<li><strong>Structured output generation:</strong> When the output is significantly shorter than the input and has a well-defined structure (e.g., extracting structured data from a long document), encoder-decoder can be more parameter-efficient.</li>
<li><strong>Efficiency with very long inputs:</strong> For tasks like summarizing a 50-page document into a paragraph, the encoder processes the document once with bidirectional attention, and the decoder generates the short summary. A decoder-only model would need to attend to the entire document at every generation step, which is less efficient.</li>
</ul>

<div class="pro-tip"><strong>PM Perspective:</strong> Understanding the architectural trade-offs is essential for making product decisions. If you are building a translation product, an encoder-decoder model may be more efficient and higher quality than prompting a decoder-only LLM. If you are building a general-purpose assistant that needs to handle open-ended user requests, a decoder-only model's versatility is invaluable. The PM should resist the temptation to use a single architecture for everything — the right architecture depends on the use case, latency requirements, and cost constraints.</div>

<h2>The Prefill-Decode Split in Modern Serving</h2>
<p>In production serving of decoder-only models, there is an important distinction between two phases of inference:</p>

<p><strong>Prefill (Prompt Processing):</strong> The model processes all prompt tokens in parallel, computing the KV cache for the entire prompt. This is compute-bound — it involves large matrix multiplications across all prompt tokens simultaneously.</p>

<p><strong>Decode (Token Generation):</strong> The model generates one token at a time, each requiring a forward pass through all layers. This is memory-bandwidth-bound — it involves reading the entire KV cache for each new token but only computing attention for a single query.</p>

<p>These two phases have very different computational profiles, which has led to sophisticated serving optimizations like:</p>
<ul>
<li><strong>Prefill-decode disaggregation:</strong> Running prefill on different hardware than decode, optimizing each for its bottleneck.</li>
<li><strong>Speculative decoding:</strong> Using a smaller draft model to propose multiple tokens, then verifying them in a single forward pass of the large model.</li>
<li><strong>Continuous batching:</strong> Dynamically batching requests at different stages (some prefilling, some decoding) to maximize GPU utilization.</li>
</ul>

<div class="pro-tip"><strong>PM Perspective:</strong> The prefill-decode split directly impacts user experience. Time-to-first-token (TTFT) is determined by the prefill phase and is proportional to prompt length. Tokens-per-second during streaming is determined by the decode phase. A PM should track both metrics separately and understand that a feature adding more context to the prompt (like RAG) increases TTFT, while a feature requiring longer outputs (like detailed analysis) affects decode time. These have different optimization strategies and user-facing tradeoffs.</div>

<h2>Looking Forward: Hybrid and New Architectures</h2>
<p>The field is not standing still. Several emerging trends suggest the decoder-only dominance may evolve:</p>

<ul>
<li><strong>State-space models (Mamba, S4):</strong> These replace attention entirely with recurrence-like mechanisms that have linear (not quadratic) scaling with sequence length. They show promising results on long-context tasks.</li>
<li><strong>Hybrid architectures:</strong> Models like Jamba combine Transformer layers with state-space layers, attempting to get the best of both worlds.</li>
<li><strong>Mixture of Experts (MoE):</strong> Models like Mixtral and Gemini use MoE layers within the Transformer, routing different tokens to different expert sub-networks. This allows scaling parameter count without proportionally scaling compute.</li>
</ul>

<div class="key-concept"><strong>Key Concept:</strong> Architecture choice is a product decision, not just a research decision. The architecture determines the model's capabilities, serving cost, latency profile, and scalability. A PM at DeepMind needs to understand these trade-offs to make informed decisions about which architecture to use for different products and features, and to evaluate proposals from research teams.</div>
`,
    quiz: {
      questions: [
        {
          question: 'Your team is building a new real-time translation feature for Google Meet using Gemini. An ML engineer proposes using the decoder-only Gemini model for translation since "it can do everything." A senior researcher suggests a specialized encoder-decoder model. As PM, how do you evaluate this?',
          type: 'scenario',
          options: null,
          correct: 'Both approaches have merit, but for real-time translation specifically, the encoder-decoder architecture has structural advantages: the encoder can build a rich bidirectional representation of the full source utterance, cross-attention provides an efficient mechanism for the decoder to selectively attend to relevant source tokens, and the architecture naturally separates the input processing from output generation. The decoder-only approach would work but may be less efficient — it must re-attend to the source at every decoding step without the benefit of bidirectional encoding. The PM should propose an evaluation: benchmark both approaches on translation quality (BLEU, human eval), latency (time-to-first-token, total generation time), and serving cost per translation. If the decoder-only model matches quality at acceptable cost, its operational simplicity (one model to maintain) could be preferred. If the encoder-decoder model is significantly better or cheaper, the specialized model is justified.',
          explanation: 'Real-time translation is a classic encoder-decoder use case: the input (source language) and output (target language) are clearly delineated, the output length is similar to the input, and the task benefits from bidirectional encoding of the source. However, modern decoder-only models at scale can also translate well, so the decision should be data-driven.',
          difficulty: 'expert',
          expertNote: 'A world-class PM would also consider the maintenance burden — if you build a specialized encoder-decoder for translation, you now have a separate model to train, evaluate, and serve alongside the main Gemini model. The operational cost of maintaining two model serving stacks could outweigh the efficiency gains of the specialized architecture.'
        },
        {
          question: 'Which of the following correctly describes the "prefill" and "decode" phases in decoder-only model inference?',
          type: 'mc',
          options: [
            'Prefill generates the output tokens, and decode processes the input prompt',
            'Prefill processes all prompt tokens in parallel (compute-bound), building the KV cache; decode generates tokens one at a time (memory-bandwidth-bound), reading the KV cache for each new token',
            'Prefill and decode are identical operations that differ only in batch size',
            'Prefill is the pre-training phase and decode is the inference phase'
          ],
          correct: 1,
          explanation: 'In production serving, prefill processes the entire prompt in a single parallel forward pass, computing and caching the K and V values for all prompt tokens. This is compute-bound because it involves large matrix multiplications. Decode then generates tokens autoregressively, one at a time, reading the full KV cache for each new token. This is memory-bandwidth-bound because the main bottleneck is reading the large KV cache from GPU memory.',
          difficulty: 'applied',
          expertNote: 'Understanding this split is crucial for PM capacity planning. Prefill cost scales with prompt length (important for RAG-heavy applications), while decode cost scales with output length (important for generation-heavy applications like coding assistants). Different use cases have very different cost profiles.'
        },
        {
          question: 'Why has the decoder-only architecture become dominant for frontier LLMs despite the theoretical advantages of encoder-decoder models for understanding tasks? Select all valid reasons.',
          type: 'multi',
          options: [
            'Scaling simplicity — one architecture, one training objective, one serving path',
            'Decoder-only models are always more accurate than encoder-decoder models on every task',
            'Emergent general-purpose capabilities at scale (in-context learning, chain-of-thought, etc.)',
            'Unified serving infrastructure is easier to optimize and deploy',
            'KV cache management is simpler with a single model stack'
          ],
          correct: [0, 2, 3, 4],
          explanation: 'Decoder-only models dominate due to scaling simplicity, emergent capabilities, unified serving, and simpler KV cache management. However, they are NOT always more accurate — encoder-decoder models can outperform on specific tasks like translation or when input is much longer than output. The decoder-only dominance is about the practical advantages of simplicity at scale, not universal superiority.',
          difficulty: 'foundational',
          expertNote: 'A top PM should recognize that the "decoder-only wins" narrative is partially driven by the AI lab business model — general-purpose models that handle many tasks through one API are more commercially viable than task-specific models, even if the latter are technically superior for individual tasks.'
        },
        {
          question: 'A product manager proposes adding "10x longer context window" as a headline feature for the next Gemini release. Based on your understanding of architecture trade-offs, what is the most critical question you should raise?',
          type: 'mc',
          options: [
            'Will the marketing team have enough time to prepare launch materials?',
            'What is the impact on time-to-first-token (prefill latency), serving cost per request, and whether the model actually maintains quality at the extended length — verified by needle-in-a-haystack and long-range reasoning evaluations?',
            'Should we file a patent for longer context windows?',
            'Can we simply increase the number of decoder layers to support longer context?'
          ],
          correct: 1,
          explanation: 'Extending context 10x has cascading impacts: O(n²) attention means ~100x more attention compute, the KV cache grows 10x (affecting memory and throughput), prefill latency increases proportionally, and the model may not maintain quality at the extended length. A responsible PM must demand evaluations like needle-in-a-haystack (can the model find specific information buried in long context?), long-range reasoning tests, and latency/cost analysis before committing to the feature.',
          difficulty: 'expert',
          expertNote: 'At DeepMind, long-context Gemini required extensive evaluation infrastructure including synthetic retrieval tasks at various depths, multi-document reasoning benchmarks, and real-user-scenario testing. The PM should also consider whether users actually need 10x longer context or whether RAG or summarization would serve the underlying use case more cost-effectively.'
        },
        {
          question: 'BERT uses an encoder-only architecture, while GPT uses a decoder-only architecture. If both models had the same number of parameters and were trained on the same data, which would you expect to perform better on a sentiment classification task where the full review text is available as input?',
          type: 'mc',
          options: [
            'GPT, because autoregressive models are always superior',
            'BERT, because bidirectional attention allows each token to attend to the full context (both before and after), producing richer representations for classification tasks',
            'They would perform identically because parameter count is the only thing that matters',
            'Neither could perform sentiment classification'
          ],
          correct: 1,
          explanation: 'For classification tasks where the full input is available (no generation needed), bidirectional attention is a natural advantage. BERT can build representations where each token is informed by its complete surrounding context, while GPT can only condition on left context. At equivalent scale, this bidirectional advantage leads to better classification performance. This is why BERT-style models dominated NLU benchmarks for years. However, at very large scale, decoder-only models partially close this gap through their massive capacity and training data.',
          difficulty: 'foundational',
          expertNote: 'The practical nuance: in the real world, decoder-only models are typically much larger than encoder-only models (175B+ vs 340M), which can compensate for the architectural disadvantage. A PM should consider whether deploying a small, efficient BERT-style model is more cost-effective for a pure classification use case than using a massive decoder-only model.'
        }
      ]
    }
  }

};
