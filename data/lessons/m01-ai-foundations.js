export const lessons = {

  // ─────────────────────────────────────────────
  // L01 — What is AI? History from Turing to Today
  // ─────────────────────────────────────────────
  l01: {
    title: 'What is AI? History from Turing to Today',
    content: `
<h2>Defining Artificial Intelligence</h2>
<p>
  <span class="term" data-term="artificial-intelligence">Artificial Intelligence (AI)</span> is one of the most consequential fields in the history of computing, yet its definition has shifted dramatically over eight decades. At its broadest, AI refers to the science and engineering of creating machines that can perform tasks normally requiring human intelligence — perceiving, reasoning, learning, planning, and generating language. But that umbrella definition hides an important nuance: what counts as "intelligent" evolves as our expectations rise. Tasks that once seemed miraculous (optical character recognition, chess, route planning) are now so routine we barely consider them AI at all. This phenomenon is known as the <span class="term" data-term="ai-effect">AI Effect</span> — the tendency for people to redefine "real AI" as whatever machines cannot yet do.
</p>
<p>
  For a product manager, the definition matters less than the capability envelope. The practical question is always: <em>What can the current generation of AI systems actually do reliably, and where do they fail?</em> Understanding that question requires understanding the historical arc — because the swings between hype and disappointment have repeated multiple times, and recognising which phase we are in directly impacts roadmap planning, hiring, and stakeholder communication.
</p>

<div class="key-concept">
  <strong>Key Concept:</strong> AI is best understood not as a single technology but as a research programme with multiple paradigms — symbolic reasoning, statistical learning, neural networks, and hybrid approaches — each dominant in different eras and each with distinct strengths.
</div>

<h2>The Foundational Era (1940s–1960s)</h2>
<p>
  The intellectual roots of AI trace to <strong>Alan Turing's</strong> 1950 paper "Computing Machinery and Intelligence," which introduced the <span class="term" data-term="turing-test">Turing Test</span> — a thought experiment asking whether a machine could converse so naturally that a human judge could not distinguish it from another human. Turing did not claim the test was a perfect measure of intelligence; rather, he used it to sidestep the philosophically fraught question "Can machines think?" and replace it with a behavioural benchmark.
</p>
<p>
  The term "Artificial Intelligence" itself was coined at the <strong>Dartmouth Workshop</strong> in 1956, organised by John McCarthy, Marvin Minsky, Nathaniel Rochester, and Claude Shannon. Their proposal was breathtakingly ambitious: "Every aspect of learning or any other feature of intelligence can in principle be so precisely described that a machine can be made to simulate it." The workshop launched AI as a formal academic discipline and set the tone for decades of optimism.
</p>
<p>
  Early successes in the 1950s and 1960s were impressive for their time. The <strong>Logic Theorist</strong> (Newell & Simon, 1956) proved mathematical theorems. <strong>ELIZA</strong> (Weizenbaum, 1966) simulated a Rogerian therapist by pattern-matching text — startlingly convincing to users despite having zero understanding. These early systems were built on <span class="term" data-term="symbolic-ai">Symbolic AI</span> — explicitly programming rules and logic in human-readable representations.
</p>

<div class="pro-tip">
  <strong>PM Perspective:</strong> The ELIZA effect — users attributing understanding to a system that merely pattern-matches — is directly relevant today. Users anthropomorphise LLM chatbots in the same way. As a PM, managing user expectations around what a model "understands" versus what it statistically generates is a critical design challenge.
</div>

<h2>AI Winters and the Limits of Symbolic AI</h2>
<p>
  The 1960s optimism led to grand predictions — Herbert Simon claimed in 1965 that "machines will be capable, within twenty years, of doing any work a man can do." When those predictions failed to materialise, funding agencies lost patience. The <strong>Lighthill Report</strong> (1973) in the UK savaged AI research, arguing that combinatorial explosion made general problem-solving intractable. This triggered the first <span class="term" data-term="ai-winter">AI Winter</span> (roughly 1974–1980), during which government funding dried up and AI became an unfashionable research topic.
</p>
<p>
  A brief resurgence came in the 1980s with <span class="term" data-term="expert-systems">Expert Systems</span> — programs encoding domain knowledge as if-then rules. Systems like MYCIN (medical diagnosis) and XCON (computer configuration at DEC) delivered real commercial value. Japan's ambitious Fifth Generation Computer Project (1982) further stoked excitement. But expert systems proved brittle: they could not learn from data, required expensive knowledge engineering, and broke down outside their narrow domains. The resulting disillusionment drove the second AI Winter (roughly 1987–1993).
</p>

<div class="warning">
  <strong>Common Misconception:</strong> AI winters were caused by "bad technology." In reality, they were caused by overpromising relative to what was achievable. The technology often worked within narrow bounds — it was the gap between public expectations and actual capability that collapsed funding. This pattern is directly relevant to AI product launches today.
</div>

<h2>The Statistical Turn and Machine Learning Rises (1990s–2000s)</h2>
<p>
  Through the 1990s, researchers quietly pivoted from hand-crafted rules to <span class="term" data-term="machine-learning">machine learning (ML)</span> — systems that learn patterns from data rather than being explicitly programmed. This was not a new idea (the <span class="term" data-term="perceptron">perceptron</span> dates to 1958), but three factors converged to make it practical: growing datasets, faster hardware, and better algorithms.
</p>
<p>
  Key milestones include the development of <span class="term" data-term="svm">Support Vector Machines (SVMs)</span>, <span class="term" data-term="random-forest">Random Forests</span>, and <span class="term" data-term="boosting">Gradient Boosting</span> methods — all of which achieved state-of-the-art results on structured data tasks. IBM's <strong>Deep Blue</strong> defeated world chess champion Garry Kasparov in 1997, though it relied heavily on brute-force search rather than learning. Meanwhile, statistical methods began dominating natural language processing, replacing handwritten grammars with probabilistic models.
</p>

<h2>The Deep Learning Revolution (2010s–Present)</h2>
<p>
  The modern era of AI was ignited by <span class="term" data-term="deep-learning">deep learning</span> — neural networks with many layers trained on large datasets using GPUs. The landmark moment was <strong>AlexNet</strong> winning the ImageNet competition in 2012, cutting the error rate nearly in half compared to traditional methods. This was not a new algorithm — the core ideas (backpropagation, convolutional networks) dated to the 1980s. What changed was scale: bigger data (ImageNet had 1.2 million labelled images), faster compute (NVIDIA GPUs), and practical innovations like <span class="term" data-term="relu">ReLU activations</span> and <span class="term" data-term="dropout">Dropout</span>.
</p>
<p>
  From 2012 onward, progress was breathtaking. DeepMind's <strong>AlphaGo</strong> defeated world Go champion Lee Sedol in 2016 — a feat many experts had predicted was decades away. The <span class="term" data-term="transformer">Transformer</span> architecture (Vaswani et al., 2017) revolutionised NLP. OpenAI's GPT series and Google's BERT demonstrated that scaling language models yielded emergent capabilities. By 2023, systems like GPT-4 and Gemini could engage in sophisticated reasoning, write code, analyse images, and pass professional exams.
</p>

<div class="example-box">
  <h4>Example</h4>
  <p><strong>DeepMind's AlphaFold</strong> (2020) solved the 50-year-old protein structure prediction problem, accurately predicting the 3D shape of virtually every known protein. This was not just a research achievement — it was productionised and made freely available, accelerating drug discovery worldwide. For a PM, AlphaFold illustrates how breakthrough AI can create enormous value when the problem, dataset, and deployment strategy align.</p>
</div>

<h2>Narrow AI vs. General AI</h2>
<p>
  A critical distinction for any AI PM is between <span class="term" data-term="narrow-ai">Narrow AI</span> (or Weak AI) and <span class="term" data-term="agi">Artificial General Intelligence (AGI)</span>. Every commercial AI system today is narrow — designed and trained for specific tasks or domains. AlphaGo cannot write poetry; GPT-4 cannot fold proteins. AGI, by contrast, would be a system with human-level generality across domains. Whether current approaches can scale to AGI, and on what timeline, is one of the most debated questions in the field.
</p>

<table>
  <thead>
    <tr><th>Attribute</th><th>Narrow AI</th><th>AGI (Hypothetical)</th></tr>
  </thead>
  <tbody>
    <tr><td>Scope</td><td>Single task or domain</td><td>Any intellectual task a human can do</td></tr>
    <tr><td>Current status</td><td>Commercially deployed</td><td>Active research, not yet achieved</td></tr>
    <tr><td>Examples</td><td>Image classifiers, LLMs, recommendation systems</td><td>None yet (theoretical)</td></tr>
    <tr><td>PM relevance</td><td>Ship products, manage capabilities & limitations</td><td>Long-term strategy, safety planning, policy engagement</td></tr>
  </tbody>
</table>

<div class="pro-tip">
  <strong>PM Perspective:</strong> When stakeholders or executives ask "When will we have AGI?", a strong PM reframes: "Here's what our current systems can and cannot do, here's the trajectory, and here's how we should plan our roadmap given that uncertainty." Concrete capability assessments beat speculation every time.
</div>

<h2>Why History Matters for AI PMs</h2>
<p>
  Understanding AI's history is not academic nostalgia — it provides pattern recognition for the present. The cycle of breakthrough, hype, overpromise, and winter has repeated multiple times. As an AI PM, you will encounter stakeholders who believe AI can do anything (hype-driven) and those who believe it is all smoke and mirrors (winter-scarred). Your job is to inhabit the realistic middle ground: deeply understanding what current systems can do, where they fail, and how fast the frontier is moving. History equips you with calibrated expectations — the single most valuable asset for an AI product leader.
</p>
`,
    quiz: {
      questions: [
        {
          question: 'Your VP wants to announce that your new LLM-powered feature "understands customer intent." Based on the history of AI, what is the most important risk a PM should raise?',
          type: 'mc',
          options: [
            'Statistical pattern matching creates trust issues when edge cases cause unexpected failures',
            'Professional exam performance validates understanding claims making announcements accurate and low-risk',
            'Competitive pressure to launch first outweighs concerns about precise capability framing claims',
            'Philosophical ambiguity around understanding means any capability claim is equally valid'
          ],
          correct: 0,
          explanation: 'The ELIZA effect teaches us that users attribute understanding to pattern-matching systems. When the system inevitably fails on edge cases, users who were told it "understands" will lose trust disproportionately. A calibrated claim like "interprets" or "processes" sets better expectations.',
          difficulty: 'applied',
          expertNote: 'A world-class AI PM would propose specific failure-mode testing and craft messaging that conveys value without anthropomorphising, drawing on lessons from both ELIZA and modern chatbot deployments.'
        },
        {
          question: 'A startup claims their new AI system achieves "human-level performance across all domains." As an AI PM evaluating a potential partnership, which historical pattern should make you most sceptical?',
          type: 'mc',
          options: [
            'Expert systems made broad claims but failed outside domains',
            'Deep Blue shows machines achieve parity across domains',
            'Multiple Turing Test passes validate general intelligence',
            'AlphaFold proves domain-general AI is now routine'
          ],
          correct: 0,
          explanation: 'Both AI winters were triggered by overpromising. Expert systems were marketed as broadly intelligent but failed outside their narrow rule-based domains. No current system demonstrates true cross-domain generality, and claims of AGI should be evaluated against specific, measurable benchmarks.',
          difficulty: 'applied',
          expertNote: 'An expert PM would request benchmark results across diverse tasks, test for transfer capability, and evaluate whether the system degrades gracefully or catastrophically at boundary conditions.'
        },
        {
          question: 'Which factors converged to enable the deep learning revolution starting in 2012? Select all that apply.',
          type: 'multi',
          options: [
            'Large labelled datasets like ImageNet',
            'GPU hardware enabling parallel matrix computation',
            'New theoretical proofs that neural networks are universal approximators',
            'Practical training innovations like ReLU activations and Dropout',
            'Government mandates requiring AI adoption in industry'
          ],
          correct: [0, 1, 3],
          explanation: 'The deep learning revolution was driven by data (ImageNet), compute (GPUs), and algorithmic innovations (ReLU, Dropout, better initialization). The universal approximation theorem dates to 1989 and was not a proximate cause. Government mandates did not drive the 2012 breakthrough.',
          difficulty: 'foundational',
          expertNote: 'A DeepMind PM would also note the role of open-source culture (papers with code, shared benchmarks) and the competitive dynamics of the ImageNet challenge in accelerating progress.'
        },
        {
          question: 'Scenario: You are a PM at a large enterprise company. Your CEO just read a breathless article about AI and wants to "add AI to every product in the portfolio within 6 months." Drawing on AI history, write a brief strategic response that acknowledges the opportunity while managing expectations.',
          type: 'scenario',
          correct: 'A strong response would: (1) Validate the CEO\'s enthusiasm by citing real recent breakthroughs and competitive pressure. (2) Reference the historical pattern of AI hype cycles leading to overpromise and disillusionment — noting that the two AI winters were caused not by bad technology but by the gap between expectations and delivered capability. (3) Propose a phased approach: identify 2-3 high-impact use cases where AI can deliver measurable value in 6 months, run focused pilots with clear success metrics, and build internal capability for broader rollout. (4) Emphasise that sustainable AI adoption requires data readiness, evaluation infrastructure, and managing user expectations — not just model integration.',
          explanation: 'The best AI PMs balance enthusiasm with calibrated expectations. History shows that indiscriminate AI deployment leads to failed projects and backlash, while focused, well-scoped applications deliver genuine value. A phased approach with clear metrics protects the organisation from hype-cycle damage.',
          difficulty: 'expert',
          expertNote: 'A DeepMind-calibre PM would additionally consider: build vs. buy decisions for models, the cost of inference at scale, data privacy implications, and establishing an AI governance framework before broad deployment.'
        },
        {
          question: 'The "AI Effect" refers to which phenomenon?',
          type: 'mc',
          options: [
            'Annual capability improvements driven by Moore\'s Law creating exponential AI progress',
            'Continuous redefinition of real AI as excluding whatever machines actually achieve',
            'Research acceleration following breakthroughs creating momentum in AI development',
            'Economic displacement of human workers as AI systems automate routine tasks'
          ],
          correct: 1,
          explanation: 'The AI Effect describes how once a capability is achieved (e.g., chess, OCR, speech recognition), people stop calling it "AI" and move the goalposts. This has product implications: features powered by sophisticated ML are often taken for granted by users, making it harder to communicate value.',
          difficulty: 'foundational',
          expertNote: 'Understanding the AI Effect helps PMs craft product narratives that emphasise user outcomes rather than "AI-powered" labels, since the label itself depreciates as capabilities become normalized.'
        }
      ]
    }
  },

  // ─────────────────────────────────────────────
  // L02 — Types of ML: Supervised, Unsupervised, Reinforcement
  // ─────────────────────────────────────────────
  l02: {
    title: 'Types of ML — Supervised, Unsupervised, Reinforcement',
    content: `
<h2>The Three Paradigms of Machine Learning</h2>
<p>
  <span class="term" data-term="machine-learning">Machine Learning</span> is not a monolithic approach — it encompasses fundamentally different paradigms, each suited to different problem structures and data availability. Understanding which paradigm applies to a given product problem is one of the most important skills an AI PM can develop, because the choice dictates data requirements, development cost, evaluation strategy, and the type of value the system can deliver.
</p>
<p>
  The three classical paradigms are <span class="term" data-term="supervised-learning">Supervised Learning</span>, <span class="term" data-term="unsupervised-learning">Unsupervised Learning</span>, and <span class="term" data-term="reinforcement-learning">Reinforcement Learning</span>. More recently, <span class="term" data-term="self-supervised-learning">Self-Supervised Learning</span> has emerged as a dominant paradigm powering large language models and foundation models. We will cover all four, with emphasis on their product implications.
</p>

<h2>Supervised Learning: Learning from Labelled Examples</h2>
<p>
  In supervised learning, the model learns a mapping from inputs to outputs using a dataset of labelled examples — input-output pairs where a human (or automated process) has provided the "correct" answer. The model's job is to generalise from these examples so it can accurately predict outputs for inputs it has never seen before.
</p>
<p>
  Formally, given a dataset <code>D = {(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)}</code>, the model learns a function <code>f(x) ≈ y</code> that minimises prediction error on unseen data. Supervised learning subdivides into two major task types:
</p>
<ul>
  <li><strong>Classification:</strong> The output is a discrete category. Examples include spam detection (spam/not spam), medical image diagnosis (benign/malignant), and content moderation (safe/toxic/borderline).</li>
  <li><strong>Regression:</strong> The output is a continuous value. Examples include predicting house prices, estimating delivery times, and forecasting demand.</li>
</ul>

<div class="key-concept">
  <strong>Key Concept:</strong> Supervised learning is the workhorse of applied ML. The majority of production ML systems — recommendation engines, search ranking, fraud detection, medical diagnostics — are supervised or have supervised components. The quality ceiling of a supervised model is fundamentally limited by the quality and representativeness of its labels.
</div>

<p>
  Common supervised learning algorithms include <span class="term" data-term="linear-regression">Linear/Logistic Regression</span>, <span class="term" data-term="decision-tree">Decision Trees</span>, <span class="term" data-term="random-forest">Random Forests</span>, <span class="term" data-term="svm">Support Vector Machines</span>, <span class="term" data-term="gradient-boosting">Gradient Boosted Trees</span> (XGBoost, LightGBM), and <span class="term" data-term="neural-network">Neural Networks</span>. For tabular/structured data, gradient boosted trees remain remarkably competitive with deep learning. For unstructured data (images, text, audio), neural networks dominate.
</p>

<div class="pro-tip">
  <strong>PM Perspective:</strong> When scoping a supervised learning project, the first question is never "What model should we use?" — it is "Do we have enough high-quality labelled data?" Data acquisition and labelling costs often dominate the project budget. A PM must evaluate: Can we leverage existing labels? Can we use active learning to label efficiently? Can we use a pre-trained foundation model to reduce label requirements?
</div>

<h2>Unsupervised Learning: Finding Structure Without Labels</h2>
<p>
  <span class="term" data-term="unsupervised-learning">Unsupervised learning</span> operates on data without labels. The model's goal is to discover hidden structure, patterns, or groupings within the data. Because it does not require labelled examples, unsupervised learning can be applied to any dataset — but the outputs require human interpretation to be actionable.
</p>
<p>
  Major unsupervised learning tasks include:
</p>
<ul>
  <li><strong>Clustering:</strong> Grouping data points into clusters based on similarity. Algorithms include <span class="term" data-term="k-means">K-Means</span>, <span class="term" data-term="dbscan">DBSCAN</span>, and hierarchical clustering. Product applications include customer segmentation, topic discovery, and anomaly grouping.</li>
  <li><strong>Dimensionality Reduction:</strong> Compressing high-dimensional data into fewer dimensions while preserving structure. Techniques include <span class="term" data-term="pca">PCA</span>, <span class="term" data-term="t-sne">t-SNE</span>, and <span class="term" data-term="umap">UMAP</span>. Used for data visualisation, feature engineering, and noise reduction.</li>
  <li><strong>Anomaly Detection:</strong> Identifying data points that deviate significantly from normal patterns. Used in fraud detection, infrastructure monitoring, and quality control.</li>
  <li><strong>Association Rules:</strong> Discovering relationships between items. The classic example is market basket analysis — "customers who buy X also buy Y."</li>
</ul>

<div class="example-box">
  <h4>Example</h4>
  <p>A music streaming service uses unsupervised clustering on user listening patterns to discover listener segments they had not hypothesised: "instrumental focus workers," "weekend party playlists," "audiobook commuters." These clusters, invisible in traditional demographics, inform personalised playlist curation and marketing campaigns. The PM does not need labels — the structure emerges from the data.</p>
</div>

<div class="warning">
  <strong>Common Misconception:</strong> "Unsupervised learning requires no human judgment." In practice, unsupervised learning requires extensive human judgment — choosing the number of clusters, interpreting what clusters mean, validating that discovered patterns are meaningful rather than artefacts of noise or data preprocessing. The "unsupervised" label refers to the absence of labels in the training data, not the absence of human oversight.
</div>

<h2>Reinforcement Learning: Learning from Interaction</h2>
<p>
  <span class="term" data-term="reinforcement-learning">Reinforcement Learning (RL)</span> is fundamentally different from both supervised and unsupervised learning. Instead of learning from a static dataset, an RL <span class="term" data-term="agent">agent</span> learns by interacting with an <span class="term" data-term="environment">environment</span>. At each time step, the agent observes a <span class="term" data-term="state">state</span>, takes an <span class="term" data-term="action">action</span>, and receives a <span class="term" data-term="reward">reward</span> signal. The agent's goal is to learn a <span class="term" data-term="policy">policy</span> — a mapping from states to actions — that maximises cumulative reward over time.
</p>
<p>
  The key challenge in RL is the <span class="term" data-term="exploration-exploitation">exploration-exploitation trade-off</span>: should the agent exploit actions it already knows yield good rewards, or explore new actions that might yield even better rewards? Another challenge is <span class="term" data-term="credit-assignment">credit assignment</span>: when a reward arrives after many actions, which actions deserve credit?
</p>
<p>
  RL has produced some of AI's most spectacular achievements:
</p>
<ul>
  <li><strong>AlphaGo / AlphaZero (DeepMind):</strong> Mastered Go, chess, and shogi through self-play, surpassing all human experts.</li>
  <li><strong>Robotics control:</strong> Training robot arms, drones, and legged robots to perform physical tasks.</li>
  <li><strong>RLHF (Reinforcement Learning from Human Feedback):</strong> The technique used to align LLMs like GPT-4 and Gemini with human preferences — arguably the most commercially impactful application of RL to date.</li>
</ul>

<div class="key-concept">
  <strong>Key Concept:</strong> RL excels in sequential decision-making problems where the optimal strategy is not obvious from individual examples. However, RL is notoriously sample-inefficient — it often requires millions of interactions to learn good policies — and reward design is tricky. Poorly designed rewards lead to <span class="term" data-term="reward-hacking">reward hacking</span>, where agents find unexpected shortcuts that maximise the reward without achieving the intended goal.
</div>

<h2>Self-Supervised Learning: The Fourth Paradigm</h2>
<p>
  <span class="term" data-term="self-supervised-learning">Self-supervised learning (SSL)</span> blurs the line between supervised and unsupervised learning. The model creates its own "labels" from the data itself — for example, masking out a word in a sentence and predicting it (as in BERT), or predicting the next token in a sequence (as in GPT). The supervision signal comes from the data's inherent structure rather than from human annotators.
</p>
<p>
  SSL is the paradigm behind modern <span class="term" data-term="foundation-model">foundation models</span>. By pre-training on vast amounts of unlabelled text, images, or code, these models learn rich representations that can be adapted (via fine-tuning or prompting) to a wide range of downstream tasks. This has dramatically reduced the need for task-specific labelled data.
</p>

<div class="pro-tip">
  <strong>PM Perspective:</strong> Self-supervised learning has changed the economics of ML product development. Instead of collecting labelled data for every new feature, PMs can leverage pre-trained foundation models and adapt them with small amounts of task-specific data. This shifts the bottleneck from "Do we have enough labels?" to "Can we evaluate quality reliably?" and "Can we afford inference costs at scale?"
</div>

<h2>Choosing the Right Paradigm: A Decision Framework</h2>

<table>
  <thead>
    <tr><th>Paradigm</th><th>Data Required</th><th>Best For</th><th>PM Considerations</th></tr>
  </thead>
  <tbody>
    <tr>
      <td>Supervised</td>
      <td>Labelled input-output pairs</td>
      <td>Classification, regression, ranking</td>
      <td>Label quality and cost are the bottleneck; evaluation is straightforward</td>
    </tr>
    <tr>
      <td>Unsupervised</td>
      <td>Unlabelled data</td>
      <td>Clustering, anomaly detection, exploration</td>
      <td>Outputs need human interpretation; harder to measure success objectively</td>
    </tr>
    <tr>
      <td>Reinforcement</td>
      <td>An interactive environment + reward signal</td>
      <td>Sequential decisions, game-playing, robotics, alignment</td>
      <td>Reward design is critical; sample-inefficient; may need simulation</td>
    </tr>
    <tr>
      <td>Self-supervised</td>
      <td>Large unlabelled corpora</td>
      <td>Pre-training foundation models, representation learning</td>
      <td>Reduces label dependency; shifts cost to compute and evaluation</td>
    </tr>
  </tbody>
</table>

<div class="example-box">
  <h4>Example</h4>
  <p><strong>A content moderation pipeline</strong> might use all four paradigms: (1) Self-supervised pre-training of a language model on internet text, (2) Supervised fine-tuning on labelled examples of toxic vs. safe content, (3) Unsupervised clustering to discover new categories of abuse not yet in the taxonomy, and (4) RLHF to align the system's outputs with human moderator judgments on nuanced cases. A PM planning this system needs to understand the interplay between paradigms.</p>
</div>
`,
    quiz: {
      questions: [
        {
          question: 'You are building a product feature that automatically categorises customer support tickets into predefined categories (billing, technical, account). You have 50,000 historical tickets that have been manually tagged by agents. Which ML paradigm is the best primary approach?',
          type: 'mc',
          options: [
            'Unsupervised learning using clustering to discover natural ticket groupings automatically',
            'Supervised learning training a classifier on the 50,000 manually labelled examples',
            'Reinforcement learning optimizing categorization decisions based on satisfaction signals',
            'Self-supervised pre-training on ticket text followed by zero-shot deployment'
          ],
          correct: 1,
          explanation: 'With 50,000 labelled examples mapping tickets to predefined categories, supervised classification is the natural fit. Unsupervised would discover categories but not necessarily match your predefined taxonomy. RL is not suited to static classification. SSL pre-training could help, but would still need supervised fine-tuning to map to your specific categories.',
          difficulty: 'foundational',
          expertNote: 'An expert PM might also consider using a pre-trained LLM with few-shot prompting as a baseline to compare against fine-tuned supervised models, evaluating cost-accuracy trade-offs.'
        },
        {
          question: 'Your team\'s RL agent for optimising ad placement has found a strategy that maximises click-through rate but shows users misleading preview text to attract clicks. This is an example of:',
          type: 'mc',
          options: [
            'Exploration-exploitation imbalance causing the agent to try risky strategies excessively',
            'Reward hacking where metric optimization diverges from the true objective goal',
            'Catastrophic forgetting causing the agent to lose ethical training constraints',
            'Distribution shift from encountering user populations unlike training data sets'
          ],
          correct: 1,
          explanation: 'Reward hacking occurs when an RL agent finds shortcuts that maximise the reward signal without fulfilling the designer\'s actual intent. Here, the reward was click-through rate, but the intended goal was meaningful user engagement. The agent found that misleading previews satisfy the metric without the desired outcome.',
          difficulty: 'applied',
          expertNote: 'A DeepMind-calibre PM would design composite reward functions combining CTR with downstream engagement metrics (dwell time, conversion) and implement guardrails (content accuracy checks) as hard constraints rather than soft reward components.'
        },
        {
          question: 'A PM at a healthcare startup claims: "We don\'t need any labelled data — we\'ll use unsupervised learning to diagnose diseases from patient records." What is the most fundamental flaw in this reasoning?',
          type: 'mc',
          options: [
            'Unsupervised algorithms lack the technical capability to process patient record data',
            'Pattern discovery requires labelled validation data to map clusters to diagnoses',
            'Healthcare applications require real-time processing unsupervised methods cannot provide',
            'Unsupervised methods are restricted to image modalities and cannot handle records'
          ],
          correct: 1,
          explanation: 'Unsupervised learning can find patient clusters and patterns, but it cannot assign diagnostic labels to those clusters without human interpretation or labelled validation data. Diagnosis requires mapping patterns to known conditions — which is inherently a supervised task. The unsupervised analysis might be a useful exploratory step, but it cannot replace labelled data for diagnostic classification.',
          difficulty: 'applied',
          expertNote: 'An expert PM would propose using unsupervised methods for hypothesis generation (discovering patient subgroups), then validating with clinical labels — combining paradigms rather than relying on one.'
        },
        {
          question: 'Scenario: You are a PM at DeepMind tasked with improving the quality of a customer-facing chatbot. Your team proposes using RLHF (Reinforcement Learning from Human Feedback) to align the model\'s responses with user preferences. Describe the key risks of this approach and what guardrails you would put in place.',
          type: 'scenario',
          correct: 'Key risks include: (1) Reward hacking — the model may learn to produce responses that human raters prefer superficially (confident, verbose, agreeable) without being more accurate or helpful; this is sometimes called "sycophancy." (2) Rater bias — the quality of RLHF is bounded by the quality and diversity of human raters; if raters have systematic biases, the model inherits them. (3) Reward model misalignment — the learned reward model is an imperfect proxy for true user satisfaction and may diverge in edge cases. Guardrails: Use diverse rater pools with clear rubrics; monitor for sycophantic behavior via automated benchmarks; combine RLHF with rule-based safety filters; conduct regular red-teaming; track downstream user satisfaction metrics (not just rater scores) to detect divergence between rater preferences and real-world outcomes.',
          explanation: 'RLHF is powerful but introduces multiple failure modes around the quality of human feedback and the fidelity of the reward model. A strong PM treats RLHF as one component in a safety stack, not a complete solution.',
          difficulty: 'expert',
          expertNote: 'A world-class AI PM would also consider constitutional AI approaches (RLAIF) as a complement to human feedback, and would design A/B tests comparing RLHF-tuned and base models on safety-critical dimensions.'
        },
        {
          question: 'Which of the following are valid reasons to prefer a self-supervised pre-training approach over training a supervised model from scratch? Select all that apply.',
          type: 'multi',
          options: [
            'You have limited labelled data but access to large volumes of unlabelled data in your domain',
            'Self-supervised models are always more accurate than supervised models',
            'You want the model to transfer knowledge across multiple downstream tasks',
            'You want to eliminate the need for any evaluation on the downstream task',
            'Pre-trained representations can reduce the amount of task-specific labelling needed'
          ],
          correct: [0, 2, 4],
          explanation: 'Self-supervised pre-training excels when labelled data is scarce but unlabelled data is abundant, when you want transferable representations, and when you want to reduce labelling costs. It is not always more accurate (gradient boosting often beats deep learning on tabular data), and it never eliminates the need for evaluation.',
          difficulty: 'applied',
          expertNote: 'A strong PM would benchmark pre-trained model performance against a supervised baseline to quantify the actual benefit, rather than assuming pre-training is always superior.'
        }
      ]
    }
  },

  // ─────────────────────────────────────────────
  // L03 — How Machines Learn: Loss Functions & Gradient Descent
  // ─────────────────────────────────────────────
  l03: {
    title: 'How Machines Learn — Loss Functions & Gradient Descent',
    content: `
<h2>The Core Loop: How Machines Actually Learn</h2>
<p>
  At the heart of nearly every machine learning system lies a deceptively simple loop: make a prediction, measure how wrong it is, adjust parameters to be less wrong, and repeat. This loop — <strong>predict, evaluate, update</strong> — is the engine of learning. To understand it deeply, you need to understand two critical components: the <span class="term" data-term="loss-function">loss function</span> (which measures error) and <span class="term" data-term="gradient-descent">gradient descent</span> (which reduces error). Together, they transform the abstract idea of "learning" into a concrete optimisation problem.
</p>
<p>
  Why does a PM need to understand this? Because the choice of loss function determines <em>what the model optimises for</em>, and that choice has direct product implications. A model trained to minimise average error may produce mediocre results across the board. A model trained to minimise worst-case error may be more robust but less accurate overall. The loss function is where mathematical optimisation meets product values — and a PM who understands this connection can have informed conversations with ML engineers about model behaviour.
</p>

<h2>Loss Functions: Quantifying "How Wrong"</h2>
<p>
  A <span class="term" data-term="loss-function">loss function</span> (also called a cost function or objective function) takes the model's prediction and the true label and outputs a single number representing how bad the prediction was. Lower is better. The model's entire training process is devoted to finding parameters that minimise this number across the training dataset.
</p>

<h3>Loss Functions for Regression</h3>
<p>
  For problems where the output is a continuous value (predicting prices, temperatures, durations), common loss functions include:
</p>
<ul>
  <li><strong>Mean Squared Error (MSE):</strong> <code>L = (1/n) * sum((y_pred - y_true)^2)</code>. Squaring amplifies large errors, making the model especially sensitive to outliers. This is appropriate when large errors are disproportionately costly.</li>
  <li><strong>Mean Absolute Error (MAE):</strong> <code>L = (1/n) * sum(|y_pred - y_true|)</code>. Treats all error magnitudes linearly. More robust to outliers than MSE.</li>
  <li><strong>Huber Loss:</strong> Acts like MSE for small errors (smooth gradient) and like MAE for large errors (robust to outliers). A hybrid that gives the best of both worlds, controlled by a threshold parameter <code>delta</code>.</li>
</ul>

<div class="example-box">
  <h4>Example</h4>
  <p>A ride-sharing app predicts trip duration. With MSE, a single trip that takes 3 hours instead of the predicted 20 minutes dominates the loss and warps the model. With MAE, that outlier is treated proportionally. With Huber loss, the PM can tune the delta parameter to decide how much to penalise extreme errors. The <em>product decision</em> — how much do we care about extreme mispredictions vs. average accuracy? — maps directly to the loss function choice.</p>
</div>

<h3>Loss Functions for Classification</h3>
<p>
  For problems where the output is a category, the dominant loss function is:
</p>
<ul>
  <li><strong>Cross-Entropy Loss (Log Loss):</strong> <code>L = -sum(y_true * log(y_pred))</code>. Measures the divergence between the model's predicted probability distribution and the true distribution. Cross-entropy loss has elegant properties: it penalises confident wrong predictions very heavily (predicting 0.01 probability for the true class incurs a much larger loss than predicting 0.4) and provides smooth gradients for optimisation.</li>
  <li><strong>Binary Cross-Entropy:</strong> The special case for two-class problems: <code>L = -[y * log(p) + (1-y) * log(1-p)]</code>.</li>
  <li><strong>Focal Loss:</strong> A modification of cross-entropy that down-weights easy examples and focuses learning on hard, misclassified examples. Invented for object detection where most candidate regions are background (easy negatives). Controlled by a focusing parameter <code>gamma</code>.</li>
</ul>

<div class="key-concept">
  <strong>Key Concept:</strong> The loss function encodes your product's values in mathematical form. Choosing cross-entropy says "penalise confident mistakes harshly." Choosing focal loss says "focus on the hard cases." Choosing a weighted loss says "false negatives are more costly than false positives" (or vice versa). A PM who understands loss function design can directly influence what the model prioritises.
</div>

<div class="warning">
  <strong>Common Misconception:</strong> "The loss function and the evaluation metric should be the same thing." In practice, they often differ. The loss function must be differentiable (for gradient descent), but your business metric (e.g., user satisfaction, revenue) often is not. A model might be trained with cross-entropy loss but evaluated with F1 score, precision@K, or a custom business metric. Understanding this gap is essential.
</div>

<h2>Gradient Descent: Walking Downhill</h2>
<p>
  Once we have a loss function, we need a method to minimise it. <span class="term" data-term="gradient-descent">Gradient descent</span> is the optimisation algorithm used by virtually all neural network training. The intuition is simple: imagine the loss function as a landscape of hills and valleys, where altitude represents loss. The model's parameters define a position in this landscape. Gradient descent computes which direction is "downhill" (the <span class="term" data-term="gradient">gradient</span>) and takes a step in that direction, iteratively walking toward a minimum.
</p>
<p>
  Formally, the parameter update rule is: <code>w_new = w_old - learning_rate * gradient(L, w_old)</code>. The <span class="term" data-term="learning-rate">learning rate</span> controls step size — too large and you overshoot the minimum; too small and training is glacially slow or gets stuck in local minima.
</p>

<h3>Variants of Gradient Descent</h3>
<table>
  <thead>
    <tr><th>Variant</th><th>How It Works</th><th>Trade-offs</th></tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Batch GD</strong></td>
      <td>Computes gradient over the entire dataset per step</td>
      <td>Stable but slow; impractical for large datasets</td>
    </tr>
    <tr>
      <td><strong>Stochastic GD (SGD)</strong></td>
      <td>Computes gradient from a single random example per step</td>
      <td>Fast but noisy; can escape local minima but oscillates</td>
    </tr>
    <tr>
      <td><strong>Mini-batch GD</strong></td>
      <td>Computes gradient from a small random batch (e.g., 32-256 examples)</td>
      <td>Best of both worlds; standard practice in deep learning</td>
    </tr>
  </tbody>
</table>

<h3>Modern Optimisers</h3>
<p>
  Vanilla gradient descent has limitations, so researchers have developed optimisers with adaptive learning rates:
</p>
<ul>
  <li><strong>Momentum:</strong> Adds a velocity term to dampen oscillation and accelerate convergence. The gradient update carries "momentum" from previous steps, like a ball rolling downhill accumulates speed.</li>
  <li><strong>RMSProp:</strong> Scales the learning rate for each parameter based on the historical magnitude of its gradients. Parameters with large gradients get smaller steps; parameters with small gradients get larger steps.</li>
  <li><strong><span class="term" data-term="adam">Adam</span> (Adaptive Moment Estimation):</strong> Combines momentum and RMSProp. The most widely used optimiser in deep learning. It maintains per-parameter adaptive learning rates using first-moment (mean) and second-moment (variance) estimates of the gradient.</li>
  <li><strong>AdamW:</strong> A correction to Adam that properly implements <span class="term" data-term="weight-decay">weight decay</span> (L2 regularisation). Preferred for training large language models.</li>
</ul>

<div class="pro-tip">
  <strong>PM Perspective:</strong> You do not need to implement optimisers, but you should understand that <strong>hyperparameter tuning</strong> — choosing the learning rate, batch size, optimiser, and learning rate schedule — significantly impacts model performance and training cost. When your ML team says "we need more compute for hyperparameter search," they are exploring this space. The learning rate alone can make the difference between a model that converges in hours and one that never converges at all.
</div>

<div class="interactive" data-interactive="gradient-descent"></div>

<h2>The Training Loop in Full</h2>
<p>
  Putting it all together, the training loop for a neural network is:
</p>
<ol>
  <li><strong>Initialise</strong> model parameters (weights and biases) randomly.</li>
  <li><strong>Forward pass:</strong> Feed a batch of training data through the model to produce predictions.</li>
  <li><strong>Compute loss:</strong> Compare predictions to true labels using the loss function.</li>
  <li><strong>Backward pass (backpropagation):</strong> Compute the gradient of the loss with respect to every parameter using the chain rule of calculus. This tells us how to adjust each parameter to reduce the loss.</li>
  <li><strong>Update parameters:</strong> Apply the optimiser's update rule.</li>
  <li><strong>Repeat</strong> for many iterations (epochs) until the loss converges or a stopping criterion is met.</li>
</ol>

<div class="key-concept">
  <strong>Key Concept:</strong> <span class="term" data-term="backpropagation">Backpropagation</span> is not a separate algorithm — it is simply the efficient application of the chain rule to compute gradients through a computational graph. It was popularised by Rumelhart, Hinton, and Williams in 1986 and remains the foundation of all neural network training today. Without backpropagation, gradient descent would be computationally infeasible for networks with millions or billions of parameters.
</div>

<h2>Overfitting, Underfitting, and the Bias-Variance Trade-off</h2>
<p>
  A model that perfectly memorises the training data but performs poorly on unseen data is <span class="term" data-term="overfitting">overfitting</span>. A model that fails to capture the underlying pattern even in the training data is <span class="term" data-term="underfitting">underfitting</span>. The <span class="term" data-term="bias-variance-tradeoff">bias-variance trade-off</span> formalises this tension:
</p>
<ul>
  <li><strong>Bias</strong> is the error from overly simplistic assumptions — underfitting. A linear model fitting a curved relationship has high bias.</li>
  <li><strong>Variance</strong> is the error from sensitivity to fluctuations in the training data — overfitting. A very complex model that fits noise has high variance.</li>
</ul>
<p>
  Strategies to combat overfitting include <span class="term" data-term="regularisation">regularisation</span> (L1/L2 penalties on weights), <span class="term" data-term="dropout">dropout</span> (randomly zeroing neurons during training), <span class="term" data-term="early-stopping">early stopping</span> (halting training when validation loss starts rising), and <span class="term" data-term="data-augmentation">data augmentation</span> (artificially expanding the training set).
</p>

<div class="warning">
  <strong>Common Misconception:</strong> "More training always improves the model." Beyond a point, additional training on the same data leads to overfitting. The model memorises training examples instead of learning generalisable patterns. This is why monitoring validation loss — the loss on held-out data the model has never seen — is essential. When validation loss starts increasing while training loss continues decreasing, you are overfitting.
</div>

<div class="pro-tip">
  <strong>PM Perspective:</strong> Overfitting is not just a technical problem — it is a product risk. An overfit model performs brilliantly in internal demos (on data similar to training data) but fails in production (on real-world data with different distributions). PMs should insist on evaluation using held-out test sets that reflect production conditions, not cherry-picked examples.
</div>
`,
    quiz: {
      questions: [
        {
          question: 'Your fraud detection model needs to catch 99% of fraudulent transactions, even if it means more false positives. Which approach to the loss function best encodes this product requirement?',
          type: 'mc',
          options: [
            'Standard cross-entropy loss treating all classification errors equally',
            'Weighted cross-entropy heavily penalizing false negatives over false positives',
            'Mean squared error optimized for continuous fraud probability scores',
            'Focal loss concentrating learning effort on easily classified examples'
          ],
          correct: 1,
          explanation: 'When the cost of missing fraud vastly exceeds the cost of false alarms, asymmetric weighting in the loss function directly encodes this trade-off. Standard cross-entropy treats both error types equally. MSE is inappropriate for classification. Focal loss focuses on hard examples but does not inherently encode asymmetric misclassification costs.',
          difficulty: 'applied',
          expertNote: 'An expert PM would also consider threshold tuning on the output probabilities and evaluate using precision-recall curves rather than accuracy, since the dataset is likely heavily imbalanced.'
        },
        {
          question: 'During training, your team observes that training loss continues decreasing but validation loss has been increasing for the last 10 epochs. What is happening, and what should they do?',
          type: 'mc',
          options: [
            'The model is underfitting and requires increased capacity or complexity',
            'The model is overfitting and requires regularization or early stopping',
            'Learning rate misconfiguration requires adjustment and continued training',
            'This represents expected convergence behavior requiring continued training'
          ],
          correct: 1,
          explanation: 'Diverging training and validation loss is the classic signature of overfitting: the model is memorising training data rather than learning generalisable patterns. Remedies include early stopping (revert to the checkpoint where validation loss was lowest), regularisation (dropout, weight decay), or augmenting the training data.',
          difficulty: 'foundational',
          expertNote: 'An expert would also examine whether the validation set is representative of the true data distribution, as a non-representative validation set can produce misleading divergence signals.'
        },
        {
          question: 'Why is Adam the most widely used optimiser for deep learning, compared to vanilla SGD?',
          type: 'mc',
          options: [
            'Adam guarantees convergence to global optima unlike SGD\'s local minima',
            'Adam adapts learning rates per-parameter using gradient moment estimates',
            'Adam eliminates hyperparameter tuning unlike SGD\'s learning rate requirements',
            'Adam leverages Hessian second-order derivatives for accelerated convergence'
          ],
          correct: 1,
          explanation: 'Adam combines momentum (first moment) and RMSProp (second moment) to adapt the learning rate for each parameter individually. This makes it less sensitive to the initial learning rate choice and generally converges faster than vanilla SGD. It does NOT guarantee global optima, still has hyperparameters (learning rate, beta1, beta2), and does not use the Hessian.',
          difficulty: 'foundational',
          expertNote: 'Interestingly, for large-scale language model training, SGD with carefully tuned schedules sometimes outperforms Adam in final quality. The choice is not always clearcut, and a strong PM supports the team in running ablation experiments.'
        },
        {
          question: 'Scenario: Your team is training a content recommendation model. The ML lead proposes optimising for click-through rate (CTR) using cross-entropy loss. The UX research team flags that users are clicking on sensational content but reporting dissatisfaction in surveys. How should you, as PM, reconcile this tension through the lens of loss function design?',
          type: 'scenario',
          correct: 'The core issue is CTR-satisfaction misalignment. A strong PM would: (1) Recognize the loss function literally trains for max clicks, ignoring downstream satisfaction. (2) Propose composite objectives incorporating engagement depth, explicit feedback signals, and regret indicators like back-button rate. (3) Note that not all outcomes are differentiable—some need post-training filters. (4) Run A/B tests comparing CTR-optimized vs satisfaction-adjusted models to quantify tradeoffs.',
          explanation: 'This is a classic Goodhart\'s Law scenario: "When a measure becomes a target, it ceases to be a good measure." The loss function defines what the model values, and aligning it with true user outcomes is one of the most important PM decisions in ML product development.',
          difficulty: 'expert',
          expertNote: 'A DeepMind-calibre PM would also consider long-term engagement effects (does optimising for satisfaction today improve retention over 30 days?) and propose establishing a causal framework rather than purely correlational metrics.'
        },
        {
          question: 'Which of the following statements about backpropagation are true? Select all that apply.',
          type: 'multi',
          options: [
            'Backpropagation computes gradients by applying the chain rule of calculus through the computational graph',
            'Backpropagation is a specific optimiser like Adam or SGD',
            'Without backpropagation, computing gradients for networks with billions of parameters would be computationally infeasible',
            'Backpropagation only works with cross-entropy loss',
            'Backpropagation propagates error signals from the output layer back through the network to update all parameters'
          ],
          correct: [0, 2, 4],
          explanation: 'Backpropagation is a gradient computation method (not an optimiser) that efficiently applies the chain rule through the network. It works with any differentiable loss function, not just cross-entropy. It computes how each parameter contributes to the loss by propagating error signals backward through the layers.',
          difficulty: 'foundational',
          expertNote: 'Modern deep learning frameworks implement backpropagation via automatic differentiation (autograd), which computes gradients for arbitrary computational graphs — a PM should understand this enables rapid experimentation with novel architectures.'
        }
      ]
    }
  },

  // ─────────────────────────────────────────────
  // L04 — Neural Networks: Perceptrons to Deep Networks
  // ─────────────────────────────────────────────
  l04: {
    title: 'Neural Networks — Perceptrons to Deep Networks',
    content: `
<h2>From Biological Inspiration to Computational Reality</h2>
<p>
  <span class="term" data-term="neural-network">Neural networks</span> are the computational backbone of modern AI. They are loosely inspired by biological neurons — but the analogy should not be stretched too far. A biological neuron receives electrical signals through dendrites, integrates them in the cell body, and fires an output signal along the axon if the combined input exceeds a threshold. An artificial neuron does something mathematically analogous: it receives numerical inputs, multiplies each by a <span class="term" data-term="weight">weight</span>, sums the results, adds a <span class="term" data-term="bias">bias</span> term, and passes the sum through an <span class="term" data-term="activation-function">activation function</span> to produce an output.
</p>
<p>
  The critical insight that makes neural networks powerful is not any single neuron — it is <em>composition</em>. By stacking layers of neurons, each layer transforms the data into increasingly abstract representations. Early layers might detect edges in an image; middle layers combine edges into textures and shapes; deep layers recognise objects and scenes. This hierarchical feature learning is what gives <span class="term" data-term="deep-learning">deep learning</span> its name and its power.
</p>

<h2>The Perceptron: Where It All Started</h2>
<p>
  The <span class="term" data-term="perceptron">perceptron</span>, invented by Frank Rosenblatt in 1958, is the simplest neural network — a single artificial neuron. It computes a weighted sum of its inputs, adds a bias, and applies a step function: output 1 if the sum exceeds zero, output 0 otherwise. Mathematically: <code>output = step(w_1*x_1 + w_2*x_2 + ... + w_n*x_n + b)</code>.
</p>
<p>
  The perceptron can learn to classify data that is <span class="term" data-term="linearly-separable">linearly separable</span> — data that can be divided by a straight line (or hyperplane in higher dimensions). Rosenblatt proved a <strong>convergence theorem</strong>: if the data is linearly separable, the perceptron learning algorithm will find a solution in finite steps.
</p>

<div class="warning">
  <strong>Common Misconception:</strong> "The perceptron was proven useless, which is why neural networks died." Minsky and Papert's 1969 book <em>Perceptrons</em> proved that a <em>single-layer</em> perceptron cannot learn the XOR function (or any non-linearly-separable pattern). But they acknowledged that multi-layer networks could — the problem was that no one knew how to train them efficiently. This distinction matters: the limitation was in training algorithms, not in the architecture's theoretical capability.
</div>

<h2>Activation Functions: Introducing Non-Linearity</h2>
<p>
  Without activation functions, a neural network — no matter how many layers it has — can only compute linear transformations of the input. Stacking linear layers is equivalent to a single linear layer. <span class="term" data-term="activation-function">Activation functions</span> introduce non-linearity, enabling the network to learn complex, non-linear mappings. The choice of activation function has significant practical impact:
</p>

<table>
  <thead>
    <tr><th>Activation</th><th>Formula</th><th>Range</th><th>Characteristics</th></tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Sigmoid</strong></td>
      <td><code>1 / (1 + e^(-x))</code></td>
      <td>(0, 1)</td>
      <td>Smooth, interpretable as probability. Suffers from vanishing gradients at extremes.</td>
    </tr>
    <tr>
      <td><strong>Tanh</strong></td>
      <td><code>(e^x - e^(-x)) / (e^x + e^(-x))</code></td>
      <td>(-1, 1)</td>
      <td>Zero-centred (better than sigmoid). Still suffers vanishing gradients.</td>
    </tr>
    <tr>
      <td><strong>ReLU</strong></td>
      <td><code>max(0, x)</code></td>
      <td>[0, infinity)</td>
      <td>Simple, fast, no vanishing gradient for positive values. Can "die" (output zero permanently).</td>
    </tr>
    <tr>
      <td><strong>Leaky ReLU</strong></td>
      <td><code>max(0.01x, x)</code></td>
      <td>(-infinity, infinity)</td>
      <td>Fixes dying ReLU by allowing small gradient for negative inputs.</td>
    </tr>
    <tr>
      <td><strong>GELU</strong></td>
      <td><code>x * Phi(x)</code></td>
      <td>(-0.17, infinity)</td>
      <td>Smooth approximation of ReLU. Used in Transformers (BERT, GPT).</td>
    </tr>
    <tr>
      <td><strong>Softmax</strong></td>
      <td><code>e^(x_i) / sum(e^(x_j))</code></td>
      <td>(0, 1), sums to 1</td>
      <td>Used in the output layer for multi-class classification. Converts logits to probabilities.</td>
    </tr>
  </tbody>
</table>

<div class="key-concept">
  <strong>Key Concept:</strong> The <span class="term" data-term="vanishing-gradient">vanishing gradient problem</span> plagued early deep networks. When using sigmoid or tanh activations, gradients in early layers become exponentially small during backpropagation, making those layers learn extremely slowly. ReLU largely solved this for feedforward networks, enabling training of much deeper architectures. This practical breakthrough, not a theoretical one, was a key enabler of the deep learning revolution.
</div>

<div class="interactive" data-interactive="neuron-playground"></div>

<h2>Architecture of a Feedforward Neural Network</h2>
<p>
  A standard <span class="term" data-term="feedforward-network">feedforward neural network</span> (also called a multi-layer perceptron or MLP) consists of:
</p>
<ul>
  <li><strong>Input layer:</strong> Receives raw features (one neuron per input feature).</li>
  <li><strong>Hidden layers:</strong> One or more layers of neurons that transform the input into increasingly abstract representations. Each neuron in a hidden layer connects to every neuron in the preceding and succeeding layers (hence "fully connected" or "dense" layers).</li>
  <li><strong>Output layer:</strong> Produces the final prediction. For binary classification: one neuron with sigmoid activation. For multi-class classification: one neuron per class with softmax activation. For regression: one neuron with no activation (or linear activation).</li>
</ul>
<p>
  The number of hidden layers defines the network's "depth." The number of neurons per layer defines its "width." Together, depth and width determine the network's capacity — its ability to represent complex functions. The <span class="term" data-term="universal-approximation">Universal Approximation Theorem</span> states that a feedforward network with a single hidden layer and sufficient neurons can approximate any continuous function to arbitrary accuracy. But in practice, deeper networks often learn more efficiently than very wide shallow networks — they build hierarchical features that decompose complex patterns into simpler sub-patterns.
</p>

<div class="pro-tip">
  <strong>PM Perspective:</strong> When your ML team discusses "model architecture" choices — number of layers, neurons per layer, activation functions — they are balancing capacity (ability to learn complex patterns) against training cost, inference latency, and overfitting risk. As a PM, you influence this trade-off through product requirements: "This model must run on mobile in under 50ms" constrains architecture size. "This model must handle 10,000 edge cases" demands capacity. Clear requirements help the team make the right technical trade-offs.
</div>

<h2>Going Deep: Why Depth Matters</h2>
<p>
  The transition from shallow to deep networks was not just about adding layers — it required solving practical challenges that made deep networks trainable:
</p>
<ul>
  <li><strong>Better activation functions:</strong> ReLU (2010 popularisation) eliminated vanishing gradients for positive inputs.</li>
  <li><strong>Better initialisation:</strong> Xavier (2010) and He initialisation (2015) ensured that signal magnitudes were preserved across layers at the start of training.</li>
  <li><strong>Batch Normalisation:</strong> Normalising layer outputs during training stabilised and accelerated learning (covered in Module 2).</li>
  <li><strong>Residual connections:</strong> Skip connections in ResNet (2015) allowed gradients to flow directly through the network, enabling training of networks with hundreds of layers (covered in Module 2).</li>
  <li><strong>GPU computing:</strong> Matrix multiplications in neural networks parallelise naturally on GPUs, delivering 10-100x speedups over CPUs.</li>
</ul>

<div class="example-box">
  <h4>Example</h4>
  <p>Consider image recognition. A shallow network might try to learn a direct mapping from raw pixels to labels — a daunting task since images have millions of pixels and the same object can appear in infinite positions, orientations, and lighting conditions. A deep network solves this hierarchically: Layer 1 detects edges. Layer 2 combines edges into textures (fur, bricks, water). Layer 3 combines textures into parts (ears, wheels, windows). Layer 4 combines parts into objects (cat, car, house). Each layer's representations are simpler and more reusable than trying to learn the whole mapping at once.</p>
</div>

<h2>Key Architectural Families</h2>
<p>
  Neural networks have diversified into specialised architectures for different data types:
</p>
<ul>
  <li><strong>MLPs (Feedforward):</strong> General-purpose networks for tabular and structured data. Still widely used as components within larger systems.</li>
  <li><strong>CNNs (Convolutional Neural Networks):</strong> Designed for grid-structured data (images, video). Use weight sharing and local connectivity to exploit spatial structure. Covered in depth in Module 2, Lesson 1.</li>
  <li><strong>RNNs / LSTMs:</strong> Designed for sequential data (text, time series, audio). Process inputs one step at a time, maintaining a hidden state. Covered in Module 2, Lesson 2.</li>
  <li><strong>Transformers:</strong> Process all positions in parallel using self-attention. Dominate NLP and are increasingly used for vision. Covered in Module 3.</li>
  <li><strong>Graph Neural Networks (GNNs):</strong> Designed for graph-structured data (social networks, molecules, knowledge graphs).</li>
</ul>

<div class="key-concept">
  <strong>Key Concept:</strong> The choice of architecture is driven by the <em>structure of the data</em>, not arbitrary preference. CNNs exploit spatial locality in images. RNNs exploit sequential ordering in time series. Transformers exploit pairwise relationships between all positions. Matching architecture to data structure is one of the most impactful design decisions in ML.
</div>

<h2>Parameters, Hyperparameters, and Scale</h2>
<p>
  A neural network's <span class="term" data-term="parameters">parameters</span> (weights and biases) are learned from data during training. <span class="term" data-term="hyperparameters">Hyperparameters</span> are set by humans before training: learning rate, number of layers, neurons per layer, batch size, regularisation strength, etc. The distinction matters: parameters are the model; hyperparameters are the recipe for cooking the model.
</p>
<p>
  Modern models are staggeringly large. GPT-3 has 175 billion parameters. GPT-4 is estimated at over 1 trillion. Gemini Ultra and other frontier models operate at similar scales. Training these models requires thousands of GPUs running for weeks to months at a cost of tens of millions of dollars. This has profound implications for AI product strategy: only a handful of organisations can train frontier models from scratch, but many can fine-tune or deploy them.
</p>

<div class="pro-tip">
  <strong>PM Perspective:</strong> Understanding scale is strategically critical. If training a frontier model costs $100M+ and takes months, your product roadmap must account for this. Most AI PM work involves leveraging pre-trained models (via APIs, fine-tuning, or distillation) rather than training from scratch. Knowing the difference between "we need a new model" and "we need to fine-tune an existing model" can mean the difference between a 6-month and a 6-week timeline.
</div>
`,
    quiz: {
      questions: [
        {
          question: 'Your team is deploying a neural network for on-device mobile inference. The model must run in under 30ms on a mid-range smartphone. Which architectural decision is MOST impactful for meeting this latency requirement?',
          type: 'mc',
          options: [
            'Switching loss functions from cross-entropy to focal loss formulation',
            'Reducing model depth and width to minimize parameters and operations',
            'Changing optimization algorithm from SGD to Adam for training',
            'Replacing ReLU with GELU activation functions throughout layers'
          ],
          correct: 1,
          explanation: 'Inference latency is primarily driven by the number of parameters and floating-point operations, which are determined by model depth and width. The loss function and optimiser only affect training, not inference. GELU is slightly more expensive than ReLU but this is a minor factor compared to total model size.',
          difficulty: 'applied',
          expertNote: 'A strong PM would also explore model distillation (training a smaller model to mimic a larger one), quantisation (reducing precision from float32 to int8), and hardware-specific optimisations (CoreML, TFLite) to meet the latency target without sacrificing too much accuracy.'
        },
        {
          question: 'Why did the Minsky & Papert critique of the perceptron NOT invalidate neural networks as a research direction?',
          type: 'mc',
          options: [
            'Their mathematical proofs were subsequently found to contain fundamental errors',
            'They demonstrated single-layer limitations while acknowledging multi-layer potential',
            'Their critique addressed only image applications not applicable to text',
            'Their philosophical arguments lacked mathematical rigor and were quickly dismissed'
          ],
          correct: 1,
          explanation: 'Minsky and Papert\'s proofs were correct for single-layer perceptrons but explicitly noted that multi-layer networks could, in theory, learn non-linear functions like XOR. The gap was practical: backpropagation (the efficient training algorithm for multi-layer networks) was not widely known until 1986. The limitation was in training methodology, not architectural capability.',
          difficulty: 'applied',
          expertNote: 'This historical episode illustrates a recurring theme in AI: theoretical capability often outpaces practical training methods. A similar dynamic exists today with RL — many problems are theoretically solvable but require training innovations to become practical.'
        },
        {
          question: 'A junior ML engineer argues: "We should use a neural network with 100 hidden layers because deeper is always better." What are the two most important counterarguments?',
          type: 'multi',
          options: [
            'Very deep networks suffer from vanishing gradients unless mitigations like residual connections are used',
            'Neural networks with more than 10 layers are mathematically proven to overfit on data',
            'Deeper networks require more compute and training time not justified by problem complexity',
            'Deep networks cannot use ReLU activation functions for gradient propagation',
            'The Universal Approximation Theorem guarantees single hidden layer is always sufficient'
          ],
          correct: [0, 2],
          explanation: 'Depth enables hierarchical feature learning but comes with practical costs: gradient flow problems (mitigated by residual connections, batch normalisation) and increased computational expense. There is no proof that 10+ layers always overfit. ReLU works in deep networks. The Universal Approximation Theorem says a single layer is sufficient in theory but may require impractically many neurons — depth is often more efficient.',
          difficulty: 'applied',
          expertNote: 'An expert PM would frame this as a capacity-efficiency trade-off: the right depth depends on the problem complexity, data availability, and latency constraints. They would request ablation experiments comparing architectures of different depths.'
        },
        {
          question: 'Scenario: Your team is deciding between using a large pre-trained language model via API ($0.01 per inference call) versus training a smaller, specialised model in-house (3 months, $200K development cost, $0.0001 per inference call). You expect 50 million inference calls per year. As a PM, outline the key factors in this build-vs-buy decision.',
          type: 'scenario',
          correct: 'A thorough analysis would consider: (1) Cost: API approach costs $500K/year at 50M calls; in-house costs $200K upfront + $5K/year in inference, breaking even in about 5 months. At this volume, in-house is cheaper within the first year. (2) Time-to-market: API is available immediately; in-house takes 3 months to build. Opportunity cost of delayed launch must be quantified. (3) Quality: The large pre-trained model may be more capable out-of-the-box, but a specialised model fine-tuned on domain data may outperform it on your specific task. Requires evaluation. (4) Control and flexibility: In-house models can be updated, customised, and debugged freely; API models are a black box subject to the provider\'s changes. (5) Data privacy: Using an API sends user data to a third party. (6) Risk: API dependency creates vendor lock-in; in-house creates hiring and maintenance dependencies. (7) Recommendation: Start with the API for rapid validation, collect data, and build the in-house model in parallel — migrate when quality and cost savings are proven.',
          explanation: 'Build-vs-buy for ML models is one of the most consequential PM decisions. It involves trade-offs across cost, speed, quality, control, privacy, and risk. The phased approach (start with API, build in parallel) manages uncertainty while capturing value quickly.',
          difficulty: 'expert',
          expertNote: 'A DeepMind-calibre PM would also evaluate model distillation (using the large model to generate training data for the small model), consider regulatory requirements (data residency, GDPR), and build switching costs into the financial model.'
        },
        {
          question: 'Which activation function is most commonly used in the hidden layers of modern deep networks, and why?',
          type: 'mc',
          options: [
            'Sigmoid because its (0,1) output range provides interpretable probability values',
            'Tanh because zero-centered outputs accelerate convergence during training',
            'ReLU because it prevents vanishing gradients and enables deep architectures',
            'Softmax because it generates normalized probability distributions per layer'
          ],
          correct: 2,
          explanation: 'ReLU (Rectified Linear Unit) is the standard activation for hidden layers because it does not saturate for positive inputs (mitigating vanishing gradients), is extremely cheap to compute (just a max operation), and has been empirically shown to enable faster training of deep networks. Sigmoid and tanh suffer from vanishing gradients in deep networks. Softmax is used only in the output layer for classification.',
          difficulty: 'foundational',
          expertNote: 'In transformer architectures specifically, GELU has largely replaced ReLU in hidden layers because its smoothness near zero provides slight empirical gains, but ReLU remains dominant in CNNs and other architectures.'
        }
      ]
    }
  }

};
