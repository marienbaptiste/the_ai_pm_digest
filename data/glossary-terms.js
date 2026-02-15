export const glossaryTerms = {
  // === AI/ML Basics ===
  'neural-network': {
    term: 'Neural Network',
    definition: 'A computational system inspired by biological neural networks, consisting of interconnected nodes (neurons) organized in layers that process information by passing weighted signals through activation functions. Each connection has a learnable weight that is adjusted during training.',
    analogy: 'Like a factory assembly line where each station (neuron) does a small transformation, and the product gets refined as it passes through each stage.',
    whyItMatters: 'As a PM, you need to understand what neural networks can and cannot do to set realistic product expectations and communicate capabilities to stakeholders.',
    related: ['deep-learning', 'weights', 'activation-function', 'backpropagation']
  },
  'deep-learning': {
    term: 'Deep Learning',
    definition: 'A subset of machine learning that uses neural networks with many layers (hence "deep") to learn hierarchical representations of data. Each layer learns increasingly abstract features, enabling the network to model complex patterns.',
    analogy: 'Like learning to recognize a face: the first layers detect edges, then shapes, then features like eyes and noses, and finally complete faces.',
    whyItMatters: 'Deep learning powers nearly every AI product today. Understanding its strengths (pattern recognition, scale) and limitations (data hunger, interpretability) is fundamental for product decisions.',
    related: ['neural-network', 'cnn', 'transformer', 'backpropagation']
  },
  'machine-learning': {
    term: 'Machine Learning',
    definition: 'A field of AI where systems learn patterns from data rather than being explicitly programmed with rules. The system improves its performance on a task as it processes more data, adjusting internal parameters to minimize prediction errors.',
    analogy: 'Like learning to cook by tasting and adjusting rather than following a recipe exactly \u2014 you learn from experience what works.',
    whyItMatters: 'ML is the foundation of AI products. A PM must understand the ML lifecycle \u2014 data collection, training, evaluation, deployment \u2014 to plan roadmaps and set expectations.',
    related: ['supervised-learning', 'unsupervised-learning', 'reinforcement-learning', 'deep-learning']
  },
  'supervised-learning': {
    term: 'Supervised Learning',
    definition: 'A ML paradigm where the model learns from labeled examples \u2014 input-output pairs where the correct answer is provided. The model learns to map inputs to outputs by minimizing the difference between its predictions and the true labels.',
    analogy: 'Like a student learning with an answer key \u2014 they see the question, make a guess, check the answer, and adjust their understanding.',
    whyItMatters: 'Most production AI features use supervised learning. PMs need to understand that quality labeled data is the bottleneck, not algorithms.',
    related: ['training-data', 'labels', 'loss-function', 'overfitting']
  },
  'unsupervised-learning': {
    term: 'Unsupervised Learning',
    definition: 'A ML paradigm where the model finds patterns in data without labeled examples. It discovers structure like clusters, anomalies, or compressed representations purely from the data distribution.',
    analogy: 'Like sorting a pile of photos into groups without anyone telling you the categories \u2014 you discover natural groupings yourself.',
    whyItMatters: 'Powers features like recommendation engines, anomaly detection, and customer segmentation. Important when labeled data is expensive or unavailable.',
    related: ['supervised-learning', 'embedding', 'autoencoder', 'features']
  },
  'reinforcement-learning': {
    term: 'Reinforcement Learning (RL)',
    definition: 'A ML paradigm where an agent learns by interacting with an environment, receiving rewards or penalties for actions. It learns a policy that maximizes cumulative reward over time through trial and error.',
    analogy: 'Like training a dog with treats \u2014 good behavior gets rewarded, bad behavior gets corrected, and the dog learns what to do without explicit instructions.',
    whyItMatters: 'RL powers game-playing AI (AlphaGo), robotics, and RLHF for LLMs. Understanding it is critical for the DeepMind PM role given their RL heritage.',
    related: ['rlhf', 'alphago', 'alignment']
  },
  'gradient-descent': {
    term: 'Gradient Descent',
    definition: 'An optimization algorithm that iteratively adjusts model parameters by computing the gradient (direction of steepest increase) of the loss function and moving in the opposite direction. It\'s how neural networks learn from their mistakes.',
    analogy: 'Like finding the lowest point in a valley while blindfolded \u2014 you feel the slope beneath your feet and always step downhill.',
    whyItMatters: 'Understanding gradient descent helps PMs grasp why training takes time, why models can get "stuck," and why hyperparameters like learning rate matter.',
    related: ['loss-function', 'learning-rate', 'backpropagation', 'batch-size']
  },
  'loss-function': {
    term: 'Loss Function',
    definition: 'A mathematical function that quantifies how wrong a model\'s predictions are compared to the true values. Training aims to minimize this loss. Common examples include cross-entropy loss for classification and mean squared error for regression.',
    analogy: 'Like a scorecard that measures how far off your dart throws are from the bullseye \u2014 lower is better.',
    whyItMatters: 'The choice of loss function shapes what the model optimizes for. PMs should understand this to ensure the model\'s objective aligns with the product\'s success metrics.',
    related: ['gradient-descent', 'backpropagation', 'overfitting']
  },
  'backpropagation': {
    term: 'Backpropagation',
    definition: 'The algorithm that computes gradients of the loss function with respect to each weight in a neural network by propagating error signals backward from the output layer to the input layer. This enables gradient descent to update all weights efficiently.',
    analogy: 'Like tracing back through an assembly line to find which station caused a defect \u2014 you assign blame to each worker proportionally.',
    whyItMatters: 'Backpropagation is why deep networks can learn at all. Understanding it helps PMs appreciate training costs and why deeper networks need more compute.',
    related: ['gradient-descent', 'loss-function', 'weights', 'neural-network']
  },
  'overfitting': {
    term: 'Overfitting',
    definition: 'When a model memorizes training data patterns (including noise) instead of learning generalizable rules, resulting in excellent training performance but poor performance on new data. The model is too complex for the amount of training data.',
    analogy: 'Like a student who memorizes test answers verbatim instead of understanding concepts \u2014 they ace practice tests but fail on new questions.',
    whyItMatters: 'A critical concept for PMs: a model that seems amazing in testing may fail in production. Understanding overfitting helps you interpret evaluation results and push for proper validation.',
    related: ['regularization', 'dropout', 'validation-set', 'underfitting']
  },
  'underfitting': {
    term: 'Underfitting',
    definition: 'When a model is too simple to capture the underlying patterns in the data, resulting in poor performance on both training and test data. The model lacks the capacity to learn the task.',
    analogy: 'Like trying to draw a portrait with only a ruler \u2014 the tool is too simple for the task.',
    whyItMatters: 'Helps PMs understand why sometimes the team needs a more powerful model, more data, or more training time.',
    related: ['overfitting', 'regularization', 'hyperparameter']
  },
  'regularization': {
    term: 'Regularization',
    definition: 'Techniques that prevent overfitting by adding constraints to the learning process. This includes L1/L2 weight penalties, dropout, data augmentation, and early stopping. Regularization trades some training accuracy for better generalization.',
    analogy: 'Like adding rules to an essay contest (word limit, no jargon) \u2014 constraints that force competitors to focus on substance over style.',
    whyItMatters: 'Understanding regularization helps PMs discuss model performance trade-offs intelligently and ask the right questions about evaluation methodology.',
    related: ['overfitting', 'dropout', 'batch-normalization']
  },
  'batch-size': {
    term: 'Batch Size',
    definition: 'The number of training examples processed together before updating model weights. Larger batches give more stable gradient estimates but require more memory. Smaller batches add noise that can help escape local minima but make training less stable.',
    analogy: 'Like grading papers: you could grade one at a time (small batch) or read 100 before providing feedback (large batch). Each approach has trade-offs.',
    whyItMatters: 'Batch size directly affects training speed, cost, and model quality. It\'s a key factor in compute budget discussions.',
    related: ['gradient-descent', 'epoch', 'learning-rate']
  },
  'epoch': {
    term: 'Epoch',
    definition: 'One complete pass through the entire training dataset. Models typically need multiple epochs to converge. Too few epochs lead to underfitting; too many can cause overfitting.',
    analogy: 'Like reading a textbook cover to cover once \u2014 you usually need multiple reads to fully absorb the material.',
    whyItMatters: 'Epochs affect training time and cost. PMs need to understand this when discussing compute budgets and training schedules.',
    related: ['batch-size', 'overfitting', 'learning-rate']
  },
  'learning-rate': {
    term: 'Learning Rate',
    definition: 'A hyperparameter that controls how much model weights are adjusted in response to each gradient update. Too high causes unstable training; too low causes painfully slow convergence or getting stuck in poor solutions.',
    analogy: 'Like the size of steps you take when walking downhill blindfolded \u2014 too big and you overshoot, too small and you\'ll never get there.',
    whyItMatters: 'One of the most impactful hyperparameters. Understanding it helps PMs grasp why training results can vary and why ML teams experiment.',
    related: ['gradient-descent', 'hyperparameter', 'batch-size']
  },
  'activation-function': {
    term: 'Activation Function',
    definition: 'A non-linear function applied to a neuron\'s output that enables neural networks to learn complex patterns. Without non-linearity, a deep network would collapse to a single linear transformation. Common activations include ReLU, sigmoid, and tanh.',
    analogy: 'Like a decision gate that decides how much signal to pass through \u2014 it\'s what gives neural networks their power beyond simple linear models.',
    whyItMatters: 'Understanding activations helps PMs grasp why certain architectures work better for specific tasks.',
    related: ['neural-network', 'deep-learning']
  },
  'weights': {
    term: 'Weights',
    definition: 'The learnable parameters in a neural network that determine how input signals are transformed as they pass between neurons. Training adjusts these weights to minimize prediction errors. A large model like GPT-4 has hundreds of billions of weights.',
    analogy: 'Like volume knobs on a mixing board \u2014 each one controls how much influence one input has on the output.',
    whyItMatters: 'Model size (parameter count) directly impacts cost, speed, and capabilities. PMs need to understand the trade-offs of larger vs smaller models.',
    related: ['neural-network', 'backpropagation', 'gradient-descent']
  },
  'features': {
    term: 'Features',
    definition: 'The individual measurable properties of the data used as input to a ML model. Feature engineering \u2014 selecting and transforming the right features \u2014 is often the most impactful part of building a ML system.',
    analogy: 'Like the attributes a real estate agent considers when pricing a house: square footage, location, number of bedrooms \u2014 each is a feature.',
    whyItMatters: 'Data quality and feature selection often matter more than model architecture. PMs should prioritize data strategy.',
    related: ['training-data', 'embedding', 'feature-map']
  },
  'labels': {
    term: 'Labels',
    definition: 'The correct answers or ground truth annotations attached to training examples in supervised learning. Creating high-quality labels is expensive and time-consuming, often requiring human annotators.',
    analogy: 'Like the answer key for a test \u2014 without it, the student (model) can\'t check if their answers are right.',
    whyItMatters: 'Label quality bottlenecks most AI projects. PMs must plan for annotation costs, quality control, and potential bias in labeling.',
    related: ['supervised-learning', 'training-data', 'human-evaluation']
  },
  'training-data': {
    term: 'Training Data',
    definition: 'The dataset used to teach a ML model. Its quality, quantity, and representativeness directly determine model performance. "Garbage in, garbage out" is the fundamental law of ML.',
    analogy: 'Like the textbooks a student uses \u2014 if they\'re incomplete, biased, or full of errors, the student will learn poorly.',
    whyItMatters: 'Data strategy is often more important than model architecture. PMs must champion data quality, diversity, and responsible sourcing.',
    related: ['labels', 'validation-set', 'test-data', 'bias-fairness']
  },
  'test-data': {
    term: 'Test Data',
    definition: 'A held-out dataset never seen during training, used to evaluate model performance on truly unseen examples. It provides the final, unbiased estimate of how well the model will perform in production.',
    analogy: 'Like the final exam that\'s different from homework and practice tests \u2014 it measures real understanding.',
    whyItMatters: 'PMs must ensure proper test methodology. Leaking test data into training gives misleadingly good results.',
    related: ['validation-set', 'training-data', 'overfitting']
  },
  'validation-set': {
    term: 'Validation Set',
    definition: 'A portion of data held out from training, used to tune hyperparameters and make architecture decisions during development. Unlike test data, it can be looked at repeatedly during model development.',
    analogy: 'Like practice exams taken during studying \u2014 they guide preparation but aren\'t the final test.',
    whyItMatters: 'Proper validation prevents overfitting to the test set. PMs should ask if the evaluation pipeline is rigorous.',
    related: ['test-data', 'training-data', 'hyperparameter']
  },
  'hyperparameter': {
    term: 'Hyperparameter',
    definition: 'Configuration settings that control the training process itself (learning rate, batch size, number of layers) as opposed to model parameters (weights) that are learned from data. They are set before training begins and significantly impact results.',
    analogy: 'Like the oven temperature and baking time for a cake recipe \u2014 they\'re not part of the ingredients but dramatically affect the outcome.',
    whyItMatters: 'Hyperparameter tuning requires compute resources. PMs should understand this cost and timeline when planning sprints.',
    related: ['learning-rate', 'batch-size', 'epoch']
  },

  // === Deep Learning ===
  'cnn': {
    term: 'Convolutional Neural Network (CNN)',
    definition: 'A neural network architecture designed for processing grid-like data (images, video) using convolutional filters that slide across the input to detect local patterns. CNNs learn hierarchical features \u2014 edges, textures, shapes, objects \u2014 automatically.',
    analogy: 'Like a magnifying glass that systematically scans across a photo, detecting patterns at each location and building up to a full understanding.',
    whyItMatters: 'CNNs power image recognition, video analysis, and visual features in products. Understanding their capabilities helps scope visual AI features.',
    related: ['convolution', 'pooling', 'feature-map', 'deep-learning']
  },
  'rnn': {
    term: 'Recurrent Neural Network (RNN)',
    definition: 'A neural network designed for sequential data that maintains a hidden state (memory) updated at each time step. This allows it to process variable-length sequences and capture temporal dependencies, though it struggles with long-range dependencies.',
    analogy: 'Like reading a book word by word while keeping a mental summary \u2014 each word updates your understanding of the story.',
    whyItMatters: 'RNNs were foundational for language and time-series tasks before Transformers. Understanding them provides context for why Transformers were revolutionary.',
    related: ['lstm', 'transformer', 'attention']
  },
  'lstm': {
    term: 'Long Short-Term Memory (LSTM)',
    definition: 'An improved RNN architecture with gating mechanisms (forget, input, output gates) that can selectively remember or forget information over long sequences, solving the vanishing gradient problem of standard RNNs.',
    analogy: 'Like a notepad where you can choose to write new notes, erase old ones, or highlight important points \u2014 controlled memory management.',
    whyItMatters: 'LSTMs were state-of-the-art for sequence tasks until Transformers. Understanding their limitations explains why the industry shifted.',
    related: ['rnn', 'transformer', 'attention']
  },
  'dropout': {
    term: 'Dropout',
    definition: 'A regularization technique that randomly deactivates (sets to zero) a percentage of neurons during each training step, forcing the network to learn redundant representations and preventing co-adaptation of neurons.',
    analogy: 'Like randomly removing team members during practice drills \u2014 everyone learns to be versatile rather than relying on one star player.',
    whyItMatters: 'A simple but powerful technique that dramatically reduces overfitting. It\'s a common lever ML teams use when models perform poorly on real data.',
    related: ['regularization', 'overfitting', 'batch-normalization']
  },
  'batch-normalization': {
    term: 'Batch Normalization',
    definition: 'A technique that normalizes the inputs to each layer to have zero mean and unit variance within each training batch. It stabilizes and accelerates training by reducing internal covariate shift.',
    analogy: 'Like calibrating measuring instruments between experiments \u2014 it ensures consistent conditions for each stage of processing.',
    whyItMatters: 'BatchNorm enables training much deeper networks. Its presence or absence affects model architecture choices and training speed.',
    related: ['dropout', 'regularization', 'layer-normalization']
  },
  'resnet': {
    term: 'ResNet (Residual Network)',
    definition: 'A groundbreaking CNN architecture that introduced skip connections (residual connections) allowing gradients to flow directly through the network. This made it possible to train networks hundreds of layers deep.',
    analogy: 'Like installing express elevators in a skyscraper \u2014 they let people (information) skip floors and reach the top faster.',
    whyItMatters: 'ResNet proved that deeper networks work better with the right architecture. Skip connections are now used everywhere, including Transformers.',
    related: ['cnn', 'skip-connection', 'deep-learning']
  },
  'gan': {
    term: 'Generative Adversarial Network (GAN)',
    definition: 'A generative model with two competing networks: a generator that creates fake data and a discriminator that tries to distinguish real from fake. Through this adversarial training, the generator learns to create increasingly realistic outputs.',
    analogy: 'Like a counterfeiter and a detective in an arms race \u2014 the counterfeiter gets better at faking, and the detective gets better at detecting, until fakes are indistinguishable.',
    whyItMatters: 'GANs pioneered AI-generated images. Understanding their instability issues provides context for why diffusion models replaced them.',
    related: ['diffusion-model', 'autoencoder', 'vae', 'latent-space']
  },
  'autoencoder': {
    term: 'Autoencoder',
    definition: 'A neural network trained to compress input data into a lower-dimensional representation (encoding) and then reconstruct the original input. The compressed representation captures the most important features.',
    analogy: 'Like creating a summary of a book \u2014 you compress the essential meaning into fewer words, then try to reconstruct the full story.',
    whyItMatters: 'Autoencoders power dimensionality reduction, anomaly detection, and are a component of VAEs used in image generation.',
    related: ['vae', 'embedding', 'latent-space']
  },
  'pooling': {
    term: 'Pooling',
    definition: 'A downsampling operation in CNNs that reduces spatial dimensions by taking the maximum or average value within local regions. It makes the network more robust to small translations and reduces computational cost.',
    analogy: 'Like creating a thumbnail of a photo \u2014 you capture the essential information while reducing the size.',
    whyItMatters: 'Pooling affects the trade-off between detail and computational cost in vision models. Understanding it helps evaluate model architecture choices.',
    related: ['cnn', 'convolution', 'feature-map']
  },
  'convolution': {
    term: 'Convolution',
    definition: 'A mathematical operation where a small filter (kernel) slides across the input, computing dot products at each position. In CNNs, learned convolutional filters detect specific patterns like edges, corners, and textures.',
    analogy: 'Like using a stencil to test different parts of a painting \u2014 the stencil (filter) reveals specific patterns wherever it finds them.',
    whyItMatters: 'Convolutions are the core operation of CNNs. Understanding them helps PMs grasp how vision models process images.',
    related: ['cnn', 'pooling', 'feature-map']
  },
  'feature-map': {
    term: 'Feature Map',
    definition: 'The output of applying a convolutional filter to an input, representing where a specific pattern or feature was detected in the input. A CNN layer produces multiple feature maps, each detecting a different pattern.',
    analogy: 'Like a heat map showing where in an image a specific pattern was found \u2014 bright spots indicate strong matches.',
    whyItMatters: 'Feature maps make CNNs partially interpretable. Understanding them helps PMs discuss model transparency and debugging.',
    related: ['cnn', 'convolution', 'features']
  },
  'skip-connection': {
    term: 'Skip Connection (Residual Connection)',
    definition: 'A shortcut that adds a layer\'s input directly to its output, bypassing the transformation. This allows gradients to flow directly through the network and enables training of very deep networks.',
    analogy: 'Like having a direct phone line to the CEO in addition to the normal chain of command \u2014 important signals don\'t get lost in bureaucracy.',
    whyItMatters: 'Skip connections are used in virtually every modern architecture (ResNet, Transformer, U-Net). They\'re a fundamental building block of modern AI.',
    related: ['resnet', 'transformer', 'deep-learning']
  },

  // === Transformers ===
  'transformer': {
    term: 'Transformer',
    definition: 'The dominant neural network architecture for processing sequences, introduced in "Attention Is All You Need" (2017). It replaces recurrence with self-attention, allowing parallel processing of entire sequences and capturing long-range dependencies. Powers LLMs, vision models, and multimodal systems.',
    analogy: 'Like a conference room where every participant can directly communicate with every other participant simultaneously, rather than passing notes one by one down a chain.',
    whyItMatters: 'Transformers are THE architecture behind Gemini, GPT, and virtually all modern AI products. Deep understanding is essential for the DeepMind PM role.',
    related: ['attention', 'self-attention', 'multi-head-attention', 'llm']
  },
  'attention': {
    term: 'Attention Mechanism',
    definition: 'A mechanism that lets a model dynamically focus on different parts of the input when producing each output element. It computes relevance scores between elements, allowing the model to weigh important information more heavily.',
    analogy: 'Like highlighting the most relevant passages in a textbook when answering a specific question \u2014 you pay more attention to what matters.',
    whyItMatters: 'Attention is the core innovation that powers modern AI. Understanding it helps PMs explain model capabilities and limitations to stakeholders.',
    related: ['self-attention', 'multi-head-attention', 'transformer', 'query-key-value']
  },
  'self-attention': {
    term: 'Self-Attention',
    definition: 'A specific type of attention where each element in a sequence attends to every other element in the same sequence, computing Query, Key, and Value vectors for each. The attention weights reveal which other elements are most relevant for understanding each position.',
    analogy: 'Like each word in a sentence looking at every other word to figure out its meaning in context \u2014 "bank" looks at "river" vs "money" to determine which meaning to use.',
    whyItMatters: 'Self-attention is computationally expensive (quadratic in sequence length), directly impacting context window limits and inference costs \u2014 key PM considerations.',
    related: ['attention', 'query-key-value', 'multi-head-attention', 'context-window']
  },
  'multi-head-attention': {
    term: 'Multi-Head Attention',
    definition: 'Running multiple self-attention operations in parallel, each with different learned projection matrices. Each "head" can attend to different types of relationships (syntax, semantics, position), and their outputs are concatenated and projected.',
    analogy: 'Like having multiple analysts review the same data, each looking for different types of patterns \u2014 together they catch more than any one alone.',
    whyItMatters: 'The number of attention heads is a key architectural decision affecting model capacity and cost. More heads = richer representation but more compute.',
    related: ['self-attention', 'attention', 'transformer']
  },
  'positional-encoding': {
    term: 'Positional Encoding',
    definition: 'Information added to input embeddings that tells the Transformer the position of each element in the sequence. Since self-attention is position-agnostic, positional encoding is necessary for the model to understand word order.',
    analogy: 'Like numbering the pages of a shuffled manuscript so the reader knows the correct order.',
    whyItMatters: 'Positional encoding innovations (RoPE, ALiBi) have been key to extending context windows \u2014 a major competitive differentiator for Gemini.',
    related: ['transformer', 'embedding', 'context-window', 'long-context']
  },
  'query-key-value': {
    term: 'Query, Key, Value (Q, K, V)',
    definition: 'The three projections used in attention. The Query represents "what am I looking for?", the Key represents "what do I contain?", and the Value represents "what information do I provide?". Attention scores are computed as Q\u00B7K\u1D40, then used to weight the Values.',
    analogy: 'Like a library search: Query is your search terms, Key is book titles/tags, and Value is the actual book content. Matching query to keys tells you which books (values) to read.',
    whyItMatters: 'Understanding Q/K/V helps PMs grasp how context retrieval works inside models and why certain prompting strategies are effective.',
    related: ['self-attention', 'attention', 'transformer']
  },
  'encoder': {
    term: 'Encoder',
    definition: 'The part of a Transformer that processes the input sequence bidirectionally (attending to all positions). It builds rich contextual representations of the input. Used in BERT-style models and the encoder-decoder architecture.',
    analogy: 'Like reading an entire document to understand it before answering questions about it.',
    whyItMatters: 'Encoder-only models (BERT) excel at understanding tasks. Knowing when to use encoder vs decoder architectures helps scope AI features correctly.',
    related: ['decoder', 'transformer', 'embedding']
  },
  'decoder': {
    term: 'Decoder',
    definition: 'The part of a Transformer that generates output tokens one at a time, each conditioned on previously generated tokens and (optionally) encoder representations. Uses masked self-attention to prevent looking at future tokens.',
    analogy: 'Like writing a story word by word, where each new word is chosen based on everything you\'ve written so far.',
    whyItMatters: 'Decoder-only models (GPT, Gemini) are dominant for generation. Understanding autoregressive generation explains latency characteristics.',
    related: ['encoder', 'transformer', 'llm', 'token']
  },
  'tokenizer': {
    term: 'Tokenizer',
    definition: 'The component that converts raw text into a sequence of tokens (subwords, characters, or words) that the model can process. Different tokenizers (BPE, SentencePiece, WordPiece) split text differently, affecting model performance and cost.',
    analogy: 'Like breaking a sentence into LEGO bricks \u2014 common words might be single bricks, while rare words get split into smaller pieces.',
    whyItMatters: 'Tokenization directly impacts cost (tokens = billing units), context limits, and multilingual performance. A PM must understand token economics.',
    related: ['token', 'embedding', 'context-window']
  },
  'embedding': {
    term: 'Embedding',
    definition: 'A dense vector representation of discrete data (words, tokens, images) in continuous space where semantic similarity corresponds to geometric proximity. Learned embeddings capture meaning: "king" - "man" + "woman" \u2248 "queen".',
    analogy: 'Like plotting cities on a map where distance represents cultural similarity rather than geographic distance.',
    whyItMatters: 'Embeddings power search, recommendations, and RAG systems. Understanding them is essential for building retrieval-based AI features.',
    related: ['token', 'vector-database', 'semantic-search', 'cosine-similarity']
  },
  'token': {
    term: 'Token',
    definition: 'The basic unit of text that LLMs process. A token is typically a subword piece (3-4 characters on average in English). Models have token limits (context windows), and API costs are typically billed per token.',
    analogy: 'Like syllables in speech \u2014 words are broken into smaller, manageable pieces that the model processes one at a time.',
    whyItMatters: 'Token counts directly determine API costs, context limits, and response latency. Critical for product pricing and UX decisions.',
    related: ['tokenizer', 'context-window', 'embedding']
  },
  'context-window': {
    term: 'Context Window',
    definition: 'The maximum number of tokens a model can process in a single forward pass, including both input and output. Gemini supports up to 1M+ tokens, while most models are 4K-128K. Longer contexts enable processing entire codebases or documents.',
    analogy: 'Like the working memory capacity of a person \u2014 some people can juggle 5 things, others 50. More capacity means handling more complex tasks.',
    whyItMatters: 'Context window size is a major competitive differentiator. It directly shapes what use cases a product can support (short chats vs full document analysis).',
    related: ['token', 'long-context', 'positional-encoding', 'gemini']
  },
  'layer-normalization': {
    term: 'Layer Normalization',
    definition: 'A normalization technique that normalizes activations across the feature dimension for each individual example. Unlike batch normalization, it doesn\'t depend on batch size, making it more suitable for sequence models.',
    analogy: 'Like auto-adjusting the brightness and contrast of a photo before each editing step.',
    whyItMatters: 'Layer norm is a standard component of Transformers. Pre-norm vs post-norm placement affects training stability.',
    related: ['batch-normalization', 'transformer']
  },

  // === LLMs ===
  'llm': {
    term: 'Large Language Model (LLM)',
    definition: 'A neural network (typically Transformer-based) with billions of parameters, pre-trained on massive text corpora to predict the next token. LLMs exhibit emergent capabilities like reasoning, code generation, and instruction following that weren\'t explicitly trained.',
    analogy: 'Like a super-well-read person who has absorbed millions of books and can discuss any topic \u2014 not because they memorized answers, but because they deeply understand language patterns.',
    whyItMatters: 'LLMs ARE the product for a Gemini PM. Deep understanding of their capabilities, limitations, and economics is non-negotiable.',
    related: ['transformer', 'pre-training', 'fine-tuning', 'rlhf', 'gemini']
  },
  'gpt': {
    term: 'GPT (Generative Pre-trained Transformer)',
    definition: 'A family of decoder-only Transformer models by OpenAI that demonstrated the power of large-scale pre-training on text followed by fine-tuning. GPT established the paradigm of "pre-train then adapt" that dominates modern AI.',
    analogy: 'Like a person who reads the entire internet before specializing \u2014 broad knowledge first, specific skills second.',
    whyItMatters: 'GPT is the primary competitor architecture to Gemini. Understanding it helps position Gemini\'s advantages (multimodality, long context).',
    related: ['llm', 'pre-training', 'transformer', 'gemini']
  },
  'pre-training': {
    term: 'Pre-training',
    definition: 'The initial training phase where an LLM learns from massive text data (often trillions of tokens) using self-supervised objectives like next-token prediction. This phase is extremely expensive (millions of dollars) and produces a base model with broad knowledge.',
    analogy: 'Like a medical student\'s general education years \u2014 broad learning before specialization.',
    whyItMatters: 'Pre-training costs shape competitive dynamics. Understanding compute requirements helps PMs reason about model updates, deprecation, and pricing.',
    related: ['fine-tuning', 'llm', 'scaling-laws', 'training-data']
  },
  'fine-tuning': {
    term: 'Fine-tuning',
    definition: 'Additional training of a pre-trained model on a smaller, task-specific or domain-specific dataset. It adapts the general model to perform well on specific tasks while retaining broad knowledge. Much cheaper than pre-training.',
    analogy: 'Like a medical student\'s residency \u2014 they specialize in cardiology after getting their general medical degree.',
    whyItMatters: 'Fine-tuning enables customization for specific products and use cases. PMs must understand when fine-tuning vs prompting is the right approach.',
    related: ['pre-training', 'lora', 'instruction-tuning', 'adapter']
  },
  'rlhf': {
    term: 'RLHF (Reinforcement Learning from Human Feedback)',
    definition: 'A training technique that aligns LLMs with human preferences by training a reward model on human comparisons, then using RL to optimize the LLM against that reward. It\'s what makes models helpful, harmless, and honest.',
    analogy: 'Like training a new employee by having experienced colleagues review their work and rate different approaches \u2014 the employee learns what "good" looks like.',
    whyItMatters: 'RLHF is crucial for product quality and safety. PMs must understand the alignment trade-offs: being too helpful (sycophantic) vs too cautious (refusing valid requests).',
    related: ['alignment', 'reinforcement-learning', 'fine-tuning', 'human-evaluation']
  },
  'prompt-engineering': {
    term: 'Prompt Engineering',
    definition: 'The practice of crafting input prompts to elicit optimal outputs from LLMs. Includes techniques like few-shot examples, chain-of-thought prompting, role specification, and output format constraints. Often more cost-effective than fine-tuning.',
    analogy: 'Like knowing exactly how to phrase a question to get the best answer from an expert \u2014 the way you ask dramatically affects what you get.',
    whyItMatters: 'Prompt engineering can unlock capabilities without expensive fine-tuning. PMs should champion it as a rapid iteration tool.',
    related: ['chain-of-thought', 'few-shot', 'zero-shot', 'in-context-learning']
  },
  'chain-of-thought': {
    term: 'Chain-of-Thought (CoT)',
    definition: 'A prompting technique where the model is encouraged to show its reasoning step-by-step before giving a final answer. This dramatically improves performance on reasoning, math, and logic tasks by decomposing complex problems.',
    analogy: 'Like asking a student to "show their work" on a math test \u2014 the step-by-step process catches errors and leads to better answers.',
    whyItMatters: 'CoT unlocks reasoning capabilities critical for complex AI features. PMs should understand when to enable it (better accuracy) vs skip it (faster responses).',
    related: ['prompt-engineering', 'in-context-learning', 'few-shot']
  },
  'in-context-learning': {
    term: 'In-Context Learning (ICL)',
    definition: 'The ability of LLMs to learn new tasks from examples provided in the prompt without any weight updates. The model adapts its behavior based on the demonstrations given in the context, an emergent capability of large models.',
    analogy: 'Like showing someone three examples of a new card game, and they immediately understand the rules and can play.',
    whyItMatters: 'ICL enables rapid prototyping without training. PMs can use it to quickly test new AI features before investing in fine-tuning.',
    related: ['few-shot', 'zero-shot', 'prompt-engineering']
  },
  'few-shot': {
    term: 'Few-Shot Learning',
    definition: 'Providing a small number of examples (2-10) in the prompt to demonstrate the desired behavior. The model generalizes from these examples to perform the task on new inputs without any weight updates.',
    analogy: 'Like showing a new employee a few completed examples before asking them to handle similar cases on their own.',
    whyItMatters: 'Few-shot is the sweet spot between zero-shot (no examples) and fine-tuning (expensive). PMs should know when it\'s sufficient vs when more is needed.',
    related: ['zero-shot', 'in-context-learning', 'prompt-engineering']
  },
  'zero-shot': {
    term: 'Zero-Shot Learning',
    definition: 'Performing a task without any task-specific examples, relying entirely on the model\'s pre-trained knowledge and instruction-following ability. Works well for common tasks but may struggle with unusual or domain-specific ones.',
    analogy: 'Like asking someone to do something they\'ve never done before, relying on their general intelligence and common sense.',
    whyItMatters: 'Zero-shot capability defines the baseline user experience. If zero-shot works well enough, you don\'t need examples or fine-tuning.',
    related: ['few-shot', 'in-context-learning', 'instruction-tuning']
  },
  'scaling-laws': {
    term: 'Scaling Laws',
    definition: 'Empirical relationships showing that model performance improves predictably with increases in model size, dataset size, and compute budget. These power laws (discovered by Kaplan et al.) enable predicting performance before training expensive models.',
    analogy: 'Like knowing that doubling the size of a solar panel predictably increases energy output \u2014 the relationship is mathematical, not random.',
    whyItMatters: 'Scaling laws guide billion-dollar compute investments. PMs at DeepMind must understand that scale is a strategic lever, not just a technical one.',
    related: ['pre-training', 'emergent-abilities', 'llm']
  },
  'emergent-abilities': {
    term: 'Emergent Abilities',
    definition: 'Capabilities that appear suddenly and unpredictably as models scale beyond certain size thresholds. Examples include complex reasoning, code generation, and multi-step planning. These abilities aren\'t present in smaller models.',
    analogy: 'Like how water suddenly changes from liquid to gas at the boiling point \u2014 gradual heating produces an abrupt state change.',
    whyItMatters: 'Emergence makes AI product roadmapping uniquely challenging. New capabilities can appear that weren\'t planned for, creating product opportunities.',
    related: ['scaling-laws', 'llm', 'pre-training']
  },
  'hallucination': {
    term: 'Hallucination',
    definition: 'When an LLM generates fluent, confident-sounding text that is factually incorrect, fabricated, or unsupported by its training data. Hallucinations are a fundamental challenge because the model doesn\'t distinguish between what it "knows" and what it\'s generating.',
    analogy: 'Like a confident storyteller who seamlessly fills gaps in their knowledge with plausible-sounding but made-up details.',
    whyItMatters: 'Hallucination is the #1 barrier to user trust and enterprise adoption. PMs must design UX that manages this risk (citations, confidence indicators, guardrails).',
    related: ['grounding', 'rag', 'guardrails', 'alignment']
  },
  'temperature': {
    term: 'Temperature',
    definition: 'A parameter controlling the randomness of LLM outputs. Lower temperature (0.0) makes outputs more deterministic and focused; higher temperature (1.0+) makes outputs more diverse and creative. It scales the logits before softmax.',
    analogy: 'Like the dial between "conservative" and "creative" on an AI writing assistant \u2014 low for factual tasks, high for brainstorming.',
    whyItMatters: 'Temperature is a key product decision. Different features need different settings: code generation (low) vs creative writing (high).',
    related: ['top-k', 'top-p', 'llm']
  },
  'top-k': {
    term: 'Top-K Sampling',
    definition: 'A decoding strategy that restricts the model to only consider the K most probable next tokens at each step, then samples among them. It prevents the model from choosing very unlikely tokens while maintaining diversity.',
    analogy: 'Like only considering the top 10 restaurants in a city rather than every single option \u2014 you limit choices to quality options.',
    whyItMatters: 'Sampling parameters directly affect output quality. PMs should understand the trade-off between creativity and reliability.',
    related: ['temperature', 'top-p', 'llm']
  },
  'top-p': {
    term: 'Top-P (Nucleus) Sampling',
    definition: 'A decoding strategy that dynamically selects the smallest set of tokens whose cumulative probability exceeds P. Unlike top-K, it adapts the number of candidates based on the probability distribution.',
    analogy: 'Like inviting "enough people to fill 90% of the seats" rather than always inviting exactly 50 people.',
    whyItMatters: 'Top-P is generally preferred over Top-K for its adaptiveness. Understanding sampling helps PMs make API configuration decisions.',
    related: ['temperature', 'top-k', 'llm']
  },
  'lora': {
    term: 'LoRA (Low-Rank Adaptation)',
    definition: 'A parameter-efficient fine-tuning technique that freezes the original model weights and trains small, low-rank decomposition matrices. It achieves near-full fine-tuning performance while training only 0.1-1% of parameters, dramatically reducing cost.',
    analogy: 'Like adding a thin specialty lens to a camera rather than rebuilding the entire camera for each type of photography.',
    whyItMatters: 'LoRA enables affordable model customization for different products and markets. PMs should consider it for domain-specific adaptation.',
    related: ['fine-tuning', 'adapter', 'pre-training']
  },
  'adapter': {
    term: 'Adapter',
    definition: 'Small trainable modules inserted between frozen layers of a pre-trained model. They allow task-specific customization without modifying the original weights, enabling efficient multi-task learning.',
    analogy: 'Like adding plug-in accessories to a power tool \u2014 the base tool stays the same, but adapters let it do different jobs.',
    whyItMatters: 'Adapters enable serving multiple customized models from a single base model, reducing infrastructure costs.',
    related: ['lora', 'fine-tuning']
  },
  'instruction-tuning': {
    term: 'Instruction Tuning',
    definition: 'Fine-tuning a model on datasets of (instruction, response) pairs to improve its ability to follow human instructions. This transforms a base model (that just predicts text) into an assistant (that follows directions).',
    analogy: 'Like training a new employee by giving them explicit tasks and showing them ideal responses.',
    whyItMatters: 'Instruction tuning is what makes LLMs actually useful as products. Quality and diversity of instruction data directly impacts user experience.',
    related: ['fine-tuning', 'rlhf', 'alignment']
  },
  'alignment': {
    term: 'Alignment',
    definition: 'The challenge of ensuring AI systems behave in accordance with human values and intentions. Includes being helpful (doing what users want), harmless (avoiding dangerous outputs), and honest (not deceiving users).',
    analogy: 'Like ensuring a powerful tool has safety mechanisms \u2014 a chainsaw needs a chain brake, an AI needs alignment.',
    whyItMatters: 'Alignment is a top priority at DeepMind. PMs must balance helpfulness with safety, and understand the ethical dimensions of product decisions.',
    related: ['rlhf', 'responsible-ai', 'guardrails', 'interpretability']
  },

  // === Diffusion ===
  'diffusion-model': {
    term: 'Diffusion Model',
    definition: 'A generative model that learns to create data by reversing a gradual noising process. It starts with random noise and iteratively denoises it to produce high-quality outputs (images, audio, video). Currently the dominant approach for image generation.',
    analogy: 'Like a sculptor who starts with a block of clay (noise) and gradually carves away material to reveal a statue (image).',
    whyItMatters: 'Diffusion models power Imagen, Stable Diffusion, and visual AI features. Understanding them is essential for the Gemini multimodal product strategy.',
    related: ['stable-diffusion', 'denoising', 'latent-space', 'classifier-free-guidance']
  },
  'stable-diffusion': {
    term: 'Stable Diffusion',
    definition: 'An open-source diffusion model that operates in a compressed latent space rather than pixel space, making it much more efficient. It combines a VAE (for compression), a U-Net (for denoising), and a text encoder (for conditioning).',
    analogy: 'Like sketching a rough outline before painting in detail \u2014 working at a lower resolution first, then upscaling.',
    whyItMatters: 'Stable Diffusion democratized image generation and created the open-source AI art ecosystem. Understanding its architecture helps position proprietary alternatives.',
    related: ['diffusion-model', 'vae', 'latent-space', 'imagen']
  },
  'vae': {
    term: 'VAE (Variational Autoencoder)',
    definition: 'A generative model that learns a continuous latent space by encoding inputs into probability distributions, sampling from them, and decoding back. The "variational" part ensures the latent space is smooth and well-structured for generation.',
    analogy: 'Like learning to describe faces using a set of sliders (roundness, hair color, age) \u2014 any setting produces a valid face.',
    whyItMatters: 'VAEs are a component of Stable Diffusion and other architectures. Understanding them helps grasp how AI compresses and generates content.',
    related: ['autoencoder', 'latent-space', 'stable-diffusion', 'diffusion-model']
  },
  'latent-space': {
    term: 'Latent Space',
    definition: 'A compressed, abstract representation space where data points are mapped to dense vectors. In this space, similar items are close together, and operations like interpolation produce meaningful results (e.g., smooth transitions between images).',
    analogy: 'Like a map where cities aren\'t placed geographically but by cultural similarity \u2014 nearby points share meaningful properties.',
    whyItMatters: 'Latent spaces are the "thought space" of generative AI. Understanding them helps PMs explain how AI creates and edits content.',
    related: ['embedding', 'vae', 'autoencoder', 'stable-diffusion']
  },
  'denoising': {
    term: 'Denoising',
    definition: 'The process of removing noise from data to recover the original signal. In diffusion models, a neural network is trained to predict and remove noise at each step, gradually transforming random noise into a coherent output.',
    analogy: 'Like cleaning a dirty photograph one pass at a time, each pass removing a bit more grime to reveal the image underneath.',
    whyItMatters: 'Denoising quality directly affects generated content quality. PMs should understand that more denoising steps = better quality but slower generation.',
    related: ['diffusion-model', 'noise-schedule']
  },
  'noise-schedule': {
    term: 'Noise Schedule',
    definition: 'The predetermined sequence of noise levels applied during the forward diffusion process and reversed during generation. It controls how quickly information is destroyed/recovered and significantly affects output quality.',
    analogy: 'Like the dimmer switch on a light \u2014 the schedule controls how gradually the light (image) fades to black (noise) and back.',
    whyItMatters: 'Noise schedule affects generation speed and quality. Understanding it helps PMs evaluate trade-offs in model serving configurations.',
    related: ['diffusion-model', 'denoising']
  },
  'classifier-free-guidance': {
    term: 'Classifier-Free Guidance (CFG)',
    definition: 'A technique that improves text-to-image generation by training the model both with and without text conditioning, then amplifying the difference at inference. Higher guidance values produce images more closely matching the text prompt.',
    analogy: 'Like turning up the "follow instructions" dial \u2014 higher values produce results that match the description more closely but may be less creative.',
    whyItMatters: 'CFG is a key quality knob for image generation products. PMs configure it to balance prompt adherence vs output diversity.',
    related: ['diffusion-model', 'text-to-image']
  },
  'text-to-image': {
    term: 'Text-to-Image',
    definition: 'AI systems that generate images from natural language descriptions. They combine language understanding (text encoders) with image generation (diffusion models or GANs) to create visual content matching textual prompts.',
    analogy: 'Like describing a painting to an artist and having them create it \u2014 except the artist is an AI that works in seconds.',
    whyItMatters: 'Text-to-image is a major AI product category. PMs must navigate quality, safety (deepfakes), and copyright challenges.',
    related: ['diffusion-model', 'stable-diffusion', 'imagen', 'multimodal']
  },
  'imagen': {
    term: 'Imagen',
    definition: 'Google\'s text-to-image diffusion model that achieves high-quality image generation using a cascade of diffusion models at increasing resolutions. It demonstrates that large language models as text encoders significantly improve image quality.',
    analogy: 'Like an artist who first sketches a rough draft, then adds medium detail, then fine detail in three passes.',
    whyItMatters: 'Imagen is part of the Gemini/Google AI ecosystem. Understanding it helps position visual generation capabilities in the product strategy.',
    related: ['diffusion-model', 'text-to-image', 'gemini']
  },

  // === RAG ===
  'rag': {
    term: 'RAG (Retrieval-Augmented Generation)',
    definition: 'A system architecture that enhances LLM responses by first retrieving relevant documents from an external knowledge base, then providing them as context for generation. RAG grounds responses in factual data, reducing hallucination.',
    analogy: 'Like an open-book exam \u2014 instead of relying on memory alone, the model looks up relevant information before answering.',
    whyItMatters: 'RAG is the primary technique for building factual, up-to-date AI products. PMs must understand its components to architect reliable AI features.',
    related: ['embedding-model', 'vector-database', 'retrieval', 'grounding', 'hallucination']
  },
  'embedding-model': {
    term: 'Embedding Model',
    definition: 'A model that converts text (or other data) into dense vector representations (embeddings) where semantic similarity is captured by geometric proximity. Used to encode both queries and documents for retrieval.',
    analogy: 'Like a translator that converts sentences into coordinates on a meaning map.',
    whyItMatters: 'Embedding model quality directly impacts retrieval quality. PMs should evaluate embedding models as carefully as generation models.',
    related: ['embedding', 'vector-database', 'semantic-search', 'cosine-similarity']
  },
  'vector-database': {
    term: 'Vector Database',
    definition: 'A specialized database designed to store and efficiently search high-dimensional vectors (embeddings). It uses approximate nearest neighbor algorithms to find the most similar vectors to a query vector in milliseconds.',
    analogy: 'Like a library where books are shelved by topic similarity rather than alphabetically \u2014 related books are always nearby.',
    whyItMatters: 'Vector DBs are core infrastructure for RAG, search, and recommendations. PMs must evaluate options (Pinecone, Weaviate, Chroma) for cost, scale, and latency.',
    related: ['embedding', 'semantic-search', 'faiss', 'cosine-similarity']
  },
  'semantic-search': {
    term: 'Semantic Search',
    definition: 'Search that finds results based on meaning rather than keyword matching. It uses embeddings to understand the intent behind queries and match them to semantically similar content, even when exact words don\'t match.',
    analogy: 'Like asking a librarian for "books about heartbreak" and getting results including "grief," "loss," and "emotional pain" \u2014 not just books with "heartbreak" in the title.',
    whyItMatters: 'Semantic search powers next-gen search experiences in AI products. Understanding it helps PMs design better information retrieval features.',
    related: ['embedding', 'vector-database', 'cosine-similarity', 'rag']
  },
  'cosine-similarity': {
    term: 'Cosine Similarity',
    definition: 'A metric that measures the angle between two vectors in embedding space, ranging from -1 (opposite) to 1 (identical). It\'s the standard measure for comparing embedding similarity, ignoring magnitude and focusing on direction.',
    analogy: 'Like comparing the direction two arrows point rather than their length \u2014 arrows pointing the same way are similar.',
    whyItMatters: 'Cosine similarity thresholds determine retrieval quality. PMs should understand how to tune them for precision vs recall trade-offs.',
    related: ['embedding', 'semantic-search', 'vector-database']
  },
  'chunking': {
    term: 'Chunking',
    definition: 'The process of splitting documents into smaller segments (chunks) for embedding and retrieval. Chunk size, overlap, and strategy (sentence, paragraph, semantic) significantly impact retrieval quality.',
    analogy: 'Like cutting a long article into index cards, each covering one idea \u2014 the right card size makes finding information efficient.',
    whyItMatters: 'Chunking strategy is a critical and often underestimated design decision in RAG systems. Poor chunking = poor retrieval = poor answers.',
    related: ['rag', 'embedding', 'retrieval']
  },
  'reranking': {
    term: 'Reranking',
    definition: 'A second-stage retrieval step where a more powerful model rescores and reorders the initial retrieval results. The reranker typically uses cross-attention between query and document, producing more accurate relevance scores.',
    analogy: 'Like having a senior editor review and reorder the research assistant\'s top picks for relevance.',
    whyItMatters: 'Reranking dramatically improves RAG quality at low cost. PMs should know about this as a high-ROI optimization.',
    related: ['rag', 'retrieval', 'hybrid-search']
  },
  'hybrid-search': {
    term: 'Hybrid Search',
    definition: 'Combining keyword-based search (BM25/TF-IDF) with semantic vector search to get the benefits of both: exact match precision from keywords and semantic understanding from embeddings.',
    analogy: 'Like searching a library using both the catalog system (keywords) and asking the librarian (semantic) \u2014 together they find more relevant results.',
    whyItMatters: 'Hybrid search consistently outperforms either approach alone. PMs should advocate for it in RAG implementations.',
    related: ['semantic-search', 'rag', 'vector-database', 'reranking']
  },
  'faiss': {
    term: 'FAISS',
    definition: 'Facebook AI Similarity Search \u2014 an open-source library for efficient similarity search and clustering of dense vectors. It supports GPU acceleration and can search billions of vectors in milliseconds.',
    analogy: 'Like a high-speed sorting machine that can instantly find the most similar item in a warehouse of billions.',
    whyItMatters: 'FAISS is the industry-standard vector search library. Understanding its capabilities helps PMs evaluate build-vs-buy for retrieval infrastructure.',
    related: ['vector-database', 'semantic-search', 'embedding']
  },
  'retrieval': {
    term: 'Retrieval',
    definition: 'The process of finding and fetching relevant information from a knowledge base in response to a query. Modern retrieval combines sparse methods (keyword matching) and dense methods (semantic similarity) for optimal results.',
    analogy: 'Like a research assistant who, given a question, quickly finds the most relevant passages from a library of documents.',
    whyItMatters: 'Retrieval quality directly determines AI response quality in RAG systems. "Garbage in, garbage out" applies even more to retrieval.',
    related: ['rag', 'semantic-search', 'hybrid-search', 'reranking']
  },
  'knowledge-base': {
    term: 'Knowledge Base',
    definition: 'A structured or unstructured collection of information that serves as the ground truth for an AI system. In RAG, it\'s the document corpus that gets embedded and searched against user queries.',
    analogy: 'Like the reference library an expert consults \u2014 the quality and coverage of the library determines the quality of answers.',
    whyItMatters: 'Building and maintaining high-quality knowledge bases is a core PM responsibility. Stale or incomplete knowledge = bad user experience.',
    related: ['rag', 'retrieval', 'grounding']
  },
  'grounding': {
    term: 'Grounding',
    definition: 'Connecting LLM outputs to verifiable sources of truth, ensuring responses are based on real data rather than model hallucination. Techniques include RAG, tool use, and citation generation.',
    analogy: 'Like requiring every claim in a news article to have a cited source \u2014 grounding prevents making things up.',
    whyItMatters: 'Grounding is essential for trustworthy AI products. PMs must design systems that balance fluency with factual accuracy.',
    related: ['rag', 'hallucination', 'retrieval']
  },

  // === Product Management ===
  'product-roadmap': {
    term: 'Product Roadmap',
    definition: 'A strategic document that outlines the vision, direction, priorities, and planned deliverables for a product over time. For AI products, roadmaps must account for model capability uncertainty and rapid technological change.',
    analogy: 'Like a GPS route for a product \u2014 it shows the destination and planned stops, but may need rerouting based on conditions.',
    whyItMatters: 'AI roadmapping is uniquely challenging because capabilities emerge unpredictably. PMs must build flexible roadmaps that can adapt.',
    related: ['okrs', 'agile', 'mvp']
  },
  'okrs': {
    term: 'OKRs (Objectives and Key Results)',
    definition: 'A goal-setting framework where ambitious Objectives (what you want to achieve) are paired with measurable Key Results (how you know you\'ve achieved it). Widely used at Google to align teams and track progress.',
    analogy: 'Like setting a destination (objective) and GPS coordinates for each milestone (key results).',
    whyItMatters: 'Google runs on OKRs. PMs at DeepMind must excel at writing clear, measurable OKRs that connect AI capabilities to business impact.',
    related: ['kpis', 'north-star-metric', 'product-roadmap']
  },
  'kpis': {
    term: 'KPIs (Key Performance Indicators)',
    definition: 'Quantitative metrics that measure a product\'s health and progress toward goals. For AI products, KPIs must capture both technical performance (latency, accuracy) and user value (task completion, satisfaction).',
    analogy: 'Like the dashboard gauges in a car \u2014 speed, fuel, temperature tell you how the car is performing at a glance.',
    whyItMatters: 'Choosing the right KPIs for AI products is nuanced. Optimizing for the wrong metric can harm user experience.',
    related: ['okrs', 'north-star-metric', 'a-b-testing']
  },
  'a-b-testing': {
    term: 'A/B Testing',
    definition: 'A controlled experiment where users are randomly split into groups that see different versions of a feature to measure causal impact on metrics. For AI, this includes testing different models, prompts, or UX treatments.',
    analogy: 'Like a taste test where half the people try recipe A and half try recipe B, then you compare which one they prefer.',
    whyItMatters: 'A/B testing is essential for data-driven AI product decisions. PMs must design experiments that account for AI-specific challenges like user adaptation effects.',
    related: ['kpis', 'user-research', 'regression-testing']
  },
  'user-research': {
    term: 'User Research',
    definition: 'Systematic study of target users through interviews, surveys, usability tests, and data analysis to understand their needs, behaviors, and pain points. For AI products, this includes studying mental models of AI capabilities.',
    analogy: 'Like a doctor taking a thorough patient history before prescribing treatment \u2014 understanding the problem before solving it.',
    whyItMatters: 'AI products require novel user research methods since users don\'t know what AI can do. PMs must discover needs users can\'t articulate.',
    related: ['product-market-fit', 'mvp', 'a-b-testing']
  },
  'mvp': {
    term: 'MVP (Minimum Viable Product)',
    definition: 'The simplest version of a product that delivers enough value to early users and provides feedback for future development. For AI, this might be a limited-scope AI feature rather than full autonomy.',
    analogy: 'Like serving a food sample at a restaurant to test if customers like the flavor before adding it to the full menu.',
    whyItMatters: 'AI MVPs are tricky because model quality thresholds determine viability. A chatbot that\'s wrong 40% of the time isn\'t an MVP, it\'s a bad product.',
    related: ['product-market-fit', 'agile', 'user-research']
  },
  'agile': {
    term: 'Agile',
    definition: 'An iterative development methodology emphasizing short cycles (sprints), continuous feedback, and adaptability. AI development requires modified agile practices to account for data dependencies and model training cycles.',
    analogy: 'Like navigating a maze by making many small moves and adjusting course, rather than planning the entire path upfront.',
    whyItMatters: 'Standard agile doesn\'t fully work for AI projects (training takes weeks, not days). PMs must adapt the methodology.',
    related: ['sprint', 'mvp', 'product-roadmap']
  },
  'sprint': {
    term: 'Sprint',
    definition: 'A fixed time period (usually 1-2 weeks) during which a specific set of work must be completed. In AI teams, sprint planning must account for experiment cycles and model training durations.',
    analogy: 'Like a focused work session with a clear deadline and defined deliverables.',
    whyItMatters: 'Sprint planning for AI teams requires balancing feature development with research exploration and model iteration.',
    related: ['agile', 'product-roadmap']
  },
  'stakeholder': {
    term: 'Stakeholder',
    definition: 'Any person or group with an interest in or influence over a product\'s direction. For AI products at DeepMind, this includes engineering, research, UX, legal, policy, marketing, and executive leadership.',
    analogy: 'Like all the people who have a say in designing a public building \u2014 architects, residents, city planners, safety inspectors.',
    whyItMatters: 'AI products have unusually many stakeholders (including ethics/policy teams). PMs must navigate complex cross-functional dynamics.',
    related: ['okrs', 'product-roadmap']
  },
  'go-to-market': {
    term: 'Go-to-Market (GTM)',
    definition: 'The strategy for launching a product to market, including positioning, pricing, distribution channels, and marketing. AI GTM must manage user expectations, communicate limitations, and plan for iterative improvement.',
    analogy: 'Like a restaurant\'s opening strategy \u2014 choosing location, menu, pricing, marketing, and how to handle opening night.',
    whyItMatters: 'AI product launches require unique GTM strategies that manage the gap between demos and real-world performance.',
    related: ['product-market-fit', 'product-roadmap']
  },
  'product-market-fit': {
    term: 'Product-Market Fit',
    definition: 'The degree to which a product satisfies strong market demand. For AI products, PMF requires not just utility but sufficient reliability, speed, and trust. A product can be technically impressive but lack PMF.',
    analogy: 'Like finding the perfect key for a lock \u2014 the product (key) must perfectly fit the market need (lock).',
    whyItMatters: 'PMF for AI is harder to achieve because user expectations are shaped by demos and hype, not by what current AI can reliably do.',
    related: ['mvp', 'user-research', 'go-to-market']
  },
  'north-star-metric': {
    term: 'North Star Metric',
    definition: 'The single metric that best captures the core value a product delivers to users. All team efforts should ultimately drive this metric. For an AI assistant, this might be "tasks successfully completed per user per week."',
    analogy: 'Like a compass that always points to true north \u2014 it ensures every team member is aligned on what matters most.',
    whyItMatters: 'Choosing the right North Star for AI products prevents optimizing for vanity metrics (usage) over real value (task completion, accuracy).',
    related: ['kpis', 'okrs']
  },
  'funnel-analysis': {
    term: 'Funnel Analysis',
    definition: 'Tracking user progression through a sequence of steps (acquisition \u2192 activation \u2192 retention \u2192 revenue) to identify where users drop off and optimize conversion. AI products have unique funnels including trust and capability discovery.',
    analogy: 'Like tracking water through a series of filters \u2014 you measure how much makes it through each stage and fix the leakiest points.',
    whyItMatters: 'AI product funnels include unique stages like "first successful AI interaction" and "trust calibration" that traditional products don\'t have.',
    related: ['kpis', 'a-b-testing', 'user-research']
  },

  // === AI Metrics ===
  'precision': {
    term: 'Precision',
    definition: 'The proportion of positive predictions that are actually correct: TP / (TP + FP). High precision means few false positives \u2014 when the model says "yes," it\'s usually right.',
    analogy: 'Like a spam filter that rarely marks real email as spam \u2014 when it flags something, you trust it.',
    whyItMatters: 'Precision matters when false positives are costly (e.g., wrongly blocking user content). PMs must choose the right precision/recall balance.',
    related: ['recall', 'f1-score', 'accuracy']
  },
  'recall': {
    term: 'Recall',
    definition: 'The proportion of actual positives that the model correctly identifies: TP / (TP + FN). High recall means few false negatives \u2014 the model catches most positive cases.',
    analogy: 'Like a search-and-rescue team that finds 95% of survivors \u2014 they miss very few.',
    whyItMatters: 'Recall matters when missing positives is costly (e.g., failing to detect harmful content). PMs trade off recall vs precision based on product needs.',
    related: ['precision', 'f1-score', 'accuracy']
  },
  'f1-score': {
    term: 'F1 Score',
    definition: 'The harmonic mean of precision and recall, providing a single metric that balances both. F1 = 2 \u00D7 (precision \u00D7 recall) / (precision + recall). Useful when you need a single number to compare models.',
    analogy: 'Like a student\'s GPA that balances performance across multiple subjects into one number.',
    whyItMatters: 'F1 is the go-to balanced metric for classification tasks. PMs should use it but also understand when precision or recall alone matters more.',
    related: ['precision', 'recall', 'accuracy']
  },
  'accuracy': {
    term: 'Accuracy',
    definition: 'The proportion of all predictions that are correct: (TP + TN) / total. While intuitive, accuracy can be misleading with imbalanced classes (99% accuracy on a 99/1 split is trivial).',
    analogy: 'Like a weather forecaster who\'s right 90% of the time \u2014 sounds good, but what if it only rains 10% of the time?',
    whyItMatters: 'PMs must look beyond accuracy to understand model performance. A model with 95% accuracy might still fail on the cases that matter most.',
    related: ['precision', 'recall', 'f1-score', 'auc-roc']
  },
  'auc-roc': {
    term: 'AUC-ROC',
    definition: 'Area Under the Receiver Operating Characteristic curve \u2014 a metric measuring classification performance across all possible thresholds. AUC of 1.0 is perfect; 0.5 is random guessing. Threshold-independent, making it useful for comparing models.',
    analogy: 'Like rating a goalkeeper not on one game but on their performance across all possible difficulty levels.',
    whyItMatters: 'AUC-ROC helps PMs understand model quality independent of threshold choices, enabling better product decisions about sensitivity settings.',
    related: ['precision', 'recall', 'accuracy']
  },
  'perplexity': {
    term: 'Perplexity',
    definition: 'A metric measuring how well a language model predicts text \u2014 lower is better. Mathematically, it\'s 2^(cross-entropy loss). A perplexity of 10 means the model is as uncertain as choosing between 10 equally likely options at each step.',
    analogy: 'Like measuring how "surprised" a model is by text \u2014 good models are rarely surprised by natural language.',
    whyItMatters: 'Perplexity is the standard LLM pre-training metric. PMs should understand it to evaluate base model quality, though it doesn\'t directly measure usefulness.',
    related: ['llm', 'pre-training', 'bleu']
  },
  'bleu': {
    term: 'BLEU Score',
    definition: 'Bilingual Evaluation Understudy \u2014 a metric that measures translation/generation quality by comparing n-gram overlap between generated text and reference texts. Ranges from 0 to 1. Widely used but has known limitations.',
    analogy: 'Like grading an essay by counting how many phrases match a model answer \u2014 useful but doesn\'t capture meaning perfectly.',
    whyItMatters: 'BLEU is widely reported but PMs should know its limitations: high BLEU doesn\'t guarantee good quality, and vice versa.',
    related: ['rouge', 'perplexity', 'human-evaluation']
  },
  'rouge': {
    term: 'ROUGE Score',
    definition: 'Recall-Oriented Understudy for Gisting Evaluation \u2014 metrics (ROUGE-1, ROUGE-2, ROUGE-L) that measure overlap between generated and reference summaries. ROUGE-L uses longest common subsequence.',
    analogy: 'Like checking how many key points from the original article appear in a student\'s summary.',
    whyItMatters: 'ROUGE is standard for evaluating summarization and information extraction. PMs should pair it with human evaluation for reliable quality assessment.',
    related: ['bleu', 'human-evaluation', 'perplexity']
  },
  'human-evaluation': {
    term: 'Human Evaluation',
    definition: 'Having humans judge AI output quality along dimensions like helpfulness, accuracy, harmlessness, and fluency. Despite being expensive and slow, it remains the gold standard because automated metrics only approximate human judgment.',
    analogy: 'Like having professional food critics evaluate a restaurant instead of relying solely on ingredient analysis.',
    whyItMatters: 'Human eval is the ultimate quality measure. PMs must design efficient human evaluation pipelines and know when automated metrics are insufficient.',
    related: ['rlhf', 'bleu', 'rouge', 'labels']
  },
  'latency': {
    term: 'Latency',
    definition: 'The time between a user sending a request and receiving a response. For AI products, this includes model inference time, retrieval time, and network overhead. Usually measured at p50 (median), p95, and p99 percentiles.',
    analogy: 'Like the wait time at a restaurant \u2014 even if most customers get served quickly, a few long waits ruin the experience.',
    whyItMatters: 'Latency directly impacts user experience and retention. PMs must set latency budgets and understand trade-offs with model quality.',
    related: ['throughput', 'token']
  },
  'throughput': {
    term: 'Throughput',
    definition: 'The number of requests a system can handle per unit time. For AI systems, this includes tokens per second for generation and queries per second for retrieval. Higher throughput = lower cost per query.',
    analogy: 'Like a factory\'s production rate \u2014 how many units it can produce per hour.',
    whyItMatters: 'Throughput determines serving costs and scalability. PMs must balance throughput with quality when choosing model sizes and infrastructure.',
    related: ['latency', 'token']
  },
  'regression-testing': {
    term: 'Regression Testing',
    definition: 'Testing to ensure that model updates or system changes haven\'t degraded performance on previously working capabilities. For AI, this includes running evaluation suites across model versions and monitoring for quality drops on specific task categories.',
    analogy: 'Like a doctor checking that a new medication doesn\'t create new problems while fixing the original one.',
    whyItMatters: 'Model regressions are a critical risk. PMs must ensure comprehensive regression testing is built into the deployment pipeline to catch degradation before users do.',
    related: ['model-degradation', 'a-b-testing', 'guardrails']
  },
  'model-degradation': {
    term: 'Model Degradation',
    definition: 'The gradual decline in model performance over time, caused by data distribution shift, concept drift, or changes in user behavior. Production models can slowly become less accurate without any code changes.',
    analogy: 'Like a map that becomes increasingly inaccurate as roads are built and cities change \u2014 the world moves but the model stays frozen.',
    whyItMatters: 'PMs must monitor for degradation and plan model refresh cycles. Ignoring it leads to gradually worsening user experience that\'s hard to detect.',
    related: ['regression-testing', 'a-b-testing', 'schema-drift']
  },
  'guardrails': {
    term: 'Guardrails',
    definition: 'Safety mechanisms that constrain AI system behavior within acceptable bounds. Includes input filters (blocking harmful prompts), output filters (catching dangerous responses), and behavioral constraints (refusing certain requests).',
    analogy: 'Like the bumpers in a bowling lane \u2014 they prevent the ball (AI output) from going into the gutter (harmful territory).',
    whyItMatters: 'Guardrails are essential for responsible AI products. PMs must balance safety (more guardrails) with capability (fewer restrictions). Too strict = useless; too loose = dangerous.',
    related: ['content-policy', 'red-teaming', 'alignment', 'responsible-ai']
  },
  'red-teaming': {
    term: 'Red Teaming',
    definition: 'Systematic adversarial testing where dedicated teams try to make an AI system produce harmful, biased, or incorrect outputs. Red teamers use creative attack strategies to find vulnerabilities before users do.',
    analogy: 'Like hiring professional burglars to test your home security system before real criminals try.',
    whyItMatters: 'Red teaming is critical for AI safety and a DeepMind priority. PMs must champion red teaming and act on findings before product launch.',
    related: ['guardrails', 'adversarial-attack', 'responsible-ai', 'content-policy']
  },
  'content-policy': {
    term: 'Content Policy',
    definition: 'Rules defining what an AI system can and cannot generate, covering areas like hate speech, violence, sexual content, misinformation, and personal information. Policies must be nuanced enough to allow legitimate use while preventing harm.',
    analogy: 'Like a TV network\'s broadcast standards \u2014 they define what\'s acceptable to air, balancing freedom of expression with public responsibility.',
    whyItMatters: 'Content policies are high-stakes PM decisions. Too restrictive = users leave; too permissive = PR crises and real harm.',
    related: ['guardrails', 'red-teaming', 'responsible-ai']
  },

  // === SDK/Platform ===
  'api': {
    term: 'API (Application Programming Interface)',
    definition: 'A set of defined protocols and tools that allow different software systems to communicate. For AI products, APIs enable developers to integrate AI capabilities (generation, analysis, search) into their own applications.',
    analogy: 'Like a restaurant menu \u2014 it defines what\'s available, how to order it, and what you\'ll get back, without requiring you to know how the kitchen works.',
    whyItMatters: 'API design determines developer adoption and ecosystem growth. For the Gemini SDK, the API IS the product for developers.',
    related: ['sdk', 'rest-api', 'endpoint', 'rate-limiting']
  },
  'sdk': {
    term: 'SDK (Software Development Kit)',
    definition: 'A collection of tools, libraries, documentation, and code samples that make it easier for developers to build on a platform. SDKs abstract away API complexity and provide language-specific interfaces.',
    analogy: 'Like a complete IKEA toolkit that comes with the furniture \u2014 it has everything you need to assemble the product.',
    whyItMatters: 'SDK quality directly impacts developer adoption. The Gemini SDK is a core product for the DeepMind PM role.',
    related: ['api', 'developer-experience', 'documentation']
  },
  'rest-api': {
    term: 'REST API',
    definition: 'An API architectural style using HTTP methods (GET, POST, PUT, DELETE) to interact with resources identified by URLs. RESTful APIs are stateless, cacheable, and the most common web API pattern.',
    analogy: 'Like the postal system \u2014 you send requests to specific addresses using standard methods and get responses back.',
    whyItMatters: 'Most AI APIs (including Gemini) are REST-based. PMs should understand REST conventions to evaluate API design quality.',
    related: ['api', 'graphql', 'endpoint']
  },
  'graphql': {
    term: 'GraphQL',
    definition: 'An API query language that lets clients request exactly the data they need in a single request, avoiding over-fetching and under-fetching. It uses a typed schema that describes all available data.',
    analogy: 'Like ordering from a buffet where you specify exactly what and how much of each item, rather than getting a fixed plate.',
    whyItMatters: 'GraphQL can reduce API calls for complex AI product UIs. PMs should evaluate whether REST or GraphQL better serves their developer audience.',
    related: ['rest-api', 'api', 'sdk']
  },
  'developer-experience': {
    term: 'Developer Experience (DX)',
    definition: 'The overall experience developers have when using a platform, including API design, documentation quality, SDK usability, error messages, onboarding flow, and community support. Good DX drives adoption.',
    analogy: 'Like the user experience (UX) but for developers \u2014 if building with your platform feels painful, developers will choose competitors.',
    whyItMatters: 'DX is the competitive moat for developer platforms. PMs must obsess over every friction point in the developer journey.',
    related: ['sdk', 'documentation', 'api']
  },
  'documentation': {
    term: 'Documentation',
    definition: 'Written resources (guides, references, tutorials, examples) that teach developers how to use a platform. For AI APIs, documentation must cover not just technical specs but also best practices, limitations, and prompt engineering guides.',
    analogy: 'Like a comprehensive cookbook that not only has recipes but explains techniques, ingredient substitutions, and common mistakes.',
    whyItMatters: 'Documentation is the #1 factor in developer adoption. PMs should treat docs as a product, not an afterthought.',
    related: ['developer-experience', 'sdk']
  },
  'rate-limiting': {
    term: 'Rate Limiting',
    definition: 'Controlling the number of API requests a client can make within a time window to prevent abuse, manage costs, and ensure fair resource allocation. For AI APIs, rate limits also manage expensive GPU compute.',
    analogy: 'Like a speed limit on a highway \u2014 it keeps the road safe and functional for everyone by preventing any one driver from going too fast.',
    whyItMatters: 'Rate limits directly impact developer experience and revenue. PMs must balance access (higher limits) with cost management and fairness.',
    related: ['api', 'endpoint', 'throughput']
  },
  'authentication': {
    term: 'Authentication',
    definition: 'The process of verifying the identity of an API user, typically through API keys, OAuth tokens, or JWTs. It controls access and enables usage tracking and billing.',
    analogy: 'Like showing your ID at a door \u2014 it proves you are who you claim to be before granting access.',
    whyItMatters: 'Auth design affects developer onboarding friction. Too complex = developers bounce. Too simple = security risks.',
    related: ['oauth', 'api', 'sdk']
  },
  'oauth': {
    term: 'OAuth',
    definition: 'An authorization framework that lets users grant third-party apps limited access to their accounts without sharing passwords. Essential for AI products that need to access user data (emails, calendar, files).',
    analogy: 'Like giving a valet your car key but not your house key \u2014 limited access for a specific purpose.',
    whyItMatters: 'OAuth is how Gemini accesses Google services on behalf of users. Understanding it is essential for integration features.',
    related: ['authentication', 'api']
  },
  'webhook': {
    term: 'Webhook',
    definition: 'A mechanism where an API sends real-time notifications to your server when events occur, instead of you repeatedly checking (polling). It\'s an "event-driven" approach to integration.',
    analogy: 'Like having a doorbell instead of repeatedly checking if someone is at the door.',
    whyItMatters: 'Webhooks enable real-time AI features and integrations. PMs should understand when webhooks vs polling is appropriate.',
    related: ['api', 'endpoint']
  },
  'endpoint': {
    term: 'Endpoint',
    definition: 'A specific URL where an API receives requests. Each endpoint typically handles a specific function (e.g., /generate for text generation, /embed for embeddings). Endpoint design is a key API architecture decision.',
    analogy: 'Like different departments in a building \u2014 each door (endpoint) leads to a specific service.',
    whyItMatters: 'Endpoint design affects API usability and versioning strategy. PMs should review and approve endpoint design decisions.',
    related: ['api', 'rest-api']
  },

  // === AI Safety ===
  'interpretability': {
    term: 'Interpretability',
    definition: 'The degree to which humans can understand how an AI system reaches its decisions. Includes techniques like attention visualization, feature attribution, and mechanistic interpretability (understanding individual neurons/circuits).',
    analogy: 'Like having a glass-walled kitchen in a restaurant \u2014 customers can see exactly how their food is prepared.',
    whyItMatters: 'Interpretability is crucial for debugging, trust, and compliance. DeepMind is a leader in this research area.',
    related: ['alignment', 'responsible-ai', 'model-card']
  },
  'robustness': {
    term: 'Robustness',
    definition: 'An AI system\'s ability to maintain performance when faced with unexpected inputs, adversarial attacks, or distribution shifts. Robust models degrade gracefully rather than failing catastrophically.',
    analogy: 'Like a bridge designed to handle not just normal traffic but also earthquakes and extreme weather.',
    whyItMatters: 'Robustness determines whether an AI product works reliably in the real world, not just in controlled demos.',
    related: ['adversarial-attack', 'graceful-degradation', 'alignment']
  },
  'adversarial-attack': {
    term: 'Adversarial Attack',
    definition: 'Carefully crafted inputs designed to fool AI models into making incorrect predictions. Examples include imperceptible image perturbations that cause misclassification, or prompt injections that bypass LLM safety filters.',
    analogy: 'Like optical illusions that fool human perception \u2014 small, carefully designed changes that exploit the system\'s blind spots.',
    whyItMatters: 'Adversarial attacks are a real threat to deployed AI products. PMs must plan for them in security and safety design.',
    related: ['robustness', 'red-teaming', 'guardrails']
  },
  'bias-fairness': {
    term: 'Bias & Fairness',
    definition: 'AI systems can perpetuate or amplify societal biases present in training data. Fairness requires ensuring equitable performance across demographic groups (gender, race, age, language). Multiple mathematical definitions of fairness exist, and they can conflict.',
    analogy: 'Like a hiring process that inadvertently favors certain backgrounds because the training examples were biased.',
    whyItMatters: 'Bias issues can cause serious harm and PR crises. PMs must champion fairness evaluation and diverse training data.',
    related: ['responsible-ai', 'training-data', 'model-card']
  },
  'eu-ai-act': {
    term: 'EU AI Act',
    definition: 'The European Union\'s comprehensive AI regulation framework that categorizes AI systems by risk level (unacceptable, high, limited, minimal) and imposes requirements including transparency, human oversight, and documentation.',
    analogy: 'Like food safety regulations that require ingredient labels and safety testing before products can be sold.',
    whyItMatters: 'The EU AI Act directly affects products launched in Europe. PMs at DeepMind (Zurich!) must ensure compliance.',
    related: ['responsible-ai', 'model-card', 'data-governance']
  },
  'responsible-ai': {
    term: 'Responsible AI',
    definition: 'An approach to developing and deploying AI that prioritizes fairness, transparency, privacy, security, and human welfare. It includes governance frameworks, ethical review processes, and impact assessments.',
    analogy: 'Like environmental sustainability in manufacturing \u2014 it\'s not just about making the product work, but making it work responsibly.',
    whyItMatters: 'Responsible AI is a core value at DeepMind. PMs must embed responsible practices throughout the product lifecycle, not just as a checkbox.',
    related: ['alignment', 'bias-fairness', 'eu-ai-act', 'model-card']
  },
  'model-card': {
    term: 'Model Card',
    definition: 'A standardized documentation framework for ML models that describes intended use, performance across demographics, limitations, ethical considerations, and training data details. Proposed by Google researchers.',
    analogy: 'Like a nutrition label for AI \u2014 it tells you what\'s inside, how it was made, and any potential allergens.',
    whyItMatters: 'Model cards are increasingly required for transparency. PMs should champion their creation for every shipped model.',
    related: ['responsible-ai', 'bias-fairness', 'documentation']
  },
  'data-governance': {
    term: 'Data Governance',
    definition: 'Policies and processes for managing data quality, privacy, security, access, and compliance throughout the data lifecycle. For AI, this includes training data provenance, consent, and right-to-deletion compliance.',
    analogy: 'Like a bank\'s vault protocols \u2014 strict rules about who can access what data, when, and how it\'s protected.',
    whyItMatters: 'Data governance is a legal requirement and trust foundation. PMs must ensure training data practices are defensible.',
    related: ['responsible-ai', 'eu-ai-act', 'training-data']
  },

  // === Gemini/DeepMind ===
  'gemini': {
    term: 'Gemini',
    definition: 'Google DeepMind\'s flagship multimodal AI model family, natively trained on text, code, images, audio, and video. Available in Ultra (most capable), Pro (balanced), and Flash (fastest) tiers. Features industry-leading context windows (1M+ tokens).',
    analogy: 'Like a polyglot genius who can simultaneously read, listen, watch, and analyze \u2014 not by translating between modes, but by understanding them natively.',
    whyItMatters: 'Gemini IS the product for this PM role. Deep understanding of its capabilities, tiers, and positioning is essential.',
    related: ['multimodal', 'long-context', 'llm', 'transformer']
  },
  'multimodal': {
    term: 'Multimodal',
    definition: 'AI systems that can process and generate multiple types of data (text, images, audio, video, code) within a single model. Gemini\'s key differentiator is being natively multimodal \u2014 trained on all modalities jointly, not as separate modules bolted together.',
    analogy: 'Like a person who can see, hear, read, and speak fluently \u2014 all senses integrated naturally, not through separate translators.',
    whyItMatters: 'Multimodality enables richer AI products (analyze images, understand video, read documents). PMs should identify high-value multimodal use cases.',
    related: ['gemini', 'transformer', 'text-to-image']
  },
  'long-context': {
    term: 'Long Context',
    definition: 'The ability to process very long input sequences (100K to 1M+ tokens). Enables analyzing entire books, codebases, or video transcripts in a single pass. Gemini\'s 1M+ token context window is a key competitive advantage.',
    analogy: 'Like having a photographic memory for an entire library versus only remembering the last paragraph you read.',
    whyItMatters: 'Long context unlocks use cases impossible with short-context models: full codebase analysis, long document Q&A, multi-hour video understanding.',
    related: ['context-window', 'gemini', 'positional-encoding']
  },
  'alphago': {
    term: 'AlphaGo',
    definition: 'DeepMind\'s Go-playing AI that defeated world champion Lee Sedol in 2016, a landmark moment in AI history. It combined deep neural networks with Monte Carlo tree search and reinforcement learning. Its successor AlphaZero learned from pure self-play.',
    analogy: 'Like a chess/Go student who became a grandmaster by playing millions of games against itself in a few days.',
    whyItMatters: 'AlphaGo established DeepMind\'s reputation and demonstrated RL\'s power. Understanding this history shows cultural awareness in interviews.',
    related: ['reinforcement-learning', 'gemini', 'alphafold']
  },
  'alphafold': {
    term: 'AlphaFold',
    definition: 'DeepMind\'s AI system that solved the 50-year-old protein structure prediction problem, accurately predicting 3D protein structures from amino acid sequences. It has predicted structures for virtually all known proteins, revolutionizing biology.',
    analogy: 'Like an architect who can look at a list of building materials and instantly know the exact shape of the resulting building.',
    whyItMatters: 'AlphaFold demonstrates DeepMind\'s mission of using AI for scientific breakthroughs. It\'s the strongest example of AI\'s transformative potential.',
    related: ['alphago', 'deep-learning', 'gemini']
  },
  'mixture-of-experts': {
    term: 'Mixture of Experts (MoE)',
    definition: 'An architecture where only a subset of model parameters are activated for each input, routed by a gating network. This enables models with very large total parameter counts but efficient inference, since only a fraction of experts process each token.',
    analogy: 'Like a hospital with many specialists \u2014 each patient sees only the relevant doctors, not the entire staff.',
    whyItMatters: 'MoE enables scaling model capacity without proportionally increasing compute costs. Gemini reportedly uses MoE, making it efficient despite its size.',
    related: ['transformer', 'gemini', 'scaling-laws']
  },

  // === Real-world PM challenges ===
  'graceful-degradation': {
    term: 'Graceful Degradation',
    definition: 'A system design principle where partial failures result in reduced functionality rather than complete system failure. For AI products integrating with third-party platforms (like Instagram), the system should handle API changes, missing data, or service outages by falling back to simpler behaviors.',
    analogy: 'Like a car that switches to lower gears when the engine struggles rather than stopping completely \u2014 reduced performance is better than no performance.',
    whyItMatters: 'Critical for Gemini\'s integrated assistance: when apps like Instagram change their UI or API, features should degrade gracefully rather than break entirely. PMs must design fallback strategies.',
    related: ['robustness', 'schema-drift', 'api-versioning', 'screen-context']
  },
  'schema-drift': {
    term: 'Schema Drift',
    definition: 'When the structure of data sources changes over time \u2014 fields added, removed, renamed, or reformatted. This breaks downstream systems that depend on the old schema. Especially common with third-party APIs and web scraping.',
    analogy: 'Like a form that keeps changing its layout \u2014 the software filling it in breaks every time fields move.',
    whyItMatters: 'Gemini\'s screen-reading features depend on parsing app layouts that change frequently (Instagram, Twitter redesigns). PMs must plan for schema drift with monitoring, versioning, and generic parsers.',
    related: ['graceful-degradation', 'api-versioning', 'model-degradation']
  },
  'api-versioning': {
    term: 'API Versioning',
    definition: 'Strategies for managing breaking changes in APIs, including URL versioning (v1/v2), header versioning, and deprecation policies. Proper versioning lets developers migrate at their own pace while the platform evolves.',
    analogy: 'Like maintaining both old and new highway exits during construction \u2014 drivers can use either until the transition is complete.',
    whyItMatters: 'For the Gemini SDK PM role: API versioning decisions directly impact developer trust and ecosystem stability. Breaking changes without proper versioning = developer churn.',
    related: ['api', 'sdk', 'developer-experience', 'graceful-degradation']
  },
  'screen-context': {
    term: 'Screen Context',
    definition: 'Understanding what\'s currently displayed on a user\'s screen (active app, visible UI elements, text, images) to provide contextual AI assistance. This involves screen parsing, OCR, UI element recognition, and semantic understanding of app state.',
    analogy: 'Like an assistant who can look over your shoulder and understand what you\'re working on to offer relevant help.',
    whyItMatters: 'Screen context is a core Gemini feature on Android/iOS. The challenge is that app UIs change constantly, making reliable parsing a moving target.',
    related: ['gemini', 'multimodal', 'graceful-degradation', 'schema-drift']
  },

  // === Additional terms (batch 2) ===

  'artificial-intelligence': {
    term: 'Artificial Intelligence',
    definition: 'The science and engineering of creating machines that can perform tasks normally requiring human intelligence — perceiving, reasoning, learning, planning, and generating language. Encompasses multiple paradigms from symbolic reasoning to neural networks.',
    analogy: 'Like building a brain out of math — not replicating biology, but achieving similar cognitive outcomes through computation.',
    whyItMatters: 'As a PM at DeepMind, you\'re literally building AI products. Understanding the breadth and limits of AI shapes every product decision you make.',
    related: ['machine-learning', 'deep-learning', 'agi', 'narrow-ai']
  },
  'agi': {
    term: 'Artificial General Intelligence (AGI)',
    definition: 'A hypothetical AI system with human-level generality across cognitive domains — able to learn any intellectual task a human can. No AGI system exists today; all current AI is narrow.',
    analogy: 'Like the difference between a calculator (narrow) and a human brain (general) — AGI would handle any problem, not just the ones it was trained for.',
    whyItMatters: 'AGI is DeepMind\'s stated long-term mission. Understanding what AGI means, how far we are, and what the risks are is essential for the role.',
    related: ['artificial-intelligence', 'narrow-ai', 'alignment']
  },
  'artificial-general-intelligence': {
    term: 'Artificial General Intelligence',
    definition: 'See AGI — a system capable of understanding or learning any intellectual task that a human being can, with the flexibility to transfer knowledge across domains.',
    analogy: 'Like a polymath who can excel at any field they turn their attention to, rather than a specialist who only knows one thing.',
    whyItMatters: 'Framing product strategy around the AGI trajectory means anticipating capability jumps that change what products are possible.',
    related: ['agi', 'narrow-ai', 'artificial-intelligence']
  },
  'narrow-ai': {
    term: 'Narrow AI',
    definition: 'AI designed and trained for a specific task or domain. Every deployed AI system today is narrow — AlphaGo cannot write poetry, GPT cannot fold proteins. Also called "Weak AI."',
    analogy: 'Like a chess grandmaster who can\'t drive a car — brilliant in one domain, helpless outside it.',
    whyItMatters: 'All AI products you\'ll ship are narrow AI. Knowing the boundaries prevents overpromising to users and stakeholders.',
    related: ['artificial-intelligence', 'agi', 'foundation-model']
  },
  'ai-effect': {
    term: 'AI Effect',
    definition: 'The tendency for people to redefine "real AI" as whatever machines cannot yet do. Once a capability becomes routine (OCR, chess, route planning), people stop considering it AI.',
    analogy: 'Like moving goalposts — no matter what AI achieves, the definition of "true intelligence" retreats just beyond its reach.',
    whyItMatters: 'Helps explain why users undervalue current AI capabilities. As a PM, managing the gap between perceived and actual intelligence is a design challenge.',
    related: ['artificial-intelligence', 'turing-test']
  },
  'ai-winter': {
    term: 'AI Winter',
    definition: 'A period of reduced funding and interest in AI research, typically following a cycle of overhyped promises that failed to materialize. Two major AI winters occurred: 1974-1980 and 1987-1993.',
    analogy: 'Like a stock market crash — a bubble of excitement inflates, reality can\'t match expectations, and investment freezes.',
    whyItMatters: 'Recognizing hype cycles prevents over-promising in product launches. The gap between expectations and capability — not bad technology — causes winters.',
    related: ['artificial-intelligence', 'expert-systems', 'symbolic-ai']
  },
  'turing-test': {
    term: 'Turing Test',
    definition: 'A test proposed by Alan Turing in 1950 where a human judge converses with both a machine and a human. If the judge cannot reliably distinguish them, the machine is said to exhibit intelligent behavior.',
    analogy: 'Like a blindfolded taste test for intelligence — can you tell which responses are human and which are machine?',
    whyItMatters: 'Frames the question of "can machines think?" as a behavioral benchmark rather than a philosophical debate. Modern chatbots regularly pass informal versions.',
    related: ['artificial-intelligence', 'ai-effect']
  },
  'symbolic-ai': {
    term: 'Symbolic AI',
    definition: 'An approach to AI based on explicit rules, logic, and human-readable representations. Dominant from the 1950s-1980s, it involves manually encoding knowledge as if-then rules rather than learning from data.',
    analogy: 'Like programming a robot by writing an instruction manual for every possible situation, versus letting it learn from experience.',
    whyItMatters: 'Understanding symbolic AI\'s limitations (brittleness, inability to learn from data) explains why modern ML approaches won and how hybrid approaches may return.',
    related: ['expert-systems', 'machine-learning', 'artificial-intelligence']
  },
  'expert-systems': {
    term: 'Expert Systems',
    definition: 'AI programs from the 1980s that encoded domain expertise as if-then rules. Systems like MYCIN (medical diagnosis) delivered real value but proved brittle outside narrow domains and required expensive knowledge engineering.',
    analogy: 'Like a very thorough FAQ — incredibly useful for known questions, but completely lost when asked something not in the script.',
    whyItMatters: 'Expert systems\' failure mode — brittleness and inability to generalize — is a cautionary tale for any AI product that relies on hard-coded rules instead of learning.',
    related: ['symbolic-ai', 'ai-winter', 'knowledge-graph']
  },
  'perceptron': {
    term: 'Perceptron',
    definition: 'The simplest neural network — a single neuron that takes weighted inputs, sums them, applies an activation function, and outputs a binary decision. Invented by Frank Rosenblatt in 1958.',
    analogy: 'Like a simple voting system — each input casts a weighted vote, and the perceptron says yes or no based on the total.',
    whyItMatters: 'The perceptron is the atomic building block of all neural networks. Understanding it demystifies deep learning as "many perceptrons stacked together."',
    related: ['neural-network', 'activation-function', 'linearly-separable']
  },
  'relu': {
    term: 'ReLU (Rectified Linear Unit)',
    definition: 'An activation function f(x) = max(0, x) that outputs the input if positive, zero otherwise. Now the default activation in most deep networks because it avoids the vanishing gradient problem and is computationally cheap.',
    analogy: 'Like a gate that lets positive signals through unchanged but blocks anything negative — simple but highly effective.',
    whyItMatters: 'ReLU\'s simplicity enabled training much deeper networks. Understanding why activation functions matter helps you grasp training challenges your ML team discusses.',
    related: ['activation-function', 'vanishing-gradient', 'neural-network']
  },
  'dropout': {
    term: 'Dropout',
    definition: 'A regularization technique where random neurons are temporarily "dropped" (set to zero) during training. This prevents co-adaptation and forces the network to learn redundant representations, reducing overfitting.',
    analogy: 'Like a sports team training with random players sitting out — the remaining players must learn to cover for missing teammates.',
    whyItMatters: 'Dropout is a standard technique for preventing overfitting. When your ML team says "we added dropout," they\'re combating the model memorizing training data.',
    related: ['regularisation', 'overfitting', 'neural-network']
  },
  'svm': {
    term: 'Support Vector Machine (SVM)',
    definition: 'A classical ML algorithm that finds the optimal hyperplane separating classes in feature space, maximizing the margin between the nearest data points (support vectors). Effective for high-dimensional data with clear margins.',
    analogy: 'Like drawing the widest possible street between two neighborhoods — the road (hyperplane) is placed to maximize distance from the nearest houses on each side.',
    whyItMatters: 'SVMs were state-of-the-art before deep learning. Understanding them provides context for why neural networks won and when simpler models still suffice.',
    related: ['machine-learning', 'kernel', 'linearly-separable']
  },
  'random-forest': {
    term: 'Random Forest',
    definition: 'An ensemble method that builds many decision trees on random subsets of data and features, then combines their predictions by majority vote. Robust, interpretable, and resistant to overfitting.',
    analogy: 'Like asking 100 doctors for their opinion and going with the majority — individual doctors may be wrong, but the crowd wisdom is usually right.',
    whyItMatters: 'Random forests remain competitive for tabular data and are far more interpretable than neural networks. Sometimes the right tool for a product isn\'t deep learning.',
    related: ['decision-tree', 'boosting', 'machine-learning']
  },
  'boosting': {
    term: 'Boosting',
    definition: 'An ensemble technique that trains models sequentially, each one focusing on correcting the errors of its predecessors. The final prediction combines all models with weighted votes. XGBoost and LightGBM are popular implementations.',
    analogy: 'Like a relay team where each runner specifically practices the sections the previous runner struggled with.',
    whyItMatters: 'Boosted models dominate tabular data competitions and many production ML systems. Not everything needs a neural network.',
    related: ['gradient-boosting', 'random-forest', 'decision-tree']
  },
  'gradient-boosting': {
    term: 'Gradient Boosting',
    definition: 'A boosting variant that builds trees to predict the residual errors (gradients) of the current model. Each new tree corrects what the existing ensemble gets wrong. XGBoost, LightGBM, and CatBoost are implementations.',
    analogy: 'Like an editor who only marks what\'s wrong in a draft — each editing pass fixes the remaining mistakes.',
    whyItMatters: 'XGBoost and similar tools power fraud detection, ranking systems, and recommendation engines. Knowing when to use these vs. deep learning is a key PM skill.',
    related: ['boosting', 'decision-tree', 'random-forest']
  },
  'decision-tree': {
    term: 'Decision Tree',
    definition: 'A tree-structured model that makes predictions by learning a series of if-then splits on features. Each internal node tests a condition, each branch represents an outcome, and each leaf assigns a prediction.',
    analogy: 'Like a flowchart for diagnosis — "Is it raining? If yes, take an umbrella. If no, is it cold? If yes, take a jacket..."',
    whyItMatters: 'Decision trees are the most interpretable ML model. When stakeholders demand explainability (healthcare, finance), trees may be preferable to black-box models.',
    related: ['random-forest', 'boosting', 'machine-learning']
  },
  'linear-regression': {
    term: 'Linear Regression',
    definition: 'The simplest predictive model — fits a straight line (or hyperplane) through data to predict continuous output values by learning weights for each feature that minimize squared prediction error.',
    analogy: 'Like drawing the best-fit line through a scatter plot — the line that\'s closest to all the points on average.',
    whyItMatters: 'Linear regression is a baseline model for any prediction task. If a simple line works well enough, there\'s no reason to deploy a neural network.',
    related: ['machine-learning', 'loss-function', 'gradient-descent']
  },
  'k-means': {
    term: 'K-Means Clustering',
    definition: 'An unsupervised algorithm that partitions data into K clusters by iteratively assigning points to the nearest centroid, then updating centroids. Simple, fast, but assumes roughly spherical clusters of similar size.',
    analogy: 'Like sorting M&Ms by color without being told the colors — you group similar ones together and refine the groups until stable.',
    whyItMatters: 'Used for customer segmentation, data exploration, and feature engineering. Often a first step before more complex analysis.',
    related: ['unsupervised-learning', 'dbscan', 'embedding']
  },
  'dbscan': {
    term: 'DBSCAN',
    definition: 'Density-Based Spatial Clustering of Applications with Noise — clusters data by finding dense regions separated by sparse areas. Unlike K-means, it discovers the number of clusters automatically and handles irregular shapes and outliers.',
    analogy: 'Like identifying islands by sea level — dense land masses become clusters, isolated rocks become noise.',
    whyItMatters: 'DBSCAN handles messy real-world data better than K-means. Useful when you don\'t know how many natural groups exist.',
    related: ['k-means', 'unsupervised-learning']
  },
  'pca': {
    term: 'PCA (Principal Component Analysis)',
    definition: 'A dimensionality reduction technique that finds the directions of maximum variance in data and projects it onto fewer dimensions while preserving as much information as possible.',
    analogy: 'Like casting a shadow of a 3D object onto a wall — you lose one dimension but the shadow captures the most important shape information.',
    whyItMatters: 'PCA helps visualize high-dimensional data and reduce computational costs. Essential for understanding embedding spaces and data exploration.',
    related: ['embedding', 't-sne', 'umap', 'unsupervised-learning']
  },
  't-sne': {
    term: 't-SNE',
    definition: 't-Distributed Stochastic Neighbor Embedding — a nonlinear dimensionality reduction technique optimized for visualizing high-dimensional data in 2D or 3D. Preserves local structure (nearby points stay nearby) better than PCA.',
    analogy: 'Like arranging photos on a table so similar photos are close together — the 2D layout reveals clusters that exist in the high-dimensional data.',
    whyItMatters: 'Used to visualize embeddings and understand what a model has learned. Helps debug and explain model behavior to stakeholders.',
    related: ['pca', 'umap', 'embedding']
  },
  'umap': {
    term: 'UMAP',
    definition: 'Uniform Manifold Approximation and Projection — a dimensionality reduction technique that preserves both local and global structure. Faster than t-SNE and often produces more meaningful layouts of high-dimensional data.',
    analogy: 'Like t-SNE\'s faster, more accurate sibling — it creates a map of your data that preserves both neighborhoods and continents.',
    whyItMatters: 'UMAP is increasingly preferred over t-SNE for embedding visualization. Understanding these tools helps you interpret what your ML team shows you.',
    related: ['t-sne', 'pca', 'embedding']
  },
  'bias': {
    term: 'Bias (Statistical)',
    definition: 'In ML, bias has two meanings: (1) a model parameter (the intercept term), and (2) systematic error from wrong assumptions, causing the model to consistently miss the target. High bias = underfitting.',
    analogy: 'Like aiming for a bullseye but always hitting the same spot to the left — your aim is consistently off in one direction.',
    whyItMatters: 'Understanding bias-variance tradeoff helps you grasp why models underfit or overfit, directly affecting product quality.',
    related: ['bias-variance-tradeoff', 'overfitting', 'data-bias']
  },
  'bias-variance-tradeoff': {
    term: 'Bias-Variance Tradeoff',
    definition: 'The fundamental tension in ML: simpler models have high bias (underfit) but low variance; complex models have low bias but high variance (overfit). The sweet spot minimizes total error.',
    analogy: 'Like adjusting a guitar string — too tight (high variance) and it breaks on new songs, too loose (high bias) and nothing sounds right.',
    whyItMatters: 'Every model your team builds navigates this tradeoff. Understanding it helps you make informed decisions about model complexity and performance expectations.',
    related: ['bias', 'overfitting', 'regularisation']
  },
  'data-bias': {
    term: 'Data Bias',
    definition: 'Systematic skew in training data that leads models to make unfair or inaccurate predictions for certain groups. Sources include selection bias, labeling bias, historical bias, and representation gaps.',
    analogy: 'Like learning about the world only from one neighborhood — your conclusions won\'t generalize to places you\'ve never seen.',
    whyItMatters: 'Data bias is the #1 source of AI fairness issues. As a PM, ensuring diverse, representative training data is a product responsibility, not just an ethics checkbox.',
    related: ['bias', 'fairness', 'responsible-ai']
  },
  'vanishing-gradient': {
    term: 'Vanishing Gradient Problem',
    definition: 'When gradients become extremely small as they propagate backward through many layers, causing early layers to learn extremely slowly or not at all. This limited deep network training until innovations like ReLU and residual connections.',
    analogy: 'Like a game of telephone — the message (gradient) gets weaker with each person it passes through until it\'s meaningless.',
    whyItMatters: 'Understanding this explains why deep networks were impractical until ~2012 and why architectural choices (ReLU, skip connections, layer norm) matter.',
    related: ['relu', 'gradient-descent', 'batch-normalisation', 'residual-connection']
  },
  'batch-normalisation': {
    term: 'Batch Normalization',
    definition: 'A technique that normalizes layer inputs to have zero mean and unit variance within each mini-batch during training. Stabilizes training, allows higher learning rates, and acts as a mild regularizer.',
    analogy: 'Like recalibrating instruments between measurements — ensuring each layer starts from a consistent baseline.',
    whyItMatters: 'BatchNorm made training deep networks much more practical. When your team discusses normalization strategies, understanding the basics helps you follow the conversation.',
    related: ['layer-norm', 'group-norm', 'internal-covariate-shift']
  },
  'layer-norm': {
    term: 'Layer Normalization',
    definition: 'Normalizes inputs across features for each individual example (rather than across a batch like BatchNorm). Used in Transformers and RNNs where batch statistics are unreliable or unavailable.',
    analogy: 'Like grading each student\'s test on their own curve rather than the class curve — each example is normalized independently.',
    whyItMatters: 'Layer norm is standard in Transformers (Gemini, GPT). Understanding it helps you grasp architectural discussions about model design.',
    related: ['batch-normalisation', 'transformer', 'group-norm']
  },
  'group-norm': {
    term: 'Group Normalization',
    definition: 'A normalization variant that divides channels into groups and normalizes within each group. Works well with small batch sizes where BatchNorm statistics are noisy.',
    analogy: 'Like normalizing test scores within study groups rather than the whole class — useful when class sizes are too small for meaningful statistics.',
    whyItMatters: 'Group norm is common in computer vision models. Understanding normalization variants helps you follow architecture decisions.',
    related: ['batch-normalisation', 'layer-norm', 'instance-norm']
  },
  'instance-norm': {
    term: 'Instance Normalization',
    definition: 'Normalizes each individual feature map independently for each example. Primarily used in style transfer and generative models where preserving per-instance style information matters.',
    analogy: 'Like adjusting the brightness of each photo individually rather than using one setting for the whole album.',
    whyItMatters: 'Relevant for understanding diffusion models and image generation — key Gemini capabilities.',
    related: ['batch-normalisation', 'group-norm', 'diffusion-models']
  },
  'internal-covariate-shift': {
    term: 'Internal Covariate Shift',
    definition: 'The change in the distribution of layer inputs during training as parameters of preceding layers update. BatchNorm was originally proposed to address this, though the exact mechanism is debated.',
    analogy: 'Like trying to hit a moving target — each layer\'s input distribution shifts as earlier layers learn, making training unstable.',
    whyItMatters: 'Understanding why normalization helps training stability is useful context for architectural decisions.',
    related: ['batch-normalisation', 'layer-norm']
  },
  'regularisation': {
    term: 'Regularization',
    definition: 'Techniques to prevent overfitting by adding constraints that discourage overly complex models. Includes L1/L2 penalties, dropout, early stopping, data augmentation, and weight decay.',
    analogy: 'Like adding speed bumps to prevent a car from going too fast — small constraints that keep the model from memorizing noise.',
    whyItMatters: 'Regularization is why models generalize to new data. When your team discusses model performance, regularization choices are often the key lever.',
    related: ['overfitting', 'dropout', 'weight-decay', 'early-stopping']
  },
  'weight-decay': {
    term: 'Weight Decay',
    definition: 'A regularization technique that adds a penalty proportional to the magnitude of weights to the loss function, encouraging smaller weights. Equivalent to L2 regularization in most optimizers.',
    analogy: 'Like a tax on complexity — the bigger the weights, the higher the penalty, encouraging the model to use simpler solutions.',
    whyItMatters: 'Weight decay is a standard hyperparameter in every training run. Understanding it helps you grasp tuning discussions.',
    related: ['regularisation', 'hyperparameters', 'overfitting']
  },
  'weight-sharing': {
    term: 'Weight Sharing',
    definition: 'Using the same set of weights across multiple positions or timesteps in a network. CNNs share weights across spatial positions; RNNs share weights across time steps. Reduces parameters and captures translation invariance.',
    analogy: 'Like using the same set of instructions for every section of a factory — efficient and ensures consistent processing regardless of position.',
    whyItMatters: 'Weight sharing is why CNNs work for images and RNNs for sequences. It\'s a fundamental design principle in neural architectures.',
    related: ['cnn', 'rnn', 'translation-invariance', 'parameters']
  },
  'weight': {
    term: 'Weight',
    definition: 'A learnable parameter in a neural network that determines the strength of connection between neurons. During training, weights are adjusted via backpropagation to minimize the loss function.',
    analogy: 'Like the volume knob on each input channel of a mixing board — adjusting how much each input contributes to the output.',
    whyItMatters: 'When people say a model has "175 billion parameters," they mostly mean weights. Model size = number of weights = compute cost = capability.',
    related: ['parameters', 'gradient-descent', 'neural-network']
  },
  'parameters': {
    term: 'Parameters',
    definition: 'The learnable values in a model (weights and biases) that are adjusted during training. Parameter count is a primary measure of model size — GPT-4 is estimated at ~1.8 trillion parameters.',
    analogy: 'Like the total number of adjustable knobs in a sound system — more knobs means more precise control but also more complexity.',
    whyItMatters: 'Parameter count drives compute costs, memory requirements, and (loosely) capability. Crucial for infrastructure planning and cost estimation.',
    related: ['weight', 'hyperparameters', 'scaling-laws']
  },
  'hyperparameters': {
    term: 'Hyperparameters',
    definition: 'Configuration values set before training begins (not learned from data). Includes learning rate, batch size, number of layers, dropout rate, and architecture choices. Tuning hyperparameters is a key part of ML engineering.',
    analogy: 'Like the settings on an oven before cooking — temperature, time, rack position. They\'re not the food (data) but they determine how it turns out.',
    whyItMatters: 'Hyperparameter tuning is often the bottleneck between a mediocre and great model. Understanding what your ML team is tuning helps you plan timelines.',
    related: ['learning-rate', 'batch-size', 'parameters']
  },
  'gradient': {
    term: 'Gradient',
    definition: 'A vector of partial derivatives indicating the direction and magnitude of steepest increase of a function. In ML, gradients of the loss function with respect to parameters tell the optimizer which direction to adjust weights.',
    analogy: 'Like a compass pointing uphill — you go the opposite direction to find the valley (minimum loss).',
    whyItMatters: 'Gradients are the mechanism by which neural networks learn. Understanding them demystifies training.',
    related: ['gradient-descent', 'backpropagation', 'loss-function']
  },
  'gradient-clipping': {
    term: 'Gradient Clipping',
    definition: 'A technique that caps gradient magnitudes during training to prevent exploding gradients, where parameter updates become so large they destabilize training. Essential for training RNNs and large models.',
    analogy: 'Like a speed limiter on a car — prevents dangerous acceleration even if the engine wants to go faster.',
    whyItMatters: 'Gradient clipping is a standard practice in LLM training. It\'s part of the infrastructure that makes training stable.',
    related: ['vanishing-gradient', 'gradient-descent', 'rnn']
  },
  'data-augmentation': {
    term: 'Data Augmentation',
    definition: 'Creating synthetic training examples by applying transformations to existing data — rotating/flipping images, adding noise, paraphrasing text, or mixing examples. Increases effective dataset size without collecting new data.',
    analogy: 'Like studying flashcards in different lighting, angles, and order — you see the same content but build more robust understanding.',
    whyItMatters: 'Data augmentation is often the cheapest way to improve model performance. As a PM, knowing this option helps when data collection is expensive.',
    related: ['training-data', 'overfitting', 'regularisation']
  },
  'early-stopping': {
    term: 'Early Stopping',
    definition: 'Halting training when performance on a validation set stops improving, even if training loss is still decreasing. Prevents the model from overfitting by memorizing training data.',
    analogy: 'Like knowing when to stop studying — at some point, more cramming hurts performance because you start memorizing errors.',
    whyItMatters: 'Early stopping is a simple, effective regularization technique. Understanding it helps you grasp why training doesn\'t just "run longer for better results."',
    related: ['overfitting', 'regularisation', 'validation']
  },
  'lr-schedule': {
    term: 'Learning Rate Schedule',
    definition: 'A strategy for changing the learning rate during training — typically starting high and decreasing. Common schedules include cosine annealing, step decay, and warmup-then-decay. Critical for training stability.',
    analogy: 'Like driving fast on the highway and slowing down near your destination — large steps at first, then finer adjustments as you approach the optimum.',
    whyItMatters: 'Learning rate scheduling is one of the most impactful hyperparameters. Discussions about training configurations almost always involve the LR schedule.',
    related: ['learning-rate', 'hyperparameters', 'adam']
  },
  'adam': {
    term: 'Adam Optimizer',
    definition: 'Adaptive Moment Estimation — the most popular optimizer for deep learning. Combines momentum (tracking direction of recent gradients) with adaptive per-parameter learning rates. Works well out of the box across many tasks.',
    analogy: 'Like a GPS that adjusts speed based on road conditions and traffic — it adapts the pace of learning for each parameter individually.',
    whyItMatters: 'Adam is the default optimizer for most projects. Understanding it helps you follow discussions about training configuration.',
    related: ['gradient-descent', 'learning-rate', 'lr-schedule']
  },
  'feedforward-network': {
    term: 'Feedforward Network',
    definition: 'A neural network where data flows in one direction — from input through hidden layers to output — with no cycles or feedback connections. The simplest network topology, contrasted with recurrent or graph networks.',
    analogy: 'Like a one-way assembly line — each station processes and passes forward, never looking back.',
    whyItMatters: 'Feedforward layers are a component of Transformers and most architectures. Understanding the building blocks helps you follow architectural discussions.',
    related: ['neural-network', 'rnn', 'transformer']
  },
  'bidirectional-rnn': {
    term: 'Bidirectional RNN',
    definition: 'An RNN variant that processes sequences in both forward and backward directions, capturing context from both past and future tokens. Two separate RNNs are run — one left-to-right, one right-to-left — and their outputs are combined.',
    analogy: 'Like reading a sentence both forwards and backwards to better understand each word from full context.',
    whyItMatters: 'Bidirectional processing is the idea behind BERT. Understanding it helps you grasp why some models need the full input before generating output.',
    related: ['rnn', 'lstm', 'encoder-decoder']
  },
  'hidden-state': {
    term: 'Hidden State',
    definition: 'In RNNs, the internal memory vector that accumulates information from previous time steps and passes it forward. In general, any internal representation a model maintains that isn\'t directly observable.',
    analogy: 'Like your working memory when reading a book — you carry forward relevant context from earlier pages to understand the current one.',
    whyItMatters: 'Hidden states are how sequence models maintain memory. Understanding them clarifies why LLMs have context windows.',
    related: ['rnn', 'cell-state', 'lstm']
  },
  'cell-state': {
    term: 'Cell State',
    definition: 'In LSTM networks, the long-term memory channel that runs through the entire sequence with minimal modification. Gates control what information is added to or removed from the cell state, solving the vanishing gradient problem.',
    analogy: 'Like a conveyor belt running through a factory — information travels along it, and workers at each station can add or remove items.',
    whyItMatters: 'Cell state is the key innovation of LSTMs that enabled practical sequence modeling before Transformers.',
    related: ['lstm', 'gates', 'hidden-state']
  },
  'gates': {
    term: 'Gates (Neural Network)',
    definition: 'Learned sigmoid-based mechanisms in LSTMs and GRUs that control information flow. Include forget gates (what to discard), input gates (what to add), and output gates (what to expose). Values between 0 (block) and 1 (pass through).',
    analogy: 'Like traffic lights controlling which cars (information) can enter or leave an intersection at any given time.',
    whyItMatters: 'Gates are the architectural innovation that made RNNs practical. The gating concept influences modern attention mechanisms.',
    related: ['lstm', 'gru', 'cell-state', 'attention']
  },
  'gru': {
    term: 'GRU (Gated Recurrent Unit)',
    definition: 'A simplified variant of LSTM that combines the forget and input gates into a single "update gate" and merges cell/hidden state. Fewer parameters than LSTM with comparable performance on many tasks.',
    analogy: 'Like a streamlined LSTM — same concept of gated memory, but with fewer moving parts.',
    whyItMatters: 'GRUs demonstrate that simpler architectures can match complex ones. Sometimes less is more in model design.',
    related: ['lstm', 'rnn', 'gates']
  },
  'bptt': {
    term: 'BPTT (Backpropagation Through Time)',
    definition: 'The algorithm for training RNNs by unrolling the network across time steps and applying standard backpropagation. Gradients flow backward through all time steps, which can cause vanishing or exploding gradients.',
    analogy: 'Like tracing cause-and-effect backward through a chain of events — how did what happened 10 steps ago influence the current outcome?',
    whyItMatters: 'BPTT\'s limitations (vanishing gradients over long sequences) are why attention mechanisms and Transformers were invented.',
    related: ['backpropagation', 'rnn', 'vanishing-gradient']
  },
  'encoder-decoder': {
    term: 'Encoder-Decoder Architecture',
    definition: 'A model structure where an encoder processes input into a latent representation, and a decoder generates output from it. Used in translation (seq2seq), image segmentation (U-Net), and original Transformer. Decoder-only (GPT) and encoder-only (BERT) are variants.',
    analogy: 'Like a translator who first understands the source language (encoder) and then formulates the translation (decoder).',
    whyItMatters: 'Understanding encoder-decoder vs. decoder-only explains fundamental differences between BERT-like and GPT-like models, including Gemini\'s architecture choices.',
    related: ['transformer', 'attention', 'unet']
  },
  'unet': {
    term: 'U-Net',
    definition: 'An encoder-decoder architecture with skip connections between corresponding encoder and decoder layers, forming a U shape. Originally designed for medical image segmentation, now widely used as the backbone of diffusion models like Stable Diffusion.',
    analogy: 'Like a funnel that compresses information then expands it back, with shortcuts connecting matching resolution levels so fine details aren\'t lost.',
    whyItMatters: 'U-Net is the backbone of image generation (Stable Diffusion, DALL-E). Understanding it helps you grasp diffusion model capabilities.',
    related: ['encoder-decoder', 'diffusion-models', 'residual-connection']
  },
  'kernel': {
    term: 'Kernel (Convolution)',
    definition: 'A small matrix of learnable weights that slides across input data in a CNN, detecting local patterns like edges, textures, or shapes. Also called a filter. In SVMs, a kernel is a function that maps data to higher dimensions.',
    analogy: 'Like a magnifying glass that scans across an image, looking for specific patterns at each position.',
    whyItMatters: 'Kernels are why CNNs can detect visual features. Understanding them helps you grasp how computer vision models work.',
    related: ['cnn', 'stride', 'padding', 'svm']
  },
  'stride': {
    term: 'Stride',
    definition: 'The step size at which a convolutional kernel moves across the input. Stride 1 means the kernel moves one pixel at a time; stride 2 skips every other position, reducing output dimensions by half.',
    analogy: 'Like reading every word (stride 1) versus every other word (stride 2) — higher stride means faster but coarser processing.',
    whyItMatters: 'Stride controls the spatial resolution tradeoff in CNN architectures — relevant for understanding computer vision capabilities.',
    related: ['kernel', 'cnn', 'padding']
  },
  'padding': {
    term: 'Padding (Convolution)',
    definition: 'Adding extra values (usually zeros) around the border of input data before convolution. "Same" padding preserves spatial dimensions; "valid" padding reduces them. Prevents information loss at edges.',
    analogy: 'Like adding a frame around a photo before scanning it — ensures the scanner can fully cover the edges.',
    whyItMatters: 'Padding choices affect model architecture dimensions. Understanding it helps follow CNN architecture discussions.',
    related: ['kernel', 'stride', 'cnn']
  },
  'translation-invariance': {
    term: 'Translation Invariance',
    definition: 'A property of CNNs meaning they can detect the same feature regardless of where it appears in the input. A cat detector works whether the cat is in the top-left or bottom-right of the image.',
    analogy: 'Like recognizing a friend\'s face whether they\'re on the left or right side of a group photo.',
    whyItMatters: 'Translation invariance is why CNNs are powerful for vision. It reduces the data needed — you don\'t need examples of cats in every possible position.',
    related: ['cnn', 'kernel', 'weight-sharing']
  },
  'dense-prediction': {
    term: 'Dense Prediction',
    definition: 'Making a prediction for every pixel/position in an input, rather than a single prediction for the whole input. Includes semantic segmentation, depth estimation, and optical flow.',
    analogy: 'Like coloring every pixel of an image with its category label, rather than just saying "this image contains a cat."',
    whyItMatters: 'Dense prediction powers AR overlays, autonomous driving, and medical imaging — capabilities that Gemini\'s multimodal models may need.',
    related: ['cnn', 'unet', 'encoder-decoder']
  },
  'linearly-separable': {
    term: 'Linearly Separable',
    definition: 'Data that can be perfectly divided into classes by a straight line (2D), plane (3D), or hyperplane (higher dimensions). Perceptrons can only solve linearly separable problems; XOR is the classic non-linearly-separable example.',
    analogy: 'Like sorting a mixed pile of red and blue marbles by drawing one straight line between them — possible only if they\'re already somewhat grouped.',
    whyItMatters: 'Understanding linear separability explains why deep networks with nonlinear activations are necessary — most real problems aren\'t linearly separable.',
    related: ['perceptron', 'svm', 'activation-function']
  },
  'universal-approximation': {
    term: 'Universal Approximation Theorem',
    definition: 'The theorem stating that a neural network with a single hidden layer and sufficient neurons can approximate any continuous function to arbitrary precision. However, it says nothing about how to find those weights or how many neurons are needed.',
    analogy: 'Like proving that enough Lego bricks can build anything — true in theory, but doesn\'t tell you which bricks to use or how many you need.',
    whyItMatters: 'This theorem is why neural networks are so versatile. But "can approximate" doesn\'t mean "easy to train" — the gap between theory and practice matters.',
    related: ['neural-network', 'deep-learning']
  },
  'classification-metrics': {
    term: 'Classification Metrics',
    definition: 'Quantitative measures for evaluating classification models: accuracy, precision, recall, F1-score, AUC-ROC. Each captures different aspects of performance — accuracy can be misleading with imbalanced classes.',
    analogy: 'Like evaluating a spam filter: precision asks "of flagged emails, how many were actually spam?" while recall asks "of all spam emails, how many were caught?"',
    whyItMatters: 'Choosing the right metric is a product decision, not just a technical one. A medical test should optimize recall; a recommendation engine might optimize precision.',
    related: ['confusion-matrix', 'precision-recall', 'evaluation']
  },
  'confusion-matrix': {
    term: 'Confusion Matrix',
    definition: 'A table showing true positives, true negatives, false positives, and false negatives for a classification model. Reveals exactly where the model gets confused and which types of errors it makes.',
    analogy: 'Like a detailed scoreboard that shows not just wins and losses, but exactly which opponents you won or lost against.',
    whyItMatters: 'Confusion matrices reveal whether errors are tolerable or catastrophic. "95% accuracy" means different things if the 5% errors are evenly spread or concentrated in one critical class.',
    related: ['classification-metrics', 'precision-recall']
  },
  'fid': {
    term: 'FID (Fréchet Inception Distance)',
    definition: 'A metric for evaluating the quality and diversity of generated images by comparing the distribution of generated images to real images in the feature space of an Inception network. Lower FID = better quality and diversity.',
    analogy: 'Like comparing the overall vibe of two photo albums — not individual photos, but whether the collection as a whole looks similar.',
    whyItMatters: 'FID is the standard metric for image generation models. Understanding it helps you evaluate and compare diffusion model performance.',
    related: ['diffusion-models', 'evaluation', 'generative-ai']
  },
  'elo-rating': {
    term: 'Elo Rating',
    definition: 'A relative ranking system where models are rated based on head-to-head comparisons. Originally designed for chess, now used in LLM evaluation (Chatbot Arena). A model\'s rating increases when it wins comparisons and decreases when it loses.',
    analogy: 'Like a chess rating for AI — you earn points by beating higher-rated opponents and lose points to lower-rated ones.',
    whyItMatters: 'Elo-based leaderboards (LMSYS Chatbot Arena) are how LLMs are compared. Understanding this helps you interpret competitive benchmarks.',
    related: ['evaluation', 'benchmarks']
  },
  'mode-collapse': {
    term: 'Mode Collapse',
    definition: 'A failure mode in GANs where the generator produces only a narrow subset of possible outputs, ignoring the full diversity of the data distribution. The generator finds a few "safe" outputs that fool the discriminator and stops exploring.',
    analogy: 'Like a student who memorizes one perfect essay and submits it for every topic — technically not wrong, but severely lacking diversity.',
    whyItMatters: 'Mode collapse is why GANs were largely replaced by diffusion models for image generation. Understanding failure modes of generative models helps evaluate alternatives.',
    related: ['gan', 'generative-ai', 'diffusion-models']
  },
  'model-drift': {
    term: 'Model Drift',
    definition: 'The degradation of model performance over time as the real-world data distribution changes from what the model was trained on. Includes concept drift (the target relationship changes) and data drift (input distribution shifts).',
    analogy: 'Like a map becoming outdated as new roads are built and old ones close — the model\'s understanding of the world slowly diverges from reality.',
    whyItMatters: 'Model drift is why AI products need monitoring and retraining pipelines. A model that\'s great at launch can silently degrade without proper observability.',
    related: ['evaluation', 'model-serving', 'monitoring']
  },
  'model-serving': {
    term: 'Model Serving',
    definition: 'The infrastructure for deploying trained models to handle real-time prediction requests. Involves API design, load balancing, batching, hardware optimization (GPU/TPU), latency management, and scaling.',
    analogy: 'Like running a restaurant kitchen — the recipe (model) is done, but you need infrastructure to serve thousands of meals (predictions) per second.',
    whyItMatters: 'Serving infrastructure determines user-facing latency and cost. A PM must understand the tradeoffs between model size, speed, and cost.',
    related: ['latency', 'inference', 'scaling']
  },
  'uncertainty': {
    term: 'Model Uncertainty',
    definition: 'The model\'s own assessment of how confident it is in its predictions. Includes epistemic uncertainty (from limited training data) and aleatoric uncertainty (from inherent noise in the task). Crucial for knowing when to defer to humans.',
    analogy: 'Like a doctor saying "I\'m 90% sure it\'s a cold, but 10% chance it\'s something else" — the confidence level matters as much as the diagnosis.',
    whyItMatters: 'Uncertainty estimation enables "I don\'t know" responses, human-in-the-loop workflows, and calibrated confidence. Essential for responsible AI products.',
    related: ['evaluation', 'calibration', 'responsible-ai']
  },
  'self-supervised-learning': {
    term: 'Self-Supervised Learning',
    definition: 'A training paradigm where the model creates its own supervision signal from unlabeled data — e.g., predicting masked words (BERT), next tokens (GPT), or matching image augmentations (CLIP). Eliminates the need for expensive human labels.',
    analogy: 'Like learning a language by reading millions of books and filling in blanks, rather than having a teacher provide translations.',
    whyItMatters: 'Self-supervised learning is how all modern LLMs are pre-trained. It\'s the key insight that enabled scaling to trillions of tokens of training data.',
    related: ['pre-training', 'llm', 'contrastive-learning']
  },
  'generative-ai': {
    term: 'Generative AI',
    definition: 'AI systems that create new content — text, images, audio, video, code — rather than just classifying or predicting. Includes LLMs (GPT, Gemini), diffusion models (DALL-E, Imagen), and audio models (MusicLM).',
    analogy: 'Like the difference between a critic who evaluates art and an artist who creates it — generative AI produces, not just judges.',
    whyItMatters: 'Generative AI is the product category you\'ll be building at DeepMind. Understanding its capabilities and limitations is your core PM competency.',
    related: ['llm', 'diffusion-models', 'multimodal']
  },
  'foundation-model': {
    term: 'Foundation Model',
    definition: 'A large model trained on broad data that can be adapted to many downstream tasks. Gemini, GPT-4, and Claude are foundation models — trained once at enormous cost, then fine-tuned or prompted for specific applications.',
    analogy: 'Like a liberal arts education — broad knowledge that can be specialized for any career, versus vocational training for one specific job.',
    whyItMatters: 'Foundation models are the product you\'re building. Understanding the economics (massive training cost, versatile application) shapes platform strategy.',
    related: ['llm', 'pre-training', 'fine-tuning', 'generative-ai']
  },
  'agent': {
    term: 'Agent (RL/AI)',
    definition: 'An entity that perceives its environment, takes actions, and receives rewards. In RL, the agent learns a policy that maps states to actions to maximize cumulative reward. In AI products, "agent" increasingly means an LLM-powered system that autonomously takes actions.',
    analogy: 'Like a player in a video game who observes the screen, chooses moves, and gets points — learning which strategies work over time.',
    whyItMatters: 'AI agents (tool-using LLMs, autonomous systems) are a major product trend. Understanding agency is central to building Gemini assistive features.',
    related: ['reinforcement-learning', 'policy', 'environment', 'reward']
  },
  'action': {
    term: 'Action (RL)',
    definition: 'A decision made by an RL agent at each time step. The set of possible actions is the "action space" — it can be discrete (turn left/right) or continuous (steering angle). The agent\'s policy determines which action to take given the current state.',
    analogy: 'Like choosing which move to make in chess — each piece you could move represents a different action in your action space.',
    whyItMatters: 'Understanding action spaces helps you grasp how RL agents operate, which is core DeepMind knowledge.',
    related: ['agent', 'state', 'policy', 'environment']
  },
  'state': {
    term: 'State (RL)',
    definition: 'The current situation of the environment as perceived by the agent — a complete or partial description of the world at a given moment. The state is the input to the agent\'s policy for choosing actions.',
    analogy: 'Like the current position of all pieces on a chess board — everything the player needs to see to make their next move.',
    whyItMatters: 'State representation design is a critical decision in RL systems. Partial observability (not seeing the full state) is a key real-world challenge.',
    related: ['agent', 'action', 'environment', 'policy']
  },
  'environment': {
    term: 'Environment (RL)',
    definition: 'Everything external to the RL agent — the world it interacts with. The environment receives actions, transitions to new states, and provides rewards. Can be a simulation (game, physics engine) or the real world (robotics).',
    analogy: 'Like the game board and rules in a board game — it responds to the player\'s moves and determines outcomes.',
    whyItMatters: 'Environment design determines what the agent can learn. Simulation environments (like DeepMind Lab) accelerate research by providing fast, safe training grounds.',
    related: ['agent', 'state', 'reward', 'action']
  },
  'reward': {
    term: 'Reward (RL)',
    definition: 'A scalar signal from the environment that tells the agent how good its action was. The agent\'s objective is to maximize cumulative reward over time. Reward design (reward shaping) is one of the hardest parts of RL.',
    analogy: 'Like the score in a game — it tells you how well you\'re doing, and you learn strategies that maximize your score.',
    whyItMatters: 'Reward misspecification is a major source of AI alignment failures. The agent optimizes exactly what you reward — which may not be what you intended.',
    related: ['agent', 'reward-hacking', 'rlhf', 'alignment']
  },
  'reward-hacking': {
    term: 'Reward Hacking',
    definition: 'When an RL agent exploits loopholes in the reward function to achieve high reward without actually solving the intended task. A symptom of misspecified objectives — the agent finds shortcuts that satisfy the letter but not spirit of the reward.',
    analogy: 'Like a student who games the grading system — getting perfect scores without actually learning the material.',
    whyItMatters: 'Reward hacking is an alignment problem that extends to RLHF-trained LLMs. Models may learn to produce outputs that "look good" to evaluators without being genuinely helpful.',
    related: ['reward', 'alignment', 'rlhf', 'exploration-exploitation']
  },
  'policy': {
    term: 'Policy (RL)',
    definition: 'A function that maps states to actions — the agent\'s strategy for behavior. Can be deterministic (state → action) or stochastic (state → probability distribution over actions). Learning the optimal policy is the goal of RL.',
    analogy: 'Like a rulebook for how to play — given any game situation, the policy tells you what move to make.',
    whyItMatters: 'Policies are what RL agents learn. AlphaGo\'s policy network was the breakthrough that enabled superhuman Go play.',
    related: ['agent', 'state', 'action', 'reinforcement-learning']
  },
  'exploration-exploitation': {
    term: 'Exploration-Exploitation Tradeoff',
    definition: 'The fundamental dilemma in RL: should the agent try new actions to discover potentially better strategies (explore) or stick with known good actions (exploit)? Too much exploration wastes time; too much exploitation misses better options.',
    analogy: 'Like choosing between your favorite restaurant (exploit) and trying a new one that might be better or worse (explore).',
    whyItMatters: 'This tradeoff appears in product decisions too: should you optimize the current feature or experiment with alternatives? Understanding it frames A/B testing and product evolution.',
    related: ['reinforcement-learning', 'agent', 'reward']
  },
  'credit-assignment': {
    term: 'Credit Assignment Problem',
    definition: 'The challenge of determining which actions in a sequence actually contributed to the eventual reward. When reward is delayed (e.g., winning a game after 100 moves), it\'s unclear which early decisions mattered.',
    analogy: 'Like figuring out which ingredient made a recipe taste great when you changed five things at once.',
    whyItMatters: 'Credit assignment is why attribution analysis matters in both RL and product analytics — knowing what caused success is harder than it seems.',
    related: ['reinforcement-learning', 'reward', 'temporal-difference']
  },
  'deepmind': {
    term: 'DeepMind',
    definition: 'A leading AI research lab founded in 2010, acquired by Google in 2014, now part of Google DeepMind. Known for AlphaGo, AlphaFold, Gemini, and fundamental contributions to reinforcement learning, neuroscience-inspired AI, and protein structure prediction.',
    analogy: 'Like the Bell Labs of AI — a research powerhouse that produces both fundamental breakthroughs and world-changing products.',
    whyItMatters: 'This is where you\'re interviewing. Understanding DeepMind\'s culture (research-first, ambitious long-term bets, safety focus) is essential.',
    related: ['gemini', 'alphago', 'alphafold']
  },
  'ai-product-lifecycle': {
    term: 'AI Product Lifecycle',
    definition: 'The end-to-end process of building AI products: problem definition → data collection → model development → evaluation → deployment → monitoring → iteration. Unlike traditional software, AI products require continuous data and model maintenance.',
    analogy: 'Like farming versus manufacturing — traditional software is built once and shipped; AI products are living systems that need ongoing cultivation.',
    whyItMatters: 'Understanding the full lifecycle prevents the common PM mistake of treating model development as the hard part when deployment and maintenance are equally challenging.',
    related: ['model-drift', 'model-serving', 'evaluation']
  },
  'developer-platform': {
    term: 'Developer Platform',
    definition: 'A set of APIs, SDKs, tools, documentation, and services that enable external developers to build on top of your technology. Examples: Google Cloud AI Platform, OpenAI API, Hugging Face. Success is measured by developer adoption and ecosystem growth.',
    analogy: 'Like providing a well-equipped kitchen to chefs — the better your tools and ingredients, the more amazing dishes (apps) developers can create.',
    whyItMatters: 'The Gemini role specifically involves SDK and developer platform strategy. Understanding what makes platforms succeed (DX, reliability, documentation) is core to the job.',
    related: ['sdk', 'developer-experience', 'developer-advocacy', 'platform-strategy']
  },
  'developer-advocacy': {
    term: 'Developer Advocacy',
    definition: 'The practice of building relationships with developer communities, creating educational content, gathering feedback, and representing developer interests internally. Developer advocates bridge the gap between the platform team and its users.',
    analogy: 'Like being an ambassador — you represent the platform to developers and represent developers to the platform team.',
    whyItMatters: 'Developer advocates are a key channel for understanding what developers need. As a platform PM, working closely with DevRel shapes your roadmap.',
    related: ['developer-platform', 'developer-experience', 'platform-strategy']
  },
  'platform-strategy': {
    term: 'Platform Strategy',
    definition: 'The strategic approach to building a technology platform that creates value by connecting producers and consumers, creating network effects, and establishing ecosystem lock-in. Platform strategy for AI includes API design, pricing, and ecosystem cultivation.',
    analogy: 'Like building a marketplace — the platform becomes more valuable as more people use it, creating a self-reinforcing cycle.',
    whyItMatters: 'Gemini\'s developer platform is competing with OpenAI, Anthropic, and open-source. Platform strategy determines who wins the AI ecosystem war.',
    related: ['developer-platform', 'network-effects', 'flywheel']
  },
  'network-effects': {
    term: 'Network Effects',
    definition: 'A phenomenon where a product becomes more valuable as more people use it. Direct network effects (more users = more value, like a phone network) and indirect effects (more developers = more apps = more users).',
    analogy: 'Like a party — the more people who come, the more fun it is, which attracts even more people.',
    whyItMatters: 'Network effects are how platforms win. For Gemini, more developers using the API → more apps → more user adoption → more data → better models.',
    related: ['platform-strategy', 'flywheel', 'developer-platform']
  },
  'flywheel': {
    term: 'Flywheel Effect',
    definition: 'A self-reinforcing cycle where each component feeds the next, building momentum over time. In AI: more users → more data → better models → better products → more users. Coined by Jim Collins.',
    analogy: 'Like pushing a heavy wheel — each push adds a bit of momentum until the wheel spins under its own power.',
    whyItMatters: 'The data flywheel is the most powerful moat in AI. Products with flywheels compound advantages over time.',
    related: ['network-effects', 'platform-strategy', 'data-moat']
  },
  'portfolio-approach': {
    term: 'Portfolio Approach',
    definition: 'A strategy of investing in multiple parallel initiatives with varying risk profiles, expecting some to fail but the winners to more than compensate. Applied to AI product bets: some safe improvements, some moonshot experiments.',
    analogy: 'Like an investment portfolio — you diversify across safe bonds and risky stocks to balance guaranteed returns with upside potential.',
    whyItMatters: 'AI roadmaps should balance incremental improvements with experimental bets. This approach manages the inherent uncertainty of AI research translating to products.',
    related: ['ai-product-lifecycle', 'platform-strategy']
  },
  'mental-model': {
    term: 'Mental Model',
    definition: 'A user\'s internal understanding of how a system works. AI products often face a mental model gap — users don\'t know what the AI can or can\'t do, leading to misuse, disappointment, or underutilization.',
    analogy: 'Like a map in your head — if your map doesn\'t match the territory, you\'ll take wrong turns even if the road is perfectly built.',
    whyItMatters: 'Designing for correct mental models is critical for AI products. Users who think chatbots "understand" them will have different expectations than those who know they pattern-match.',
    related: ['user-research', 'developer-experience']
  },
  'wizard-of-oz': {
    term: 'Wizard of Oz Testing',
    definition: 'A prototyping technique where users interact with what appears to be an AI system, but a human behind the scenes is actually generating the responses. Used to test AI product concepts before building the model.',
    analogy: 'Like the Wizard of Oz — the user sees impressive "AI" but there\'s a person behind the curtain pulling the levers.',
    whyItMatters: 'WOz testing is the fastest way to validate AI product concepts. You can test user interactions, identify edge cases, and refine prompts before investing in model development.',
    related: ['user-research', 'ai-product-lifecycle']
  },
  'third-party-integration': {
    term: 'Third-Party Integration',
    definition: 'Connecting your AI system with external services (Instagram, WhatsApp, banking apps) to provide contextual assistance. Challenges include API instability, authentication, rate limits, and maintaining functionality when third parties change their interfaces.',
    analogy: 'Like building bridges between islands — each island (service) can change its coastline anytime, potentially breaking your bridge.',
    whyItMatters: 'Gemini as an assistant layer on Android/iOS depends heavily on third-party app integrations. When Instagram changes its UI, Gemini\'s screen-reading features can break.',
    related: ['graceful-degradation', 'schema-drift', 'screen-context']
  },
  'sse': {
    term: 'SSE (Server-Sent Events)',
    definition: 'A web technology enabling a server to push real-time updates to a client over HTTP. Used in streaming LLM responses — tokens appear as they\'re generated rather than waiting for the complete response.',
    analogy: 'Like watching a letter being typed live versus waiting for the entire letter to arrive in an envelope.',
    whyItMatters: 'SSE enables the streaming chat experience users expect from AI products. Understanding it helps you design responsive API interfaces.',
    related: ['api', 'sdk', 'latency']
  },
  'vertex-ai': {
    term: 'Vertex AI',
    definition: 'Google Cloud\'s managed ML platform that provides tools for building, deploying, and scaling ML models. Includes AutoML, custom training, model deployment, and increasingly, access to Gemini foundation models via API.',
    analogy: 'Like a turnkey factory for AI — all the machines, tools, and logistics are set up so you can focus on what to build.',
    whyItMatters: 'Vertex AI is a key distribution channel for Gemini models. Understanding Google\'s platform ecosystem is essential for the DeepMind PM role.',
    related: ['model-serving', 'developer-platform', 'gemini']
  }
};
