export const modules = [
  {
    id: 'm01',
    number: 1,
    title: 'AI & ML Foundations',
    description: 'From Turing to modern machine learning - understand the building blocks of artificial intelligence.',
    icon: 'brain',
    color: '#00d4aa',
    lessons: [
      { id: 'l01', title: 'What is AI? History from Turing to Today', diagram: 'ai-timeline' },
      { id: 'l02', title: 'Types of ML - Supervised, Unsupervised, Reinforcement', diagram: 'ml-types' },
      { id: 'l03', title: 'How Machines Learn - Loss Functions & Gradient Descent', diagram: 'training-loop' },
      { id: 'l04', title: 'Neural Networks - Perceptrons to Deep Networks', diagram: 'neural-network' }
    ]
  },
  {
    id: 'm02',
    number: 2,
    title: 'Deep Learning',
    description: 'Convolutional networks, recurrent architectures, and the training techniques that make them work.',
    icon: 'layers',
    color: '#3b82f6',
    lessons: [
      { id: 'l01', title: 'CNNs - Convolutions, Pooling & Feature Maps', diagram: 'cnn' },
      { id: 'l02', title: 'RNNs & LSTMs - Sequential Data & Memory', diagram: 'lstm' },
      { id: 'l03', title: 'Training Deep Networks - BatchNorm, Dropout & Schedules', diagram: 'training-loop' },
      { id: 'l04', title: 'Architectures That Changed Everything - ResNet, U-Net, GAN', diagram: 'neural-network' }
    ]
  },
  {
    id: 'm03',
    number: 3,
    title: 'Transformers & Attention',
    description: 'The architecture that powers modern AI - understand self-attention, multi-head attention, and positional encoding.',
    icon: 'zap',
    color: '#a855f7',
    lessons: [
      { id: 'l01', title: 'The Attention Revolution - Why Attention Is All You Need', diagram: 'transformer' },
      { id: 'l02', title: 'Self-Attention - Q, K, V Matrices Step by Step', diagram: 'transformer' },
      { id: 'l03', title: 'Multi-Head Attention & Positional Encoding', diagram: 'transformer' },
      { id: 'l04', title: 'Encoder-Decoder vs Decoder-Only Architectures', diagram: 'transformer' }
    ]
  },
  {
    id: 'm04',
    number: 4,
    title: 'Large Language Models',
    description: 'GPT, RLHF, prompting, fine-tuning, scaling laws - the complete LLM landscape.',
    icon: 'message-square',
    color: '#f59e0b',
    lessons: [
      { id: 'l01', title: 'From Word2Vec to GPT - The Evolution', diagram: 'word-embeddings' },
      { id: 'l02', title: 'Pre-training - Next Token Prediction at Scale', diagram: 'transformer' },
      { id: 'l03', title: 'RLHF & Alignment - Making Models Helpful and Safe', diagram: 'rlhf' },
      { id: 'l04', title: 'Prompting, In-Context Learning & Chain-of-Thought', diagram: 'transformer' },
      { id: 'l05', title: 'Fine-tuning, LoRA & Adaptation Techniques', diagram: 'scaling-laws' },
      { id: 'l06', title: 'Scaling Laws, Emergent Abilities & Frontier Models', diagram: 'scaling-laws' }
    ]
  },
  {
    id: 'm05',
    number: 5,
    title: 'Diffusion Models',
    description: 'How AI generates images - VAEs, GANs, diffusion processes, and multimodal models.',
    icon: 'sparkles',
    color: '#ec4899',
    lessons: [
      { id: 'l01', title: 'Generative Models Landscape - VAEs, GANs, Flows, Diffusion', diagram: 'diffusion' },
      { id: 'l02', title: 'How Diffusion Works - Forward & Reverse Process', diagram: 'diffusion' },
      { id: 'l03', title: 'Stable Diffusion, DALL-E, Imagen - Architecture Deep Dive', diagram: 'diffusion' },
      { id: 'l04', title: 'Multimodal Models - Connecting Vision and Language', diagram: 'multimodal' }
    ]
  },
  {
    id: 'm06',
    number: 6,
    title: 'RAG & Retrieval Systems',
    description: 'Embeddings, vector databases, and retrieval-augmented generation - grounding AI in real data.',
    icon: 'database',
    color: '#06b6d4',
    lessons: [
      { id: 'l01', title: 'Embeddings - From Words to Vectors', diagram: 'word-embeddings' },
      { id: 'l02', title: 'Vector Databases - FAISS, Pinecone, Chroma', diagram: 'vector-db' },
      { id: 'l03', title: 'RAG Architecture - End to End', diagram: 'rag-pipeline' },
      { id: 'l04', title: 'Advanced RAG - Reranking, Hybrid Search, Agentic RAG', diagram: 'rag-pipeline' }
    ]
  },
  {
    id: 'm07',
    number: 7,
    title: 'AI Product Management',
    description: 'Roadmapping, metrics, evaluation, and go-to-market strategies specific to AI products.',
    icon: 'kanban',
    color: '#00d4aa',
    lessons: [
      { id: 'l01', title: 'AI Product Lifecycle - From Research to Production', diagram: 'product-lifecycle' },
      { id: 'l02', title: 'Defining Success - Metrics & Evaluation for AI Products', diagram: 'product-lifecycle' },
      { id: 'l03', title: 'Roadmapping Under Uncertainty - AI-Specific Challenges', diagram: 'product-lifecycle' },
      { id: 'l04', title: 'Go-to-Market for AI - Launch Strategies & Developer Adoption', diagram: 'platform-flywheel' },
      { id: 'l05', title: 'User Research for AI Products - Novel Interaction Paradigms', diagram: 'stakeholder-map' }
    ]
  },
  {
    id: 'm08',
    number: 8,
    title: 'SDK & Developer Platforms',
    description: 'API design, developer experience, documentation strategy, and platform ecosystem thinking.',
    icon: 'code-2',
    color: '#3b82f6',
    lessons: [
      { id: 'l01', title: 'What Makes a Great Developer Platform', diagram: 'platform-flywheel' },
      { id: 'l02', title: 'API Design Principles - REST, GraphQL, SDKs', diagram: 'platform-flywheel' },
      { id: 'l03', title: 'Developer Experience - Documentation, Onboarding, Community', diagram: 'platform-flywheel' },
      { id: 'l04', title: 'Platform Strategy - Ecosystem Flywheel Effects', diagram: 'platform-flywheel' }
    ]
  },
  {
    id: 'm09',
    number: 9,
    title: 'AI Ethics, Safety & Responsible AI',
    description: 'Bias, fairness, alignment, regulation, and building safety into AI products.',
    icon: 'shield',
    color: '#ef4444',
    lessons: [
      { id: 'l01', title: 'Bias, Fairness & Representation in AI Systems', diagram: 'safety-layers' },
      { id: 'l02', title: 'AI Safety - Alignment, Interpretability, Robustness', diagram: 'safety-layers' },
      { id: 'l03', title: 'Regulatory Landscape - EU AI Act & Responsible AI Frameworks', diagram: 'safety-layers' },
      { id: 'l04', title: 'Building Safety Into Products - Red Teaming & Guardrails', diagram: 'safety-layers' }
    ]
  },
  {
    id: 'm10',
    number: 10,
    title: 'Cross-functional Leadership',
    description: 'Leading without authority, stakeholder management, and translating research into product.',
    icon: 'users',
    color: '#f59e0b',
    lessons: [
      { id: 'l01', title: 'Leading Without Authority - Influence & Decision Frameworks', diagram: 'stakeholder-map' },
      { id: 'l02', title: 'Working with Research Teams - Translating Research to Product', diagram: 'product-lifecycle' },
      { id: 'l03', title: 'Stakeholder Management - Engineering, UX, Legal, Marketing', diagram: 'stakeholder-map' },
      { id: 'l04', title: 'Communicating Technical Concepts to Non-Technical Audiences', diagram: 'stakeholder-map' }
    ]
  },
  {
    id: 'm13',
    number: 11,
    title: 'Frontier AI & Cutting-Edge Concepts',
    description: 'MoE architectures, state space models, reasoning at inference, AI agents, and the trends shaping 2025\u20132026.',
    icon: 'rocket',
    color: '#14b8a6',
    lessons: [
      { id: 'l01', title: 'Mixture of Experts - How Gemini & Mixtral Scale Efficiently', diagram: 'frontier-tech' },
      { id: 'l02', title: 'State Space Models - Mamba, S4 & Alternatives to Transformers', diagram: 'frontier-tech' },
      { id: 'l03', title: 'Reasoning Models & Test-Time Compute - o1, R1 & Thinking at Inference', diagram: 'frontier-tech' },
      { id: 'l04', title: 'AI Agents, Tool Use & Multi-Agent Systems', diagram: 'frontier-tech' },
      { id: 'l05', title: 'Frontier Trends - Synthetic Data, World Models, Video Gen & What\u2019s Next', diagram: 'frontier-tech' }
    ]
  }
];

export function getModule(moduleId) {
  return modules.find(m => m.id === moduleId);
}

export function getLesson(moduleId, lessonId) {
  const mod = getModule(moduleId);
  if (!mod) return null;
  return mod.lessons.find(l => l.id === lessonId) || null;
}

export function getTotalLessons() {
  return modules.reduce((sum, m) => sum + m.lessons.length, 0);
}
