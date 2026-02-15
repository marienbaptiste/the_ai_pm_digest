export const modules = [
  {
    id: 'm01',
    number: 1,
    title: 'AI & ML Foundations',
    description: 'From Turing to modern machine learning — understand the building blocks of artificial intelligence.',
    icon: '\u{1F9E0}',
    color: '#00d4aa',
    lessons: [
      { id: 'l01', title: 'What is AI? History from Turing to Today', diagram: 'ai-timeline' },
      { id: 'l02', title: 'Types of ML \u2014 Supervised, Unsupervised, Reinforcement', diagram: 'ml-types' },
      { id: 'l03', title: 'How Machines Learn \u2014 Loss Functions & Gradient Descent', diagram: 'training-loop' },
      { id: 'l04', title: 'Neural Networks \u2014 Perceptrons to Deep Networks', diagram: 'neural-network' }
    ]
  },
  {
    id: 'm02',
    number: 2,
    title: 'Deep Learning',
    description: 'Convolutional networks, recurrent architectures, and the training techniques that make them work.',
    icon: '\u{1F50D}',
    color: '#3b82f6',
    lessons: [
      { id: 'l01', title: 'CNNs \u2014 Convolutions, Pooling & Feature Maps', diagram: 'cnn' },
      { id: 'l02', title: 'RNNs & LSTMs \u2014 Sequential Data & Memory', diagram: 'lstm' },
      { id: 'l03', title: 'Training Deep Networks \u2014 BatchNorm, Dropout & Schedules', diagram: 'training-loop' },
      { id: 'l04', title: 'Architectures That Changed Everything \u2014 ResNet, U-Net, GAN', diagram: 'neural-network' }
    ]
  },
  {
    id: 'm03',
    number: 3,
    title: 'Transformers & Attention',
    description: 'The architecture that powers modern AI \u2014 understand self-attention, multi-head attention, and positional encoding.',
    icon: '\u26A1',
    color: '#a855f7',
    lessons: [
      { id: 'l01', title: 'The Attention Revolution \u2014 Why Attention Is All You Need', diagram: 'transformer' },
      { id: 'l02', title: 'Self-Attention \u2014 Q, K, V Matrices Step by Step', diagram: 'transformer' },
      { id: 'l03', title: 'Multi-Head Attention & Positional Encoding', diagram: 'transformer' },
      { id: 'l04', title: 'Encoder-Decoder vs Decoder-Only Architectures', diagram: 'transformer' }
    ]
  },
  {
    id: 'm04',
    number: 4,
    title: 'Large Language Models',
    description: 'GPT, RLHF, prompting, fine-tuning, scaling laws \u2014 the complete LLM landscape.',
    icon: '\u{1F4AC}',
    color: '#f59e0b',
    lessons: [
      { id: 'l01', title: 'From Word2Vec to GPT \u2014 The Evolution', diagram: 'word-embeddings' },
      { id: 'l02', title: 'Pre-training \u2014 Next Token Prediction at Scale', diagram: 'transformer' },
      { id: 'l03', title: 'RLHF & Alignment \u2014 Making Models Helpful and Safe', diagram: 'rlhf' },
      { id: 'l04', title: 'Prompting, In-Context Learning & Chain-of-Thought', diagram: 'transformer' },
      { id: 'l05', title: 'Fine-tuning, LoRA & Adaptation Techniques', diagram: 'scaling-laws' },
      { id: 'l06', title: 'Scaling Laws, Emergent Abilities & Frontier Models', diagram: 'scaling-laws' }
    ]
  },
  {
    id: 'm05',
    number: 5,
    title: 'Diffusion Models',
    description: 'How AI generates images \u2014 VAEs, GANs, diffusion processes, and multimodal models.',
    icon: '\u{1F3A8}',
    color: '#ec4899',
    lessons: [
      { id: 'l01', title: 'Generative Models Landscape \u2014 VAEs, GANs, Flows, Diffusion', diagram: 'diffusion' },
      { id: 'l02', title: 'How Diffusion Works \u2014 Forward & Reverse Process', diagram: 'diffusion' },
      { id: 'l03', title: 'Stable Diffusion, DALL-E, Imagen \u2014 Architecture Deep Dive', diagram: 'diffusion' },
      { id: 'l04', title: 'Multimodal Models \u2014 Connecting Vision and Language', diagram: 'multimodal' }
    ]
  },
  {
    id: 'm06',
    number: 6,
    title: 'RAG & Retrieval Systems',
    description: 'Embeddings, vector databases, and retrieval-augmented generation \u2014 grounding AI in real data.',
    icon: '\u{1F50E}',
    color: '#06b6d4',
    lessons: [
      { id: 'l01', title: 'Embeddings \u2014 From Words to Vectors', diagram: 'word-embeddings' },
      { id: 'l02', title: 'Vector Databases \u2014 FAISS, Pinecone, Chroma', diagram: 'vector-db' },
      { id: 'l03', title: 'RAG Architecture \u2014 End to End', diagram: 'rag-pipeline' },
      { id: 'l04', title: 'Advanced RAG \u2014 Reranking, Hybrid Search, Agentic RAG', diagram: 'rag-pipeline' }
    ]
  },
  {
    id: 'm07',
    number: 7,
    title: 'AI Product Management',
    description: 'Roadmapping, metrics, evaluation, and go-to-market strategies specific to AI products.',
    icon: '\u{1F4CB}',
    color: '#00d4aa',
    lessons: [
      { id: 'l01', title: 'AI Product Lifecycle \u2014 From Research to Production', diagram: 'product-lifecycle' },
      { id: 'l02', title: 'Defining Success \u2014 Metrics & Evaluation for AI Products', diagram: 'product-lifecycle' },
      { id: 'l03', title: 'Roadmapping Under Uncertainty \u2014 AI-Specific Challenges', diagram: 'product-lifecycle' },
      { id: 'l04', title: 'Go-to-Market for AI \u2014 Launch Strategies & Developer Adoption', diagram: 'platform-flywheel' },
      { id: 'l05', title: 'User Research for AI Products \u2014 Novel Interaction Paradigms', diagram: 'stakeholder-map' }
    ]
  },
  {
    id: 'm08',
    number: 8,
    title: 'SDK & Developer Platforms',
    description: 'API design, developer experience, documentation strategy, and platform ecosystem thinking.',
    icon: '\u{1F528}',
    color: '#3b82f6',
    lessons: [
      { id: 'l01', title: 'What Makes a Great Developer Platform', diagram: 'platform-flywheel' },
      { id: 'l02', title: 'API Design Principles \u2014 REST, GraphQL, SDKs', diagram: 'platform-flywheel' },
      { id: 'l03', title: 'Developer Experience \u2014 Documentation, Onboarding, Community', diagram: 'platform-flywheel' },
      { id: 'l04', title: 'Platform Strategy \u2014 Ecosystem Flywheel Effects', diagram: 'platform-flywheel' }
    ]
  },
  {
    id: 'm09',
    number: 9,
    title: 'AI Ethics, Safety & Responsible AI',
    description: 'Bias, fairness, alignment, regulation, and building safety into AI products.',
    icon: '\u{1F6E1}\uFE0F',
    color: '#ef4444',
    lessons: [
      { id: 'l01', title: 'Bias, Fairness & Representation in AI Systems', diagram: 'safety-layers' },
      { id: 'l02', title: 'AI Safety \u2014 Alignment, Interpretability, Robustness', diagram: 'safety-layers' },
      { id: 'l03', title: 'Regulatory Landscape \u2014 EU AI Act & Responsible AI Frameworks', diagram: 'safety-layers' },
      { id: 'l04', title: 'Building Safety Into Products \u2014 Red Teaming & Guardrails', diagram: 'safety-layers' }
    ]
  },
  {
    id: 'm10',
    number: 10,
    title: 'Cross-functional Leadership',
    description: 'Leading without authority, stakeholder management, and translating research into product.',
    icon: '\u{1F91D}',
    color: '#f59e0b',
    lessons: [
      { id: 'l01', title: 'Leading Without Authority \u2014 Influence & Decision Frameworks', diagram: 'stakeholder-map' },
      { id: 'l02', title: 'Working with Research Teams \u2014 Translating Research to Product', diagram: 'product-lifecycle' },
      { id: 'l03', title: 'Stakeholder Management \u2014 Engineering, UX, Legal, Marketing', diagram: 'stakeholder-map' },
      { id: 'l04', title: 'Communicating Technical Concepts to Non-Technical Audiences', diagram: 'stakeholder-map' }
    ]
  },
  {
    id: 'm11',
    number: 11,
    title: 'DeepMind & Gemini Deep Dive',
    description: 'DeepMind\'s journey, Gemini\'s architecture, product ecosystem, and competitive landscape.',
    icon: '\u{1F48E}',
    color: '#a855f7',
    lessons: [
      { id: 'l01', title: 'DeepMind\'s History \u2014 AlphaGo to Gemini', diagram: 'gemini-ecosystem' },
      { id: 'l02', title: 'Gemini Architecture & Capabilities \u2014 Multimodal, Long Context', diagram: 'multimodal' },
      { id: 'l03', title: 'Gemini Product Ecosystem \u2014 Android, iOS, Web, API', diagram: 'gemini-ecosystem' },
      { id: 'l04', title: 'Competitive Landscape \u2014 OpenAI, Meta, Anthropic, Mistral', diagram: 'gemini-ecosystem' }
    ]
  },
  {
    id: 'm12',
    number: 12,
    title: 'Interview Prep & Case Studies',
    description: 'PM interview frameworks, AI case studies, and mock interview practice.',
    icon: '\u{1F3AF}',
    color: '#ec4899',
    lessons: [
      { id: 'l01', title: 'PM Interview Frameworks \u2014 CIRCLES, RICE, Execution', diagram: 'interview-framework' },
      { id: 'l02', title: 'AI PM Case Studies \u2014 Real Product Decisions', diagram: 'product-lifecycle' },
      { id: 'l03', title: 'Technical Deep Dive Prep \u2014 Explaining AI to Interviewers', diagram: 'transformer' },
      { id: 'l04', title: 'Mock Questions with Model Answers', diagram: 'interview-framework' }
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
