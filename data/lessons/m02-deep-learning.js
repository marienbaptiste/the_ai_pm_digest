export const lessons = {

  // ─────────────────────────────────────────────
  // L01 — CNNs: Convolutions, Pooling & Feature Maps
  // ─────────────────────────────────────────────
  l01: {
    title: 'CNNs — Convolutions, Pooling & Feature Maps',
    content: `
<h2>Why Standard Neural Networks Fail on Images</h2>
<p>
  Consider a modest 256x256 colour image. It contains 256 * 256 * 3 = 196,608 pixel values. If you fed this into a standard fully connected neural network with 1,000 neurons in the first hidden layer, that single layer would require 196,608 * 1,000 = ~197 million parameters — just for the first layer. This is computationally wasteful and, more importantly, ignores a fundamental property of images: <strong>spatial structure</strong>. Nearby pixels are more related than distant pixels. An eye is made of locally grouped pixels, not random pixels scattered across the frame.
</p>
<p>
  <span class="term" data-term="cnn">Convolutional Neural Networks (CNNs)</span> were designed to exploit this spatial structure. Instead of connecting every input pixel to every neuron, CNNs use small, learnable filters that slide across the image, detecting local patterns. This achieves two critical properties: <span class="term" data-term="weight-sharing">weight sharing</span> (the same filter is applied everywhere, drastically reducing parameters) and <span class="term" data-term="translation-invariance">translation equivariance</span> (a pattern detected in one part of the image can be recognised anywhere else).
</p>

<div class="key-concept">
  <strong>Key Concept:</strong> CNNs encode an <em>inductive bias</em> — the assumption that spatially local patterns are important and that the same pattern can appear at different locations. This bias matches the structure of visual data, which is why CNNs dominate image tasks. When the inductive bias matches the data structure, the model learns more efficiently from less data.
</div>

<h2>The Convolution Operation</h2>
<p>
  The core operation in a CNN is the <span class="term" data-term="convolution">convolution</span>. A small matrix called a <span class="term" data-term="kernel">kernel</span> (or filter) — typically 3x3, 5x5, or 7x7 — slides across the input image (or the output of a previous layer). At each position, the kernel performs an element-wise multiplication with the overlapping input region and sums the results into a single output value. The collection of all such output values forms a <span class="term" data-term="feature-map">feature map</span> (also called an activation map).
</p>
<p>
  Mathematically, for a 2D convolution with input <code>I</code> and kernel <code>K</code>: <code>output(i,j) = sum_m sum_n I(i+m, j+n) * K(m,n)</code>. Each kernel learns to detect a specific local pattern — edges, corners, textures, colour gradients — depending on the values of its learned weights.
</p>
<p>
  A convolutional layer typically contains multiple kernels, each producing a separate feature map. If a layer has 64 kernels, it produces 64 feature maps — a 64-channel output representing 64 different detected patterns. The number of parameters in a convolutional layer is <code>(kernel_height * kernel_width * input_channels + 1) * num_kernels</code> — orders of magnitude fewer than a fully connected layer.
</p>

<div class="example-box">
  <h4>Example</h4>
  <p>A 3x3 kernel with weights <code>[[-1, -1, -1], [0, 0, 0], [1, 1, 1]]</code> acts as a horizontal edge detector. When slid over an image, it produces large positive values at horizontal edges where the image transitions from dark (top) to light (bottom), large negative values at the reverse transition, and values near zero in uniform regions. The network <em>learns</em> these kernel values during training — you do not hand-design them. Early layers learn simple patterns (edges, gradients); deeper layers learn complex patterns (textures, object parts).</p>
</div>

<h2>Padding and Stride</h2>
<p>
  Two hyperparameters control how the kernel traverses the input:
</p>
<ul>
  <li><strong><span class="term" data-term="padding">Padding</span>:</strong> Adding rows/columns of zeros around the input border. "Same" padding preserves the spatial dimensions (output size = input size). "Valid" padding uses no padding, shrinking the output. Padding ensures that border pixels participate in as many convolutions as interior pixels.</li>
  <li><strong><span class="term" data-term="stride">Stride</span>:</strong> The step size of the sliding kernel. Stride 1 moves the kernel one pixel at a time. Stride 2 skips every other position, halving the output dimensions. Higher stride reduces computation and output size, acting as a form of downsampling.</li>
</ul>
<p>
  The output spatial dimension is calculated as: <code>output_size = floor((input_size + 2*padding - kernel_size) / stride) + 1</code>.
</p>

<h2>Pooling: Downsampling Feature Maps</h2>
<p>
  <span class="term" data-term="pooling">Pooling</span> layers reduce the spatial dimensions of feature maps, decreasing computation and providing a degree of translation invariance. The two most common types are:
</p>
<ul>
  <li><strong>Max Pooling:</strong> Takes the maximum value in each spatial window (typically 2x2 with stride 2). Preserves the strongest activation — "if this feature was detected anywhere in this region, keep it." This halves the spatial dimensions at each pooling layer.</li>
  <li><strong>Average Pooling:</strong> Takes the mean value in each window. Smoother than max pooling, sometimes used in the final layers. <strong>Global Average Pooling (GAP)</strong> averages each feature map into a single value, often used as the final layer before classification to reduce spatial dimensions to 1x1.</li>
</ul>

<div class="warning">
  <strong>Common Misconception:</strong> "Pooling makes the network invariant to the position of objects." Pooling provides only <em>local</em> translation invariance within the pooling window. True position invariance emerges from the combination of many pooling operations across layers and from data augmentation during training. A CNN that has only seen cats in the centre of images may still struggle with cats in corners — this is why data augmentation (random crops, flips, shifts) is essential.
</div>

<h2>The Full CNN Architecture</h2>
<p>
  A typical CNN follows a pattern of alternating convolutional and pooling layers, gradually increasing the number of feature channels while decreasing spatial dimensions, followed by fully connected layers for the final classification:
</p>
<ol>
  <li><strong>Input:</strong> Raw image (e.g., 224x224x3 for RGB).</li>
  <li><strong>Conv blocks:</strong> Each block contains one or more convolutional layers (with ReLU activation) followed by pooling. Channels increase (64 → 128 → 256 → 512) while spatial size decreases (224 → 112 → 56 → 28 → 14 → 7).</li>
  <li><strong>Flatten or Global Average Pooling:</strong> Convert the final 3D feature map into a 1D vector.</li>
  <li><strong>Fully connected layers:</strong> One or more dense layers mapping features to output classes.</li>
  <li><strong>Output:</strong> Softmax layer producing class probabilities.</li>
</ol>

<div class="key-concept">
  <strong>Key Concept:</strong> The <em>hierarchical feature extraction</em> in CNNs is their defining strength. Layer 1 kernels learn to detect edges and colour gradients. Layer 2 combines these into textures and simple shapes. Layer 3 detects object parts (eyes, wheels, handles). Layer 4+ recognises whole objects and scenes. This hierarchical decomposition mirrors how the human visual cortex processes information — simple features in early areas (V1) building to complex object representations in higher areas (IT cortex).
</div>

<h2>Landmark CNN Architectures</h2>
<p>
  The evolution of CNN architectures tells the story of the field's progress:
</p>

<table>
  <thead>
    <tr><th>Architecture</th><th>Year</th><th>Key Innovation</th><th>Depth</th></tr>
  </thead>
  <tbody>
    <tr><td><strong>LeNet-5</strong></td><td>1998</td><td>First practical CNN (handwritten digit recognition)</td><td>5 layers</td></tr>
    <tr><td><strong>AlexNet</strong></td><td>2012</td><td>GPU training, ReLU, dropout; sparked the deep learning revolution</td><td>8 layers</td></tr>
    <tr><td><strong>VGGNet</strong></td><td>2014</td><td>Uniform 3x3 convolutions; showed depth matters</td><td>16-19 layers</td></tr>
    <tr><td><strong>GoogLeNet/Inception</strong></td><td>2014</td><td>Inception modules with parallel multi-scale convolutions</td><td>22 layers</td></tr>
    <tr><td><strong>ResNet</strong></td><td>2015</td><td>Residual (skip) connections enabling 100+ layer training</td><td>50-152 layers</td></tr>
    <tr><td><strong>EfficientNet</strong></td><td>2019</td><td>Compound scaling of depth, width, and resolution</td><td>Variable</td></tr>
  </tbody>
</table>

<div class="pro-tip">
  <strong>PM Perspective:</strong> As a PM, you will rarely design CNN architectures from scratch. Instead, you will make strategic decisions about which pre-trained CNN to use as a backbone for your application. The trade-offs are accuracy vs. latency vs. model size. A medical imaging product needing maximum accuracy might use ResNet-152; a mobile app needing real-time performance might use MobileNet or EfficientNet-B0. Understanding these trade-offs helps you specify requirements that your ML team can act on.
</div>

<h2>CNNs Beyond Image Classification</h2>
<p>
  While CNNs were developed for image classification, their ability to extract spatial features has made them foundational across computer vision:
</p>
<ul>
  <li><strong>Object Detection:</strong> Models like YOLO, Faster R-CNN, and SSD use CNN backbones to detect and localise multiple objects in an image with bounding boxes.</li>
  <li><strong>Semantic Segmentation:</strong> Models like U-Net and DeepLab classify every pixel in an image (e.g., "this pixel is road, this pixel is pedestrian").</li>
  <li><strong>Image Generation:</strong> CNNs form the backbone of GANs and diffusion model decoders.</li>
  <li><strong>Video Understanding:</strong> 3D CNNs extend convolutions to the temporal dimension for action recognition.</li>
  <li><strong>Beyond vision:</strong> 1D CNNs are effective for time series, audio, and even text classification.</li>
</ul>

<div class="example-box">
  <h4>Example</h4>
  <p><strong>Autonomous driving</strong> uses CNNs in multiple ways simultaneously: a CNN backbone detects pedestrians, vehicles, and signs (object detection); another segments the road surface and lane markings (semantic segmentation); and a third estimates the depth of objects from stereo camera pairs (depth estimation). A PM for self-driving must understand how these CNN-powered components interact, their individual failure modes, and how to set performance thresholds for safety-critical deployment.</p>
</div>
`,
    quiz: {
      questions: [
        {
          question: 'Your team is building a visual inspection system for a factory production line. The system needs to detect tiny defects (scratches, dents) on manufactured parts. The ML engineer proposes using aggressive max pooling (4x4 windows) to speed up inference. What product risk does this introduce?',
          type: 'mc',
          options: [
            'Max pooling will cause the model to overfit on training data',
            'Large pooling windows discard fine spatial detail, potentially causing the model to miss small defects that span only a few pixels',
            'Max pooling will make the model too slow for real-time inspection',
            'Max pooling prevents the use of pre-trained backbones like ResNet'
          ],
          correct: 1,
          explanation: 'Max pooling with a 4x4 window reduces spatial resolution by 75% at each application. For defect detection where defects may be only a few pixels across, this aggressive downsampling can discard the very features the model needs to detect. A better approach might be smaller pooling windows, dilated convolutions, or feature pyramid networks that preserve multi-scale information.',
          difficulty: 'applied',
          expertNote: 'A world-class PM would require the team to analyse defect size distribution in the training data and ensure the effective receptive field of the network at the detection layer is appropriate for the smallest defect size the product must catch.'
        },
        {
          question: 'Why do CNNs use dramatically fewer parameters than fully connected networks for image processing?',
          type: 'mc',
          options: [
            'CNNs use smaller images as input through mandatory preprocessing',
            'CNNs use weight sharing (same kernel applied across all spatial positions) and local connectivity (each neuron connects to only a small input region)',
            'CNNs only process grayscale images, reducing input dimensionality',
            'CNNs replace multiplication with addition, halving parameter count'
          ],
          correct: 1,
          explanation: 'Weight sharing means a single 3x3 kernel with 9 parameters is applied at every spatial position, rather than learning separate weights for each position. Local connectivity means each output neuron depends on only a small patch of the input rather than the entire image. Together, these reduce parameters by orders of magnitude while encoding the inductive bias that local spatial patterns are important.',
          difficulty: 'foundational',
          expertNote: 'A PM should understand that this parameter efficiency is not free — it encodes an assumption (spatial locality) that must match the data. For data without spatial structure (e.g., tabular data), CNNs offer no advantage over fully connected networks.'
        },
        {
          question: 'Scenario: You are a PM at a medical imaging startup. Your CNN model achieves 97% accuracy on the test set for detecting lung nodules in CT scans. The radiologist on your advisory board raises concerns that the model has never been tested on images from hospitals outside your training data. What is the specific technical risk, and how should you address it as a PM?',
          type: 'scenario',
          correct: 'The technical risk is distribution shift (also called domain shift). CT scanners from different manufacturers produce images with different noise characteristics, resolution, contrast, and artefacts. A CNN trained only on images from one hospital\'s scanner may learn features specific to that scanner rather than features of actual lung nodules, causing performance to degrade dramatically on images from other hospitals. As PM, you should: (1) Immediately commission evaluation on external datasets from different hospitals and scanner types. (2) Build a diverse training dataset spanning multiple scanner manufacturers and institutions. (3) Implement data augmentation that simulates scanner variability. (4) Consider domain adaptation techniques. (5) Establish ongoing monitoring that tracks model performance stratified by scanner type after deployment. (6) Never launch with a single-institution evaluation — regulatory bodies (FDA) also require multi-site validation.',
          explanation: 'Distribution shift is one of the most common failure modes for deployed ML systems, especially in healthcare. A model that performs well on in-distribution test data can fail catastrophically on data from different sources. Multi-site validation is both a technical best practice and a regulatory requirement.',
          difficulty: 'expert',
          expertNote: 'A DeepMind-calibre PM would also consider federated learning (training across hospitals without sharing data) and would build scanner-type metadata into the evaluation pipeline to detect performance degradation proactively.'
        },
        {
          question: 'Which of the following are valid reasons that ResNet can train networks with 150+ layers while VGGNet struggles beyond 19 layers? Select all that apply.',
          type: 'multi',
          options: [
            'ResNet uses residual (skip) connections that allow gradients to flow directly through the network, mitigating vanishing gradients',
            'ResNet uses a fundamentally different type of convolution that is more stable',
            'Skip connections allow layers to learn residual mappings (corrections to the identity), which are easier to optimise than learning the full transformation from scratch',
            'ResNet uses larger kernels that capture more context per layer',
            'The identity shortcuts in ResNet ensure that adding layers can never hurt performance in theory — the extra layers can always learn the identity function'
          ],
          correct: [0, 2, 4],
          explanation: 'ResNet\'s key innovation is the skip (residual) connection: the output of a block is the sum of the block\'s transformation and the original input. This allows gradients to flow unimpeded (addressing vanishing gradients), makes the learning target easier (learn a small correction rather than a full transformation), and ensures that deeper networks are at least as good as shallower ones (extra layers can learn to be identity mappings). ResNet uses standard 3x3 convolutions, not different or larger ones.',
          difficulty: 'applied',
          expertNote: 'Understanding ResNet is not just historical — residual connections are now ubiquitous in transformers, diffusion models, and nearly every modern architecture. A PM encountering "residual connections" in architecture discussions should recognise them as a foundational training stability technique.'
        },
        {
          question: 'Your team must choose a CNN backbone for a mobile app that identifies plant species from photos. The app runs on-device with no internet connection. Inference must complete in under 100ms on a typical smartphone. Which architectural consideration should MOST heavily influence your choice?',
          type: 'mc',
          options: [
            'The number of training epochs required to reach convergence',
            'The model\'s parameter count and floating-point operations (FLOPs), which determine inference speed and memory usage on mobile hardware',
            'Whether the model was originally trained on ImageNet or on a plant-specific dataset',
            'The activation function used in hidden layers'
          ],
          correct: 1,
          explanation: 'For on-device mobile inference with a strict latency budget, the primary constraint is the model\'s computational footprint: parameter count (affects memory and download size) and FLOPs (affects inference latency). Architectures like MobileNet and EfficientNet-B0 are specifically designed for this constraint. Training details, original dataset, and activation choice are secondary to the fundamental size/speed constraint.',
          difficulty: 'applied',
          expertNote: 'An expert PM would also consider model quantisation (int8 reduces model size by 4x), hardware acceleration APIs (CoreML on iOS, NNAPI on Android), and whether to use on-device inference exclusively or add a cloud fallback for ambiguous cases.'
        }
      ]
    }
  },

  // ─────────────────────────────────────────────
  // L02 — RNNs & LSTMs: Sequential Data & Memory
  // ─────────────────────────────────────────────
  l02: {
    title: 'RNNs & LSTMs — Sequential Data & Memory',
    content: `
<h2>The Challenge of Sequential Data</h2>
<p>
  Many real-world data types are inherently sequential — they have an ordering where context from earlier elements matters for understanding later ones. Text is sequential: the meaning of "bank" depends on whether earlier words mention "river" or "finance." Time series are sequential: tomorrow's stock price depends on the trend of preceding days. Audio is sequential: a phoneme's identity depends on the sounds before and after it. Speech, music, sensor data, user clickstreams, DNA sequences — all are sequential.
</p>
<p>
  Standard feedforward networks and CNNs process fixed-size inputs with no notion of order or memory. If you fed a sentence into a feedforward network, you would need to flatten all words into a single vector, losing their order entirely. <span class="term" data-term="rnn">Recurrent Neural Networks (RNNs)</span> were designed to process sequences by maintaining a <span class="term" data-term="hidden-state">hidden state</span> — a form of memory that carries information from earlier time steps to later ones.
</p>

<h2>The Vanilla RNN</h2>
<p>
  An RNN processes a sequence one element at a time. At each time step <code>t</code>, it takes the current input <code>x_t</code> and the previous hidden state <code>h_(t-1)</code>, combines them through learnable weight matrices, applies a non-linearity (typically tanh), and produces a new hidden state <code>h_t</code>:
</p>
<p>
  <code>h_t = tanh(W_hh * h_(t-1) + W_xh * x_t + b_h)</code>
</p>
<p>
  The output at each step can be derived from the hidden state: <code>y_t = W_hy * h_t + b_y</code>. The same weight matrices (<code>W_hh</code>, <code>W_xh</code>, <code>W_hy</code>) are shared across all time steps — this is weight sharing in the temporal dimension, analogous to CNN weight sharing in the spatial dimension.
</p>
<p>
  Different sequence tasks use different output patterns:
</p>
<ul>
  <li><strong>Many-to-one:</strong> Input is a sequence, output is a single value. Example: sentiment analysis (sequence of words → positive/negative).</li>
  <li><strong>One-to-many:</strong> Input is a single value, output is a sequence. Example: image captioning (image → sequence of words).</li>
  <li><strong>Many-to-many:</strong> Both input and output are sequences. Example: machine translation (English sentence → French sentence), speech recognition (audio frames → text).</li>
</ul>

<div class="key-concept">
  <strong>Key Concept:</strong> The hidden state <code>h_t</code> is the RNN's memory — a compressed representation of everything the network has seen so far in the sequence. In theory, it can carry information from the very first token to the very last. In practice, vanilla RNNs struggle to maintain information over long sequences due to the <span class="term" data-term="vanishing-gradient">vanishing gradient problem</span>.
</div>

<h2>The Vanishing Gradient Problem in RNNs</h2>
<p>
  When an RNN is trained using <span class="term" data-term="bptt">Backpropagation Through Time (BPTT)</span> — unrolling the network across time steps and computing gradients — the gradient signal must flow backward through every time step. At each step, the gradient is multiplied by the weight matrix <code>W_hh</code> and the derivative of the activation function. Over many time steps, these repeated multiplications cause the gradient to either:
</p>
<ul>
  <li><strong>Vanish</strong> (when the multiplied values are consistently less than 1): The gradient shrinks exponentially, and the network cannot learn long-range dependencies. A word at position 3 may have no influence on the prediction at position 100.</li>
  <li><strong>Explode</strong> (when the multiplied values are consistently greater than 1): The gradient grows exponentially, causing numerical instability and wild parameter updates. This is typically addressed by <span class="term" data-term="gradient-clipping">gradient clipping</span> — capping the gradient magnitude at a threshold.</li>
</ul>

<div class="warning">
  <strong>Common Misconception:</strong> "Vanishing gradients mean the RNN forgets — it only has short-term memory." More precisely, vanishing gradients mean the RNN cannot <em>learn</em> long-term dependencies from the training data. The hidden state can technically carry information forward, but the gradient signal used to train the network to exploit long-range patterns becomes too weak. It is a training problem, not an inference problem per se — but the practical effect is the same: vanilla RNNs behave as if they have limited memory.
</div>

<h2>LSTMs: Gated Memory Cells</h2>
<p>
  The <span class="term" data-term="lstm">Long Short-Term Memory (LSTM)</span> network, introduced by Hochreiter and Schmidhuber in 1997, was specifically designed to solve the vanishing gradient problem. The key innovation is the <span class="term" data-term="cell-state">cell state</span> — a dedicated memory highway that runs through the entire sequence with minimal transformation, allowing information to flow unchanged over many time steps.
</p>
<p>
  LSTMs control information flow through three <span class="term" data-term="gates">gates</span> — learned mechanisms that decide what information to store, discard, and output:
</p>

<table>
  <thead>
    <tr><th>Gate</th><th>Function</th><th>Analogy</th></tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Forget gate</strong> <code>f_t</code></td>
      <td>Decides what information to discard from the cell state. <code>f_t = sigmoid(W_f * [h_(t-1), x_t] + b_f)</code></td>
      <td>Like deciding what to erase from a whiteboard before writing new notes</td>
    </tr>
    <tr>
      <td><strong>Input gate</strong> <code>i_t</code></td>
      <td>Decides what new information to write to the cell state. <code>i_t = sigmoid(W_i * [h_(t-1), x_t] + b_i)</code></td>
      <td>Like deciding which of your new notes are important enough to put on the whiteboard</td>
    </tr>
    <tr>
      <td><strong>Output gate</strong> <code>o_t</code></td>
      <td>Decides what information from the cell state to expose as the hidden state output. <code>o_t = sigmoid(W_o * [h_(t-1), x_t] + b_o)</code></td>
      <td>Like deciding which whiteboard notes to share with your colleague at this moment</td>
    </tr>
  </tbody>
</table>

<p>
  The cell state update is: <code>C_t = f_t * C_(t-1) + i_t * tanh(W_c * [h_(t-1), x_t] + b_c)</code>. The forget gate can be close to 1 (retain everything) or close to 0 (erase everything). This additive structure — adding new information rather than repeatedly multiplying — is what prevents gradients from vanishing. Gradients flow through the cell state with minimal decay, enabling LSTMs to learn dependencies spanning hundreds of time steps.
</p>

<div class="example-box">
  <h4>Example</h4>
  <p>In the sentence: "The author, who grew up in Paris and later moved to Berlin where she studied philosophy before returning to her hometown, wrote the novel in <strong>French</strong>." A vanilla RNN might forget that "Paris" was mentioned by the time it reaches "French" — the gradient signal needed to connect "Paris" with the language prediction has vanished through 20+ intervening tokens. An LSTM's cell state can carry the "Paris → French language" information through the entire sentence, with the forget gate keeping it alive and the input gate filtering out irrelevant intervening details.</p>
</div>

<h2>GRUs: A Simpler Alternative</h2>
<p>
  The <span class="term" data-term="gru">Gated Recurrent Unit (GRU)</span>, introduced by Cho et al. in 2014, simplifies the LSTM by merging the forget and input gates into a single <strong>update gate</strong> and combining the cell state and hidden state into one. GRUs have fewer parameters than LSTMs and often perform comparably on many tasks, making them a popular lightweight alternative.
</p>

<table>
  <thead>
    <tr><th>Feature</th><th>LSTM</th><th>GRU</th></tr>
  </thead>
  <tbody>
    <tr><td>Gates</td><td>3 (forget, input, output)</td><td>2 (reset, update)</td></tr>
    <tr><td>Memory mechanism</td><td>Separate cell state and hidden state</td><td>Merged into single hidden state</td></tr>
    <tr><td>Parameters</td><td>More (4 weight matrices per layer)</td><td>Fewer (3 weight matrices per layer)</td></tr>
    <tr><td>Performance</td><td>Slightly better on very long sequences</td><td>Comparable on most tasks; faster to train</td></tr>
  </tbody>
</table>

<h2>Bidirectional and Multi-Layer RNNs</h2>
<p>
  Standard RNNs process sequences left-to-right, meaning predictions at position <code>t</code> can only use information from positions <code>1</code> through <code>t</code>. <span class="term" data-term="bidirectional-rnn">Bidirectional RNNs</span> run two separate RNNs — one left-to-right and one right-to-left — and concatenate their hidden states. This allows each position to incorporate context from both before and after it, significantly improving performance on tasks like named entity recognition and machine translation where future context matters.
</p>
<p>
  <strong>Stacked (multi-layer) RNNs</strong> place multiple RNN layers on top of each other, with the hidden state of one layer serving as the input to the next. This adds depth, enabling the network to learn increasingly abstract temporal representations. Stacking 2-4 LSTM layers was standard practice for state-of-the-art NLP before transformers.
</p>

<div class="pro-tip">
  <strong>PM Perspective:</strong> Before transformers, LSTMs powered virtually every production NLP system — machine translation (Google Translate used stacked LSTMs from 2016-2019), speech recognition, text summarisation, and chatbots. Understanding RNN/LSTM architecture helps you appreciate (1) why transformers were such a breakthrough (parallelisation, no vanishing gradients) and (2) why LSTMs are still used in specific domains where transformers are overkill — low-resource edge deployments, simple time-series forecasting, and streaming applications where processing must be incremental.
</div>

<h2>The Limitations That Led to Transformers</h2>
<p>
  Despite LSTMs' improvements over vanilla RNNs, they have fundamental limitations that motivated the development of the <span class="term" data-term="transformer">Transformer</span> architecture:
</p>
<ul>
  <li><strong>Sequential processing:</strong> RNNs must process tokens one at a time (each step depends on the previous hidden state), preventing parallelisation. This makes training on long sequences slow — you cannot utilise GPU parallelism effectively.</li>
  <li><strong>Finite memory bottleneck:</strong> All information about the sequence must be compressed into a fixed-size hidden state vector. For very long sequences, this bottleneck limits what can be remembered.</li>
  <li><strong>Practical context window:</strong> Despite LSTMs' theoretical ability to model long dependencies, in practice performance degrades significantly beyond a few hundred tokens.</li>
  <li><strong>No direct pairwise comparison:</strong> An RNN at position 500 can only access position 10 through the hidden state chain — it cannot directly compare the two positions. Transformers solve this with attention mechanisms that allow any position to directly attend to any other position.</li>
</ul>

<div class="key-concept">
  <strong>Key Concept:</strong> The transition from RNNs to Transformers was driven primarily by <strong>parallelisation</strong> and <strong>direct long-range attention</strong>. Transformers process all positions simultaneously (massively parallelisable on GPUs) and use attention to directly connect any two positions regardless of distance (no vanishing gradient over sequence length). This enabled training on much longer sequences and much larger datasets, unlocking the era of large language models.
</div>

<div class="example-box">
  <h4>Example</h4>
  <p>Consider training a language model on a document with 10,000 tokens. An LSTM must process all 10,000 tokens sequentially — 10,000 steps, each dependent on the last. A Transformer processes all 10,000 tokens in parallel (though with quadratic attention cost). On a modern GPU cluster, the Transformer might train 100x faster, enabling the use of 100x more training data in the same wall-clock time. This computational advantage, not a theoretical one, is what made billion-parameter language models feasible.</p>
</div>
`,
    quiz: {
      questions: [
        {
          question: 'Your team is building a predictive maintenance system for industrial equipment that processes real-time sensor streams on an edge device with limited compute. The input is a continuous time series from 12 sensors, and you need to flag anomalies within 50ms. Would you recommend an LSTM or a Transformer, and why?',
          type: 'mc',
          options: [
            'Transformer — because transformers are always superior to LSTMs',
            'LSTM — because it processes inputs incrementally (one time step at a time), is computationally lighter for streaming data, and can run within the latency budget on edge hardware',
            'Neither — time series require CNNs exclusively',
            'LSTM — because transformers cannot process numerical sensor data'
          ],
          correct: 1,
          explanation: 'For streaming time series on edge devices, LSTMs are often the better choice. They process data incrementally (maintaining a hidden state updated with each new reading) and have a much smaller computational footprint than transformers. Transformers require all positions to attend to all other positions (quadratic cost), which is expensive for long streams, and they struggle with truly incremental processing without modifications.',
          difficulty: 'applied',
          expertNote: 'A senior PM would also evaluate temporal convolutional networks (TCNs) as an alternative — they can be more parallelisable than LSTMs while still being lighter than full transformers. The choice should be validated empirically with latency profiling on the target hardware.'
        },
        {
          question: 'What specific problem does the LSTM cell state solve that the vanilla RNN hidden state cannot?',
          type: 'mc',
          options: [
            'The cell state allows the LSTM to process longer input sequences by compressing data more efficiently',
            'The cell state provides an additive information pathway through time, preventing gradient vanishing that occurs with the multiplicative hidden state updates in vanilla RNNs',
            'The cell state stores a copy of the original input sequence for reference',
            'The cell state enables parallel processing of sequence elements, which vanilla RNNs cannot do'
          ],
          correct: 1,
          explanation: 'The cell state is updated additively (new information is added; old information is retained or erased via gates) rather than through repeated matrix multiplication. In vanilla RNNs, the hidden state is repeatedly multiplied by the weight matrix, causing gradients to vanish or explode. The LSTM\'s additive cell state allows gradients to flow through time with minimal decay, enabling learning of long-range dependencies.',
          difficulty: 'foundational',
          expertNote: 'This additive gradient highway concept reappears in ResNets (residual connections) and Transformers (residual streams). Understanding it as a general principle — additive pathways preserve gradient flow — is more valuable than memorising LSTM-specific equations.'
        },
        {
          question: 'Scenario: You are PM for a real-time speech transcription product. Your current system uses a bidirectional LSTM, which achieves excellent accuracy but introduces 2 seconds of latency because it needs to "see" future context before outputting text. Users are complaining about the delay. How do you approach this latency-accuracy trade-off?',
          type: 'scenario',
          correct: 'A strong response addresses: (1) Root cause: Bidirectional LSTMs require the full input sequence (or a large chunk) before producing output because the backward pass needs future context. This inherently introduces latency equal to the chunk size. (2) Options: (a) Switch to a unidirectional LSTM or streaming transformer that outputs text incrementally with no lookahead — trades some accuracy for near-zero latency. (b) Use a small lookahead window (e.g., 200ms instead of 2s) — a bidirectional model over a short chunk provides some future context with much less delay. (c) Use a two-pass system: display preliminary unidirectional results immediately, then revise with bidirectional results after a short delay. (d) Evaluate a CTC-based or attention-based streaming architecture designed for online transcription. (3) Decision framework: Quantify the accuracy loss from each option on your test set, measure user satisfaction with different latency levels, and choose the option that maximises the product of accuracy and user experience. (4) The right answer depends on use case: live captioning tolerates more errors for low latency; medical dictation needs high accuracy even with some delay.',
          explanation: 'Bidirectional models achieve higher accuracy by using future context, but this fundamentally conflicts with real-time output. This is a classic accuracy-latency trade-off that requires PM judgment about user needs, not just engineering optimisation.',
          difficulty: 'expert',
          expertNote: 'A DeepMind-calibre PM would also evaluate emerging streaming transformer architectures (e.g., Emformer, chunked attention) that achieve near-bidirectional accuracy with bounded latency, and would set up A/B experiments measuring both word error rate and user-reported satisfaction.'
        },
        {
          question: 'Which of the following were fundamental limitations of RNNs/LSTMs that motivated the development of the Transformer architecture? Select all that apply.',
          type: 'multi',
          options: [
            'Sequential processing prevents GPU parallelisation during training',
            'RNNs cannot process text data, only numerical sequences',
            'The fixed-size hidden state creates an information bottleneck for very long sequences',
            'RNNs require more parameters than transformers for the same task',
            'There is no mechanism for directly comparing distant positions — information must flow through the entire chain of hidden states'
          ],
          correct: [0, 2, 4],
          explanation: 'The three fundamental limitations are: (1) Sequential processing — each step depends on the previous, preventing parallelisation on GPUs; (2) Information bottleneck — all sequence information must be compressed into a fixed-size vector; (3) No direct long-range connections — distant positions can only interact through the hidden state chain. RNNs can process text (via embeddings), and parameter count comparison is not a fundamental limitation.',
          difficulty: 'applied',
          expertNote: 'Understanding these limitations is critical because they explain why transformers dominate: self-attention provides direct pairwise connections (solving #3), operates in parallel (solving #1), and the attention pattern allows flexible, position-dependent memory (partially addressing #2).'
        },
        {
          question: 'An LSTM forget gate with an output consistently near 1.0 for a particular cell state dimension indicates that the network has learned to:',
          type: 'mc',
          options: [
            'Completely erase that dimension of information at every time step',
            'Retain that dimension of information across time steps, treating it as a long-term memory',
            'Ignore the current input entirely and rely on the previous hidden state',
            'Output that dimension of the cell state to the next layer'
          ],
          correct: 1,
          explanation: 'The forget gate controls how much of the previous cell state to retain. A value near 1.0 means "keep almost everything" — that dimension of the cell state carries its information forward through time with minimal decay. This is the mechanism that enables LSTMs to maintain long-term memories. A value near 0.0 would mean "erase this memory."',
          difficulty: 'foundational',
          expertNote: 'Analysing gate activations is a useful interpretability technique. A PM can ask the ML team to visualise which gates activate for which types of input, providing insight into what the model has learned to remember and forget.'
        }
      ]
    }
  },

  // ─────────────────────────────────────────────
  // L03 — Training Deep Networks: BatchNorm, Dropout & Schedules
  // ─────────────────────────────────────────────
  l03: {
    title: 'Training Deep Networks — BatchNorm, Dropout & Schedules',
    content: `
<h2>The Challenge of Training Deep Networks</h2>
<p>
  Having a deep network architecture is necessary but not sufficient — you must also be able to <em>train</em> it effectively. Deep networks with dozens or hundreds of layers face training challenges that do not afflict shallow networks: vanishing and exploding gradients, <span class="term" data-term="internal-covariate-shift">internal covariate shift</span>, overfitting, sensitivity to hyperparameters, and long training times. The techniques in this lesson — <span class="term" data-term="batch-normalisation">Batch Normalisation</span>, <span class="term" data-term="dropout">Dropout</span>, learning rate schedules, and other regularisation methods — are the practical innovations that made deep learning reliable and reproducible. Without them, modern architectures simply would not converge.
</p>
<p>
  For a PM, these techniques determine training cost, model reliability, and the gap between "works in research" and "works in production." Understanding them helps you assess timelines, costs, and risk when your ML team discusses training infrastructure.
</p>

<h2>Batch Normalisation (BatchNorm)</h2>
<p>
  Introduced by Ioffe and Szegedy in 2015, <span class="term" data-term="batch-normalisation">Batch Normalisation</span> was one of the most impactful practical innovations in deep learning. The idea is simple: normalise the inputs to each layer so they have zero mean and unit variance, computed across the current mini-batch during training. This is followed by a learnable scale (<code>gamma</code>) and shift (<code>beta</code>) that allow the network to undo the normalisation if that is beneficial.
</p>
<p>
  The BatchNorm computation for a given feature: <code>x_normalized = (x - mean_batch) / sqrt(var_batch + epsilon)</code>, then <code>output = gamma * x_normalized + beta</code>.
</p>
<p>
  Why does this help? The original paper attributed it to reducing "internal covariate shift" — the phenomenon where each layer's input distribution changes as the preceding layers' parameters are updated, forcing every layer to continuously adapt to shifting inputs. While the theoretical explanation has been debated (some researchers argue BatchNorm's benefit comes from smoothing the loss landscape rather than reducing covariate shift), the practical benefits are clear and dramatic:
</p>
<ul>
  <li><strong>Faster convergence:</strong> Models train significantly faster (often 2-10x), allowing higher learning rates without instability.</li>
  <li><strong>Reduced sensitivity to initialisation:</strong> The network is more forgiving of imperfect weight initialisation.</li>
  <li><strong>Regularisation effect:</strong> The mini-batch statistics introduce noise (each batch has slightly different statistics), providing a mild regularisation effect that reduces overfitting.</li>
  <li><strong>Enables deeper networks:</strong> Without BatchNorm, training very deep networks (50+ layers) is often unstable; with it, training becomes reliable.</li>
</ul>

<div class="key-concept">
  <strong>Key Concept:</strong> During training, BatchNorm uses per-batch statistics (mean and variance). During inference, there are no batches — so BatchNorm uses <em>running averages</em> of the training statistics (accumulated via exponential moving average during training). This train/inference discrepancy can cause subtle bugs if the running statistics do not represent the deployment data well. Your ML team should test that BatchNorm statistics generalise to production data.
</div>

<div class="warning">
  <strong>Common Misconception:</strong> "BatchNorm is always beneficial." BatchNorm can cause problems with very small batch sizes (noisy statistics), in RNNs (sequence lengths vary), and when there is a significant distribution shift between training and deployment data (the running statistics become stale). Alternatives like <span class="term" data-term="layer-norm">Layer Normalisation</span> (used in Transformers), <span class="term" data-term="group-norm">Group Normalisation</span>, and <span class="term" data-term="instance-norm">Instance Normalisation</span> (used in style transfer) address different limitations.
</div>

<table>
  <thead>
    <tr><th>Normalisation</th><th>Computes Statistics Over</th><th>Best For</th></tr>
  </thead>
  <tbody>
    <tr><td><strong>BatchNorm</strong></td><td>Batch dimension (for each feature)</td><td>CNNs with large batch sizes</td></tr>
    <tr><td><strong>LayerNorm</strong></td><td>Feature dimension (for each example)</td><td>Transformers, RNNs, small batches</td></tr>
    <tr><td><strong>GroupNorm</strong></td><td>Groups of channels (for each example)</td><td>Object detection, small batch sizes</td></tr>
    <tr><td><strong>InstanceNorm</strong></td><td>Each channel individually (for each example)</td><td>Style transfer, image generation</td></tr>
  </tbody>
</table>

<h2>Dropout: Regularisation Through Random Deactivation</h2>
<p>
  <span class="term" data-term="dropout">Dropout</span>, introduced by Hinton et al. in 2014, is a regularisation technique that prevents overfitting by randomly setting a fraction of neurons to zero during each training forward pass. A typical dropout rate is 0.1-0.5 (10-50% of neurons deactivated). During inference, all neurons are active but their outputs are scaled by <code>(1 - dropout_rate)</code> to compensate for the fact that more neurons are active than during training (this is called "inverted dropout" when applied during training instead).
</p>
<p>
  Why does randomly removing neurons reduce overfitting? Several complementary intuitions:
</p>
<ul>
  <li><strong>Prevents co-adaptation:</strong> Without dropout, neurons can develop complex co-dependencies — specific groups of neurons that only work together. Dropout forces each neuron to be useful independently, since it cannot rely on specific peers being present.</li>
  <li><strong>Implicit ensemble:</strong> Each training step with dropout uses a different random sub-network. Inference with all neurons is approximately equivalent to averaging the predictions of exponentially many sub-networks — similar to an ensemble, which is known to reduce variance and improve generalisation.</li>
  <li><strong>Noise injection:</strong> Dropout adds noise to the training process, which prevents the network from memorising individual training examples.</li>
</ul>

<div class="example-box">
  <h4>Example</h4>
  <p>Consider training a text classifier. Without dropout, the network might learn that whenever the words "terrible" and "movie" appear together, the sentiment is negative — but it memorises this specific word pair rather than learning a general concept of negative sentiment. With dropout, sometimes the neuron responding to "terrible" is dropped, forcing the network to also learn that "awful," "horrendous," and "dreadful" are relevant. This produces a more robust model that generalises better to unseen reviews.</p>
</div>

<div class="pro-tip">
  <strong>PM Perspective:</strong> Dropout is typically used during training only. If your team reports that a model performs well in training but poorly in production, one potential cause is that dropout was accidentally left enabled during inference (a surprisingly common bug). Another is that the dropout rate was too high, preventing the network from learning complex patterns. These are debugging details a PM does not need to fix but should recognise as diagnostic clues.
</div>

<h2>Learning Rate Schedules</h2>
<p>
  The <span class="term" data-term="learning-rate">learning rate</span> is arguably the single most important hyperparameter in neural network training. Too high, and the model diverges or oscillates wildly. Too low, and training is painfully slow or gets stuck in poor local minima. The solution: <span class="term" data-term="lr-schedule">learning rate schedules</span> that adjust the learning rate over the course of training.
</p>
<p>
  Common scheduling strategies:
</p>
<ul>
  <li><strong>Step decay:</strong> Reduce the learning rate by a factor (e.g., 10x) at predetermined epochs. Simple but effective. Requires manual tuning of when to decay.</li>
  <li><strong>Cosine annealing:</strong> Smoothly decreases the learning rate following a cosine curve from the initial value to near zero. Widely used for training transformers and large models. <code>lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T))</code></li>
  <li><strong>Warmup:</strong> Start with a very small learning rate and linearly increase it over the first few thousand steps, then decay. The "warmup + cosine decay" schedule is standard for transformer training. Warmup prevents the model from making destructive large updates when the weights are randomly initialised and the gradients are noisy.</li>
  <li><strong>Reduce on plateau:</strong> Monitor validation loss and reduce the learning rate when it stops improving. Adaptive and requires no preset schedule.</li>
  <li><strong>Cyclical learning rates:</strong> Oscillate the learning rate between bounds. Can help escape local minima and has been shown to speed convergence in some cases.</li>
</ul>

<div class="key-concept">
  <strong>Key Concept:</strong> The "warmup + cosine decay" schedule is the de facto standard for training large language models and transformers. The warmup phase (typically 1-5% of total steps) allows the model to stabilise before the learning rate reaches its maximum. The cosine decay then gradually reduces the rate, enabling fine-grained optimisation in later stages. This schedule was key to stable training of models like GPT-3 and Gemini.
</div>

<h2>Weight Initialisation</h2>
<p>
  How you initialise the network's weights before training begins has a significant impact on whether training succeeds. If weights are initialised too large, activations and gradients explode. Too small, and they vanish. Good initialisation sets the scale of weights so that the variance of activations is approximately preserved across layers at the start of training.
</p>
<ul>
  <li><strong>Xavier/Glorot initialisation:</strong> Weights drawn from <code>N(0, sqrt(2 / (fan_in + fan_out)))</code>. Designed for layers with tanh or sigmoid activations.</li>
  <li><strong>He/Kaiming initialisation:</strong> Weights drawn from <code>N(0, sqrt(2 / fan_in))</code>. Designed for ReLU activations (accounts for the fact that ReLU zeros out half of activations). Standard for modern CNNs.</li>
</ul>

<h2>Data Augmentation</h2>
<p>
  <span class="term" data-term="data-augmentation">Data augmentation</span> artificially expands the training dataset by applying random transformations to training examples. For images, common augmentations include random cropping, horizontal flipping, rotation, colour jittering, cutout (masking random patches), and mixup (blending two images and their labels). For text, augmentations include synonym replacement, random insertion/deletion, back-translation, and paraphrasing.
</p>
<p>
  Data augmentation serves two purposes: (1) it increases the effective dataset size, reducing overfitting, and (2) it teaches the model invariances — if you randomly flip images during training, the model learns that objects look the same when flipped.
</p>

<div class="example-box">
  <h4>Example</h4>
  <p><strong>CutMix and MixUp</strong> are advanced augmentation strategies used in state-of-the-art image training. MixUp creates new training examples by linearly interpolating between two images and their labels: <code>image_new = 0.7 * image_A + 0.3 * image_B</code>, <code>label_new = 0.7 * label_A + 0.3 * label_B</code>. CutMix replaces a random rectangular patch of one image with a patch from another, with labels weighted by the area ratio. Both techniques have been shown to improve accuracy and calibration, especially on small datasets.</p>
</div>

<h2>Other Regularisation Techniques</h2>
<ul>
  <li><strong>Weight decay (L2 regularisation):</strong> Adds a penalty proportional to the squared magnitude of weights: <code>L_total = L_data + lambda * sum(w^2)</code>. Discourages large weights, promoting simpler models. In AdamW (the optimiser used for most LLMs), weight decay is implemented as a direct decay of weights rather than an L2 penalty on the loss.</li>
  <li><strong>Early stopping:</strong> Monitor validation loss during training and stop when it begins to increase, reverting to the checkpoint with the lowest validation loss. Simple, effective, and widely used.</li>
  <li><strong>Label smoothing:</strong> Instead of training with hard labels (1 for the correct class, 0 for all others), use soft labels (e.g., 0.9 for the correct class, 0.1/(K-1) for others). This prevents the model from becoming overconfident and improves calibration — the model's predicted probabilities more accurately reflect true likelihoods.</li>
  <li><strong>Stochastic depth:</strong> Randomly skip (bypass via residual connections) entire layers during training. Similar in spirit to dropout but applied at the layer level. Used in very deep networks.</li>
</ul>

<div class="pro-tip">
  <strong>PM Perspective:</strong> These training techniques collectively determine how long training takes, how much compute it costs, and how robust the resulting model is. When your team says "we need another week of training" or "we need 2x more GPUs," the reason is often hyperparameter search over these choices: batch size, learning rate schedule, dropout rate, augmentation strategy, weight decay strength. Supporting your team with compute resources for proper hyperparameter tuning is one of the highest-ROI investments a PM can make — a well-tuned model can outperform a poorly-tuned model with 10x more parameters.
</div>
`,
    quiz: {
      questions: [
        {
          question: 'Your ML team is training a large image classifier and reports that the model performs well on training data but poorly on the validation set. They propose three interventions. Which combination is MOST likely to address overfitting?',
          type: 'mc',
          options: [
            'Increase the learning rate and remove all augmentation',
            'Add dropout, increase data augmentation, and apply early stopping',
            'Switch from BatchNorm to LayerNorm and increase model depth',
            'Reduce the dataset size and increase the number of training epochs'
          ],
          correct: 1,
          explanation: 'Overfitting (high training accuracy, low validation accuracy) is addressed by regularisation: dropout prevents co-adaptation, data augmentation increases effective dataset diversity, and early stopping halts training before the model memorises the training set. Increasing learning rate risks divergence. Reducing data size worsens overfitting. Switching normalisation and adding depth do not directly address overfitting.',
          difficulty: 'foundational',
          expertNote: 'An expert PM would also ask the team to check whether the validation set is representative of production data, since apparent "overfitting" can sometimes be distribution mismatch rather than true memorisation.'
        },
        {
          question: 'Why does modern transformer training use a "warmup + cosine decay" learning rate schedule rather than a constant learning rate?',
          type: 'mc',
          options: [
            'A constant learning rate causes the model to learn only the first few examples in the dataset',
            'Warmup prevents destructively large gradient updates when weights are randomly initialised, and cosine decay enables progressively finer optimisation as training proceeds',
            'The schedule is required by the Adam optimiser and cannot be changed',
            'Warmup reduces memory usage and cosine decay reduces training time'
          ],
          correct: 1,
          explanation: 'At the start of training, weights are random and gradients are noisy and potentially large. A high learning rate at this point can push the model into a poor region of parameter space from which it cannot recover. Warmup starts small and ramps up, allowing the model to stabilise. Cosine decay gradually reduces the learning rate later in training, enabling finer-grained optimisation as the model approaches a good solution.',
          difficulty: 'applied',
          expertNote: 'The warmup length, peak learning rate, and total training steps are interrelated hyperparameters that significantly affect final model quality. A PM should understand that these are not "set and forget" — they require tuning for each model scale and dataset.'
        },
        {
          question: 'Scenario: Your team has deployed a BatchNorm-based image classifier in production. Performance was excellent during testing but has degraded over 6 months. Investigation reveals that the types of images submitted by users have gradually shifted (e.g., more low-light images than in the training data). How does BatchNorm specifically contribute to this degradation, and what should you do?',
          type: 'scenario',
          correct: 'BatchNorm uses running statistics (mean and variance) computed during training to normalise inputs at inference time. These statistics reflect the training data distribution. As the production data distribution shifts (more low-light images with different pixel value distributions), the running statistics become stale — they no longer accurately normalise the incoming data, causing feature maps to have unexpected scales that degrade downstream predictions. Solutions: (1) Periodically retrain or fine-tune the model on recent production data to update both the model weights and BatchNorm running statistics. (2) Implement online BatchNorm statistic updates using recent production data (test-time adaptation). (3) Consider replacing BatchNorm with GroupNorm or LayerNorm, which compute statistics per-example and are therefore immune to distribution shift in the batch statistics. (4) Set up monitoring that detects distribution shift early (e.g., tracking the distribution of pixel intensities, model confidence scores, or prediction entropy over time).',
          explanation: 'This is a concrete example of how a training technique (BatchNorm) creates a deployment vulnerability. The running statistics are a snapshot of the training distribution — when production data drifts away from that distribution, the normalisation becomes inappropriate. This is a subtle but common production issue.',
          difficulty: 'expert',
          expertNote: 'A DeepMind-calibre PM would implement a continuous monitoring pipeline comparing the distribution of incoming data against the training distribution (using metrics like KL divergence or Population Stability Index) and establish automated alerts when drift exceeds a threshold, triggering model retraining.'
        },
        {
          question: 'Label smoothing replaces hard targets like [0, 0, 1, 0] with soft targets like [0.033, 0.033, 0.9, 0.033]. Which of the following are valid benefits of this technique? Select all that apply.',
          type: 'multi',
          options: [
            'Prevents the model from becoming overconfident, improving calibration of predicted probabilities',
            'Makes the model train faster by providing more gradient signal',
            'Improves generalisation by encouraging the model to learn from uncertain examples',
            'Reduces the model\'s parameter count',
            'Helps when labels in the training data contain noise (some may be incorrect)'
          ],
          correct: [0, 2, 4],
          explanation: 'Label smoothing prevents overconfidence (the model does not try to push logits to infinity for the correct class), improves generalisation (the model maintains some uncertainty, which translates to better performance on borderline cases), and provides natural robustness to label noise (the soft targets already acknowledge that the "correct" label may not be 100% certain). It does not reduce parameter count and does not necessarily speed up training.',
          difficulty: 'applied',
          expertNote: 'Label smoothing is now standard in large-scale training (used in training Inception, EfficientNet, and many transformers). A PM should understand that the smoothing parameter (e.g., 0.1) is another hyperparameter that affects model calibration and should be tuned.'
        },
        {
          question: 'A junior engineer proposes using a dropout rate of 0.9 (dropping 90% of neurons) to eliminate overfitting. Why is this likely to be counterproductive?',
          type: 'mc',
          options: [
            'Dropout rates above 0.5 cause numerical overflow in the weight gradients',
            'With 90% of neurons dropped, the effective capacity of the network at each training step is so small that it cannot learn the underlying patterns — it underfits severely',
            'Dropout rates must be exactly 0.5 to maintain the mathematical guarantees of the technique',
            'High dropout rates prevent BatchNorm from computing accurate statistics'
          ],
          correct: 1,
          explanation: 'With a dropout rate of 0.9, only 10% of neurons are active during each training step. This creates an extremely low-capacity sub-network at each step, likely insufficient to capture the complexity of the data. The model will underfit — failing to learn even the training data, let alone generalising. Typical dropout rates range from 0.1-0.5, balancing regularisation against learning capacity.',
          difficulty: 'foundational',
          expertNote: 'The optimal dropout rate depends on model size and data quantity. Larger models with less data benefit from higher dropout (up to 0.5); smaller models or data-rich regimes may need little to no dropout. This is an empirical choice guided by validation performance.'
        }
      ]
    }
  },

  // ─────────────────────────────────────────────
  // L04 — Architectures That Changed Everything: ResNet, U-Net, GAN
  // ─────────────────────────────────────────────
  l04: {
    title: 'Architectures That Changed Everything — ResNet, U-Net, GAN',
    content: `
<h2>Why Architecture Innovation Matters</h2>
<p>
  In deep learning, the <em>structure</em> of the network — how layers are connected, how information flows, what operations are performed — is as important as the data it is trained on or the optimiser used. Architectural innovations have repeatedly unlocked capabilities that were previously impossible, not by changing the mathematical primitives (convolution, matrix multiplication, non-linear activation) but by rearranging them in ways that enable better gradient flow, richer feature extraction, or entirely new capabilities like image generation.
</p>
<p>
  This lesson covers three architectures that fundamentally changed the field: <span class="term" data-term="resnet">ResNet</span> (which showed you can train networks with hundreds of layers), <span class="term" data-term="unet">U-Net</span> (which enabled pixel-level predictions for medical imaging and beyond), and <span class="term" data-term="gan">GANs</span> (which demonstrated that neural networks can generate photorealistic images). Each represents a different architectural insight with lasting influence.
</p>

<h2>ResNet: Learning Residual Functions</h2>
<p>
  Before ResNet, a paradoxical empirical observation hindered deep learning: adding more layers to a network should never hurt — the extra layers could simply learn the identity function. In practice, however, networks deeper than ~20 layers often performed <em>worse</em> than their shallower counterparts. This was not overfitting (training error was also higher); it was a fundamental optimisation difficulty — deeper networks were harder to train.
</p>
<p>
  <span class="term" data-term="resnet">Residual Networks (ResNet)</span>, introduced by Kaiming He et al. in 2015, solved this with a beautifully simple idea: <span class="term" data-term="skip-connection">skip connections</span> (also called shortcut connections or residual connections). Instead of learning a full transformation <code>H(x)</code>, each block learns a residual <code>F(x) = H(x) - x</code>, and the block's output is <code>F(x) + x</code>. The input <code>x</code> is added directly to the block's output via a skip connection.
</p>
<p>
  Why does this matter? Three complementary explanations:
</p>
<ul>
  <li><strong>Easy identity:</strong> If the optimal transformation is close to the identity, the block only needs to learn <code>F(x) ≈ 0</code>, which is much easier than learning <code>H(x) ≈ x</code> from scratch. The skip connection provides a "default" identity path.</li>
  <li><strong>Gradient highway:</strong> During backpropagation, the gradient flows through the skip connection unmodified (the derivative of <code>x</code> is 1). This provides a direct gradient path from loss to early layers, mitigating vanishing gradients regardless of network depth.</li>
  <li><strong>Ensemble effect:</strong> A ResNet can be viewed as an implicit ensemble of many networks of different effective depths, since the skip connections create exponentially many paths through the network. This redundancy improves robustness.</li>
</ul>

<div class="key-concept">
  <strong>Key Concept:</strong> ResNet's skip connections did not just enable deeper networks — they introduced a design principle now used everywhere in deep learning. Transformers use residual connections around every attention and feedforward block. Diffusion models use them throughout. U-Nets use them to connect encoder and decoder. The residual connection is arguably the single most influential architectural innovation since backpropagation itself.
</div>

<p>
  ResNet architectures come in several sizes: ResNet-18, ResNet-34, ResNet-50, ResNet-101, and ResNet-152, with the number indicating the number of layers. ResNet-50 and ResNet-101 are the most commonly used backbones in computer vision. ResNet-50 uses "bottleneck" blocks (1x1 → 3x3 → 1x1 convolutions) to reduce computation while maintaining depth.
</p>

<div class="example-box">
  <h4>Example</h4>
  <p>Before ResNet, the ImageNet competition winners used 8-22 layer networks. ResNet-152 — a 152-layer network — won the 2015 competition with a 3.57% top-5 error rate, surpassing human performance (estimated at ~5%). This was a stunning proof that depth, when coupled with skip connections, enables capabilities unreachable by shallower networks. More importantly, ResNet's pre-trained weights became the go-to "backbone" for nearly every downstream vision task — object detection, segmentation, medical imaging — via transfer learning.</p>
</div>

<h2>U-Net: Encoder-Decoder with Skip Connections for Dense Prediction</h2>
<p>
  Image classification assigns one label to an entire image. But many real-world tasks require <span class="term" data-term="dense-prediction">dense prediction</span> — assigning a label or value to every pixel. Medical image segmentation (outlining tumour boundaries), autonomous driving (segmenting road, sky, pedestrians), and satellite imagery analysis all require pixel-level output. The challenge: classification networks progressively downsample the image to extract high-level features, losing the fine spatial detail needed for pixel-level prediction.
</p>
<p>
  <span class="term" data-term="unet">U-Net</span>, introduced by Ronneberger, Fischer, and Brox in 2015, solved this with an elegant <span class="term" data-term="encoder-decoder">encoder-decoder</span> architecture shaped like a "U":
</p>
<ul>
  <li><strong>Encoder (contracting path):</strong> Standard CNN (convolution + pooling) that progressively downsamples the image while increasing feature channels. This extracts high-level semantic features but loses spatial precision.</li>
  <li><strong>Decoder (expanding path):</strong> Progressively upsamples the feature maps (using transposed convolutions or upsampling + convolution) back to the original image resolution, recovering spatial detail.</li>
  <li><strong>Skip connections between corresponding encoder and decoder layers:</strong> The key innovation. Feature maps from the encoder are concatenated with the upsampled feature maps in the decoder at each scale level. This directly provides the decoder with the fine-grained spatial information lost during downsampling, combined with the high-level semantic information from the decoder's own upsampled features.</li>
</ul>

<div class="key-concept">
  <strong>Key Concept:</strong> U-Net's skip connections (different from ResNet's) concatenate feature maps across the encoder-decoder divide. This combines "what" information (semantic features from deep layers — "this region is a tumour") with "where" information (spatial features from shallow layers — "the boundary is precisely here"). This combination is what enables U-Net to produce sharp, accurate segmentation boundaries.
</div>

<p>
  U-Net was originally designed for biomedical image segmentation and was particularly notable for working well with very small training datasets — a common constraint in medical imaging. Key variants and descendants include:
</p>
<ul>
  <li><strong>V-Net:</strong> 3D extension for volumetric medical data (CT/MRI scans).</li>
  <li><strong>Attention U-Net:</strong> Adds attention gates that learn to focus on relevant regions.</li>
  <li><strong>nnU-Net:</strong> "No New Net" — an auto-configuring framework that adapts U-Net to any medical segmentation task without manual architecture design. Won numerous medical imaging challenges.</li>
  <li><strong>U-Net in diffusion models:</strong> The U-Net architecture (with modifications) is used as the denoising network in Stable Diffusion and many other image generation models.</li>
</ul>

<div class="pro-tip">
  <strong>PM Perspective:</strong> U-Net's dominance in medical imaging makes it one of the most commercially important architectures. If your product involves image segmentation — medical, satellite, industrial, or scientific — U-Net variants are likely the starting point. The nnU-Net framework is particularly relevant: it automates architecture decisions, reducing the need for ML expertise in medical teams and accelerating time-to-deployment.
</div>

<h2>GANs: The Adversarial Framework</h2>
<p>
  <span class="term" data-term="gan">Generative Adversarial Networks (GANs)</span>, introduced by Ian Goodfellow in 2014, represent a fundamentally different approach to generative modelling. Instead of explicitly modelling the data distribution (as VAEs do), GANs set up a <strong>game between two neural networks</strong>:
</p>
<ul>
  <li><strong>Generator (G):</strong> Takes random noise as input and generates synthetic data (e.g., images). Its goal is to produce data indistinguishable from real data.</li>
  <li><strong>Discriminator (D):</strong> Takes an image (either real or generated) and classifies it as real or fake. Its goal is to correctly distinguish generated data from real data.</li>
</ul>
<p>
  The two networks are trained adversarially. The generator tries to fool the discriminator; the discriminator tries not to be fooled. Formally, they play a minimax game: <code>min_G max_D [E[log D(x)] + E[log(1 - D(G(z)))]]</code>. At equilibrium, the generator produces data so realistic that the discriminator cannot do better than random chance (50/50).
</p>

<div class="example-box">
  <h4>Example</h4>
  <p>Think of a GAN as a counterfeiter (generator) and a detective (discriminator). The counterfeiter creates fake banknotes. The detective examines notes and declares each real or fake. Over time, the counterfeiter learns to produce increasingly convincing fakes to fool the detective, and the detective learns to detect increasingly subtle forgery markers. The competitive pressure drives both to improve. At the end, the counterfeiter can produce notes that even an expert cannot distinguish from real ones.</p>
</div>

<h3>Training Challenges</h3>
<p>
  GANs are notoriously difficult to train. The adversarial dynamics create several pathologies:
</p>
<ul>
  <li><strong><span class="term" data-term="mode-collapse">Mode collapse</span>:</strong> The generator produces only a few types of output that fool the discriminator, ignoring the diversity of the real data. For example, a face-generating GAN might produce only one type of face.</li>
  <li><strong>Training instability:</strong> The generator-discriminator balance is delicate. If the discriminator is too strong, the generator receives no useful gradient signal (all its outputs are easily classified as fake). If the generator is too strong, the discriminator cannot provide meaningful feedback.</li>
  <li><strong>Non-convergence:</strong> The minimax game may oscillate rather than converge to equilibrium.</li>
  <li><strong>Evaluation difficulty:</strong> Unlike supervised learning with clear accuracy metrics, measuring GAN quality requires specialised metrics like <span class="term" data-term="fid">Frechet Inception Distance (FID)</span> and Inception Score (IS), which are imperfect proxies for human perceptual quality.</li>
</ul>

<div class="warning">
  <strong>Common Misconception:</strong> "GANs are the best method for image generation." While GANs produce sharp, high-quality images, diffusion models have largely surpassed them in diversity, controllability, and training stability since 2021. DALL-E 2, Stable Diffusion, and Imagen all use diffusion models rather than GANs. GANs remain relevant for real-time generation (they are faster at inference — one forward pass vs. many diffusion steps) and for specific applications like style transfer, but they are no longer the dominant generative paradigm.
</div>

<h3>Landmark GAN Variants</h3>
<table>
  <thead>
    <tr><th>Variant</th><th>Year</th><th>Key Innovation</th></tr>
  </thead>
  <tbody>
    <tr><td><strong>DCGAN</strong></td><td>2015</td><td>Convolutional architecture guidelines for stable GAN training</td></tr>
    <tr><td><strong>Conditional GAN (cGAN)</strong></td><td>2014</td><td>Conditions generation on a class label or input image (pix2pix)</td></tr>
    <tr><td><strong>CycleGAN</strong></td><td>2017</td><td>Unpaired image-to-image translation (horse→zebra without paired examples)</td></tr>
    <tr><td><strong>Progressive GAN</strong></td><td>2017</td><td>Grows resolution progressively during training for high-res images</td></tr>
    <tr><td><strong>StyleGAN / StyleGAN2</strong></td><td>2018-19</td><td>Style-based generator producing the most photorealistic faces ever at the time</td></tr>
    <tr><td><strong>WGAN</strong></td><td>2017</td><td>Wasserstein distance loss for more stable training</td></tr>
  </tbody>
</table>

<h2>The Broader Impact: Architecture as Product Strategy</h2>
<p>
  These three architectures illustrate a broader principle: <strong>architectural innovation is a competitive moat</strong>. ResNet unlocked the ability to train far deeper networks, creating a foundation used in thousands of applications. U-Net made pixel-level prediction accessible and reliable, enabling entire industries (medical imaging, satellite analysis). GANs opened the door to photorealistic generation, spawning applications from art to fashion to gaming.
</p>
<p>
  For a PM, the lesson is not about the technical details of skip connections or adversarial training — it is about recognising that architectural breakthroughs create new capability envelopes. When a new architecture emerges (as Transformers did in 2017, diffusion models in 2020), it reshapes what products are possible. Staying informed about architectural trends is not academic curiosity — it is strategic intelligence.
</p>

<div class="pro-tip">
  <strong>PM Perspective:</strong> When evaluating whether to adopt a new architecture, ask three questions: (1) Does it solve a problem our current approach cannot? (Not "is it newer" but "is it better for our use case?") (2) Is it mature enough for production? (Research papers are exciting but production-ready frameworks, tooling, and community support matter more.) (3) What is the migration cost? (Retraining, re-evaluation, infrastructure changes.) The best PM decisions are driven by capability gaps, not hype.
</div>

<div class="example-box">
  <h4>Example</h4>
  <p><strong>Pix2Pix (Conditional GAN)</strong> demonstrated paired image-to-image translation: given a sketch, generate a photorealistic image; given a daytime photo, generate a nighttime version; given a satellite image, generate a street map. This architecture directly inspired product features in design tools (concept art generation), gaming (automatic texture generation), and architecture (rendering building facades from sketches). A PM who understood cGANs could immediately see the product applications when the paper was published.</p>
</div>
`,
    quiz: {
      questions: [
        {
          question: 'Your team is designing a medical image segmentation model to outline tumour boundaries in MRI scans. The output must be a pixel-level mask at the original image resolution. Which architecture is the most natural fit, and why?',
          type: 'mc',
          options: [
            'ResNet-152 — its depth provides the highest feature extraction capability',
            'A standard GAN — adversarial training produces the most realistic-looking segmentations',
            'U-Net — its encoder-decoder structure with skip connections combines high-level semantic features with fine spatial detail for precise pixel-level predictions',
            'A vanilla CNN classifier applied to each pixel independently'
          ],
          correct: 2,
          explanation: 'U-Net was specifically designed for medical image segmentation. Its encoder captures "what" (semantic understanding of tissue types), its decoder recovers "where" (spatial precision), and the skip connections combine both for precise boundaries. ResNet is a classification backbone, not designed for dense prediction. GANs are for generation, not segmentation. Per-pixel classification ignores spatial context.',
          difficulty: 'foundational',
          expertNote: 'A strong PM would also evaluate nnU-Net (auto-configuring U-Net), consider 3D U-Net for volumetric scans, and ensure the evaluation metric matches clinical needs (Dice coefficient for overlap, Hausdorff distance for boundary accuracy).'
        },
        {
          question: 'A startup pitches you on a GAN-based product that generates synthetic medical images for training data augmentation. What is the MOST significant risk you should evaluate as a PM?',
          type: 'mc',
          options: [
            'GANs cannot generate high-resolution images',
            'Mode collapse — the GAN may produce limited diversity in synthetic images, failing to represent the full variability of real medical data, and potentially introducing systematic biases into models trained on the synthetic data',
            'GANs are too slow for batch generation of training data',
            'Synthetic medical images are illegal in all jurisdictions'
          ],
          correct: 1,
          explanation: 'Mode collapse is the critical risk: if the GAN generates only a narrow range of pathologies, patient demographics, or imaging conditions, models trained on this synthetic data will inherit those gaps. In medical contexts, this could mean the model fails on underrepresented patient populations or rare conditions — with potentially life-threatening consequences. The other options are either false or secondary concerns.',
          difficulty: 'applied',
          expertNote: 'A DeepMind-calibre PM would require quantitative diversity analysis of the synthetic data (comparing feature distributions against real data), conduct separate evaluation of downstream models trained on synthetic vs. real data, and consider regulatory implications (FDA guidance on synthetic data is evolving).'
        },
        {
          question: 'Scenario: Your company uses a ResNet-50 backbone for an image recognition API. A competitor announces they have switched to a Vision Transformer (ViT) and claims superior accuracy. Your engineering lead wants to immediately retrain with ViT. How should you evaluate this decision?',
          type: 'scenario',
          correct: 'A thoughtful evaluation would cover: (1) Verify the claim: Do ViTs outperform ResNet on YOUR specific task and dataset, or just on the competitor\'s? Academic benchmarks may not reflect production conditions. Run a head-to-head comparison on your own evaluation set. (2) Understand the trade-offs: ViTs typically require larger datasets to match CNN performance (they have less inductive bias for spatial structure). If your dataset is smaller, ResNet may still be superior. ViTs also have different compute/memory profiles — quadratic attention cost matters for high-resolution images. (3) Assess infrastructure impact: Does your serving infrastructure support the new model? What is the latency difference? Memory requirements? (4) Consider migration cost: Retraining, re-evaluation, updating monitoring and alerts, potential regression testing. (5) Evaluate incrementally: Consider using a ViT backbone for a shadow deployment (running alongside the current model, comparing outputs) before full migration. (6) The decision should be driven by measured performance on your task, cost-benefit analysis of migration, and strategic alignment — not by the competitor\'s marketing claim.',
          explanation: 'Architectural decisions in production systems should be evidence-based, not hype-driven. A new architecture may be superior in research benchmarks but inferior for your specific use case, dataset size, or latency requirements. Rigorous evaluation on your own conditions is essential.',
          difficulty: 'expert',
          expertNote: 'A world-class PM would also evaluate hybrid approaches (CNN backbone with transformer heads), consider whether the claimed accuracy gain is statistically significant and meaningful for the user experience, and assess whether the performance difference justifies the engineering investment.'
        },
        {
          question: 'Which of the following correctly describe the difference between skip connections in ResNet and skip connections in U-Net? Select all that apply.',
          type: 'multi',
          options: [
            'ResNet skip connections ADD the input to the block output; U-Net skip connections CONCATENATE encoder features with decoder features',
            'ResNet skip connections connect adjacent layers within the encoder; U-Net skip connections bridge across the encoder-decoder divide at corresponding resolution levels',
            'Both serve the exact same purpose and are implemented identically',
            'ResNet skip connections primarily address gradient flow for deep network training; U-Net skip connections primarily recover spatial detail lost during downsampling',
            'U-Net skip connections are only used in medical imaging applications'
          ],
          correct: [0, 1, 3],
          explanation: 'ResNet uses additive skip connections (element-wise addition) within the encoder to enable gradient flow through deep networks. U-Net uses concatenation-based skip connections across the encoder-decoder boundary to combine semantic features (from deep layers) with spatial features (from shallow layers). Their purposes are different: ResNet solves the optimisation problem of deep networks; U-Net solves the spatial precision problem of dense prediction.',
          difficulty: 'applied',
          expertNote: 'Understanding these as distinct mechanisms is important because both appear in modern architectures simultaneously — diffusion model U-Nets use U-Net-style encoder-decoder skips AND ResNet-style residual connections within each block.'
        },
        {
          question: 'A GAN trained on celebrity faces consistently generates only young, light-skinned female faces despite the training dataset containing diverse demographics. What is the most likely technical cause?',
          type: 'mc',
          options: [
            'The discriminator is too weak to detect demographic differences',
            'Mode collapse — the generator has found a narrow distribution of outputs that consistently fools the discriminator, ignoring underrepresented modes in the data',
            'The training data was not preprocessed correctly',
            'GANs are inherently unable to generate diverse outputs'
          ],
          correct: 1,
          explanation: 'Mode collapse is the GAN-specific pathology where the generator collapses to producing only a narrow range of outputs. If certain demographics are overrepresented in the training data or are easier for the generator to model (e.g., young female faces may be more prevalent in celebrity datasets), the generator can satisfy the discriminator by producing only those types, ignoring the rest of the distribution. This is both a technical problem (mode collapse) and a data problem (dataset imbalance).',
          difficulty: 'applied',
          expertNote: 'A PM for a generative product must audit both the training data demographics AND the model outputs for representational fairness. Techniques like conditional generation, diversity-promoting losses, and post-generation filtering can mitigate this, but the root cause often lies in training data curation.'
        },
        {
          question: 'Why has the residual connection design pattern from ResNet become ubiquitous across nearly all modern deep learning architectures, including Transformers?',
          type: 'mc',
          options: [
            'It reduces the number of parameters needed in the network',
            'It provides a direct gradient pathway that prevents vanishing gradients, ensures adding depth never hurts performance in theory, and enables the training of much deeper networks',
            'It was mandated by major deep learning frameworks like PyTorch and TensorFlow',
            'It eliminates the need for activation functions in the network'
          ],
          correct: 1,
          explanation: 'Residual connections provide three universal benefits: (1) gradient highway — gradients flow through the identity path without decay, enabling training of very deep networks; (2) easy identity — blocks can default to passing information through unchanged, so added depth cannot hurt in theory; (3) ensemble effect — multiple paths create implicit ensembles. These benefits are not specific to CNNs — they apply to any deep architecture, which is why Transformers, diffusion models, and modern RNNs all use them.',
          difficulty: 'foundational',
          expertNote: 'The universality of residual connections underscores a meta-lesson: the most impactful architectural innovations are often the simplest. Skip connections are just element-wise addition — but the insight to use them transformed the entire field.'
        }
      ]
    }
  }

};
