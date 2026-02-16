export const lessons = {

  // ─────────────────────────────────────────────
  // L01 — Generative Models Landscape
  // ─────────────────────────────────────────────
  l01: {
    title: 'Generative Models Landscape — VAEs, GANs, Flows, Diffusion',
    content: `
<h2>Why Generative Models Matter for AI PMs</h2>
<p>
  Generative models sit at the heart of the creative-AI revolution. As a product manager at a frontier lab
  like DeepMind, you will regularly evaluate trade-offs between different generative architectures.
  Understanding <em>why</em> the field moved from <span class="term" data-term="gan">GANs</span> to
  <span class="term" data-term="diffusion-model">diffusion models</span> is not just academic trivia —
  it shapes compute budgets, latency targets, safety constraints, and product capabilities you can promise
  to users.
</p>
<p>
  This lesson surveys the four dominant families of generative models, compares their strengths and
  weaknesses, and explains the trajectory that ultimately led to the diffusion-based systems powering
  products like DALL-E, Imagen, and Stable Diffusion.
</p>

<h2>Family 1: Variational Autoencoders (VAEs)</h2>
<p>
  A <span class="term" data-term="vae">Variational Autoencoder</span> learns to compress high-dimensional
  data (e.g., a 512x512 image) into a compact <span class="term" data-term="latent-space">latent space</span>
  and then reconstruct it. The architecture has two halves:
</p>
<ul>
  <li><strong>Encoder</strong> — maps input <code>x</code> to a distribution <code>q(z|x)</code> in latent space,
      parameterized by a mean vector <code>&mu;</code> and variance <code>&sigma;&sup2;</code>.</li>
  <li><strong>Decoder</strong> — maps a sampled latent vector <code>z</code> back to pixel space,
      producing <code>p(x|z)</code>.</li>
</ul>
<p>
  Training minimizes the <em>Evidence Lower Bound (ELBO)</em>, which balances reconstruction fidelity
  against how close <code>q(z|x)</code> stays to a simple prior (usually a standard Gaussian).
  Formally: <code>L = E[log p(x|z)] - KL(q(z|x) || p(z))</code>.
</p>

<div class="key-concept">
  <strong>Key Concept:</strong> The KL-divergence term in the VAE objective acts as a regularizer,
  forcing the latent space to be smooth and continuous. This is what allows you to <em>sample</em> from the
  latent space at generation time — walk between two points and get meaningful interpolations.
</div>

<p>
  <strong>Strengths:</strong> Stable training, principled probabilistic framework, useful latent
  representations for downstream tasks.
  <br/>
  <strong>Weaknesses:</strong> Outputs tend to be blurry because the model optimizes an average
  reconstruction loss rather than producing sharp, high-frequency details.
</p>

<div class="pro-tip">
  <strong>PM Perspective:</strong> VAEs remain relevant in production pipelines even though they
  rarely serve as the final generative output. In Stable Diffusion, a VAE compresses images into
  a latent space where diffusion then operates — dramatically reducing the compute needed. When
  scoping a project, ask: "Are we using the VAE as the generator or as a compression stage?"
</div>

<h2>Family 2: Generative Adversarial Networks (GANs)</h2>
<p>
  <span class="term" data-term="gan">GANs</span>, introduced by Ian Goodfellow in 2014, frame generation
  as a two-player game. A <strong>generator</strong> <code>G</code> produces synthetic samples, and a
  <strong>discriminator</strong> <code>D</code> tries to distinguish real from fake. Both networks improve
  iteratively:
</p>
<ul>
  <li><code>D</code> maximizes <code>log D(x) + log(1 - D(G(z)))</code></li>
  <li><code>G</code> minimizes <code>log(1 - D(G(z)))</code></li>
</ul>
<p>
  At the Nash equilibrium, <code>G</code> perfectly captures the data distribution and <code>D</code>
  outputs 0.5 for every sample (it cannot tell real from fake).
</p>

<div class="warning">
  <strong>Common Misconception:</strong> Many people assume GANs were abandoned because diffusion models
  are strictly better. In reality, GANs still achieve lower latency at inference and are used in
  real-time applications (e.g., face filters, super-resolution). Diffusion won on <em>quality</em>
  and <em>diversity</em>, not on speed.
</div>

<p>
  <strong>Key GAN variants:</strong>
</p>
<table>
  <thead>
    <tr><th>Variant</th><th>Innovation</th><th>Impact</th></tr>
  </thead>
  <tbody>
    <tr><td>DCGAN (2015)</td><td>Convolutional architecture with batch norm</td><td>First reliable image GAN</td></tr>
    <tr><td>WGAN (2017)</td><td>Wasserstein distance instead of JS divergence</td><td>Stabilized training</td></tr>
    <tr><td>StyleGAN (2019)</td><td>Style-based generator with adaptive instance norm</td><td>Photorealistic faces at 1024x1024</td></tr>
    <tr><td>GigaGAN (2023)</td><td>Scaled GAN to 1B+ params with text conditioning</td><td>Showed GANs can compete at scale</td></tr>
  </tbody>
</table>

<p>
  <strong>Strengths:</strong> Extremely sharp outputs, fast single-pass inference.
  <br/>
  <strong>Weaknesses:</strong> Mode collapse (generator learns to produce only a subset of the
  distribution), training instability, no principled likelihood estimate.
</p>

<h2>Family 3: Normalizing Flows</h2>
<p>
  Normalizing flows define an <em>invertible</em> mapping between a simple base distribution
  (e.g., Gaussian) and the complex data distribution. Because the mapping is bijective, you can
  compute exact log-likelihoods using the change-of-variables formula:
</p>
<p>
  <code>log p(x) = log p(z) + log |det(dz/dx)|</code>
</p>
<p>
  The model chains together multiple invertible transformations — each "flow step" warps the
  distribution a little further. Popular building blocks include affine coupling layers (RealNVP),
  autoregressive flows, and continuous normalizing flows (neural ODEs).
</p>

<div class="example-box">
  <h4>Example</h4>
  <p>Glow (Kingma & Dhariwal, 2018) used 1x1 invertible convolutions and affine coupling layers
  to generate 256x256 face images with exact likelihood computation. While quality did not match
  StyleGAN, the model enabled meaningful latent-space interpolation and attribute manipulation
  with theoretical guarantees no GAN could offer.</p>
</div>

<p>
  <strong>Strengths:</strong> Exact likelihoods, invertibility, principled density estimation.
  <br/>
  <strong>Weaknesses:</strong> Architectural constraints (must be invertible) limit expressiveness;
  high memory cost; quality gap compared to GANs and diffusion.
</p>

<h2>Family 4: Diffusion Models</h2>
<p>
  <span class="term" data-term="diffusion-model">Diffusion models</span> (also called score-based
  generative models) represent the current state of the art for image, video, and audio generation.
  The core idea is deceptively simple:
</p>
<ol>
  <li><strong>Forward process:</strong> Gradually add Gaussian noise to data over <code>T</code> steps
      until it becomes pure noise.</li>
  <li><strong>Reverse process:</strong> Train a neural network to reverse each noise-addition step,
      thereby learning to generate data from noise.</li>
</ol>
<p>
  The forward process is fixed (no learned parameters). All learning happens in the reverse
  <span class="term" data-term="denoising">denoising</span> network. The training objective reduces
  to a simple form: predict the noise <code>&epsilon;</code> that was added at each step.
</p>

<div class="key-concept">
  <strong>Key Concept:</strong> Unlike GANs (which require a delicate adversarial balance) or flows
  (which require invertible architectures), diffusion models use a straightforward denoising objective.
  This simplicity is a major reason they scaled so successfully — you can pour more compute into a
  larger U-Net or Transformer without worrying about training collapse.
</div>

<h2>Comparing the Families</h2>
<table>
  <thead>
    <tr>
      <th>Criterion</th><th>VAE</th><th>GAN</th><th>Flow</th><th>Diffusion</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Sample quality</td><td>Blurry</td><td>Sharp</td><td>Good</td><td>State-of-the-art</td></tr>
    <tr><td>Diversity</td><td>High</td><td>Mode collapse risk</td><td>High</td><td>High</td></tr>
    <tr><td>Training stability</td><td>Stable</td><td>Fragile</td><td>Stable</td><td>Stable</td></tr>
    <tr><td>Inference speed</td><td>Fast (1 pass)</td><td>Fast (1 pass)</td><td>Fast (1 pass)</td><td>Slow (many steps)</td></tr>
    <tr><td>Likelihood</td><td>Lower bound</td><td>None</td><td>Exact</td><td>Lower bound</td></tr>
    <tr><td>Controllability</td><td>Limited</td><td>Moderate</td><td>Good</td><td>Excellent (guidance)</td></tr>
  </tbody>
</table>

<div class="pro-tip">
  <strong>PM Perspective:</strong> When choosing a generative architecture for a new product feature,
  map requirements to this table. Need real-time generation (e.g., live camera filter)? A GAN or
  distilled diffusion model may be necessary. Need maximal quality and controllability for an
  offline creative tool? Full diffusion is the clear winner. The "best" model depends entirely on
  the product context.
</div>

<h2>The Historical Arc: How We Got to Diffusion</h2>
<p>
  The generative modeling timeline looks roughly like this:
</p>
<ul>
  <li><strong>2013-2014:</strong> VAEs and GANs published, igniting the field.</li>
  <li><strong>2015-2018:</strong> GAN variants dominate image generation; flows offer theoretical elegance.</li>
  <li><strong>2019:</strong> StyleGAN produces photorealistic faces; GANs seem unbeatable.</li>
  <li><strong>2020:</strong> DDPM (Ho et al.) shows diffusion can match GAN quality on small benchmarks.</li>
  <li><strong>2021:</strong> Dhariwal & Nichol prove "Diffusion Models Beat GANs" on ImageNet with classifier guidance.</li>
  <li><strong>2022:</strong> DALL-E 2, Imagen, and Stable Diffusion bring diffusion to products. The era of text-to-image begins.</li>
  <li><strong>2023-2024:</strong> Diffusion extends to video (Sora), 3D (DreamFusion), and audio (AudioLDM). Consistency models and distillation close the speed gap.</li>
</ul>

<div class="warning">
  <strong>Common Misconception:</strong> Diffusion models are often described as "completely new."
  In fact, the mathematical foundations — stochastic differential equations and score matching — date
  back decades. The breakthroughs were in architecture (U-Nets, Transformers), conditioning (classifier-free
  guidance), and scaling, not in the underlying theory itself.
</div>
`,
    quiz: {
      questions: [
        {
          question: 'You are scoping a new real-time background replacement feature for video calls. The feature must run at 30fps on consumer GPUs with no noticeable latency. Which generative approach is most appropriate to evaluate first, and why?',
          type: 'scenario',
          options: [
            'A diffusion model, because it produces the highest quality results',
            'A GAN-based model, because single-pass inference enables real-time speeds',
            'A normalizing flow, because exact likelihoods help quality assessment',
            'A VAE, because stable training is the most important criterion'
          ],
          correct: 1,
          explanation: 'Real-time (30fps) requires inference in roughly 33ms per frame. Standard diffusion models require dozens of sequential denoising steps, making them too slow without heavy distillation. GANs produce outputs in a single forward pass, making them the natural starting point for latency-constrained applications. You might later explore distilled diffusion, but evaluating GANs first is the correct scoping decision.',
          difficulty: 'applied',
          expertNote: 'In practice, teams often use a GAN for real-time inference and a diffusion model for offline quality benchmarks. Consistency models and latent consistency models have narrowed this gap to 1-4 steps, but as of 2024 GANs remain faster for hard real-time constraints.'
        },
        {
          question: 'What is the primary reason VAE outputs tend to appear blurry compared to GAN outputs?',
          type: 'mc',
          options: [
            'VAEs use smaller neural networks than GANs',
            'The reconstruction loss optimizes for the average of possible outputs, smoothing high-frequency details',
            'VAEs cannot learn hierarchical features due to the KL divergence term',
            'The latent space in a VAE is too high-dimensional'
          ],
          correct: 1,
          explanation: 'VAEs minimize a pixel-wise reconstruction loss (typically MSE) plus a KL regularizer. When the true data distribution is multimodal — e.g., a face could have slightly different hair positions — the model learns to average over modes, producing blurry outputs. GANs avoid this by using an adversarial loss that rewards sharpness directly.',
          difficulty: 'foundational',
          expertNote: 'This is why modern systems like Stable Diffusion use the VAE only for compression (encoding to latent space), not for final generation. The diffusion process in latent space handles the actual generation, sidestepping the blurriness problem.'
        },
        {
          question: 'A researcher proposes replacing the GAN in your image generation pipeline with a normalizing flow to gain exact log-likelihood computation. What is the most significant trade-off you should raise?',
          type: 'mc',
          options: [
            'Normalizing flows cannot generate images at all — they are only density estimators',
            'The invertibility constraint limits architectural flexibility, likely reducing sample quality',
            'Normalizing flows require adversarial training which is even less stable than GANs',
            'Flow models cannot use GPUs efficiently due to the Jacobian computation'
          ],
          correct: 1,
          explanation: 'Normalizing flows require every layer to be invertible with a tractable Jacobian determinant. This architectural constraint eliminates many powerful building blocks (e.g., standard convolutions, attention without modification). The result is typically lower sample quality compared to unconstrained architectures used in GANs or diffusion models.',
          difficulty: 'applied',
          expertNote: 'Flow matching — a more recent approach used in Meta\'s Voicebox and other systems — relaxes some of these constraints by learning a vector field rather than an explicit invertible map. This hybridizes flow-like training with diffusion-like flexibility.'
        },
        {
          question: 'Which of the following are true about diffusion models compared to GANs? Select all that apply.',
          type: 'multi',
          options: [
            'Diffusion models have more stable training dynamics',
            'Diffusion models always produce higher quality than GANs on every benchmark',
            'Diffusion models natively produce a diversity of outputs without mode collapse',
            'Diffusion models require significantly more inference-time compute by default',
            'Diffusion models provide a principled probabilistic framework with a tractable training objective'
          ],
          correct: [0, 2, 3, 4],
          explanation: 'Diffusion models train stably (no adversarial dynamics), avoid mode collapse through their probabilistic formulation, and optimize a well-defined denoising objective. However, they require many sequential denoising steps at inference. The claim that diffusion "always" beats GANs is false — on specific benchmarks and with specific metrics (e.g., FID on constrained domains like faces), well-tuned GANs can still be competitive.',
          difficulty: 'foundational',
          expertNote: 'The "always better quality" myth persists because diffusion models excel on diverse benchmarks like ImageNet. On narrow domains (faces, specific textures), the quality difference is much smaller, and GANs win on speed.'
        },
        {
          question: 'Your team at DeepMind is building a generative feature that requires both high-quality output AND the ability to compute how likely a given image is under the model. Which architecture or combination should you recommend?',
          type: 'mc',
          options: [
            'A pure GAN — it can be modified to approximate likelihoods',
            'A diffusion model — it provides exact likelihoods natively',
            'A hybrid approach using a diffusion model for generation and evaluating likelihoods via the ELBO or probability flow ODE',
            'A normalizing flow — it is the only architecture that can generate high-quality images with exact likelihoods'
          ],
          correct: 2,
          explanation: 'Diffusion models provide a lower-bound on log-likelihood via the ELBO and can compute exact likelihoods through the probability flow ODE formulation. This gives you state-of-the-art generation quality with principled likelihood computation. GANs have no native likelihood, and normalizing flows sacrifice too much quality. The hybrid framing (diffusion for generation, ODE for likelihood) is the practical approach used in research.',
          difficulty: 'expert',
          expertNote: 'Song et al. (2021) showed that the reverse diffusion process can be interpreted as a probability flow ODE, enabling exact likelihood computation. This made diffusion models the first architecture to combine top-tier sample quality with principled density estimation.'
        }
      ]
    }
  },

  // ─────────────────────────────────────────────
  // L02 — How Diffusion Works
  // ─────────────────────────────────────────────
  l02: {
    title: 'How Diffusion Works — Forward & Reverse Process',
    content: `
<h2>The Core Intuition</h2>
<p>
  Imagine you have a photograph. You photocopy it, but each time you introduce a tiny bit of random
  static. After thousands of copies-of-copies, the image is pure noise — indistinguishable from
  random TV static. <span class="term" data-term="diffusion-model">Diffusion models</span> learn to
  run this process in reverse: given pure noise, iteratively remove the static until a coherent image
  emerges.
</p>
<p>
  This two-phase structure — a <strong>forward process</strong> that destroys information and a
  <strong>reverse process</strong> that reconstructs it — is the mathematical backbone of all
  diffusion-based generative models.
</p>

<h2>The Forward (Noising) Process</h2>
<p>
  The forward process defines a Markov chain that gradually adds Gaussian noise to data
  <code>x<sub>0</sub></code> over <code>T</code> timesteps:
</p>
<p>
  <code>q(x<sub>t</sub> | x<sub>t-1</sub>) = N(x<sub>t</sub>; &radic;(1-&beta;<sub>t</sub>) x<sub>t-1</sub>, &beta;<sub>t</sub> I)</code>
</p>
<p>
  Here, <code>&beta;<sub>t</sub></code> is a <strong>noise schedule</strong> — a small positive value
  that controls how much noise is added at step <code>t</code>. Common schedules include:
</p>
<ul>
  <li><strong>Linear schedule:</strong> <code>&beta;<sub>t</sub></code> increases linearly from
      <code>&beta;<sub>1</sub> = 10<sup>-4</sup></code> to <code>&beta;<sub>T</sub> = 0.02</code>.</li>
  <li><strong>Cosine schedule:</strong> Proposed by Nichol & Dhariwal (2021), it preserves more signal
      in early steps and is the default in many modern systems.</li>
</ul>

<div class="key-concept">
  <strong>Key Concept:</strong> A critical property is that you can jump directly to any timestep
  <code>t</code> without iterating through all previous steps. Define
  <code>&alpha;<sub>t</sub> = 1 - &beta;<sub>t</sub></code> and
  <code>&alpha;&#772;<sub>t</sub> = &prod; &alpha;<sub>s</sub></code> (cumulative product).
  Then: <code>q(x<sub>t</sub> | x<sub>0</sub>) = N(x<sub>t</sub>; &radic;&alpha;&#772;<sub>t</sub> x<sub>0</sub>, (1-&alpha;&#772;<sub>t</sub>) I)</code>.
  This "closed-form sampling" is essential for efficient training — you randomly pick a timestep and
  directly compute the noisy version.
</div>

<p>
  As <code>t &rarr; T</code>, the signal-to-noise ratio approaches zero, and
  <code>x<sub>T</sub> &sim; N(0, I)</code> — pure isotropic Gaussian noise. The original image is
  completely destroyed.
</p>

<h2>The Reverse (Denoising) Process</h2>
<p>
  The reverse process is where all the learning happens. We want to approximate:
</p>
<p>
  <code>p<sub>&theta;</sub>(x<sub>t-1</sub> | x<sub>t</sub>) = N(x<sub>t-1</sub>; &mu;<sub>&theta;</sub>(x<sub>t</sub>, t), &Sigma;<sub>&theta;</sub>(x<sub>t</sub>, t))</code>
</p>
<p>
  A neural network with parameters <code>&theta;</code> takes the current noisy image
  <code>x<sub>t</sub></code> and the timestep <code>t</code> as input, and predicts the parameters
  of the Gaussian distribution for the slightly-less-noisy image <code>x<sub>t-1</sub></code>.
</p>

<h2>The Training Objective: Noise Prediction</h2>
<p>
  Ho et al. (2020) showed that the variational lower bound simplifies to a remarkably clean objective.
  Instead of predicting <code>&mu;</code> directly, the network predicts the noise
  <code>&epsilon;</code> that was added:
</p>
<p>
  <code>L<sub>simple</sub> = E<sub>t, x<sub>0</sub>, &epsilon;</sub> [ || &epsilon; - &epsilon;<sub>&theta;</sub>(x<sub>t</sub>, t) ||&sup2; ]</code>
</p>
<p>
  Training procedure:
</p>
<ol>
  <li>Sample a clean image <code>x<sub>0</sub></code> from the dataset.</li>
  <li>Sample a random timestep <code>t ~ Uniform(1, T)</code>.</li>
  <li>Sample noise <code>&epsilon; ~ N(0, I)</code>.</li>
  <li>Compute <code>x<sub>t</sub> = &radic;&alpha;&#772;<sub>t</sub> x<sub>0</sub> + &radic;(1-&alpha;&#772;<sub>t</sub>) &epsilon;</code>.</li>
  <li>Feed <code>(x<sub>t</sub>, t)</code> to the network, get predicted noise <code>&epsilon;<sub>&theta;</sub></code>.</li>
  <li>Compute MSE loss between <code>&epsilon;</code> and <code>&epsilon;<sub>&theta;</sub></code>. Backpropagate.</li>
</ol>

<div class="warning">
  <strong>Common Misconception:</strong> People often think the model is trained sequentially — first
  learning to denoise from step T, then T-1, etc. In reality, training samples random timesteps in
  every batch. The model simultaneously learns to denoise from any noise level. This is why timestep
  <code>t</code> is provided as an input — the network must know <em>how noisy</em> the input is.
</div>

<h2>The Denoising Network Architecture</h2>
<p>
  The workhorse architecture for diffusion is the <strong>U-Net</strong> — an encoder-decoder
  convolutional network with skip connections. Key components:
</p>
<ul>
  <li><strong>Downsampling path:</strong> Extracts multi-scale features via convolutions and
      attention blocks.</li>
  <li><strong>Upsampling path:</strong> Reconstructs spatial resolution with transposed convolutions.</li>
  <li><strong>Skip connections:</strong> Concatenate encoder features to decoder features at
      matching resolutions, preserving fine-grained detail.</li>
  <li><strong>Timestep embedding:</strong> The timestep <code>t</code> is encoded (typically via
      sinusoidal positional encoding, similar to Transformers) and injected into every residual block.</li>
  <li><strong>Attention layers:</strong> Self-attention at lower resolutions captures global structure;
      cross-attention enables text conditioning.</li>
</ul>

<div class="example-box">
  <h4>Example</h4>
  <p>In Stable Diffusion's U-Net, there are approximately 860M parameters. The model operates on
  64x64 latent representations (not 512x512 pixels). At each residual block, the timestep embedding
  is added to the feature maps via adaptive group normalization, and text embeddings from CLIP are
  injected through cross-attention layers. This architecture allows simultaneous conditioning on
  "how noisy" (timestep) and "what to generate" (text prompt).</p>
</div>

<h2>Sampling (Inference)</h2>
<p>
  To generate an image:
</p>
<ol>
  <li>Sample <code>x<sub>T</sub> ~ N(0, I)</code> — start from pure noise.</li>
  <li>For <code>t = T, T-1, ..., 1</code>:
    <ul>
      <li>Predict noise: <code>&epsilon;<sub>&theta;</sub>(x<sub>t</sub>, t)</code></li>
      <li>Compute the mean of <code>x<sub>t-1</sub></code> using the predicted noise</li>
      <li>Add a small amount of random noise (except at the final step)</li>
    </ul>
  </li>
  <li>Return <code>x<sub>0</sub></code> — the generated image.</li>
</ol>
<p>
  Original DDPM used <code>T = 1000</code> steps, making inference slow. Modern approaches
  dramatically reduce this:
</p>

<table>
  <thead>
    <tr><th>Method</th><th>Steps</th><th>Approach</th></tr>
  </thead>
  <tbody>
    <tr><td>DDPM</td><td>1000</td><td>Original Markov chain</td></tr>
    <tr><td>DDIM</td><td>50-100</td><td>Deterministic non-Markovian process</td></tr>
    <tr><td>DPM-Solver</td><td>10-25</td><td>Fast ODE solver for probability flow</td></tr>
    <tr><td>Consistency Models</td><td>1-2</td><td>Direct mapping to clean data</td></tr>
    <tr><td>Latent Consistency Models</td><td>1-4</td><td>Consistency distillation in latent space</td></tr>
  </tbody>
</table>

<div class="pro-tip">
  <strong>PM Perspective:</strong> The number of sampling steps directly determines your inference
  cost and user-facing latency. A 50-step DDIM sampler on an A100 GPU takes roughly 3-5 seconds for
  a 512x512 image. A 4-step LCM brings this under 1 second. When building product requirements,
  define latency budgets early and work backwards to determine which sampler (and how many steps)
  you need. This is often the single most impactful product-engineering trade-off in diffusion systems.
</div>

<h2>Guidance: Steering Generation</h2>
<p>
  Raw diffusion models generate diverse but uncontrolled outputs. <strong>Guidance</strong> mechanisms
  steer generation toward desired attributes:
</p>
<ul>
  <li><strong>Classifier guidance</strong> (Dhariwal & Nichol, 2021): Train a separate classifier on
      noisy images. During sampling, use its gradients to push generation toward a target class.
      Quality improves dramatically but requires an external classifier.</li>
  <li><strong><span class="term" data-term="classifier-free-guidance">Classifier-free guidance</span></strong>
      (Ho & Salimans, 2022): During training, randomly drop the conditioning signal (e.g., text prompt)
      some fraction of the time. At inference, compute both conditional and unconditional predictions,
      then extrapolate: <code>&epsilon; = &epsilon;<sub>uncond</sub> + s &middot; (&epsilon;<sub>cond</sub> - &epsilon;<sub>uncond</sub>)</code>,
      where <code>s</code> is the guidance scale (typically 7-15).</li>
</ul>

<div class="key-concept">
  <strong>Key Concept:</strong> Classifier-free guidance is the dominant conditioning mechanism in
  modern diffusion products. The guidance scale <code>s</code> trades off fidelity to the prompt
  (higher = more adherent but less diverse) against variety (lower = more creative but may ignore
  parts of the prompt). This is directly exposed to users as a "creativity" or "adherence" slider
  in many products.
</div>

<h2>The Score Function Perspective</h2>
<p>
  There is an elegant alternative view of diffusion through <strong>score matching</strong>. The
  <em>score function</em> <code>&nabla;<sub>x</sub> log p(x)</code> points toward regions of higher
  data density. Instead of predicting noise, you can equivalently train a model to predict the score.
  The two are related by: <code>s<sub>&theta;</sub>(x<sub>t</sub>, t) = -&epsilon;<sub>&theta;</sub>(x<sub>t</sub>, t) / &radic;(1-&alpha;&#772;<sub>t</sub>)</code>.
</p>
<p>
  Song et al. (2021) unified discrete-step diffusion (DDPM) and continuous-time score models under
  a single framework of <strong>stochastic differential equations (SDEs)</strong>. The forward process
  is an SDE that adds noise; the reverse is another SDE that removes it. This unification enabled
  the probability flow ODE — a deterministic trajectory that allows exact likelihood computation.
</p>
`,
    quiz: {
      questions: [
        {
          question: 'During diffusion model training, a random timestep t is sampled for each training example. Why is this approach used instead of training sequentially from t=T down to t=1?',
          type: 'mc',
          options: [
            'Sequential training would require T times more training data',
            'Random timestep sampling allows the model to learn all noise levels simultaneously in a single pass, massively improving training efficiency',
            'The forward process must be computed sequentially, so training must be too',
            'Sequential training would cause the model to forget earlier timesteps (catastrophic forgetting)'
          ],
          correct: 1,
          explanation: 'Because the closed-form expression q(x_t | x_0) lets you jump to any timestep directly, there is no need to iterate through all steps. Random timestep sampling means every batch contains examples from across the entire noise spectrum, allowing the model to learn all denoising levels simultaneously. This is both more efficient and produces better gradients.',
          difficulty: 'foundational',
          expertNote: 'Some recent work (like P2 weighting) suggests that not all timesteps are equally important — intermediate timesteps contribute most to perceptual quality. Importance sampling over timesteps can further improve training efficiency.'
        },
        {
          question: 'Your team is building a consumer image generation product. Testing shows that 50-step DDIM sampling takes 4 seconds on target hardware, but the product requirement is under 1.5 seconds. Which approach should you prioritize investigating?',
          type: 'scenario',
          options: [
            'Switch from DDIM to the original DDPM sampler with 1000 steps for better quality',
            'Evaluate distillation approaches like Latent Consistency Models that reduce steps to 4-8 while preserving quality',
            'Reduce image resolution from 512x512 to 128x128 to meet latency targets',
            'Deploy on 4x more GPUs to parallelize the denoising steps'
          ],
          correct: 1,
          explanation: 'Distillation methods like LCM compress the sampling trajectory into 1-8 steps, often achieving comparable quality to 50-step DDIM. Going from 50 steps to 4-8 steps would roughly achieve the 3-4x speedup needed. Reducing resolution would degrade the product experience. Denoising steps are inherently sequential (each depends on the previous), so parallelization across GPUs does not help. DDPM with 1000 steps would be 20x slower.',
          difficulty: 'applied',
          expertNote: 'In practice, many production systems use a two-tier approach: a fast distilled model for real-time preview (1-4 steps) and a full-step model for the final high-quality render. This gives users immediate feedback while delivering maximum quality.'
        },
        {
          question: 'What does increasing the classifier-free guidance scale (e.g., from 3 to 15) primarily do to generated outputs?',
          type: 'mc',
          options: [
            'Increases generation speed by skipping unnecessary denoising steps',
            'Increases adherence to the conditioning signal (e.g., text prompt) while reducing output diversity',
            'Improves the resolution of generated images without changing content',
            'Reduces the number of sampling steps needed for convergence'
          ],
          correct: 1,
          explanation: 'The guidance scale amplifies the difference between conditional and unconditional predictions. Higher values push the model more strongly toward what the text prompt describes, producing outputs that are more faithful to the prompt but less diverse. Very high values (>20) can cause oversaturation and artifacts. This is a fundamental quality-diversity trade-off that PMs must understand.',
          difficulty: 'foundational',
          expertNote: 'The guidance scale is one of the most user-facing hyperparameters in diffusion products. Midjourney, for instance, exposes this as a "stylize" parameter. Finding the right default guidance scale for your product\'s use case is a critical PM decision that affects user perception of quality.'
        },
        {
          question: 'Which of the following correctly describe the relationship between the noise prediction and score function perspectives of diffusion? Select all that apply.',
          type: 'multi',
          options: [
            'They are mathematically equivalent — predicting noise is proportional to predicting the score',
            'The score function points toward higher-density regions of the data distribution',
            'Only the noise prediction formulation can be used for image generation',
            'The SDE framework unifies both perspectives and enables exact likelihood computation via the probability flow ODE'
          ],
          correct: [0, 1, 3],
          explanation: 'The noise prediction epsilon_theta and score function are related by a simple scaling factor involving the noise schedule. Both can be used for generation. The SDE unification by Song et al. showed that the probability flow ODE — a deterministic version of the reverse SDE — enables exact log-likelihood computation, bridging the gap with normalizing flows.',
          difficulty: 'expert',
          expertNote: 'The SDE/ODE duality is powerful: the stochastic sampler (SDE) often produces higher quality due to the injected randomness acting as a form of exploration, while the deterministic sampler (ODE) enables likelihood computation and more predictable outputs. Some systems let users choose between them.'
        },
        {
          question: 'A DeepMind researcher proposes changing the noise schedule from linear to cosine for your image generation model. What is the primary motivation for this change?',
          type: 'mc',
          options: [
            'Cosine schedules are computationally cheaper to implement in production',
            'Cosine schedules destroy information more gradually in early steps, preserving more coarse structure for longer and improving sample quality',
            'Cosine schedules make the forward process deterministic instead of stochastic',
            'Cosine schedules eliminate the need for classifier-free guidance entirely'
          ],
          correct: 1,
          explanation: 'The linear schedule adds noise somewhat aggressively in early steps, destroying important coarse structure before the model has a chance to capture it. The cosine schedule proposed by Nichol & Dhariwal (2021) preserves more signal-to-noise ratio in early timesteps, giving the reverse process more information to work with. This leads to measurably better FID scores, especially on high-resolution images.',
          difficulty: 'applied',
          expertNote: 'The noise schedule is one of the most underappreciated hyperparameters in diffusion models. Different schedules can have significant effects on generation quality, and the optimal schedule can vary by data domain (natural images vs. medical images vs. audio). Some modern approaches learn the schedule end-to-end.'
        }
      ]
    }
  },

  // ─────────────────────────────────────────────
  // L03 — Stable Diffusion, DALL-E, Imagen
  // ─────────────────────────────────────────────
  l03: {
    title: 'Stable Diffusion, DALL-E, Imagen — Architecture Deep Dive',
    content: `
<h2>The Text-to-Image Revolution</h2>
<p>
  In 2022, three systems arrived in rapid succession and permanently changed what was possible with
  <span class="term" data-term="text-to-image">text-to-image</span> generation:
  <strong>DALL-E 2</strong> (OpenAI, April 2022), <strong>Imagen</strong> (Google Brain, May 2022),
  and <strong><span class="term" data-term="stable-diffusion">Stable Diffusion</span></strong>
  (Stability AI / CompVis / Runway, August 2022). Each took a fundamentally different architectural
  approach while achieving remarkably similar quality. Understanding these differences is essential
  for any AI PM evaluating generative image capabilities.
</p>

<h2>DALL-E 2: The Prior + Decoder Approach</h2>
<p>
  DALL-E 2 (technically called "unCLIP") has a distinctive two-stage architecture:
</p>
<ol>
  <li><strong>Prior:</strong> Takes a text caption, encodes it via CLIP text encoder, and generates
      a CLIP <em>image</em> embedding that matches the text. This can be either an autoregressive
      transformer prior or a diffusion prior (OpenAI found diffusion performed better).</li>
  <li><strong>Decoder:</strong> Takes the CLIP image embedding and generates a high-resolution image
      via a diffusion model (modified GLIDE architecture). Two-stage upsampling: 64x64 → 256x256 → 1024x1024.</li>
</ol>

<div class="key-concept">
  <strong>Key Concept:</strong> DALL-E 2's key insight was to leverage CLIP's joint text-image embedding
  space as an intermediary. Rather than going directly from text to pixels, it goes text → CLIP text
  embedding → CLIP image embedding → pixels. This decouples the text-understanding problem from the
  image-generation problem.
</div>

<p>
  <strong>Architecture details:</strong>
</p>
<ul>
  <li>CLIP ViT-H/14 as the image and text encoder (trained on 400M image-text pairs)</li>
  <li>Diffusion prior: 1B parameter Transformer that produces CLIP image embeddings</li>
  <li>Decoder: Modified GLIDE model (3.5B parameters) with CLIP conditioning</li>
  <li>Two super-resolution diffusion models for upsampling</li>
</ul>

<div class="warning">
  <strong>Common Misconception:</strong> DALL-E 2 and DALL-E 1 use completely different architectures.
  DALL-E 1 was an autoregressive transformer (like GPT) that generated image tokens. DALL-E 2 is a
  diffusion model with CLIP guidance. DALL-E 3 later moved to a different architecture again with
  improved text understanding via recaptioning. Do not conflate them.
</div>

<h2>Imagen: Text Encoder Supremacy</h2>
<p>
  <span class="term" data-term="imagen">Imagen</span> from Google Brain took a simpler architectural
  path but made a profound discovery: <strong>scaling the text encoder matters more than scaling the
  image model</strong>.
</p>
<p>
  Architecture:
</p>
<ol>
  <li><strong>Text encoder:</strong> T5-XXL (4.6B parameters), a frozen large language model.
      Critically, this is a <em>text-only</em> model — never trained on images.</li>
  <li><strong>Base diffusion model:</strong> A U-Net conditioned on T5 embeddings via cross-attention.
      Generates 64x64 images.</li>
  <li><strong>Super-resolution models:</strong> Two cascaded diffusion models upscale to 256x256
      and then 1024x1024, each conditioned on the text and the previous resolution.</li>
</ol>

<div class="key-concept">
  <strong>Key Concept:</strong> Imagen's most important finding — published in the "Photorealistic
  Text-to-Image Diffusion Models with Deep Language Understanding" paper — was that increasing the
  size of the text encoder (from T5-Small to T5-XXL) improved image quality and text-image alignment
  far more than increasing the size of the U-Net diffusion model. This suggests that text
  <em>understanding</em> is the bottleneck, not image <em>generation</em>.
</div>

<p>
  Imagen also introduced <strong>dynamic thresholding</strong> — a technique for preventing pixel
  saturation when using high guidance scales. Without it, classifier-free guidance at scale 15+
  produces washed-out or artifact-heavy images. Dynamic thresholding clips the predicted
  <code>x<sub>0</sub></code> to a percentile-based range at each step.
</p>

<div class="pro-tip">
  <strong>PM Perspective:</strong> Imagen's finding has direct product implications. If you are
  allocating compute budget between the text encoder and the image generator, Imagen's research
  suggests investing in text understanding first. A user who types "a corgi wearing a chef's hat
  while riding a skateboard on the moon" cares most about whether the model <em>understands</em>
  every element of the prompt. Compositional understanding is the text encoder's job.
</div>

<h2>Stable Diffusion: Latent Diffusion at Scale</h2>
<p>
  Stable Diffusion, based on the <strong>Latent Diffusion Model (LDM)</strong> paper by Rombach
  et al. (2022), introduced the most commercially impactful architectural innovation: performing
  diffusion in a compressed <span class="term" data-term="latent-space">latent space</span> rather
  than pixel space.
</p>
<p>
  Architecture:
</p>
<ol>
  <li><strong>VAE encoder:</strong> Compresses a 512x512x3 image into a 64x64x4
      <span class="term" data-term="latent-space">latent</span> representation (8x spatial compression).
      This single step reduces the number of elements the diffusion model must process by 48x.</li>
  <li><strong>U-Net:</strong> Performs the forward and reverse diffusion process entirely in this
      compressed latent space (~860M parameters).</li>
  <li><strong>Text encoder:</strong> CLIP ViT-L/14 text encoder (123M parameters). Text embeddings
      are injected via cross-attention in the U-Net.</li>
  <li><strong>VAE decoder:</strong> Decompresses the denoised latent back to pixel space.</li>
</ol>

<table>
  <thead>
    <tr><th>Component</th><th>DALL-E 2</th><th>Imagen</th><th>Stable Diffusion v1.5</th></tr>
  </thead>
  <tbody>
    <tr><td>Text encoder</td><td>CLIP ViT-H/14</td><td>T5-XXL (4.6B)</td><td>CLIP ViT-L/14 (123M)</td></tr>
    <tr><td>Diffusion space</td><td>Pixel (64x64 base)</td><td>Pixel (64x64 base)</td><td>Latent (64x64x4)</td></tr>
    <tr><td>Main model params</td><td>~3.5B</td><td>~2B (base U-Net)</td><td>~860M</td></tr>
    <tr><td>Total params</td><td>~5.5B</td><td>~7B+</td><td>~1B</td></tr>
    <tr><td>Super-resolution</td><td>Two separate models</td><td>Two cascaded models</td><td>Not needed (VAE decode)</td></tr>
    <tr><td>Open source</td><td>No</td><td>No</td><td>Yes</td></tr>
    <tr><td>Consumer GPU</td><td>No</td><td>No</td><td>Yes (6GB+ VRAM)</td></tr>
  </tbody>
</table>

<div class="key-concept">
  <strong>Key Concept:</strong> The latent diffusion approach is why Stable Diffusion could run on
  consumer GPUs while DALL-E 2 and Imagen required massive infrastructure. By compressing to latent
  space first, the diffusion U-Net operates on 48x fewer elements. This is not just a speedup — it
  is an entire paradigm shift that democratized generative AI. The approach proved so successful that
  subsequent versions of virtually every major system adopted latent diffusion.
</div>

<h2>Evolution: SDXL and Beyond</h2>
<p>
  Stable Diffusion evolved rapidly:
</p>
<ul>
  <li><strong>SD v1.5</strong> (Oct 2022): 860M U-Net, CLIP text encoder, 512x512 native resolution.</li>
  <li><strong>SD v2.0/2.1</strong> (Nov 2022): OpenCLIP text encoder, 768x768 support, depth-conditioned models.</li>
  <li><strong>SDXL</strong> (Jul 2023): Dual text encoders (CLIP ViT-L + OpenCLIP ViT-bigG), 2.6B U-Net,
      1024x1024 native resolution, refiner model for detail enhancement.</li>
  <li><strong>SDXL Turbo</strong> (Nov 2023): Adversarial diffusion distillation — 1-4 step generation.</li>
  <li><strong>SD3</strong> (2024): Replaced U-Net with a Diffusion Transformer (DiT) using flow matching,
      triple text encoders (CLIP ViT-L, OpenCLIP ViT-bigG, T5-XXL).</li>
</ul>

<div class="example-box">
  <h4>Example</h4>
  <p>The progression from SD v1.5 to SD3 illustrates a key trend: text encoders are becoming the most
  important component. SD v1.5 used a single 123M CLIP encoder and struggled with compositional prompts.
  SD3 uses three encoders totaling over 5B parameters, with T5-XXL providing deep language understanding.
  This mirrors Imagen's original finding and shows how research insights propagate across the industry.</p>
</div>

<h2>DALL-E 3 and Recaptioning</h2>
<p>
  DALL-E 3 (October 2023) addressed the persistent problem of text-image misalignment through a
  data-centric approach: <strong>recaptioning</strong>. Instead of training on noisy web-scraped
  alt-text, OpenAI used a purpose-built captioning model to generate highly detailed, accurate
  descriptions of every training image. This dramatically improved the model's ability to follow
  complex, multi-element prompts.
</p>

<div class="pro-tip">
  <strong>PM Perspective:</strong> DALL-E 3's success with recaptioning underscores a crucial lesson:
  <strong>data quality often matters more than architecture</strong>. As a PM, when your team is
  deciding between "improve the model architecture" and "improve the training data," the data
  investment frequently delivers higher ROI. Budget accordingly. This is especially true for
  text-to-image where the bottleneck is often the quality of text-image pairs, not model capacity.
</div>

<h2>Diffusion Transformers (DiT): The Next Architecture</h2>
<p>
  The latest architectural shift replaces the U-Net with a <strong>Transformer</strong>. The Diffusion
  Transformer (DiT), introduced by Peebles & Xie (2023), treats image patches as tokens and applies
  standard Transformer blocks with adaptive layer normalization for timestep conditioning.
</p>
<p>
  Advantages of DiT over U-Net:
</p>
<ul>
  <li>Scales more predictably with compute (follows Transformer scaling laws)</li>
  <li>Simpler architecture — no need for hand-designed skip connections and resolution stages</li>
  <li>Better at handling variable resolutions and aspect ratios</li>
  <li>More compatible with existing Transformer infrastructure and optimizations</li>
</ul>
<p>
  DiT is the backbone of <strong>Sora</strong> (OpenAI's video model), <strong>SD3</strong>,
  and many next-generation systems. The trend is clear: the Transformer is becoming the universal
  architecture for all modalities — text, images, video, and audio.
</p>

<h2>Conditioning Beyond Text</h2>
<p>
  Modern diffusion systems accept many types of conditioning signals beyond text:
</p>
<ul>
  <li><strong>ControlNet:</strong> Condition on spatial maps (edges, depth, pose, segmentation) by
      adding a trainable copy of the U-Net encoder.</li>
  <li><strong>IP-Adapter:</strong> Image prompt conditioning — use a reference image to guide style.</li>
  <li><strong>Inpainting:</strong> Condition on a masked image to fill in regions.</li>
  <li><strong>Img2img:</strong> Start from a partially noised version of an existing image instead
      of pure noise, enabling editing.</li>
</ul>

<div class="pro-tip">
  <strong>PM Perspective:</strong> These conditioning mechanisms are what transform a research model
  into a product. Raw text-to-image is a creative toy. ControlNet + inpainting + img2img enable
  professional workflows — architectural visualization, product photography, fashion design. When
  building a product roadmap, the conditioning modalities you support define your market. Think of
  the base diffusion model as a platform and conditioning mechanisms as features built on top.
</div>
`,
    quiz: {
      questions: [
        {
          question: 'You are a PM at DeepMind evaluating whether to build your next image generation system as a pixel-space cascade (like Imagen) or a latent diffusion model (like Stable Diffusion). Your primary product targets are: (1) consumer-grade hardware deployment, (2) 1024x1024 output resolution, (3) strong compositional text understanding. Which architecture should you recommend and what is the key trade-off?',
          type: 'scenario',
          options: [
            'Pixel-space cascade — higher theoretical quality ceiling justifies the compute cost',
            'Latent diffusion with a large text encoder — 48x fewer diffusion elements enables consumer deployment while a scaled text encoder addresses compositional understanding',
            'Pixel-space with model compression — quantize the model to fit on consumer GPUs',
            'A GAN-based approach — consumer hardware requires single-pass inference'
          ],
          correct: 1,
          explanation: 'Latent diffusion reduces the computational burden by ~48x through VAE compression, making consumer GPU deployment feasible. The lesson from Imagen is that text understanding is the bottleneck for compositional prompts, so pairing latent diffusion with a large text encoder (like T5-XXL, as SD3 does) addresses both requirements simultaneously. Pixel-space cascades require too much compute for consumer hardware, and model compression alone is insufficient. GANs sacrifice the quality and controllability that users expect.',
          difficulty: 'applied',
          expertNote: 'This is exactly the architectural trajectory the industry followed: SD3 and FLUX combine latent diffusion with large text encoders and DiT backbones. The synthesis of Imagen\'s insight (scale the text encoder) with LDM\'s insight (compress to latent space) became the industry standard.'
        },
        {
          question: 'What was Imagen\'s most significant research finding regarding model scaling?',
          type: 'mc',
          options: [
            'Larger U-Net models produced proportionally better image quality across all benchmarks',
            'Scaling the text encoder improved image quality and text-image alignment more than scaling the image generation model',
            'Super-resolution cascades were unnecessary when using larger base models exclusively',
            'CLIP-based text encoders were superior to language model-based encoders for all text understanding tasks'
          ],
          correct: 1,
          explanation: 'Imagen systematically demonstrated that scaling T5 (the text encoder) from Small to XXL improved both FID scores and text-image alignment far more than scaling the U-Net. This finding reshaped the field — subsequent systems invested heavily in text encoders, with SD3 using three text encoders totaling 5B+ parameters.',
          difficulty: 'foundational',
          expertNote: 'This finding is counterintuitive — most engineers instinctively want to scale the "image part" of an image generation model. Imagen showed that the bottleneck was understanding what the user wanted, not generating the pixels. This insight has analogs across AI product development: the interface between intent and execution is often the hardest problem.'
        },
        {
          question: 'DALL-E 3 significantly improved text-image alignment compared to DALL-E 2. What was the primary technique responsible?',
          type: 'mc',
          options: [
            'Switching from a diffusion to an autoregressive architecture for generation',
            'Training a custom captioning model to recaption all training images with highly detailed descriptions',
            'Doubling the size of the CLIP text encoder parameters',
            'Using classifier guidance instead of classifier-free guidance mechanisms'
          ],
          correct: 1,
          explanation: 'DALL-E 3\'s main innovation was data-centric: they trained a specialized captioning model to create detailed, accurate descriptions for training images, replacing noisy web-scraped alt-text. This "recaptioning" approach dramatically improved the model\'s ability to follow complex prompts. It is a textbook example of data quality mattering more than architectural changes.',
          difficulty: 'foundational',
          expertNote: 'The recaptioning approach has become widespread. Training synthetic captions is now a standard preprocessing step for image generation models. This also highlights a meta-lesson: sometimes the biggest gains come from improving the data pipeline, not the model.'
        },
        {
          question: 'Which of the following are true about the Diffusion Transformer (DiT) architecture compared to the traditional U-Net? Select all that apply.',
          type: 'multi',
          options: [
            'DiT treats image patches as tokens and applies standard Transformer blocks',
            'DiT scales more predictably with compute, following established Transformer scaling laws',
            'DiT cannot use classifier-free guidance for text conditioning',
            'DiT handles variable resolutions and aspect ratios more naturally',
            'DiT requires significantly more parameters than a U-Net for equivalent quality'
          ],
          correct: [0, 1, 3],
          explanation: 'DiT patchifies images into tokens and uses standard Transformer blocks with adaptive layer normalization for timestep conditioning. It follows Transformer scaling laws (predictable improvement with more compute), handles variable resolutions naturally, and supports all standard conditioning mechanisms including classifier-free guidance. It does not inherently require more parameters — in fact, at equivalent parameter counts, DiT models often match or exceed U-Net quality.',
          difficulty: 'applied',
          expertNote: 'DiT\'s adoption in Sora and SD3 suggests it will become the dominant architecture for diffusion. For PMs, this means the same Transformer infrastructure (hardware, optimization techniques, serving frameworks) can be shared across text and image models — a significant operational advantage.'
        },
        {
          question: 'A product designer asks you why users cannot simply "edit one object" in a generated image without affecting the rest. Which architectural capability would you point to as the solution?',
          type: 'mc',
          options: [
            'Increasing the guidance scale to focus the model on specific objects exclusively',
            'Using ControlNet with a segmentation mask combined with inpainting to isolate and regenerate specific regions',
            'Switching from latent diffusion to pixel-space diffusion for significantly higher fidelity',
            'Retraining the model on a dataset that contains only single-object images exclusively'
          ],
          correct: 1,
          explanation: 'Inpainting allows the model to regenerate only masked regions while keeping the rest of the image fixed. Combining this with ControlNet (which can accept segmentation masks, edge maps, or depth maps) gives precise spatial control over which object is regenerated and how. This is the standard approach for object-level editing in production diffusion systems.',
          difficulty: 'applied',
          expertNote: 'This is a common PM scenario — translating user needs ("I just want to change the color of the car") into architectural requirements (inpainting + segmentation conditioning). Products like Adobe Firefly and Canva\'s Magic Edit implement exactly this pipeline.'
        },
        {
          question: 'Your team has built a latent diffusion model. During testing, users report that images look slightly "soft" or lacking fine detail compared to pixel-space models. What is the most likely cause and the standard mitigation?',
          type: 'mc',
          options: [
            'The U-Net is too small — double its parameters to improve quality',
            'The VAE decoder introduces reconstruction artifacts when decompressing from latent space; a refiner model or VAE fine-tuning can address this',
            'The text encoder is not powerful enough to describe fine details accurately',
            'Classifier-free guidance is set too low, causing the model to generate generic outputs consistently'
          ],
          correct: 1,
          explanation: 'The VAE compression is lossy — going from 512x512x3 to 64x64x4 and back inevitably loses some high-frequency detail. This manifests as slight softness. Standard mitigations include: (1) fine-tuning the VAE decoder for sharper reconstruction, (2) using a refiner model (as SDXL does) that adds detail in a second pass, or (3) training with a higher-resolution latent space. This is a known trade-off of the latent diffusion approach.',
          difficulty: 'expert',
          expertNote: 'SDXL addressed this with its "refiner" model — a second diffusion model specialized for adding high-frequency detail. SD3 used an improved VAE with a higher-dimensional latent space (16 channels vs. 4). Understanding where quality loss occurs in the pipeline is crucial for debugging generative systems.'
        }
      ]
    }
  },

  // ─────────────────────────────────────────────
  // L04 — Multimodal Models
  // ─────────────────────────────────────────────
  l04: {
    title: 'Multimodal Models — Connecting Vision and Language',
    content: `
<h2>What Are Multimodal Models?</h2>
<p>
  A <span class="term" data-term="multimodal">multimodal model</span> is a system that can process,
  understand, and generate content across multiple data types — typically text, images, audio, and
  video. While text-to-image models like Stable Diffusion are technically multimodal (they connect
  text and vision), the term increasingly refers to models that can <em>reason</em> across modalities:
  answer questions about images, generate images from conversational context, transcribe speech while
  understanding visual context, etc.
</p>
<p>
  The trajectory is clear: frontier AI is converging toward unified multimodal systems. GPT-4V,
  Gemini, and Claude can all process images alongside text. Understanding how these systems work
  is critical for any PM building products on top of them.
</p>

<h2>CLIP: The Foundation of Vision-Language Alignment</h2>
<p>
  CLIP (Contrastive Language-Image Pre-training, Radford et al., 2021) is the foundational model
  that enabled the modern multimodal era. Its architecture and training approach are elegant:
</p>
<ul>
  <li><strong>Two encoders:</strong> A vision encoder (ViT or ResNet) and a text encoder (Transformer).</li>
  <li><strong>Contrastive objective:</strong> Given a batch of N (image, text) pairs, CLIP learns to
      maximize the cosine similarity of the N correct pairs while minimizing similarity of the
      N&sup2; - N incorrect pairs.</li>
  <li><strong>Training data:</strong> 400 million (image, text) pairs scraped from the internet.</li>
</ul>
<p>
  The result is a <strong>shared embedding space</strong> where images and text describing similar
  concepts land near each other. This space is used for zero-shot classification, image search,
  and — critically — as the text conditioning mechanism in diffusion models.
</p>

<div class="key-concept">
  <strong>Key Concept:</strong> CLIP's shared embedding space is what makes text-to-image generation
  possible. When you type "a sunset over mountains" into Stable Diffusion, the CLIP text encoder
  maps your words to a point in embedding space. The diffusion model then generates an image whose
  CLIP image embedding is close to that point. CLIP is the bridge between language and vision.
</div>

<h2>Contrastive vs. Generative Multimodal Training</h2>
<p>
  There are two fundamental approaches to connecting modalities:
</p>
<table>
  <thead>
    <tr><th>Approach</th><th>How It Works</th><th>Strengths</th><th>Limitations</th></tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Contrastive</strong> (CLIP, ALIGN, SigLIP)</td>
      <td>Learn to match corresponding pairs while pushing apart non-matching pairs</td>
      <td>Efficient training, excellent retrieval, strong zero-shot transfer</td>
      <td>Cannot generate content; understanding is limited to similarity matching</td>
    </tr>
    <tr>
      <td><strong>Generative</strong> (Flamingo, GPT-4V, Gemini)</td>
      <td>Train a model to generate text conditioned on visual (and/or other) input</td>
      <td>Deep reasoning, question answering, flexible generation</td>
      <td>More compute-intensive, requires careful data curation</td>
    </tr>
  </tbody>
</table>

<div class="pro-tip">
  <strong>PM Perspective:</strong> Contrastive models are best for search, ranking, and classification
  tasks. Generative multimodal models are needed for conversational understanding, reasoning, and
  explanation. Most production systems use both: CLIP-like models for fast retrieval and filtering,
  and generative models for deep understanding. Know which you need for each product feature.
</div>

<h2>Architecture Patterns for Multimodal LLMs</h2>
<p>
  Modern multimodal LLMs follow a few key architectural patterns:
</p>

<h3>Pattern 1: Visual Encoder + Adapter + LLM</h3>
<p>
  The most common pattern connects a pretrained vision encoder to a pretrained LLM via a
  lightweight adapter:
</p>
<ul>
  <li><strong>LLaVA (Large Language and Vision Assistant):</strong> ViT-L/14 → linear projection → Vicuna LLM.
      Remarkably, a single linear layer is sufficient to connect the visual encoder to the language model.</li>
  <li><strong>Flamingo (DeepMind):</strong> Uses Perceiver Resampler modules to compress visual features
      and gated cross-attention layers interleaved with frozen LLM layers.</li>
  <li><strong>BLIP-2:</strong> Introduces a Q-Former (Querying Transformer) — a lightweight Transformer
      with learned query vectors that extract a fixed number of visual tokens from the frozen image encoder.</li>
</ul>

<div class="example-box">
  <h4>Example</h4>
  <p>In LLaVA, an image is processed by CLIP ViT-L/14 to produce 576 visual tokens (24x24 grid of
  patch embeddings). A linear projection maps these to the LLM's embedding dimension. The visual
  tokens are then prepended to the text tokens and processed by the LLM as if they were "visual words."
  The LLM does not know it is looking at visual tokens — it simply processes a longer sequence where
  the first 576 positions happen to encode visual information.</p>
</div>

<h3>Pattern 2: Native Multimodal (Single Architecture)</h3>
<p>
  Instead of bolting a vision encoder onto an LLM, some systems are designed multimodal from the start:
</p>
<ul>
  <li><strong>Gemini (Google DeepMind):</strong> Natively multimodal Transformer trained on interleaved
      text, image, audio, and video data from the beginning. All modalities share the same
      Transformer backbone.</li>
  <li><strong>Fuyu (Adept):</strong> Feeds raw image patches directly into a decoder-only Transformer
      without a separate vision encoder. Image patches are linearly projected into the token space.</li>
</ul>

<div class="key-concept">
  <strong>Key Concept:</strong> The distinction between "bolted-on" and "native" multimodality matters
  for product capabilities. Bolted-on systems (Pattern 1) can leverage the best pretrained vision and
  language models but may have a "seam" between modalities — they can describe what they see but may
  struggle with fine-grained spatial reasoning. Native multimodal systems (Pattern 2) can develop
  deeper cross-modal understanding but require training from scratch on massive multimodal datasets.
</div>

<h2>Key Capabilities and Benchmarks</h2>
<p>
  Multimodal models are evaluated on a range of capabilities:
</p>
<ul>
  <li><strong>Visual Question Answering (VQA):</strong> "What color is the car in the background?"</li>
  <li><strong>Optical Character Recognition (OCR):</strong> Reading text in images — signs, documents, code.</li>
  <li><strong>Spatial reasoning:</strong> "Is the cat on the left or right side of the table?"</li>
  <li><strong>Chart/diagram understanding:</strong> Interpreting data visualizations and technical diagrams.</li>
  <li><strong>Multi-image reasoning:</strong> Comparing, contrasting, or synthesizing information across
      multiple images.</li>
  <li><strong>Grounding:</strong> Localizing objects referenced in text (bounding boxes).</li>
</ul>

<table>
  <thead>
    <tr><th>Benchmark</th><th>What It Tests</th><th>Why It Matters for PMs</th></tr>
  </thead>
  <tbody>
    <tr><td>MMMU</td><td>Multi-discipline multimodal understanding</td><td>College-level reasoning across subjects</td></tr>
    <tr><td>MathVista</td><td>Mathematical reasoning in visual contexts</td><td>Chart/diagram interpretation for business tools</td></tr>
    <tr><td>DocVQA</td><td>Document understanding and question answering</td><td>Critical for enterprise document processing</td></tr>
    <tr><td>RealWorldQA</td><td>Real-world image understanding</td><td>Practical everyday visual reasoning</td></tr>
  </tbody>
</table>

<h2>From Understanding to Generation: Unified Multimodal Models</h2>
<p>
  The next frontier is models that can both <em>understand</em> and <em>generate</em> across
  modalities. Current examples:
</p>
<ul>
  <li><strong>Gemini with Imagen integration:</strong> Can understand images in conversation and
      generate new images within the same interaction.</li>
  <li><strong>GPT-4o:</strong> Processes and generates text, images, and audio natively within a
      single model.</li>
  <li><strong>Chameleon (Meta):</strong> An early-fusion token-based model that interleaves image
      and text tokens throughout the architecture.</li>
</ul>

<div class="warning">
  <strong>Common Misconception:</strong> Many people assume that multimodal models like GPT-4V literally
  "see" images the way humans do. In reality, the vision encoder converts the image into a grid of
  embedding vectors that the language model processes as a sequence of tokens. The model has no
  built-in understanding of 3D geometry, physics, or causation — it learns statistical associations
  between visual patterns and language. This matters for product decisions: do not promise capabilities
  (like reliable spatial reasoning or accurate counting) that the architecture does not guarantee.
</div>

<h2>Multimodal Applications and Product Opportunities</h2>
<p>
  The product landscape for multimodal AI is vast and rapidly expanding:
</p>
<ul>
  <li><strong>Visual assistants:</strong> Google Lens, smartphone camera understanding, accessibility tools for visually impaired users.</li>
  <li><strong>Document intelligence:</strong> Extracting information from invoices, contracts, medical records.</li>
  <li><strong>Creative tools:</strong> Unified platforms where users describe, generate, edit, and iterate on visual content through conversation.</li>
  <li><strong>Autonomous agents:</strong> Models that can see and interact with GUIs, web pages, and physical environments.</li>
  <li><strong>Scientific research:</strong> Analyzing microscopy images, protein structures, satellite imagery with natural language queries.</li>
</ul>

<div class="pro-tip">
  <strong>PM Perspective:</strong> The convergence of understanding and generation into unified
  multimodal models is the most important architectural trend for AI product development. It means
  the user interface can be a single conversational thread where the user describes what they want,
  the model generates it, the user provides feedback (possibly as an annotated image), and the model
  iterates. This is qualitatively different from discrete "upload → process → download" workflows.
  Design your products for this conversational, iterative paradigm.
</div>

<h2>Challenges and Open Problems</h2>
<p>
  Despite rapid progress, significant challenges remain:
</p>
<ul>
  <li><strong>Hallucination in visual reasoning:</strong> Models confidently describe objects that
      are not present in an image. This is analogous to text hallucination but harder to detect
      because it requires comparing model output against visual ground truth.</li>
  <li><strong>Fine-grained spatial understanding:</strong> Counting objects, understanding relative
      positions, reading small text — these remain unreliable even in frontier models.</li>
  <li><strong>Multimodal safety:</strong> Images can be used to jailbreak text safety filters. A model
      that refuses to describe how to pick a lock via text might comply when the instructions are
      embedded in an image.</li>
  <li><strong>Evaluation difficulty:</strong> Benchmarks for multimodal capabilities are still
      immature compared to text-only benchmarks. It is hard to measure whether a model truly
      "understands" an image or has memorized patterns from training data.</li>
</ul>

<div class="key-concept">
  <strong>Key Concept:</strong> Multimodal hallucination is one of the hardest unsolved problems in AI
  safety. When a text model hallucinates, you can sometimes catch it with retrieval or fact-checking.
  When a multimodal model hallucinates about image content, there is no straightforward automated
  way to verify the claim against the image — you need either a second multimodal model or human review.
  This is a first-order product concern for any application involving visual understanding.
</div>
`,
    quiz: {
      questions: [
        {
          question: 'You are building a document processing product at DeepMind that needs to extract structured data from scanned invoices. The system must handle diverse layouts, handwritten annotations, and tables. Which multimodal architecture pattern is most appropriate?',
          type: 'scenario',
          options: [
            'CLIP-based contrastive model — embed documents and search for matching templates',
            'A generative multimodal LLM (like GPT-4V/Gemini) that can reason about document structure and extract fields through natural language interaction',
            'A text-only LLM that processes OCR output — multimodal capabilities are unnecessary',
            'A diffusion model conditioned on document images to generate cleaned-up versions'
          ],
          correct: 1,
          explanation: 'Document extraction with diverse layouts, handwriting, and tables requires deep visual reasoning — understanding spatial relationships, table structure, and handwritten text. A generative multimodal LLM can be prompted to extract specific fields, handle layout variations flexibly, and even explain its reasoning. CLIP is too limited (similarity matching only). Text-only LLM after OCR loses spatial/layout information. Diffusion models generate images, not structured data.',
          difficulty: 'applied',
          expertNote: 'In practice, you would likely combine a specialized OCR model with a multimodal LLM. The OCR handles character recognition, while the LLM handles layout understanding and field extraction. This hybrid approach is more robust than either alone.'
        },
        {
          question: 'What is the fundamental difference between CLIP\'s contrastive training and Flamingo\'s generative training?',
          type: 'mc',
          options: [
            'CLIP uses a larger vision encoder while Flamingo uses a smaller one',
            'CLIP learns to match image-text pairs via similarity; Flamingo learns to generate text conditioned on visual input, enabling open-ended reasoning',
            'CLIP can process multiple images while Flamingo is limited to one',
            'CLIP is trained on more data while Flamingo uses a curated smaller dataset'
          ],
          correct: 1,
          explanation: 'CLIP learns a shared embedding space through contrastive loss — it can tell you how similar an image and text are, but it cannot generate explanations, answer questions, or reason. Flamingo (and similar generative multimodal models) learn to produce text tokens conditioned on visual input, enabling open-ended visual question answering, description, and reasoning.',
          difficulty: 'foundational',
          expertNote: 'This distinction maps directly to product capabilities. If your feature needs ranking or retrieval (e.g., "find the most relevant image"), CLIP is sufficient and far more efficient. If your feature needs understanding (e.g., "explain what is happening in this photo"), you need a generative model.'
        },
        {
          question: 'Your team wants to add image understanding to an existing text-based AI assistant. Which of the following approaches would allow the fastest time-to-deployment while preserving the quality of the existing text capabilities?',
          type: 'mc',
          options: [
            'Retrain the entire model from scratch on a multimodal dataset',
            'Use a bolted-on approach: add a pretrained vision encoder connected via a lightweight adapter to the frozen LLM',
            'Replace the text model with a natively multimodal model like Gemini',
            'Use CLIP to convert images to text descriptions, then feed those to the existing text model'
          ],
          correct: 1,
          explanation: 'The bolted-on approach (Pattern 1) preserves the existing LLM weights entirely — the text capabilities remain unchanged. Only the vision encoder and adapter need training, which requires far less compute and data than training from scratch. This is exactly the approach taken by LLaVA, which achieved strong results with just a linear projection layer. Retraining from scratch is slow and risks regressing text quality. Replacing the model entirely changes the product unpredictably. CLIP-to-text loses too much visual information.',
          difficulty: 'applied',
          expertNote: 'LLaVA demonstrated that even a simple linear projection between a frozen CLIP encoder and a frozen LLM could achieve surprisingly good results with minimal training. This validates the "bolt-on" approach as a legitimate rapid deployment strategy.'
        },
        {
          question: 'Which of the following are genuine challenges with current multimodal models that a PM must account for in product design? Select all that apply.',
          type: 'multi',
          options: [
            'Visual hallucination — models describe objects or attributes not present in the image',
            'Multimodal models cannot process images larger than 256x256 pixels',
            'Images can be used as attack vectors to bypass text-based safety filters',
            'Fine-grained spatial reasoning (counting, relative positioning) remains unreliable',
            'Evaluating multimodal capabilities is harder than evaluating text-only capabilities due to benchmark immaturity'
          ],
          correct: [0, 2, 3, 4],
          explanation: 'All options except the image size limitation are genuine challenges. Modern multimodal models process images at various resolutions (GPT-4V handles up to 2048 pixels; Gemini handles variable resolutions). Visual hallucination, multimodal jailbreaking, unreliable spatial reasoning, and benchmark immaturity are all well-documented challenges that directly affect product quality and safety.',
          difficulty: 'foundational',
          expertNote: 'Multimodal safety is particularly concerning because it is an underexplored attack surface. Safety teams have invested years in text-based red teaming, but visual jailbreaks are newer and less well-defended. As a PM, ensure your safety roadmap explicitly covers multimodal attack vectors.'
        },
        {
          question: 'Google DeepMind built Gemini as a natively multimodal model rather than bolting vision onto an existing LLM. What is the primary architectural advantage of this approach?',
          type: 'mc',
          options: [
            'Native multimodality requires fewer parameters than bolted-on approaches',
            'The model can develop deeper cross-modal reasoning because modalities interact throughout the entire architecture from pretraining, rather than being connected through a narrow adapter',
            'Native multimodal models are faster at inference because they skip the vision encoding step',
            'Only native multimodal models can generate images — bolted-on models cannot'
          ],
          correct: 1,
          explanation: 'When modalities are trained jointly from the beginning, the model can learn rich cross-modal representations at every layer of the network. In a bolted-on approach, the connection between vision and language is limited to the adapter layer — a potential bottleneck for complex reasoning that requires deep integration of visual and textual information. Native multimodality allows the model to develop "visual thinking" throughout its entire depth.',
          difficulty: 'applied',
          expertNote: 'The trade-off is that native multimodal training requires massive multimodal datasets and compute from the start. You cannot easily reuse an existing strong text model. Google DeepMind invested in this approach because they had both the data and compute, and believed the deeper integration would yield fundamentally better cross-modal reasoning.'
        }
      ]
    }
  }
};
