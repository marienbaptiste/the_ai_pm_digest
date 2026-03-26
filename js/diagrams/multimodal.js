export function render(container) {
  container.innerHTML = `
    <div style="text-align: center; width: 100%;">
      <svg viewBox="0 0 850 520" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto;">
        <style>
          .mm-fade { animation: mmFadeIn 0.6s ease forwards; opacity: 0; }
          .mm-slide-l { animation: mmSlideLeft 0.7s ease forwards; opacity: 0; transform: translateX(-20px); }
          .mm-slide-r { animation: mmSlideRight 0.7s ease forwards; opacity: 0; transform: translateX(20px); }
          .mm-arrow { stroke-dasharray: 8 4; animation: mmFlow 1.2s linear infinite; }
          .mm-label { font-family: var(--font-mono); font-size: 10px; fill: var(--text-dim); }
          .mm-title { font-family: var(--font-heading); font-size: 12px; font-weight: 600; }
          .mm-vec { animation: mmVecFlow 0.4s ease forwards; opacity: 0; transform: translateY(10px); }
          @keyframes mmFadeIn { to { opacity: 1; } }
          @keyframes mmSlideLeft { to { opacity: 1; transform: translateX(0); } }
          @keyframes mmSlideRight { to { opacity: 1; transform: translateX(0); } }
          @keyframes mmFlow { to { stroke-dashoffset: -24; } }
          @keyframes mmVecFlow { to { opacity: 1; transform: translateY(0); } }
        </style>

        <defs>
          <marker id="mm-arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="var(--border-medium)"/>
          </marker>
          <marker id="mm-arrowhead-teal" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="#9CCFA4"/>
          </marker>
          <linearGradient id="mm-fusion-grad" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stop-color="#E8553A" stop-opacity="0.2"/>
            <stop offset="50%" stop-color="#C4A7E7" stop-opacity="0.3"/>
            <stop offset="100%" stop-color="#7EB8DA" stop-opacity="0.2"/>
          </linearGradient>
        </defs>

        <!-- Title -->
        <g class="mm-fade" style="animation-delay: 0s;">
          <text x="425" y="30" text-anchor="middle" font-family="var(--font-heading)" font-size="15" fill="var(--text-primary)" font-weight="700">Multimodal Architecture</text>
        </g>

        <!-- ========== LEFT STREAM: Vision ========== -->

        <!-- Image input icon -->
        <g class="mm-slide-l" style="animation-delay: 0s;">
          <rect x="30" y="65" width="80" height="65" rx="8" fill="#E8553A15" stroke="#E8553A" stroke-width="1.5"/>
          <rect x="45" y="75" width="50" height="35" rx="4" fill="none" stroke="#E8553A" stroke-width="1.2"/>
          <circle cx="58" cy="87" r="5" fill="#E8553A40" stroke="#E8553A" stroke-width="1"/>
          <polyline points="47,105 60,93 72,100 82,90 92,103" fill="none" stroke="#E8553A" stroke-width="1.2"/>
          <text x="70" y="148" text-anchor="middle" class="mm-label" fill="#E8553A">Image Input</text>
        </g>

        <!-- Arrow: Image -> Vision Encoder -->
        <line x1="115" y1="97" x2="158" y2="97" stroke="var(--border-medium)" stroke-width="1.5" class="mm-arrow" marker-end="url(#mm-arrowhead)"/>

        <!-- Vision Encoder box -->
        <g class="mm-slide-l" style="animation-delay: 0.4s;">
          <rect x="162" y="50" width="160" height="95" rx="10" fill="#E8553A12" stroke="#E8553A" stroke-width="2"/>
          <text x="242" y="72" text-anchor="middle" class="mm-title" fill="#E8553A">Vision Encoder</text>
          <text x="242" y="87" text-anchor="middle" class="mm-label" fill="#E8553A">ViT / CNN</text>
          <rect x="178" y="96" width="128" height="6" rx="2" fill="#E8553A25" stroke="none">
            <animate attributeName="opacity" values="0.3;0.7;0.3" dur="3s" repeatCount="indefinite"/>
          </rect>
          <rect x="178" y="106" width="128" height="6" rx="2" fill="#E8553A30" stroke="none">
            <animate attributeName="opacity" values="0.3;0.7;0.3" dur="3s" repeatCount="indefinite" begin="0.3s"/>
          </rect>
          <rect x="178" y="116" width="128" height="6" rx="2" fill="#E8553A35" stroke="none">
            <animate attributeName="opacity" values="0.3;0.7;0.3" dur="3s" repeatCount="indefinite" begin="0.6s"/>
          </rect>
          <rect x="178" y="126" width="128" height="6" rx="2" fill="#E8553A40" stroke="none">
            <animate attributeName="opacity" values="0.3;0.7;0.3" dur="3s" repeatCount="indefinite" begin="0.9s"/>
          </rect>
        </g>

        <!-- Vision feature vectors -->
        <g class="mm-vec" style="animation-delay: 1.0s;">
          <rect x="242" y="165" width="8" height="14" rx="2" fill="#E8553A" opacity="0.9"/>
          <rect x="253" y="169" width="8" height="10" rx="2" fill="#E8553A" opacity="0.7"/>
          <rect x="264" y="163" width="8" height="16" rx="2" fill="#E8553A" opacity="0.8"/>
          <rect x="275" y="167" width="8" height="12" rx="2" fill="#E8553A" opacity="0.6"/>
          <text x="264" y="196" text-anchor="middle" class="mm-label" fill="#E8553A">Image features</text>
        </g>

        <!-- Arrow: Vision vectors -> Fusion -->
        <path d="M 264 200 L 264 230 L 340 250" fill="none" stroke="var(--border-medium)" stroke-width="1.5" class="mm-arrow" marker-end="url(#mm-arrowhead)"/>


        <!-- ========== RIGHT STREAM: Text ========== -->

        <!-- Text tokens input -->
        <g class="mm-slide-r" style="animation-delay: 0.6s;">
          <rect x="640" y="65" width="80" height="65" rx="8" fill="#7EB8DA15" stroke="#7EB8DA" stroke-width="1.5"/>
          <rect x="650" y="75" width="26" height="16" rx="3" fill="#7EB8DA30" stroke="#7EB8DA" stroke-width="0.8"/>
          <text x="663" y="87" text-anchor="middle" font-family="var(--font-mono)" font-size="7" fill="#7EB8DA">The</text>
          <rect x="680" y="75" width="30" height="16" rx="3" fill="#7EB8DA30" stroke="#7EB8DA" stroke-width="0.8"/>
          <text x="695" y="87" text-anchor="middle" font-family="var(--font-mono)" font-size="7" fill="#7EB8DA">cat</text>
          <rect x="650" y="95" width="26" height="16" rx="3" fill="#7EB8DA30" stroke="#7EB8DA" stroke-width="0.8"/>
          <text x="663" y="107" text-anchor="middle" font-family="var(--font-mono)" font-size="7" fill="#7EB8DA">sat</text>
          <rect x="680" y="95" width="30" height="16" rx="3" fill="#7EB8DA30" stroke="#7EB8DA" stroke-width="0.8"/>
          <text x="695" y="107" text-anchor="middle" font-family="var(--font-mono)" font-size="7" fill="#7EB8DA">on</text>
          <text x="680" y="148" text-anchor="middle" class="mm-label" fill="#7EB8DA">Text Tokens</text>
        </g>

        <!-- Arrow: Text -> Text Encoder -->
        <line x1="638" y1="97" x2="600" y2="97" stroke="var(--border-medium)" stroke-width="1.5" class="mm-arrow" marker-end="url(#mm-arrowhead)"/>

        <!-- Text Encoder box -->
        <g class="mm-slide-r" style="animation-delay: 1.0s;">
          <rect x="438" y="50" width="160" height="95" rx="10" fill="#7EB8DA12" stroke="#7EB8DA" stroke-width="2"/>
          <text x="518" y="72" text-anchor="middle" class="mm-title" fill="#7EB8DA">Text Encoder</text>
          <text x="518" y="87" text-anchor="middle" class="mm-label" fill="#7EB8DA">Transformer</text>
          <rect x="454" y="96" width="128" height="6" rx="2" fill="#7EB8DA25" stroke="none">
            <animate attributeName="opacity" values="0.3;0.7;0.3" dur="3s" repeatCount="indefinite" begin="0.2s"/>
          </rect>
          <rect x="454" y="106" width="128" height="6" rx="2" fill="#7EB8DA30" stroke="none">
            <animate attributeName="opacity" values="0.3;0.7;0.3" dur="3s" repeatCount="indefinite" begin="0.5s"/>
          </rect>
          <rect x="454" y="116" width="128" height="6" rx="2" fill="#7EB8DA35" stroke="none">
            <animate attributeName="opacity" values="0.3;0.7;0.3" dur="3s" repeatCount="indefinite" begin="0.8s"/>
          </rect>
          <rect x="454" y="126" width="128" height="6" rx="2" fill="#7EB8DA40" stroke="none">
            <animate attributeName="opacity" values="0.3;0.7;0.3" dur="3s" repeatCount="indefinite" begin="1.1s"/>
          </rect>
        </g>

        <!-- Text feature vectors -->
        <g class="mm-vec" style="animation-delay: 1.5s;">
          <rect x="498" y="165" width="8" height="12" rx="2" fill="#7EB8DA" opacity="0.8"/>
          <rect x="509" y="163" width="8" height="16" rx="2" fill="#7EB8DA" opacity="0.9"/>
          <rect x="520" y="167" width="8" height="10" rx="2" fill="#7EB8DA" opacity="0.65"/>
          <rect x="531" y="164" width="8" height="14" rx="2" fill="#7EB8DA" opacity="0.85"/>
          <text x="518" y="196" text-anchor="middle" class="mm-label" fill="#7EB8DA">Text features</text>
        </g>

        <!-- Arrow: Text vectors -> Fusion -->
        <path d="M 518 200 L 518 230 L 470 250" fill="none" stroke="var(--border-medium)" stroke-width="1.5" class="mm-arrow" marker-end="url(#mm-arrowhead)"/>


        <!-- ========== CENTER: Fusion Layer ========== -->

        <g class="mm-fade" style="animation-delay: 2.0s;">
          <rect x="260" y="245" width="330" height="100" rx="14" fill="url(#mm-fusion-grad)" stroke="#C4A7E7" stroke-width="2"/>
          <text x="425" y="275" text-anchor="middle" class="mm-title" fill="#C4A7E7" font-size="14">Cross-Attention / Fusion</text>
          <text x="425" y="295" text-anchor="middle" class="mm-label">Image tokens attend to text tokens &amp; vice versa</text>

          <!-- Attention lines inside fusion -->
          <g opacity="0.4">
            <line x1="310" y1="310" x2="360" y2="320" stroke="#E8553A" stroke-width="1">
              <animate attributeName="opacity" values="0.2;0.8;0.2" dur="2s" repeatCount="indefinite"/>
            </line>
            <line x1="370" y1="310" x2="425" y2="320" stroke="#C4A7E7" stroke-width="1">
              <animate attributeName="opacity" values="0.2;0.8;0.2" dur="2s" repeatCount="indefinite" begin="0.3s"/>
            </line>
            <line x1="440" y1="310" x2="490" y2="320" stroke="#7EB8DA" stroke-width="1">
              <animate attributeName="opacity" values="0.2;0.8;0.2" dur="2s" repeatCount="indefinite" begin="0.6s"/>
            </line>
            <line x1="500" y1="310" x2="540" y2="320" stroke="#C4A7E7" stroke-width="1">
              <animate attributeName="opacity" values="0.2;0.8;0.2" dur="2s" repeatCount="indefinite" begin="0.9s"/>
            </line>
          </g>

          <!-- Q K V labels -->
          <rect x="350" y="315" width="28" height="16" rx="3" fill="#E8553A40"/>
          <text x="364" y="327" text-anchor="middle" font-family="var(--font-mono)" font-size="8" fill="#E8553A" font-weight="600">Q</text>
          <rect x="383" y="315" width="28" height="16" rx="3" fill="#7EB8DA40"/>
          <text x="397" y="327" text-anchor="middle" font-family="var(--font-mono)" font-size="8" fill="#7EB8DA" font-weight="600">K</text>
          <rect x="416" y="315" width="28" height="16" rx="3" fill="#C4A7E740"/>
          <text x="430" y="327" text-anchor="middle" font-family="var(--font-mono)" font-size="8" fill="#C4A7E7" font-weight="600">V</text>
        </g>

        <!-- Arrow: Fusion -> Unified representation -->
        <line x1="425" y1="345" x2="425" y2="375" stroke="var(--border-medium)" stroke-width="1.5" class="mm-arrow" marker-end="url(#mm-arrowhead)"/>

        <!-- Unified representation -->
        <g class="mm-fade" style="animation-delay: 2.6s;">
          <rect x="310" y="380" width="230" height="40" rx="8" fill="#C4A7E710" stroke="#C4A7E7" stroke-width="1" stroke-dasharray="4"/>
          <!-- Mixed color vectors -->
          <rect x="340" y="388" width="6" height="10" rx="1" fill="#E8553A" opacity="0.7"/>
          <rect x="350" y="386" width="6" height="12" rx="1" fill="#C4A7E7" opacity="0.8"/>
          <rect x="360" y="389" width="6" height="9" rx="1" fill="#7EB8DA" opacity="0.7"/>
          <rect x="370" y="387" width="6" height="11" rx="1" fill="#E8553A" opacity="0.6"/>
          <rect x="380" y="385" width="6" height="13" rx="1" fill="#C4A7E7" opacity="0.9"/>
          <text x="425" y="407" text-anchor="middle" class="mm-label" fill="#C4A7E7">Unified Multimodal Representation</text>
        </g>

        <!-- Arrow: Unified -> Output -->
        <line x1="425" y1="420" x2="425" y2="445" stroke="#9CCFA4" stroke-width="2" class="mm-arrow" marker-end="url(#mm-arrowhead-teal)"/>

        <!-- Multimodal Output -->
        <g class="mm-fade" style="animation-delay: 3.0s;">
          <rect x="290" y="448" width="270" height="55" rx="12" fill="#9CCFA415" stroke="#9CCFA4" stroke-width="2"/>
          <text x="425" y="472" text-anchor="middle" class="mm-title" fill="#9CCFA4" font-size="14">Multimodal Output</text>
          <text x="425" y="490" text-anchor="middle" class="mm-label">Caption, VQA, Generation, Grounding</text>
          <rect x="292" y="450" width="266" height="51" rx="11" fill="none" stroke="#9CCFA4" stroke-width="1" opacity="0.3">
            <animate attributeName="opacity" values="0.1;0.4;0.1" dur="3s" repeatCount="indefinite"/>
          </rect>
        </g>

        <!-- Side annotations -->
        <g class="mm-fade" style="animation-delay: 3.4s;">
          <!-- Vision side label -->
          <rect x="30" y="255" width="110" height="40" rx="6" fill="#0C0A0990" stroke="var(--border-subtle)" stroke-width="1"/>
          <text x="85" y="273" text-anchor="middle" font-family="var(--font-mono)" font-size="8" fill="#E8553A">Patch embeddings</text>
          <text x="85" y="286" text-anchor="middle" font-family="var(--font-mono)" font-size="8" fill="#E8553A">+ position enc.</text>
          <line x1="140" y1="275" x2="260" y2="285" stroke="#E8553A" stroke-width="0.8" stroke-dasharray="3" opacity="0.4"/>

          <!-- Text side label -->
          <rect x="710" y="255" width="110" height="40" rx="6" fill="#0C0A0990" stroke="var(--border-subtle)" stroke-width="1"/>
          <text x="765" y="273" text-anchor="middle" font-family="var(--font-mono)" font-size="8" fill="#7EB8DA">Token embeddings</text>
          <text x="765" y="286" text-anchor="middle" font-family="var(--font-mono)" font-size="8" fill="#7EB8DA">+ position enc.</text>
          <line x1="710" y1="275" x2="590" y2="285" stroke="#7EB8DA" stroke-width="0.8" stroke-dasharray="3" opacity="0.4"/>
        </g>

      </svg>
      <p style="font-family: var(--font-mono); font-size: var(--text-xs); color: var(--text-dim); margin-top: var(--space-3);">
        Multimodal Fusion - Vision + Language Cross-Attention Architecture
      </p>
    </div>
  `;
}
