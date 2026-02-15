export function render(container) {
  container.innerHTML = `
    <div style="text-align: center; width: 100%;">
      <svg viewBox="0 0 750 430" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto;">
        <style>
          .mm-fade { animation: mmFadeIn 0.6s ease forwards; opacity: 0; }
          .mm-slide-l { animation: mmSlideLeft 0.7s ease forwards; opacity: 0; transform: translateX(-20px); }
          .mm-slide-r { animation: mmSlideRight 0.7s ease forwards; opacity: 0; transform: translateX(20px); }
          .mm-arrow { stroke-dasharray: 8 4; animation: mmFlow 1.2s linear infinite; }
          .mm-label { font-family: var(--font-mono); font-size: 10px; fill: var(--text-dim); }
          .mm-title { font-family: var(--font-heading); font-size: 12px; font-weight: 600; }
          .mm-vec { animation: mmVecFlow 0.4s ease forwards; opacity: 0; transform: translateY(10px); }
          .mm-pulse { animation: mmPulse 2s ease-in-out infinite; }
          @keyframes mmFadeIn { to { opacity: 1; } }
          @keyframes mmSlideLeft { to { opacity: 1; transform: translateX(0); } }
          @keyframes mmSlideRight { to { opacity: 1; transform: translateX(0); } }
          @keyframes mmFlow { to { stroke-dashoffset: -24; } }
          @keyframes mmVecFlow { to { opacity: 1; transform: translateY(0); } }
          @keyframes mmPulse { 0%, 100% { opacity: 0.6; } 50% { opacity: 1; } }
          @keyframes mmLayerShimmer { 0%, 100% { opacity: 0.3; } 50% { opacity: 0.7; } }
        </style>

        <defs>
          <marker id="mm-arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="var(--border-medium)"/>
          </marker>
          <marker id="mm-arrowhead-teal" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="#00d4aa"/>
          </marker>
          <linearGradient id="mm-fusion-grad" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stop-color="#ec4899" stop-opacity="0.2"/>
            <stop offset="50%" stop-color="#a855f7" stop-opacity="0.3"/>
            <stop offset="100%" stop-color="#3b82f6" stop-opacity="0.2"/>
          </linearGradient>
        </defs>

        <!-- ========== LEFT STREAM: Vision ========== -->

        <!-- Image input icon -->
        <g class="mm-slide-l" style="animation-delay: 0s;">
          <rect x="30" y="50" width="80" height="65" rx="8" fill="#ec489915" stroke="#ec4899" stroke-width="1.5"/>
          <!-- Simple image icon -->
          <rect x="45" y="60" width="50" height="35" rx="4" fill="none" stroke="#ec4899" stroke-width="1.2"/>
          <circle cx="58" cy="72" r="5" fill="#ec489940" stroke="#ec4899" stroke-width="1"/>
          <polyline points="47,90 60,78 72,85 82,75 92,88" fill="none" stroke="#ec4899" stroke-width="1.2"/>
          <text x="70" y="108" text-anchor="middle" class="mm-label" fill="#ec4899">Image Input</text>
        </g>

        <!-- Arrow: Image -> Vision Encoder -->
        <line x1="110" y1="82" x2="148" y2="82" stroke="var(--border-medium)" stroke-width="1.5" class="mm-arrow" marker-end="url(#mm-arrowhead)"/>

        <!-- Vision Encoder box -->
        <g class="mm-slide-l" style="animation-delay: 0.4s;">
          <rect x="152" y="35" width="140" height="95" rx="10" fill="#ec489912" stroke="#ec4899" stroke-width="2"/>
          <text x="222" y="55" text-anchor="middle" class="mm-title" fill="#ec4899">Vision Encoder</text>
          <text x="222" y="70" text-anchor="middle" class="mm-label" fill="#ec4899">ViT / CNN</text>
          <!-- Internal layers (stacked lines) -->
          <rect x="168" y="78" width="108" height="6" rx="2" fill="#ec489925" stroke="none">
            <animate attributeName="opacity" values="0.3;0.7;0.3" dur="3s" repeatCount="indefinite"/>
          </rect>
          <rect x="168" y="88" width="108" height="6" rx="2" fill="#ec489930" stroke="none">
            <animate attributeName="opacity" values="0.3;0.7;0.3" dur="3s" repeatCount="indefinite" begin="0.3s"/>
          </rect>
          <rect x="168" y="98" width="108" height="6" rx="2" fill="#ec489935" stroke="none">
            <animate attributeName="opacity" values="0.3;0.7;0.3" dur="3s" repeatCount="indefinite" begin="0.6s"/>
          </rect>
          <rect x="168" y="108" width="108" height="6" rx="2" fill="#ec489940" stroke="none">
            <animate attributeName="opacity" values="0.3;0.7;0.3" dur="3s" repeatCount="indefinite" begin="0.9s"/>
          </rect>
        </g>

        <!-- Vision feature vectors -->
        <g class="mm-vec" style="animation-delay: 1.0s;">
          <rect x="310" y="56" width="8" height="14" rx="2" fill="#ec4899" opacity="0.9"/>
          <rect x="321" y="60" width="8" height="10" rx="2" fill="#ec4899" opacity="0.7"/>
          <rect x="332" y="52" width="8" height="18" rx="2" fill="#ec4899" opacity="0.8"/>
          <rect x="343" y="58" width="8" height="12" rx="2" fill="#ec4899" opacity="0.6"/>
          <rect x="354" y="54" width="8" height="16" rx="2" fill="#ec4899" opacity="0.85"/>
          <text x="340" y="86" text-anchor="middle" class="mm-label" fill="#ec4899">Image features</text>
        </g>

        <!-- Arrow: Vision vectors -> Fusion -->
        <path d="M 340 95 L 340 140 L 375 160" fill="none" stroke="var(--border-medium)" stroke-width="1.5" class="mm-arrow" marker-end="url(#mm-arrowhead)"/>


        <!-- ========== RIGHT STREAM: Text ========== -->

        <!-- Text tokens input -->
        <g class="mm-slide-r" style="animation-delay: 0.6s;">
          <rect x="530" y="50" width="80" height="65" rx="8" fill="#3b82f615" stroke="#3b82f6" stroke-width="1.5"/>
          <!-- Token boxes -->
          <rect x="540" y="60" width="26" height="16" rx="3" fill="#3b82f630" stroke="#3b82f6" stroke-width="0.8"/>
          <text x="553" y="72" text-anchor="middle" font-family="var(--font-mono)" font-size="7" fill="#3b82f6">The</text>
          <rect x="570" y="60" width="30" height="16" rx="3" fill="#3b82f630" stroke="#3b82f6" stroke-width="0.8"/>
          <text x="585" y="72" text-anchor="middle" font-family="var(--font-mono)" font-size="7" fill="#3b82f6">cat</text>
          <rect x="540" y="80" width="26" height="16" rx="3" fill="#3b82f630" stroke="#3b82f6" stroke-width="0.8"/>
          <text x="553" y="92" text-anchor="middle" font-family="var(--font-mono)" font-size="7" fill="#3b82f6">sat</text>
          <rect x="570" y="80" width="30" height="16" rx="3" fill="#3b82f630" stroke="#3b82f6" stroke-width="0.8"/>
          <text x="585" y="92" text-anchor="middle" font-family="var(--font-mono)" font-size="7" fill="#3b82f6">on</text>
          <text x="570" y="108" text-anchor="middle" class="mm-label" fill="#3b82f6">Text Tokens</text>
        </g>

        <!-- Arrow: Text -> Text Encoder -->
        <line x1="530" y1="82" x2="500" y2="82" stroke="var(--border-medium)" stroke-width="1.5" class="mm-arrow" marker-end="url(#mm-arrowhead)"/>

        <!-- Text Encoder box -->
        <g class="mm-slide-r" style="animation-delay: 1.0s;">
          <rect x="358" y="35" width="140" height="95" rx="10" fill="#3b82f612" stroke="#3b82f6" stroke-width="2"/>
          <text x="428" y="55" text-anchor="middle" class="mm-title" fill="#3b82f6">Text Encoder</text>
          <text x="428" y="70" text-anchor="middle" class="mm-label" fill="#3b82f6">Transformer</text>
          <!-- Internal layers -->
          <rect x="374" y="78" width="108" height="6" rx="2" fill="#3b82f625" stroke="none">
            <animate attributeName="opacity" values="0.3;0.7;0.3" dur="3s" repeatCount="indefinite" begin="0.2s"/>
          </rect>
          <rect x="374" y="88" width="108" height="6" rx="2" fill="#3b82f630" stroke="none">
            <animate attributeName="opacity" values="0.3;0.7;0.3" dur="3s" repeatCount="indefinite" begin="0.5s"/>
          </rect>
          <rect x="374" y="98" width="108" height="6" rx="2" fill="#3b82f635" stroke="none">
            <animate attributeName="opacity" values="0.3;0.7;0.3" dur="3s" repeatCount="indefinite" begin="0.8s"/>
          </rect>
          <rect x="374" y="108" width="108" height="6" rx="2" fill="#3b82f640" stroke="none">
            <animate attributeName="opacity" values="0.3;0.7;0.3" dur="3s" repeatCount="indefinite" begin="1.1s"/>
          </rect>
        </g>

        <!-- Text feature vectors -->
        <g class="mm-vec" style="animation-delay: 1.5s;">
          <rect x="520" y="56" width="8" height="12" rx="2" fill="#3b82f6" opacity="0.8"/>
          <rect x="531" y="52" width="8" height="16" rx="2" fill="#3b82f6" opacity="0.9"/>
          <rect x="542" y="58" width="8" height="10" rx="2" fill="#3b82f6" opacity="0.65"/>
          <rect x="553" y="54" width="8" height="14" rx="2" fill="#3b82f6" opacity="0.85"/>
          <rect x="564" y="60" width="8" height="10" rx="2" fill="#3b82f6" opacity="0.7"/>
          <text x="546" y="86" text-anchor="middle" class="mm-label" fill="#3b82f6">Text features</text>
        </g>

        <!-- Arrow: Text vectors -> Fusion -->
        <path d="M 546 95 L 546 140 L 480 160" fill="none" stroke="var(--border-medium)" stroke-width="1.5" class="mm-arrow" marker-end="url(#mm-arrowhead)"/>


        <!-- ========== CENTER: Fusion Layer ========== -->

        <g class="mm-fade" style="animation-delay: 2.0s;">
          <rect x="255" y="155" width="290" height="90" rx="14" fill="url(#mm-fusion-grad)" stroke="#a855f7" stroke-width="2"/>
          <text x="400" y="185" text-anchor="middle" class="mm-title" fill="#a855f7" font-size="14">Cross-Attention / Fusion</text>
          <text x="400" y="205" text-anchor="middle" class="mm-label">Image tokens attend to text tokens &amp; vice versa</text>

          <!-- Attention lines inside fusion -->
          <g opacity="0.4">
            <line x1="300" y1="220" x2="340" y2="230" stroke="#ec4899" stroke-width="1">
              <animate attributeName="opacity" values="0.2;0.8;0.2" dur="2s" repeatCount="indefinite"/>
            </line>
            <line x1="340" y1="220" x2="400" y2="230" stroke="#a855f7" stroke-width="1">
              <animate attributeName="opacity" values="0.2;0.8;0.2" dur="2s" repeatCount="indefinite" begin="0.3s"/>
            </line>
            <line x1="400" y1="220" x2="460" y2="230" stroke="#3b82f6" stroke-width="1">
              <animate attributeName="opacity" values="0.2;0.8;0.2" dur="2s" repeatCount="indefinite" begin="0.6s"/>
            </line>
            <line x1="460" y1="220" x2="500" y2="230" stroke="#a855f7" stroke-width="1">
              <animate attributeName="opacity" values="0.2;0.8;0.2" dur="2s" repeatCount="indefinite" begin="0.9s"/>
            </line>
          </g>

          <!-- Q K V labels -->
          <rect x="275" y="222" width="28" height="16" rx="3" fill="#ec489940"/>
          <text x="289" y="234" text-anchor="middle" font-family="var(--font-mono)" font-size="8" fill="#ec4899" font-weight="600">Q</text>
          <rect x="308" y="222" width="28" height="16" rx="3" fill="#3b82f640"/>
          <text x="322" y="234" text-anchor="middle" font-family="var(--font-mono)" font-size="8" fill="#3b82f6" font-weight="600">K</text>
          <rect x="341" y="222" width="28" height="16" rx="3" fill="#a855f740"/>
          <text x="355" y="234" text-anchor="middle" font-family="var(--font-mono)" font-size="8" fill="#a855f7" font-weight="600">V</text>
        </g>

        <!-- Arrow: Fusion -> Unified representation -->
        <line x1="400" y1="245" x2="400" y2="280" stroke="var(--border-medium)" stroke-width="1.5" class="mm-arrow" marker-end="url(#mm-arrowhead)"/>

        <!-- Unified representation -->
        <g class="mm-fade" style="animation-delay: 2.6s;">
          <rect x="310" y="283" width="180" height="40" rx="8" fill="#a855f710" stroke="#a855f7" stroke-width="1" stroke-dasharray="4"/>
          <text x="400" y="307" text-anchor="middle" class="mm-label" fill="#a855f7">Unified Multimodal Representation</text>
          <!-- Mixed color vectors -->
          <rect x="330" y="290" width="6" height="10" rx="1" fill="#ec4899" opacity="0.7"/>
          <rect x="338" y="288" width="6" height="12" rx="1" fill="#a855f7" opacity="0.8"/>
          <rect x="346" y="291" width="6" height="9" rx="1" fill="#3b82f6" opacity="0.7"/>
          <rect x="354" y="289" width="6" height="11" rx="1" fill="#ec4899" opacity="0.6"/>
          <rect x="362" y="287" width="6" height="13" rx="1" fill="#a855f7" opacity="0.9"/>
        </g>

        <!-- Arrow: Unified -> Output -->
        <line x1="400" y1="323" x2="400" y2="348" stroke="#00d4aa" stroke-width="2" class="mm-arrow" marker-end="url(#mm-arrowhead-teal)"/>

        <!-- Multimodal Output -->
        <g class="mm-fade" style="animation-delay: 3.0s;">
          <rect x="280" y="350" width="240" height="60" rx="12" fill="#00d4aa15" stroke="#00d4aa" stroke-width="2"/>
          <text x="400" y="375" text-anchor="middle" class="mm-title" fill="#00d4aa" font-size="14">Multimodal Output</text>
          <text x="400" y="395" text-anchor="middle" class="mm-label">Caption, VQA, Generation, Grounding</text>
          <!-- Subtle glow -->
          <rect x="282" y="352" width="236" height="56" rx="11" fill="none" stroke="#00d4aa" stroke-width="1" opacity="0.3">
            <animate attributeName="opacity" values="0.1;0.4;0.1" dur="3s" repeatCount="indefinite"/>
          </rect>
        </g>

        <!-- Side annotations -->
        <g class="mm-fade" style="animation-delay: 3.4s;">
          <!-- Vision side label -->
          <rect x="30" y="155" width="100" height="50" rx="6" fill="#11182790" stroke="var(--border-subtle)" stroke-width="1"/>
          <text x="80" y="175" text-anchor="middle" font-family="var(--font-mono)" font-size="8" fill="#ec4899">Patch embeddings</text>
          <text x="80" y="190" text-anchor="middle" font-family="var(--font-mono)" font-size="8" fill="#ec4899">+ position enc.</text>
          <line x1="130" y1="180" x2="255" y2="190" stroke="#ec4899" stroke-width="0.8" stroke-dasharray="3" opacity="0.4"/>

          <!-- Text side label -->
          <rect x="620" y="155" width="110" height="50" rx="6" fill="#11182790" stroke="var(--border-subtle)" stroke-width="1"/>
          <text x="675" y="175" text-anchor="middle" font-family="var(--font-mono)" font-size="8" fill="#3b82f6">Token embeddings</text>
          <text x="675" y="190" text-anchor="middle" font-family="var(--font-mono)" font-size="8" fill="#3b82f6">+ position enc.</text>
          <line x1="620" y1="180" x2="545" y2="190" stroke="#3b82f6" stroke-width="0.8" stroke-dasharray="3" opacity="0.4"/>
        </g>

        <!-- Title -->
        <g class="mm-fade" style="animation-delay: 0s;">
          <text x="400" y="22" text-anchor="middle" font-family="var(--font-heading)" font-size="14" fill="var(--text-primary)" font-weight="700">Multimodal Architecture</text>
        </g>

      </svg>
      <p style="font-family: var(--font-mono); font-size: var(--text-xs); color: var(--text-dim); margin-top: var(--space-3);">
        Multimodal Fusion \u2014 Vision + Language Cross-Attention Architecture
      </p>
    </div>
  `;
}
