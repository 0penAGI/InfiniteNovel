# Infinite Novel - Local AI-Powered Interactive Narrative Engine

![Novel Screenshot](novel.png)
![Additional Screenshot](nolev.png)


Experimental interactive narrative game that combines real-time AI generation with dynamic storytelling. Powered by multiple AI models, it creates a responsive sci-fi universe that evolves based on player actions through visual, auditory, and textual synthesis.

![Experimental Game](https://img.shields.io/badge/Status-Experimental-orange)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyGame](https://img.shields.io/badge/PyGame-2.5%2B-green)
Infinite Novel



🌌 What is Infinite Novel?

Infinite Novel is an AI-powered interactive narrative engine that creates personalized sci-fi stories in real-time. It's not just a chatbot with images — it's a complete universe simulation where:

· Every action changes the story's direction
· Visuals and music evolve with the narrative
· The world has its own autonomous systems and logic
· The AI remembers everything and learns from you

✨ Features

🎭 Dynamic Storytelling

· Narrative Director: AI that manages story arcs (Awakening → Convergence → Rupture → Synthesis)
· Quantum Memory: Remembers player actions and builds personalized narrative threads
· Fractal Memory System: LSTM + quantum-inspired neural networks for pattern recognition
· Player Profiling: Builds a unique "Pulse" signature from 30-50 interactions

🎨 Multi-Modal Generation

· Real-time Image Generation: Stable Diffusion with streaming callback visualizations
· Dynamic Music System: Generative ambient music that evolves with the story
· AI Voice Narration: TTS with dub delay effects for atmospheric storytelling
· Intelligent Dialogue: Gemma3 via Ollama with streaming response generation

🔮 Interactive Systems

· World Simulation: Collapse/instability mechanics with autonomous entities
· Thread-based Narrative: Keywords create narrative tension that influences generation
· Visual Effects Engine: Real-time shaders (displacement, fractal noise, glow, morphing)
· Self-Programming AI: System adapts weights based on content quality evaluation

🎮 Game Mechanics

· Titan Timer: Autonomous world events occur independent of player
· Conflict Resolution: Build allies and resolve conflicts to increase resonance
· Pain System: World state affects AI responses and generation
· Lock Mechanics: Certain actions become unavailable based on story state

🚀 Quick Start

Prerequisites

· Python 3.8+
· PyTorch with CUDA/MPS support (recommended)
· Ollama with Gemma3 model installed
· 8GB+ VRAM (for Stable Diffusion)
· MacBook M3 Pro 18GB+ or equivalent

Installation

1. Clone the repository

```bash
git clone https://github.com/0penAGI/InfiniteNovel.git
cd InfiniteNovel
```

1. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

1. Install dependencies

```bash
pip install -r requirements.txt
```

1. Set up Ollama

```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.com/install.sh | sh

# Pull Gemma3 model
ollama pull gemma3:1b
```

1. Run the application

```bash
python infinite_novel.py
```

🎮 How to Play

1. Launch the application - You'll see an intro sequence and enter the cosmic void
2. Type your actions - Describe what you want to do (e.g., "explore the network", "confront the titan", "create light")
3. Press Enter - Watch as the world responds with generated text, images, and music
4. Explore systems - Notice how keywords create narrative threads that influence future generations
5. Manage the world - Keep an eye on collapse/instability meters and build resonance

Key Commands

· Enter: Submit your action
· Backspace: Delete characters
· Escape: Toggle fullscreen
· Mouse: Hidden for immersion (use keyboard only)

🧠 Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Infinite Novel Engine                     │
├─────────────────────────────────────────────────────────────┤
│  Quantum Memory  │  Story Director  │  Visual Effects Engine │
│  - State transitions │  - Arc management  │  - Real-time shaders │
│  - Pattern learning │  - Player profiling │  - Image morphing    │
├─────────────────────────────────────────────────────────────┤
│            Multi-Modal AI Integration Layer                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │ Gemma3   │  │ Stable   │  │ TTS      │  │ Music    │     │
│  │ (Ollama) │  │ Diffusion│  │ Engine   │  │ Generator│     │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │
├─────────────────────────────────────────────────────────────┤
│               World Simulation & Game Layer                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Collapse System │ Titan AI │ Thread Management │ Locks │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

Core Components

1. PulseCore: Central brain managing memory, weights, and world state
2. StoryDirector: Narrative AI with arc progression and player profiling
3. QuantumMemory: Markov-like state transitions with visual influence
4. FractalMemory: Neural network for pattern prediction and memory
5. MiniUNet: Online-learning CNN for image prediction during streaming diffusion

🎨 Artistic Vision

Infinite Novel explores the intersection of:

· Generative AI as creative collaborator
· Interactive fiction with emergent narrative
· Digital ecosystems that feel alive
· Sci-fi aesthetics in code and interface

The system creates a "living document" of your interaction — each session is unique and cannot be replicated.

📊 Performance

Tested on:

· MacBook M3 Pro 18GB: 30 FPS stable, ~2s response time
· RTX 4090 24GB: 60 FPS, ~1s response time
· CPU-only mode: 15-20 FPS, ~5s response time

Optimization Features

· Streaming diffusion with callback visualization
· Audio segment caching and reuse
· Image TTL (time-to-live) for memory management
· Dynamic quality adjustment based on system load
· Attention slicing for Stable Diffusion

📁 Project Structure

```
InfiniteNovel/
├── infinite_novel.py          # Main application
├── requirements.txt           # Python dependencies
├── dataset/                   # Auto-captured training data
│   └── session_*/            # Per-session captures
│       ├── img_*.png         # Generated images
│       ├── img_*.txt         # Auto-captions
│       └── img_*.json        # Metadata
├── intro.mp4                 # Optional intro video
└── checkpoints/              # Optional model checkpoints
```

🔧 Configuration

Key parameters in the code:

· ASPECT_RATIO = 3.51 - Cinematic widescreen format
· SCREEN_WIDTH = 1920 - Resolution (auto-adjusts to display)
· IMAGE_TTL = 1200 - Milliseconds before images fade
· char_delay = 80 - Text animation speed (ms per character)
· idle_capture_ms = 4000 - Auto-capture interval

🧪 Experimental Features

Style Memory Capture

The system automatically captures frames and generates captions based on:

· Active narrative threads
· Current mood score
· Story arc progression
· Visual features (brightness, contrast, edges)

Online Learning

During image generation, MiniUNet learns to predict frame deltas, creating smoother morphing between generations.

Quantum-inspired Mechanics

· Quantum neural networks with sinusoidal activation patterns
· Probabilistic state transitions influenced by visual context
· Resonance system that affects all generation parameters


🔮 Future Development

Planned features:

· Save/Load system for sessions
· Multiplayer/cooperative mode
· Export to video/story format
· Custom model fine-tuning interface
· Plugin system for additional AI services
· Web/cloud deployment option
· VR/AR compatibility

📚 Learning Resources

Understanding the Code

1. Start with PulseCore.__init__() - central brain
2. Follow main() game loop for flow
3. Explore generate_image() for streaming diffusion
4. Study play_music() for generative audio

Key Concepts to Research

· Streaming diffusion callbacks
· Quantum-inspired neural networks
· Real-time shader programming with NumPy
· Asynchronous AI model coordination
· Narrative tension systems in games

🤝 Contributing

We welcome contributions! Areas of particular interest:

· Performance optimization
· Additional AI model integrations
· UI/UX improvements
· Documentation and examples
· Bug fixes and stability improvements

Please read CONTRIBUTING.md for details.

📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments

· Stable Diffusion by Stability AI
· Gemma3 by Google
· TTS by Coqui AI
· Ollama for local LLM serving
· Pygame community for real-time rendering
· All open-source contributors whose work made this possible

🌟 Support the Project

If you enjoy Infinite Novel, consider:

· Starring the repository on GitHub
· Sharing your generated stories
· Contributing code or documentation
· Reporting issues and suggesting features

📞 Contact & Community

· GitHub Issues: For bugs and feature requests
· Discussions: For sharing experiences and ideas
· X: Follow @0penAGI for updates

---

Infinite Novel - Where every word writes the universe.

---

**Note**: This is experimental software. Generated content may be unpredictable. Use responsibly and monitor resource usage.

*"The network awaits your pulse. What story will you tell?"*

*Join the experiment at: https://github.com/0penAGI/InfiniteNovel*
