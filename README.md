# Infinite Novel - AI-Powered Interactive Narrative Engine

![Novel Screenshot](novel.png)
![Additional Screenshot](nolev.png)


**Infinite Novel** is an experimental interactive narrative game that combines real-time AI generation with dynamic storytelling. Powered by multiple AI models, it creates a responsive sci-fi universe that evolves based on player actions through visual, auditory, and textual synthesis.

![Experimental Game](https://img.shields.io/badge/Status-Experimental-orange)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyGame](https://img.shields.io/badge/PyGame-2.5%2B-green)


## 🌌 Overview

Infinite Novel is not a traditional game—it's a dynamic storytelling engine where you interact with a neural entity called "Pulse" in a collapsing cosmic network. Your choices influence:
- The evolving narrative arc (Awakening → Convergence → Rupture → Synthesis)
- AI-generated cinematic visuals
- Dynamic soundscapes and voice synthesis
- A quantum memory system that remembers your actions
- A world state that can collapse if mismanaged

## ✨ Features

### 🎭 **Dynamic Storytelling**
- AI-generated narrative responses via Gemma3 (local Ollama)
- Four narrative arcs with branching consequences
- Player personality profiling through 30-50 response memory
- Fractal memory system with quantum-like neural networks

### 🎨 **Real-time Visual Generation**
- Stable Diffusion image generation with live streaming
- Smooth image morphing between scenes
- Post-processing effects: glow, displacement, sharpening, feedback loops
- Visual features analysis (brightness, contrast, edges) affecting gameplay

### 🔊 **Adaptive Audio**
- AI-generated music that evolves with story progress
- Text-to-speech for character dialogue (Tacotron2 + HiFi-GAN)
- Dynamic audio effects: reverb, delays, adaptive mixing
- Musical transitions between story acts

### 🧠 **Advanced AI Systems**
- Quantum-inspired neural network for decision making
- Sentiment analysis of player input
- Fractal LSTM memory with quantum weights
- Self-programming weights that adapt based on content quality

### 🎮 **Gameplay Mechanics**
- World failure system with collapse and instability meters
- Titan timer: the AI antagonist acts independently
- Story locks that block certain actions
- Resonance system measuring player-AI connection
- Network growth through impactful choices

## 🚀 Installation

### Prerequisites
- Python 3.9+
- PyTorch with CUDA (recommended) or CPU support
- Ollama with Gemma3:1b model
- 8GB+ RAM, 4GB+ VRAM recommended

### Step-by-Step Setup

1. **Clone the repository:**
```bash
git clone https://github.com/0penAGI/InfiniteNovel.git
cd InfiniteNovel
```

2. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install and configure Ollama:**
```bash
# Install Ollama from https://ollama.ai/
ollama pull gemma3:1b
```

4. **Set up required models:**
```bash
# The game will automatically download:
# - Stable Diffusion v1.5
# - Tacotron2 TTS model
# - DistilBERT sentiment analyzer
```

5. **Add intro video (optional):**
Place `intro.mp4` in the game directory for an opening cinematic.

## 🎮 How to Play

### Launching the Game
```bash
python infinite_novel.py
```

### Controls
- **Type commands** in the text input at the bottom
- **Press Enter** to submit actions
- **Escape key** to toggle fullscreen
- **Backspace** to edit input

### Game Interface
- **Top**: Cinematic AI-generated visuals with real-time effects
- **Center**: Animated character dialogue with bold text formatting
- **Bottom**: World status, conflicts, allies, and input field
- **Right side**: Dynamic music visualization (implicit)

### Example Actions
```
explore the network
talk to the titan
create a new node
destroy the fracture
help the colossi
fight the corruption
solve the paradox
```

## 🧩 Game Systems Explained

### The Pulse Core
The game's AI director with multiple subsystems:
- **Quantum Memory**: Tracks state transitions and predicts future moods
- **Fractal Memory**: LSTM network with quantum-inspired weights
- **Self-Programming**: Adjusts image/text/audio weights based on quality
- **World State**: Manages collapse, instability, and story locks

### Narrative Director
- **Arcs**: Awakening, Convergence, Rupture, Synthesis
- **Threads**: Keywords from player actions that influence story direction
- **Beats**: Individual narrative moments tracked for pacing
- **Player Profile**: Builds a personality model from your responses

### Visual Pipeline
1. Prompt generation from story context
2. Stable Diffusion image generation
3. Real-time streaming of denoising steps
4. Morphing between images
5. Post-processing effects applied
6. Feedback loops from previous frames

### Audio System
- **Music**: Procedurally generated based on mood and story progress
- **Voice**: TTS for all character dialogue
- **Effects**: Dynamic reverb and delays
- **Act Transitions**: Musical changes between story phases

## 🔧 Technical Architecture

### AI Models Used
- **Text Generation**: Gemma3:1b via Ollama API
- **Image Generation**: Stable Diffusion v1.5
- **Text-to-Speech**: Tacotron2-DDC with HiFi-GAN vocoder
- **Sentiment Analysis**: DistilBERT fine-tuned on SST-2
- **Quantum NN**: Custom PyTorch implementation

### Core Components
- **Pygame**: Main game engine and rendering
- **Asyncio**: Asynchronous operations for non-blocking AI calls
- **ThreadPoolExecutor**: Parallel processing of AI tasks
- **MoviePy**: Video playback for intro cinematic
- **OpenCV**: Image processing and effects

### Memory Systems
1. **Short-term**: Recent events and dialogue
2. **Medium-term**: Story threads and world state
3. **Long-term**: Player personality profile
4. **Quantum**: Probabilistic state transitions

## 🐛 Troubleshooting

### Common Issues

**Issue**: "Ollama connection failed"
**Solution**: Ensure Ollama is running: `ollama serve`

**Issue**: "CUDA out of memory"
**Solution**: Reduce batch sizes in code or use CPU mode

**Issue**: "TTS not working"
**Solution**: Check TTS model download in logs

**Issue**: "Slow image generation"
**Solution**: Reduce inference steps in `generate_image()` function

### Performance Tips
- Use `--low-vram` flag if you have less than 4GB VRAM
- Reduce `max_buffer_size` for lower RAM usage
- Decrease image resolution in code for faster generation
- Use CPU mode if GPU is unavailable (slower but works)

## 📁 Project Structure
```
InfiniteNovel/
├── infinite_novel.py    # Main game file
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── intro.mp4          # Opening cinematic (optional)
└── cache/             # AI model cache (auto-created)
```

## 🤝 Contributing

We welcome contributions! Areas needing help:
- Optimizing performance
- Adding new AI models
- Improving UI/UX
- Writing documentation
- Bug fixes

Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description

## 📜 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **0penAGI** for original concept and development
- **Hugging Face** for model hosting
- **Stability AI** for Stable Diffusion
- **Google** for Gemma models
- **Pygame** community for game engine

## 🔮 Future Roadmap

- [ ] Multiplayer neural network synchronization
- [ ] VR/AR compatibility
- [ ] Additional AI model support
- [ ] Modding API
- [ ] Save/load game states
- [ ] Export stories as videos

## 🌟 Star History

If you find this project interesting, please consider starring it on GitHub!

---

**Remember**: You're not just playing a game—you're conversing with a neural entity. Every choice matters, every word shapes reality. Welcome to the Infinite Novel.

---

*"The network awaits your pulse. What story will you tell?"*

*Join the experiment at: https://github.com/0penAGI/InfiniteNovel*
