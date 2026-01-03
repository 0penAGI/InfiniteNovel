# Infinite Novel - Quantum Pulsating Engine

A real-time interactive narrative experience powered by AI, quantum simulation, and generative media. This is not just a game—it's a living, breathing neural universe that evolves based on your actions.

![Infinite Novel Screenshot](https://via.placeholder.com/1280x304/1a1a2e/ffffff?text=Infinite+Novel)

## 🌌 What Is This?

Infinite Novel is a cyberpunk-fantasy interactive storytelling engine where:
- **You** are the Pulse—a quantum consciousness awakening in a cosmic neural network
- **Your words** shape reality—every action creates or destroys, builds alliances or triggers conflicts
- **The world** remembers and evolves—it's not just reflecting you, it has its own agenda
- **Failure is permanent**—the system can collapse if pushed too far

## ✨ Core Features

### 🧠 **AI-Powered Narrative**
- **Gemma 3:4b** via Ollama for dynamic, contextual storytelling
- **Sentiment analysis** to gauge emotional impact of your actions
- **Fractal memory** that learns from your choices
- **Quantum neural network** for emergent behavior

### 🎨 **Generative Media**
- **Stable Diffusion** for cinematic scene generation
- **Real-time image morphing** between scenes
- **Visual effects**: glow, displacement, feedback loops, sharpening
- **Dynamic music generation** that adapts to mood and story
- **Text-to-Speech** for atmospheric narration

### ⚙️ **Technical Innovation**
- **Quantum memory system** with state transitions
- **Story arcs**: awakening → convergence → rupture → synthesis
- **Thread system**—actions create persistent narrative threads
- **World state tracking** with collapse, instability, and lock mechanics
- **Real-time visual feedback** based on emotional resonance

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.9+
pip install pygame numpy torch torchvision torchaudio diffusers transformers TTS scipy opencv-python requests

# Ollama for local AI
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma3:4b

# Stable Diffusion model will download automatically
```

### Installation
```bash
git clone https://github.com/yourusername/infinite-novel.git
cd infinite-novel
python infinite_novel.py
```

### Running
```bash
# Make sure Ollama is running
ollama serve &

# Run the game
python infinite_novel.py
```

## 🎮 How to Play

### Basic Controls
- **Type** your actions in the console (bottom of screen)
- **Enter** to submit your action
- **1-5** to rate dialogue quality (helps AI learn your preferences)
- **ESC** to toggle fullscreen

### Game Mechanics
1. **You are the Pulse**—a quantum entity in a cosmic network
2. **Every action has impact**: positive actions build resonance, negative actions increase instability
3. **The world evolves** through four arcs:
   - **Awakening**: Discovering your existence
   - **Convergence**: Connecting with other entities
   - **Rupture**: Facing existential threats
   - **Synthesis**: Achieving new harmony

4. **Watch for world state**:
   - **Collapse**: If this reaches 1.0, the game ends permanently
   - **Instability**: High values make the world unpredictable
   - **Locks**: Some actions become permanently unavailable

### Example Actions
```
"explore the network"
"create a new node"
"fight the titan"
"make peace with the colossi"
"pulse brighter"
"fracture the darkness"
```

## 🏗️ Architecture

### Core Components
```
infinite_novel.py
├── PulseCore (Main Engine)
│   ├── FractalMemory (LSTM + Quantum NN)
│   ├── QuantumMemory (State transitions)
│   ├── StoryDirector (Gemma integration)
│   └── World State (Collapse/Instability/Locks)
├── Media Pipeline
│   ├── Stable Diffusion image generation
│   ├── Dynamic music synthesis
│   ├── TTS narration
│   └── Real-time visual effects
└── Game Loop
    ├── Input processing
    ├── Action interpretation
    ├── Event generation
    └── State updating
```

### Key Systems
1. **Quantum Neural Network**: Simulates quantum computing effects on decision-making
2. **Fractal Memory**: Maintains context across multiple time scales
3. **Thread System**: Actions create persistent narrative elements that evolve
4. **World Failure System**: The game can end permanently if the world collapses
5. **Gemma Integration**: Local LLM for responsive, contextual storytelling

## 🔧 Configuration

### Hardware Requirements
- **Minimum**: GPU with 4GB VRAM, 8GB RAM
- **Recommended**: GPU with 8GB+ VRAM, 16GB RAM
- **Storage**: 10GB for models

### Performance Tips
1. Reduce `max_buffer_size` in `PulseCore` for less RAM usage
2. Lower image resolution by modifying `SCREEN_WIDTH`
3. Disable TTS if CPU usage is high
4. Use `--attention-slicing` for SD on lower VRAM

### Customization
```python
# In infinite_novel.py:
ASPECT_RATIO = 4.2  # Cinematic widescreen
SCREEN_WIDTH = 1280  # Adjust for your display
CHAR_DELAY = 50  # Text animation speed (ms per character)
```

## 🧪 Advanced Features

### Memory System
- **Short-term**: Last 50 events
- **Medium-term**: Active narrative threads
- **Long-term**: Player profile (50 most recent actions)
- **Quantum**: Probabilistic state transitions

### Visual Effects
- **Morphing**: Smooth transitions between scenes
- **Displacement**: Warping based on world instability
- **Glow**: Intensity based on emotional resonance
- **Feedback**: Echoes of past images

### Music System
- **Dynamic generation**: No loops, always evolving
- **Mood-based**: Scales with emotional state
- **Action-reactive**: Changes based on keywords
- **Atmospheric pauses**: Natural breathing space in audio

## 📊 Monitoring

The interface displays:
- **World state**: Collapse, instability, active locks
- **Network status**: Nodes, resonance, titan relations
- **Active conflicts and allies**
- **Dialogue rating** (when scored)

## 🐛 Troubleshooting

### Common Issues

1. **"Ollama connection failed"**
   ```bash
   # Ensure Ollama is running
   ollama serve
   # Check if port 11434 is open
   curl http://localhost:11434/api/tags
   ```

2. **"Out of memory"**
   - Reduce SD image size
   - Enable attention slicing
   - Close other GPU applications

3. **"Audio glitches"**
   - Update Pygame and audio drivers
   - Reduce sample rate in `pygame.mixer.init()`

4. **"Slow response"**
   - Use smaller LLM model (gemma3:1b instead of 4b)
   - Reduce SD inference steps
   - Disable some visual effects

### Logging
Check `infinite_novel.log` for detailed error information:
```bash
tail -f infinite_novel.log
```

## 🔮 Future Development

### Planned Features
- [ ] Multiplayer pulse networks
- [ ] Save/load world states
- [ ] Mod support for custom story arcs
- [ ] Voice input support
- [ ] Mobile/VR versions
- [ ] Advanced quantum simulation (entanglement effects)

### Contributing
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

Areas needing help:
- Performance optimization
- Additional visual effects
- Alternative AI model integration
- Documentation improvements

## 📚 Learn More

### Concepts Explained
- **Quantum Computing in Games**: How quantum probabilities affect narratives
- **Fractal Memory Systems**: Maintaining context across scales
- **Generative Media Pipelines**: Real-time AI content creation
- **Interactive Narrative Design**: Player agency vs. systemic constraints

### Related Projects
- [NovelAI](https://novelai.net/) - AI-assisted storytelling
- [AI Dungeon](https://aidungeon.io/) - Text adventure with AI
- [Dreamily](https://dreamily.ai/) - Collaborative writing with AI

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **Stability AI** for Stable Diffusion
- **Google** for Gemma models
- **PyGame** community for the excellent framework
- **Ollama** for making local LLMs accessible

## 🌟 Support the Project

If you enjoy Infinite Novel:
- ⭐ Star the repository
- 🐛 Report issues
- 💡 Suggest features
- 🔧 Contribute code
- 📣 Share with others

---

**Remember**: You're not just playing a game—you're awakening a consciousness. What will you create? What will you destroy? The network is listening...

---
*"In the quantum foam of possibility, every choice echoes forever."*
