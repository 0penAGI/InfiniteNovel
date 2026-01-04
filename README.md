# Infinite Novel - AI-Powered Interactive Narrative Engine

![Novel Screenshot](novel.png)
![Additional Screenshot](nolev.png)


**Infinite Novel** is an experimental interactive narrative game that combines real-time AI generation with dynamic storytelling. Powered by multiple AI models, it creates a responsive sci-fi universe that evolves based on player actions through visual, auditory, and textual synthesis.

![Experimental Game](https://img.shields.io/badge/Status-Experimental-orange)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyGame](https://img.shields.io/badge/PyGame-2.5%2B-green)

## 🌟 Features

### **Real-Time AI Generation**
- **Visual Synthesis**: Stable Diffusion-powered cinematic scene generation
- **Narrative Generation**: Gemma3 (via Ollama) for dynamic story responses
- **Voice Synthesis**: TTS with HiFi-GAN vocoder for spoken dialogue
- **Music Generation**: Procedural audio that adapts to narrative mood

### **Dynamic Story Systems**
- **Fractal Memory**: LSTM networks with quantum-inspired layers for narrative continuity
- **Quantum Memory**: State transitions and prediction based on player actions
- **Story Director**: Multi-arc narrative system with player profiling
- **World State Simulation**: Collapse mechanics, instability, and narrative locks

### **Immersive Visual Effects**
- Real-time image morphing and displacement
- Glow and sharpness post-processing
- Feedback loops and visual memory buffers
- Streaming image generation with micro-shaders

### **Interactive Systems**
- Player action interpretation with sentiment analysis
- Thread-based narrative tension system
- Conflict resolution and ally management
- Dynamic dialogue with bold text formatting

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+**
- **Ollama** (for Gemma3 text generation)
- **CUDA/MPS** compatible GPU (recommended)
- **18GB+ RAM** (tested on MacBook M3 Pro 18GB)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/0penAGI/InfiniteNovel.git
cd InfiniteNovel
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up Ollama:**
```bash
# Install Ollama from https://ollama.ai/
ollama pull gemma3:1b  # or gemma3:4b
```

### Running the Game
```bash
python infinite_novel.py
```

## 🎮 Gameplay

### Controls
- **Type** commands and press **Enter** to interact
- **Escape** to toggle fullscreen
- **Backspace** to edit input

### Narrative Mechanics
1. **Start**: You awaken in a cosmic void as "The Pulse"
2. **Interact**: Type actions, questions, or commands
3. **Influence**: Your choices affect world state, alliances, and conflicts
4. **Progress**: Unlock narrative arcs (Awakening → Convergence → Rupture → Synthesis)

### World States
- **Collapse**: Measures world stability (≥1.0 = game over)
- **Instability**: Chaos level affecting generation
- **Locks**: Narrative paths that become unavailable
- **Titan Timer**: Megatitan interventions

## 🏗️ Architecture

### Core Systems

```
PulseCore
├── FractalMemory (LSTM + Quantum NN)
├── QuantumMemory (State transitions)
├── StoryDirector (Narrative arcs)
└── WorldState (Collapse/Instability)
```

### AI Integration
| Component | Model | Purpose |
|-----------|-------|---------|
| **Text** | Gemma3 via Ollama | Narrative responses |
| **Images** | Stable Diffusion v1.5 | Scene generation |
| **Audio** | TTS (Tacotron2-DDC) | Voice synthesis |
| **Music** | Procedural generation | Adaptive soundtrack |
| **Sentiment** | DistilBERT | Action interpretation |

### Rendering Pipeline
1. **Image Generation** → 2. **Morphing** → 3. **Displacement** → 4. **Post-processing** → 5. **Display**

## ⚙️ Configuration

### Key Parameters
```python
# Display
ASPECT_RATIO = 4.2
SCREEN_WIDTH = 1280

# Generation
MAX_BUFFER_SIZE = 5
CHAR_DELAY = 50  # ms per character

# World
TITAN_TIMER = 120
COLLAPSE_THRESHOLD = 1.0
```

### Device Support
- **CUDA**: NVIDIA GPUs (recommended)
- **MPS**: Apple Silicon (tested on M3 Pro)
- **CPU**: Fallback (slower)

## 📁 Project Structure

```
InfiniteNovel/
├── infinite_novel.py      # Main game engine
├── requirements.txt       # Dependencies
├── README.md             # This file
└── assets/               # (Optional) Static assets
```

## 🔧 Advanced Usage

### Custom Models
Edit the model initialization sections:
```python
# For different SD model:
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",  # Change model
    torch_dtype=torch.float32
)

# For different TTS:
tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False)
```

### Narrative Tuning
Adjust `StoryDirector` parameters:
```python
class StoryDirector:
    def __init__(self):
        self.arc = "awakening"  # Starting arc
        self.player_profile_size = 50  # Memory length
```

### Performance Optimization
- Reduce `MAX_BUFFER_SIZE` for lower memory usage
- Decrease `num_inference_steps` for faster image generation
- Adjust `executor.max_workers` based on CPU cores

## 🐛 Known Issues

### Current Limitations
1. **Memory Intensive**: Requires 18GB+ RAM for optimal performance
2. **Slow Initial Load**: Models download on first run (~10GB)
3. **Ollama Dependency**: Must run Ollama server locally
4. **MPS Performance**: Apple Silicon support is experimental

### Common Fixes
- **Black screen**: Check Ollama server status
- **Slow generation**: Reduce image resolution in code
- **Audio glitches**: Adjust `pygame.mixer.init` parameters

## 🔮 Future Development

### Planned Features
- [ ] Save/Load game states
- [ ] Mod support for custom narratives
- [ ] Multiplayer synchronization
- [ ] Export generated stories
- [ ] Web deployment option

### Research Directions
- Reinforcement learning for narrative optimization
- Cross-modal consistency models
- Real-time style transfer
- Emotional resonance mapping

## 🤝 Contributing

This is an experimental project by **0penAGI**. Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description

### Development Guidelines
- Use type hints where possible
- Add logging for new features
- Test on multiple devices if possible
- Document novel approaches

## 📄 License

This project is shared for research and educational purposes. See LICENSE for details.

## 🙏 Acknowledgments

- **0penAGI** team for the original implementation
- **Stability AI** for Stable Diffusion
- **Google** for Gemma models
- **PyGame** community for multimedia framework
- **Ollama** for local LLM serving

## 📚 Citation

If you use this code in research, please cite:

```bibtex
@software{infinite_novel_2024,
  title = {Infinite Novel: AI-Powered Interactive Narrative Engine},
  author = {0penAGI},
  year = {2024},
  url = {https://github.com/0penAGI/InfiniteNovel}
}
```

## 📞 Support

- **Issues**: GitHub Issues tracker
- **Email**: Check repository description

---

**Note**: This is experimental software. Generated content may be unpredictable. Use responsibly and monitor resource usage.

*"The network awaits your pulse. What story will you tell?"*

*Join the experiment at: https://github.com/0penAGI/InfiniteNovel*
