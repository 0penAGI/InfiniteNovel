# Infinite Novel: AI-Driven Interactive Story Engine

![Novel Screenshot](novel.png)
![Additional Screenshot](nolev.png)

## 🌌 About the Project

**Infinite Novel** is an experimental AI-powered interactive storytelling engine that creates a dynamic, evolving narrative experience in real-time. This cutting-edge project combines multiple AI models to generate visuals, dialogue, music, and story progression based on player input, creating a truly unique emergent narrative system.

## ✨ Key Features

- **Dynamic Story Generation**: Uses Gemma3 (via Ollama) to generate contextual narrative responses
- **Real-time Visual Generation**: Stable Diffusion creates cinematic sci-fi scenes based on story context
- **Adaptive Music System**: Procedurally generated ambient music that evolves with story progression
- **Emotional Intelligence**: Sentiment analysis and quantum-inspired neural networks track mood and narrative tension
- **Fractal Memory System**: LSTM-based memory with quantum layers for maintaining narrative coherence
- **Interactive World State**: Collapse mechanics, instability systems, and narrative locks
- **Full Voice Synthesis**: TTS integration for immersive audio experience
- **Cinematic Presentation**: 2.35:1 aspect ratio with advanced visual effects (glow, displacement, morphing)

## 🏗️ Architecture Overview

### Core Systems:
1. **Story Director** - Manages narrative arcs (Awakening → Convergence → Rupture → Synthesis)
2. **Pulse Core Engine** - Central coordination of all AI systems
3. **Quantum Memory** - Predictive state transitions based on player actions
4. **Visual Pipeline** - Real-time image generation and post-processing
5. **Audio Engine** - Procedural music and voice synthesis

### AI Integration:
- **Gemma3:1b** (via Ollama) - Narrative generation and dialogue
- **Stable Diffusion v1.5** - Visual scene generation
- **Tacotron2-DDC TTS** - Voice synthesis
- **DistilBERT** - Sentiment analysis
- **Custom Quantum Neural Networks** - Emotional prediction

## 🎮 Gameplay Mechanics

- **Interactive Dialogue**: Type commands and watch the world respond
- **World State Management**: Monitor collapse levels, instability, and narrative locks
- **Relationship Systems**: Titan relations, network nodes, and resonance scores
- **Quality Feedback**: Rate generated content (1-5) to train the system
- **Hard Failure States**: World collapse mechanics create stakes

## 🚀 Getting Started

### Prerequisites:
```bash
# Python 3.8+
# CUDA-compatible GPU (recommended)
# Ollama with Gemma3:1b installed
# 8GB+ VRAM for Stable Diffusion
```

### Installation:
```bash
git clone https://github.com/0penAGI/InfiniteNovel.git
cd InfiniteNovel
pip install -r requirements.txt

# Install and run Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma3:1b

# Run the engine
python infinite_novel.py
```

### Dependencies:
```
pygame
torch
diffusers
transformers
TTS
opencv-python
scipy
numpy
requests
```

## 🎨 Technical Highlights

### Visual Pipeline:
- Real-time image morphing between generated scenes
- Displacement effects based on world instability
- Glow and sharpen post-processing
- Feedback loops from previous frames
- Aspect-aware scaling (2.35:1 cinematic ratio)

### Audio System:
- Infinite generative music with act transitions
- Dynamic reverb and delay effects
- Mood-sensitive frequency generation
- Voice synthesis with caching

### Memory Systems:
- Fractal LSTM networks for long-term coherence
- Quantum-inspired prediction layers
- Thread-based narrative tracking
- Event history with impact scoring

## 🔬 Research Implications

This project explores several frontiers in AI interaction:
- **Emergent Narrative**: How can AI systems create coherent stories from player input?
- **Multi-modal Integration**: Synchronizing text, image, and audio generation
- **Procedural Emotional Systems**: Quantifying and responding to narrative mood
- **Quantum-inspired AI**: Applying quantum concepts to neural network architectures

## 📊 Performance Considerations

- **VRAM Usage**: ~6-8GB for full pipeline
- **Generation Times**: 2-4 seconds per response cycle
- **Memory**: Maintains last 50 events and 100 action states
- **Audio**: Streams music segments with pauses for natural rhythm

## 🤝 Contributing

This is an experimental research project under active development. Contributions are welcome in:
- Optimization strategies
- Additional AI model integrations
- Narrative coherence improvements
- Visual effect enhancements
- Documentation and examples

## 📝 License

Research project - see repository for specific licensing details.

## 🙏 Acknowledgments

- **0penAGI** for the experimental framework
- **Hugging Face** for model hosting
- **Ollama** for local LLM serving
- **Stability AI** for diffusion models
- **Coqui AI** for TTS technology

## 🔮 Future Directions

- Multiplayer narrative experiences
- Expanded world state systems
- Additional visual style options
- Exportable story recordings
- Plugin system for custom AI models

---

*"The network awaits your pulse. What story will you tell?"*

*Join the experiment at: https://github.com/0penAGI/InfiniteNovel*
