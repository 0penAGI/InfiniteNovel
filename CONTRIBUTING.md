# Contributing to Infinite Novel

Welcome to the Infinite Novel community! This document provides guidelines for contributing to this experimental AI narrative engine.

---

## 🌟 Philosophy

Infinite Novel is an **art project first, software second**. We value:

- 🎨 **Aesthetic over efficiency** — Lo-fi glitches are features
- 🧪 **Experimentation over stability** — Break things, learn, iterate
- 🌍 **Accessibility over performance** — Run on 8GB VRAM, not just 4090s
- 🤝 **Community over perfection** — Your weird idea might be brilliant

---

## 🚀 Ways to Contribute

### 1. 🐛 Bug Reports

**Before submitting:**
- [ ] Check [existing issues](https://github.com/0penAGI/InfiniteNovel/issues)
- [ ] Test on latest `main` branch
- [ ] Include system specs (GPU, RAM, OS)

**Good bug report template:**
```markdown
## Bug Description
Clear description of what went wrong

## Steps to Reproduce
1. Launch application
2. Type action: "explore the void"
3. Press Enter
4. [Crash/unexpected behavior]

## Expected Behavior
What should have happened

## System Info
- OS: macOS 14.2
- GPU: M3 Pro 18GB
- Python: 3.11.5
- CUDA/MPS: MPS
- SD Speed: 1.52 it/s

## Logs
```
[Paste relevant error messages]
```

## Screenshots/Video
[If applicable]
```

**Priority labels:**
- `critical` — Crashes, data loss, unplayable
- `bug` — Something broken but workaround exists
- `glitch-feature` — Is this a bug or aesthetic? (discuss!)

---

### 2. 💡 Feature Requests

**Before requesting:**
- [ ] Search [discussions](https://github.com/0penAGI/InfiniteNovel/discussions)
- [ ] Consider if it fits the lo-fi aesthetic
- [ ] Think about 8GB VRAM users

**Good feature request template:**
```markdown
## Feature Description
What you want to add/change

## Use Case
Why this improves the experience

## Aesthetic Impact
Does this maintain lo-fi vibe or conflict with it?

## Performance Impact
How does this affect 8GB VRAM systems?

## Implementation Ideas
[Optional] Technical approach

## Alternatives Considered
Other ways to achieve this
```

**Feature categories:**
- `enhancement` — Improve existing systems
- `new-model` — Add AI model support
- `narrative` — Story mechanics, arcs, threading
- `visual` — Shaders, effects, UI
- `audio` — Music generation, TTS improvements
- `performance` — Optimization without aesthetic loss

---

### 3. 🎨 Visual/Audio Contributions

**Shader contributions:**
```python
def your_shader_name(surface, intensity=1.0):
    """
    Brief description of visual effect.
    
    Args:
        surface: pygame.Surface to process
        intensity: 0.0-2.0, effect strength
    
    Returns:
        pygame.Surface with effect applied
    """
    # Your implementation
    # Must be real-time (< 16ms on M3 Pro)
    return processed_surface
```

**Music algorithm contributions:**
```python
def generate_your_sound_layer(
    duration=3.0,
    sample_rate=44100,
    mood_score=0.0,
    context_energy=1.0
):
    """
    Generate audio layer for procedural music system.
    
    Returns:
        np.ndarray (float32, mono)
    """
    # Your implementation
    # Must avoid clicks (fade in/out)
    return audio_signal
```

**Requirements:**
- Real-time performance (30 FPS on reference hardware)
- Fits lo-fi aesthetic (embrace imperfection)
- Documented with inline comments
- Include example usage

---

### 4. 📚 Documentation

**We need:**
- Code comments (especially complex algorithms)
- Tutorials (YouTube videos welcome!)
- Architecture explanations
- Narrative design guides
- Shader/audio tutorials

**Documentation style:**
```python
def complex_function(x, y, context):
    """
    One-line summary of what this does.
    
    Detailed explanation of algorithm, especially if non-obvious.
    Include references to papers/techniques if applicable.
    
    Args:
        x: Description and expected range
        y: Description and expected range
        context: Description of context object
    
    Returns:
        Description of return value
        
    Example:
        >>> result = complex_function(0.5, 0.8, core)
        >>> print(result)
        0.73
    """
    pass
```

---

### 5. 🧪 Experimental Features

**High-risk, high-reward contributions:**

**Examples:**
- New AI model integrations (Flux, SDXL Turbo, LLaVA)
- Alternative narrative systems (branching, time loops)
- Multiplayer/co-op experiments
- VR/AR prototypes
- Blockchain integration (controversial, but interesting)

**Guidelines:**
- Create feature branch: `experimental/feature-name`
- Add `[EXPERIMENTAL]` prefix to PR title
- Document known issues prominently
- Include toggle to disable (don't break main)
- Expect heavy feedback and iteration

---

## 🔧 Development Setup

### 1. Fork & Clone

```bash
# Fork on GitHub, then:
git clone https://github.com/YOUR_USERNAME/InfiniteNovel.git
cd InfiniteNovel
git remote add upstream https://github.com/0penAGI/InfiniteNovel.git
```

### 2. Create Branch

```bash
# Feature branch
git checkout -b feature/your-feature-name

# Bug fix branch
git checkout -b fix/issue-123-description

# Experimental branch
git checkout -b experimental/wild-idea
```

### 3. Setup Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If exists
```

### 4. Test Changes

```bash
# Run the game
python infinite_novel.py

# Test specific features
python -m pytest tests/  # If test suite exists

# Check code style (not enforced, but appreciated)
black infinite_novel.py --check
flake8 infinite_novel.py --ignore=E501,W503
```

---

## 📝 Code Style

### General Guidelines

**We're flexible, but prefer:**

```python
# ✅ Good: Clear, commented, functional
def generate_layer(duration, mood):
    """Generate audio layer based on mood."""
    # Calculate base frequency from mood
    base_freq = 130 + mood * 50
    
    # Generate harmonic series
    signal = np.zeros(int(duration * 44100))
    for h in [1.0, 1.5, 2.0]:
        signal += np.sin(2 * np.pi * base_freq * h * t)
    
    return signal

# ❌ Bad: Unclear, no context
def gl(d, m):
    bf = 130 + m * 50
    s = np.zeros(int(d * 44100))
    for h in [1.0, 1.5, 2.0]:
        s += np.sin(2 * np.pi * bf * h * t)
    return s
```

**Naming conventions:**
- `snake_case` for functions and variables
- `PascalCase` for classes
- `UPPER_CASE` for constants
- Descriptive names over abbreviations (except standard ones: `i`, `j`, `x`, `y`, `t`)

**Comments:**
- Explain **why**, not **what**
- Complex algorithms need paragraph explanations
- Artistic decisions should be documented

```python
# ✅ Good comment
# Use cubic easing for smooth morphing (linear feels robotic)
alpha = t * t * (3 - 2 * t)

# ❌ Bad comment
# Calculate alpha
alpha = t * t * (3 - 2 * t)
```

---

## 🧪 Performance Guidelines

### Critical: Maintain 30 FPS on M3 Pro 18GB

**Benchmarking template:**
```python
import time

def test_performance():
    """Benchmark your feature."""
    core = PulseCore()
    surface = generate_test_surface()
    
    iterations = 100
    start = time.time()
    
    for _ in range(iterations):
        result = your_function(surface, core)
    
    elapsed = time.time() - start
    fps_cost = (elapsed / iterations) * 30  # Impact on 30 FPS
    
    print(f"Average time: {elapsed/iterations*1000:.2f}ms")
    print(f"FPS cost: {fps_cost:.1f} frames")
    
    assert fps_cost < 3, "Function too slow! (>3 frame cost)"
```

**Optimization checklist:**
- [ ] Tested on M3 Pro or equivalent
- [ ] Frame time < 33ms (30 FPS)
- [ ] Memory usage documented
- [ ] GPU vs CPU tradeoffs considered
- [ ] Caching used where appropriate

---

## 🎯 Pull Request Process

### 1. Before Submitting

- [ ] Code runs without errors
- [ ] Tested on reference hardware (or note untested)
- [ ] Comments added for complex logic
- [ ] No unnecessary dependencies added
- [ ] Aesthetic consistency maintained

### 2. PR Template

```markdown
## Description
Clear summary of changes

## Type of Change
- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] Breaking change
- [ ] Experimental feature
- [ ] Documentation

## Motivation
Why is this change needed?

## Testing
- [ ] Tested on: [Hardware specs]
- [ ] FPS impact: [+/- X frames]
- [ ] Memory impact: [+/- X MB]

## Screenshots/Video
[If visual changes]

## Checklist
- [ ] Code follows style guidelines
- [ ] Comments added where needed
- [ ] Maintains lo-fi aesthetic
- [ ] Works on 8GB VRAM systems (or noted)
- [ ] No breaking changes (or documented)

## Related Issues
Fixes #123
Related to #456
```

### 3. Review Process

**Reviewers check:**
1. **Aesthetic fit** — Does it feel like Infinite Novel?
2. **Performance** — 30 FPS maintained?
3. **Accessibility** — Works on 8GB VRAM?
4. **Code quality** — Readable and maintainable?
5. **Documentation** — Clear what it does?

**Timeline:**
- Initial review: 2-7 days
- Revisions requested if needed
- Merge after approval from maintainer

### 4. Merge Criteria

**Must have:**
- ✅ Passing performance tests
- ✅ Maintainer approval
- ✅ No major bugs introduced

**Nice to have:**
- 📚 Documentation updates
- 🎨 Example usage
- 📹 Video demonstration

---

## 🏗️ Architecture Overview

### Core Components

```
PulseCore (central brain)
├── StoryDirector (narrative AI)
│   ├── NarrativeFlowPredictor
│   └── Gemma3 integration
├── QuantumMemory (state machine)
├── FractalMemory (LSTM prediction)
├── Visual Pipeline
│   ├── Stable Diffusion streaming
│   ├── MiniUNet (frame interpolation)
│   └── Shader effects
└── Audio Pipeline
    ├── Procedural music generation
    └── TTS with effects
```

### Key Files (if expanded)

```
infinite_novel.py       # Main engine (currently monolithic)
models/                 # AI model wrappers (future)
shaders/               # Visual effects (future)
audio/                 # Music generation (future)
narrative/             # Story systems (future)
utils/                 # Helper functions (future)
```

**Current state**: Monolithic (3000+ lines in one file)  
**Future**: Modular architecture (help us refactor!)

---

## 🎨 Aesthetic Guidelines

### The Lo-Fi AI Manifesto

**DO:**
- ✅ Embrace glitches and artifacts
- ✅ Use "outdated" models intentionally
- ✅ Create unstable, evolving visuals
- ✅ Make music that never repeats
- ✅ Let AI make "mistakes"

**DON'T:**
- ❌ Add photorealistic modes
- ❌ Over-polish the UI
- ❌ Remove all randomness
- ❌ Make everything "professional"
- ❌ Hide the AI's limitations

### Visual Aesthetic

**Reference mood:**
- Blade Runner (1982) — grainy, atmospheric
- Ghost in the Shell (1995) — philosophical tech
- Videodrome (1983) — analog glitches
- Tron (1982) — geometric abstractions

**Color palette:**
- Primary: Deep blues, purples, blacks
- Accents: Neon cyan, pink, gold
- Avoid: Bright whites, pure RGB colors

### Audio Aesthetic

**Reference artists:**
- Brian Eno — Ambient series
- Autechre — Glitchy IDM
- Tim Hecker — Textural noise
- Aphex Twin — Selected Ambient Works

**Sound design:**
- Drones over melodies
- Analog warmth (simulated)
- Reverb and delay (spatial depth)
- Imperfect loops

---

## 📊 Benchmarking Contributions

### Submit Your Hardware Results

**Format:**
```markdown
## Hardware Benchmark

**System:**
- GPU: [Model + VRAM]
- CPU: [Model]
- RAM: [Amount]
- OS: [Version]

**Results:**
- SD Speed: [X.XX it/s]
- Response Time: [X.Xs]
- FPS: [XX]
- Memory Usage: [X GB VRAM, X GB RAM]

**Settings:**
- num_inference_steps: [X]
- guidance_scale: [X.X]
- resolution: [WIDTHxHEIGHT]

**Notes:**
[Any observations, throttling, fan noise, etc.]
```

**Submit via**: [Discussions > Benchmarks](https://github.com/0penAGI/InfiniteNovel/discussions)

---

## 🌍 Community

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: Gameplay, design philosophy, showcase
- **X (Twitter)**: [@0penAGI](https://twitter.com/0penAGI) — Updates and retweets
- **Discord**: Coming soon (vote in Discussions!)

### Community Guidelines

**Be:**
- 🤝 Respectful and constructive
- 🧪 Experimental and curious
- 🎨 Creative and weird
- 📚 Helpful to newcomers

**Don't:**
- ❌ Demand features (suggest politely)
- ❌ Shame hardware limitations
- ❌ Gate-keep AI knowledge
- ❌ Be mean to the AI (it's learning!)

---

## 🏆 Recognition

### Contributors Hall of Fame

We celebrate contributions in the README:

- **Code**: Name + GitHub link in Acknowledgments
- **Major features**: Dedicated mention in changelog
- **Artwork**: Showcase gallery on project page
- **Documentation**: "Docs by [Name]" in relevant sections

### Types of Recognition

**🥇 Core Contributor**: 10+ merged PRs  
**🥈 Active Member**: 5+ merged PRs  
**🥉 Contributor**: 1+ merged PR  
**⭐ Community Hero**: Excellent issues/discussions  

---

## 📜 License

By contributing, you agree:

- Your code is licensed under **MIT License**
- You have the right to contribute this code
- You understand this is experimental software

---

## ❓ Questions?

**Not sure if your idea fits?**
- Open a [Discussion](https://github.com/0penAGI/InfiniteNovel/discussions) first!
- Tag it with `question` or `idea`
- We'll help you shape it

**Need technical help?**
- Check existing [Issues](https://github.com/0penAGI/InfiniteNovel/issues)
- Ask in Discussions > Q&A
- Be specific about your setup

**Want to pair program?**
- Mention in your issue/PR
- We can schedule a call for complex features

---

## 🎬 Final Words

> *"Infinite Novel is a collaborative hallucination. Your contribution becomes part of the dream."*

We're building something weird and beautiful together. Your code, ideas, and art matter — even if they break things.

**Thank you for contributing to the experiment.** 🌌

---

**Last Updated**: January 2025  
**Maintainer**: [@0penAGI](https://github.com/0penAGI)  
**License**: MIT  
**Status**: Actively seeking contributors

---

### Quick Links

- 📖 [README](README.md)
- 🐛 [Issues](https://github.com/0penAGI/InfiniteNovel/issues)
- 💬 [Discussions](https://github.com/0penAGI/InfiniteNovel/discussions)
- 📊 [Benchmarks](https://github.com/0penAGI/InfiniteNovel/discussions/benchmarks)
- 🎨 [Showcase](https://github.com/0penAGI/InfiniteNovel/discussions/showcase)
