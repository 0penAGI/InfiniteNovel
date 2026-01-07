def sanitize_tts(text):
    return (
        text
        .replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", "-")
        .replace("…", "...")
    )
#infinite_novel.py - https://github.com/0penAGI/InfiniteNovel/tree/main 
# open-source by 0penAGI tested on MacBook M3 Pro 18gb 
import pygame
import numpy as np
import torch
import asyncio
from diffusers import StableDiffusionPipeline
from transformers import pipeline
from TTS.api import TTS
from io import BytesIO
from PIL import Image
import tempfile
import os
import logging
from concurrent.futures import ThreadPoolExecutor
import torch.nn as nn
from collections import defaultdict
import random
import textwrap
import re
import cv2
import scipy.signal
import scipy.io.wavfile
import requests
import threading
import functools
import collections
import time
import moviepy.editor as mp




logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initial Pygame
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2)

# Aspect ratio and resolution
ASPECT_RATIO = 3.51
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = int(SCREEN_WIDTH / ASPECT_RATIO)

# Chech resolution
display_info = pygame.display.Info()
max_width = display_info.current_w
if max_width < SCREEN_WIDTH:
    SCREEN_WIDTH = max_width
    SCREEN_HEIGHT = int(SCREEN_WIDTH / ASPECT_RATIO)

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN | pygame.SCALED)
pygame.display.set_caption("Infinite Novel")

# Dynamic fonts
font_size = int(SCREEN_HEIGHT / 25)
font = pygame.font.SysFont("arial", font_size)
clock = pygame.time.Clock()

# Which technology, gpu or cpu?
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
logging.info(f"Используется устройство: {device}")

# Global pull 
executor = ThreadPoolExecutor(max_workers=2)

# Stable Diffusion
try:
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32,
        use_safetensors=True
    ).to(device)
    pipe.enable_attention_slicing()
    pipe.safety_checker = None
except Exception as e:
    logging.error(f"Ошибка загрузки Stable Diffusion: {e}")
    pipe = None

# TTS с HiFi-GAN вокодером
try:
    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=(device == "cuda"))
except Exception as e:
    logging.error(f"Ошибка загрузки TTS: {e}")
    tts = None

# Mood analysis
try:
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device if device != "mps" else -1)
except Exception as e:
    logging.error(f"Ошибка загрузки sentiment-analyzer: {e}")
    sentiment_analyzer = None


def render_bold_wrapped(text, font, color, screen, x, y, line_spacing, max_width=None, mask=None):
    """
    Рендер текста с *bold* выделением, перенос по ширине экрана.
    Можно применять маску (Surface с alpha) к тексту.
    """
    pattern = r"\*(.*?)\*"
    parts = re.split(pattern, text)
    bold = False
    current_x = x
    current_y = y
    if max_width is None:
        max_width = screen.get_width() - x * 2

    # рисуем на отдельный surface, чтобы потом маску наложить
    text_surface = pygame.Surface((screen.get_width(), screen.get_height()), pygame.SRCALPHA)
    
    for part in parts:
        if not part:
            bold = not bold
            continue
        f = pygame.font.SysFont("arial", font.get_height(), bold=bold)
        words = part.split(" ")
        for word in words:
            word_surface = f.render(word + " ", True, color)
            word_width = word_surface.get_width()
            if current_x + word_width > x + max_width:
                current_x = x
                current_y += line_spacing
            text_surface.blit(word_surface, (current_x, current_y))
            current_x += word_width
        bold = not bold

    if mask:
        text_surface.blit(mask, (0,0), special_flags=pygame.BLEND_RGBA_MULT)

    screen.blit(text_surface, (0,0))


# Quantum like neural network
class QuantumNeuralNetwork(nn.Module):
    def __init__(self, input_size=10, hidden_size=20):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)
        self.quantum_weights = torch.randn(hidden_size, requires_grad=True).to(device)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        quantum_layer = x * torch.cos(self.quantum_weights) + torch.sin(self.quantum_weights)
        return self.fc2(quantum_layer)

# Fractal Memory
class FractalMemory(nn.Module):
    def __init__(self, input_size=10, hidden_size=20):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.quantum = QuantumNeuralNetwork(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, input_size)
        self.hidden_size = hidden_size

    def forward(self, x, hidden=None):
        # x: (seq_len, batch, input_size) or (batch, input_size) or (input_size,)
        if x.dim() == 2:
            # (batch, input_size) → (seq_len=1, batch, input_size)
            x = x.unsqueeze(0)
        elif x.dim() == 1:
            # (input_size,) → (seq_len=1, batch=1, input_size)
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() != 3:
            raise ValueError(f"Unexpected input dim: {x.shape}")
        if hidden is None:
            hidden = (
                torch.zeros(1, x.size(1), self.hidden_size, device=x.device),
                torch.zeros(1, x.size(1), self.hidden_size, device=x.device)
            )
        out, hidden = self.lstm(x, hidden)
        out = self.quantum(out)
        out = self.fc(out)
        return out, hidden

# Quantum like Crystals
class QuantumMemory:
    def __init__(self):
        self.transitions = defaultdict(lambda: defaultdict(int))
        self.states = ["POSITIVE", "NEGATIVE", "NEUTRAL", "CURIOUS", "AGGRESSIVE"]
        self.action_history = []

    def update(self, prev_state, curr_state, action_text):
        self.transitions[prev_state][curr_state] += 1
        self.action_history.append((action_text, curr_state))
        if len(self.action_history) > 100:
            self.action_history.pop(0)

    def predict(self, curr_state, visual_features=None):
        if curr_state not in self.transitions:
            return np.random.choice(self.states)
        total = sum(self.transitions[curr_state].values())
        probs = [self.transitions[curr_state][s] / total for s in self.states]
        recent_states = [s for _, s in self.action_history[-10:]]
        state_weights = [1.0 + 0.2 * recent_states.count(s) for s in self.states]
        if visual_features:
            brightness = visual_features.get("brightness", 0.0)
            contrast = visual_features.get("contrast", 0.0)
            visual_boost = 1.0 + brightness * 0.5 + contrast * 0.5
            state_weights = [w * visual_boost if s in ["POSITIVE", "CURIOUS"] else w for w, s in zip(state_weights, self.states)]
        probs = np.array(probs) * np.array(state_weights) * np.cos(np.random.randn(5)) + np.sin(np.random.randn(5))
        probs = np.clip(probs, 0, None)
        if probs.sum() == 0:
            probs = np.ones_like(probs) / len(probs)
        else:
            probs /= probs.sum()
        return np.random.choice(self.states, p=probs)

# --- Gemma3:4b streaming generation through Ollama ---
def gemma3_generate(prompt: str, history=None, model="gemma3:1b", on_token=None):
    """
    Стриминговый вызов Ollama Gemma.
    on_token(token: str) вызывается при каждом новом фрагменте текста.
    Возвращает полный текст после завершения.
    """
    import json
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "stream": True,
        "messages": []
    }
    if history:
        payload["messages"].extend(history)
    payload["messages"].append({"role": "user", "content": prompt})

    full_text = ""

    try:
        with requests.post(url, json=payload, stream=True, timeout=60) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue

                if "message" in data and "content" in data["message"]:
                    token = data["message"]["content"]
                    full_text += token
                    if on_token:
                        on_token(token)

                if data.get("done", False):
                    break

        return full_text.strip()

    except Exception as e:
        logging.error(f"Gemma3 stream failed: {e}")
        return full_text.strip()


#
# ---  StoryDirector with memory Gemma3 and user profile ---
class NarrativeFlowPredictor:
    """
    Tracks story states and generates plot suggestions based on history.
    """
    def __init__(self):
        self.story_states = []
        self.plot_suggestions = []

    def update(self, state, intent):
        self.story_states.append((state, intent))
        if len(self.story_states) > 50:
            self.story_states.pop(0)

    def predict(self):
        # Very simple logic: suggest a plot twist if same arc repeats > 3 times
        if len(self.story_states) < 4:
            return None
        last_arc = self.story_states[-1][0]
        if all(s[0] == last_arc for s in self.story_states[-4:]):
            return f"Consider a plot twist or escalation in the '{last_arc}' arc."
        return None


class StoryDirector:
    def __init__(self):
        self.arc = "awakening"
        self.beat = 0
        self.flags = set()
        self.history = []
        self.thread_influence = defaultdict(float)
        self.last_visual_context = {}
        self.conversation_memory = []  # last conversation
        self.player_profile = []       # 30-50 replies for forming «Pulse»
        # Add NarrativeFlowPredictor for story state tracking and plot suggestions
        self.narrative_flow = NarrativeFlowPredictor()

    def advance_arc(self, impact, mood, threads=None, visual=None):
        thread_boost = sum(min(v, 1.0) for v in threads.values()) * 0.1 if threads else 0.0
        visual_boost = (visual.get("contrast", 0) + visual.get("edges", 0)) * 0.1 if visual else 0.0
        total_impact = impact + thread_boost + visual_boost
        if self.arc == "awakening" and (total_impact > 0.12 or mood > 0.18):
            self.arc = "convergence"
        elif self.arc == "convergence" and (total_impact < -0.12):
            self.arc = "rupture"
        elif self.arc == "rupture" and (total_impact > 0.22):
            self.arc = "synthesis"

    def next_beat(self, intent, threads=None):
        self.beat += 1
        self.history.append((self.arc, self.beat, intent))
        if len(self.history) > 20:
            self.history.pop(0)
        if threads:
            for k, v in threads.items():
                self.thread_influence[k] += v * 0.1
                if self.thread_influence[k] > 1.0:
                    self.thread_influence[k] = 1.0
        # Update narrative flow history
        self.update_story_history(self.arc, intent)

    def respond(self, intent, mood, resonance, core):
        self.player_profile.append({"role": "player", "content": intent})
        if len(self.player_profile) > 50:
            self.player_profile = self.player_profile[-50:]

        top_threads = sorted(core.memory["threads"].items(), key=lambda x: -x[1])[:3]
        threads_str = ", ".join(f"{k}({v:.1f})" for k, v in top_threads) if top_threads else "none"
        recent_events = core.memory["events"][-3:]
        events_str = " → ".join(e["action"] for e in recent_events) if recent_events else "beginning"
        collapse = core.world_state["collapse"]
        instability = core.world_state["instability"]
        locks = list(core.world_state["locks"])

        context = (
            f"Story Arc: {self.arc}\n"
            f"Active Threads: {threads_str}\n"
            f"Recent Events: {events_str}\n"
            f"World State: collapse={collapse:.1f}, instability={instability:.1f}, locks={locks}\n"
            f"Mood: {mood:.2f} | Resonance: {resonance:.2f}\n"
            f"Player Action: {intent}\n\n"
            f"Respond as the world itself, remembering the player's past actions. Be human, direct, emotional. 1-2 sentences."
        )

        self.conversation_memory.append({"role": "user", "content": context})
        if len(self.conversation_memory) > 50:
            self.conversation_memory = self.conversation_memory[-50:]

        streamed_text = ""

        def _on_token(tok):
            nonlocal streamed_text
            streamed_text += tok
            core.set_dialogue(streamed_text)

        text = gemma3_generate(
            context,
            history=self.conversation_memory + self.player_profile,
            on_token=_on_token
        )
        self.conversation_memory.append({"role": "assistant", "content": text})

        return text.strip()

    # --- NarrativeFlowPredictor integration ---
    def update_story_history(self, state, intent):
        """
        Update the narrative flow predictor with the current story state and intent.
        """
        if hasattr(self, "narrative_flow") and self.narrative_flow:
            self.narrative_flow.update(state, intent)

    def predict_narrative_flow(self):
        """
        Use the narrative flow predictor to suggest the next plot development.
        Returns a suggestion string or None.
        """
        if hasattr(self, "narrative_flow") and self.narrative_flow:
            return self.narrative_flow.predict()
        return None



# Core
class PulseCore:
    def __init__(self):
        self.memory = {
            "prompts": [],
            "sentiments": [],
            "mood_score": 0.0,
            "dialogue_scores": [],
            "events": [],          # что произошло (короткие факты)
            "threads": defaultdict(float),  # активные сюжетные нити и их «напряжение»
            "world_state": {}       # изменяемые факты мира
        }
        self.fractal_memory = FractalMemory().to(device)
        self.quantum_memory = QuantumMemory()
        self.quantum_nn = QuantumNeuralNetwork().to(device)
        self.weights = {"image": 0.3, "audio": 0.3, "text": 0.4}
        self.audio_cache = {}
        self.network_nodes = 0
        self.resonance = 0.5
        self.titan_relation = 0.0
        self.current_world = "Cosmic Void"
        self.conflicts = []
        self.allies = []
        self.story_progress = 0
        self.preferred_phrases = []
        self.current_dialogue = ""
        self.displayed_dialogue = ""
        self.text_animation_index = 0
        self.last_char_time = 0
        self.char_delay = 80  
        self.image_buffer = []
        self.max_buffer_size = 5
        self.feedback_opacity = 0.2
        self.displacement_strength = 10.0
        self.visual_features = {"brightness": 0.0, "contrast": 0.0, "edges": 0.0}
        self.morphing_state = {"active": False, "start_time": 0, "duration": 12000}  # longer for smoother morph
        self.morphing_old_image = None
        self.morphing_new_image = None
        self.stream_buffer = []
        self.next_image_task = None
        self.next_image_prompt = None
        self.next_image_surface = None
        self.story = StoryDirector()
        self.image_task = None
        self.is_streaming = False
        self.last_stream_surface = None
        self.stream_fade_alpha = 0.0
        self.stream_fade_speed = 0.08
        self.stream_last_surface = None

        # === STYLE MEMORY CAPTURE ===
        self.capture_dir = "dataset"
        self.session_id = int(time.time())
        self.last_input_time = pygame.time.get_ticks()
        self.idle_capture_ms = 4000
        os.makedirs(f"{self.capture_dir}/session_{self.session_id}", exist_ok=True)

        self.music_state = {
            "mode": "drone",
            "mutation": 0.0,
            "age": 0
        }

        # === WORLD FAILURE STATE (GAME LAYER) ===
        self.world_state = {
            "collapse": 0.0,
            "instability": 0.0,
            "locks": set(),
            "titan_timer": 120
        }
        self.pain_level = 0.0  # Subjectivity pain of system

    def capture_frame(self, surface):
        if surface is None:
            return

        ts = pygame.time.get_ticks()
        base = f"{self.capture_dir}/session_{self.session_id}/img_{ts}"

        # save image
        pygame.image.save(surface, base + ".png")

        # --- auto-caption from threads ---
        threads = self.memory.get("threads", {})
        sorted_threads = sorted(
            [(k, v) for k, v in threads.items() if v >= 0.25],
            key=lambda x: x[1],
            reverse=True
        )[:6]

        thread_tokens = []
        for name, weight in sorted_threads:
            if name in ["pulse", "signal", "network"]:
                thread_tokens.append("pulse network")
            elif name in ["fracture", "break", "collapse"]:
                thread_tokens.append("fractured structure")
            elif name in ["void", "silence", "emptiness"]:
                thread_tokens.append("void atmosphere")
            else:
                thread_tokens.append(name.replace("_", " "))

        # mood modifier
        mood = float(self.memory.get("mood_score", 0.0))
        if mood < -0.3:
            mood_tag = "dark, tense"
        elif mood > 0.3:
            mood_tag = "luminous, calm"
        else:
            mood_tag = "neutral, abstract"

        arc = getattr(self.story, "arc", None)
        arc_tag = f"{arc} arc" if arc else "unknown arc"

        caption = (
            "abstract sci-fi scene, "
            + ", ".join(thread_tokens)
            + f", {arc_tag}, {mood_tag}, hypnotic, neural dream"
        )

        # save caption
        with open(base + ".txt", "w") as f:
            f.write(caption)

        # save meta
        meta = {
            "arc": arc,
            "mood": mood,
            "resonance": float(self.resonance),
            "threads": threads,
            "time": ts
        }

        with open(base + ".json", "w") as f:
            import json
            json.dump(meta, f, indent=2)

    def evaluate_quality(self, data, data_type):
        if data_type == "image":
            img_array = np.array(data.convert("L")).astype(np.float32)
            edges = np.abs(np.gradient(img_array)[0])
            return min(np.var(edges) / 1000, 1.0)
        elif data_type == "audio":
            return 0.8
        elif data_type == "text":
            return 0.8 if len(data.split()) > 3 and any(p in data for p in self.preferred_phrases) else 0.4
        return 0.5

    def self_program(self, image_quality, audio_quality, text_quality):
        feedback = np.array([image_quality, audio_quality, text_quality])
        target = np.array([1.0, 1.0, 1.0])
        grad = (target - feedback) * 0.1
        visual_boost = 1.0 + self.visual_features["contrast"] * 0.5 + self.visual_features["brightness"] * 0.3
        grad[0] *= visual_boost
        for i, k in enumerate(["image", "audio", "text"]):
            self.weights[k] = np.clip(self.weights[k] + grad[i], 0.1, 0.5)
        logging.info(f"Self-programmed weights: {self.weights}")

    def grow_network(self, action_impact):
        self.network_nodes += int(action_impact * 5)

        self.world_state["instability"] += abs(action_impact) * 0.2
        if action_impact < 0:
            self.world_state["collapse"] += abs(action_impact) * 0.3

        self.resonance = np.clip(self.resonance + action_impact * 0.05, 0.0, 1.0)
        self.titan_relation += action_impact * 0.05
        self.story_progress += abs(action_impact) * 8

    def record_event(self, action_text, action_impact, inferred_state):
        """
        Фиксирует факт произошедшего и усиливает сюжетные нити.
        """
        event = {
            "action": action_text,
            "impact": action_impact,
            "state": inferred_state,
            "time": pygame.time.get_ticks()
        }
        self.memory["events"].append(event)
        if len(self.memory["events"]) > 50:
            self.memory["events"].pop(0)

        # Выделяем нити из действия
        for word in action_text.lower().split():
            if len(word) > 4:
                self.memory["threads"][word] += abs(action_impact) + 0.1

        # Медленно затухают старые нити
        for k in list(self.memory["threads"].keys()):
            self.memory["threads"][k] *= 0.98
            if self.memory["threads"][k] < 0.05:
                del self.memory["threads"][k]

    def add_conflict(self, conflict):
        self.conflicts.append(conflict)

    def resolve_conflict(self, conflict):
        if conflict in self.conflicts:
            self.conflicts.remove(conflict)
            self.resonance += 0.1

    def add_ally(self, ally):
        if ally not in self.allies:
            self.allies.append(ally)
            self.titan_relation += 0.1

    def update_dialogue_preference(self, dialogue, score):
        self.memory["dialogue_scores"].append((dialogue, score))
        if score > 0.7:
            self.preferred_phrases.append(dialogue[:20])
        if len(self.preferred_phrases) > 50:
            self.preferred_phrases.pop(0)

    def animate_text(self, current_time):
        if self.text_animation_index < len(self.current_dialogue):
            if current_time - self.last_char_time >= self.char_delay:
                self.text_animation_index += 1
                self.displayed_dialogue = self.current_dialogue[:self.text_animation_index]
                self.last_char_time = current_time
        return self.displayed_dialogue

    def set_dialogue(self, dialogue):
        self.current_dialogue = dialogue
        self.displayed_dialogue = ""
        self.text_animation_index = 0
        self.last_char_time = pygame.time.get_ticks()

    def add_image_to_buffer(self, image_surface):
        if image_surface:
            self.image_buffer.append(image_surface)
            if len(self.image_buffer) > self.max_buffer_size:
                self.image_buffer.pop(0)

    def apply_feedback_loop(self, current_surface):
        if not self.image_buffer or not current_surface:
            return current_surface

        feedback_surface = current_surface.copy()
        for i, old_surface in enumerate(self.image_buffer):
            opacity = self.feedback_opacity * (1.0 - i / len(self.image_buffer))
            old_surface.set_alpha(int(255 * opacity))

            old_array = pygame.surfarray.array3d(old_surface)
            old_array = np.transpose(old_array, (1, 0, 2))
            blurred = cv2.GaussianBlur(old_array, (9, 9), 2)
            blurred = np.transpose(blurred, (1, 0, 2))
            blurred_surface = pygame.surfarray.make_surface(blurred)
            blurred_surface.set_alpha(int(255 * opacity))

            feedback_surface.blit(blurred_surface, (0, 0))

        return feedback_surface

    def apply_glow(self, image_surface):
        img_array = pygame.surfarray.array3d(image_surface)
        img_array = np.transpose(img_array, (1, 0, 2))
        
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, int(50 + self.memory["mood_score"] * 50))
        v = np.clip(v, 0, 255)
        hsv = cv2.merge([h, s, v])
        glowed = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        glowed = cv2.GaussianBlur(glowed, (15, 15), 0)
        
        glowed = np.transpose(glowed, (1, 0, 2))
        glowed_surface = pygame.surfarray.make_surface(glowed)
        glowed_surface.set_alpha(100)
        result_surface = image_surface.copy()
        result_surface.blit(glowed_surface, (0, 0))
        
        return result_surface

    def update_visual_features(self, image):
        if image is None:
            return
        img_array = pygame.surfarray.array3d(image)
        img_array = np.transpose(img_array, (1, 0, 2))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        self.visual_features["brightness"] = np.mean(gray) / 255.0
        self.visual_features["contrast"] = np.std(gray) / 255.0
        edges = cv2.Canny(gray, 100, 200)
        self.visual_features["edges"] = np.mean(edges) / 255.0

    def start_morphing(self, old_image, new_image):
        self.morphing_state["active"] = True
        self.morphing_state["start_time"] = pygame.time.get_ticks()
        self.morphing_old_image = old_image
        self.morphing_new_image = new_image

    def apply_morphing(self, current_time):
        if not self.morphing_state["active"] or not self.morphing_old_image or not self.morphing_new_image:
            self.morphing_state["active"] = False
            return self.morphing_new_image or self.morphing_old_image

        elapsed = current_time - self.morphing_state["start_time"]
        # ease-in-out cubic for smoother morphing
        t = min(elapsed / self.morphing_state["duration"], 1.0)
        alpha = t * t * (3 - 2 * t)
        logging.info(f"Morphing alpha: {alpha:.2f}, elapsed: {elapsed}, duration: {self.morphing_state['duration']}")

        if alpha >= 1.0:
            self.morphing_state["active"] = False
            logging.info("Morphing completed")
            return self.morphing_new_image

        old_array = pygame.surfarray.array3d(self.morphing_old_image)
        new_array = pygame.surfarray.array3d(self.morphing_new_image)

        if old_array.shape != new_array.shape:
            logging.warning("Image size mismatch in morphing")
            self.morphing_state["active"] = False
            return self.morphing_new_image

        old_array = np.transpose(old_array, (1, 0, 2))
        new_array = np.transpose(new_array, (1, 0, 2))

        morphed = cv2.addWeighted(old_array, 1.0 - alpha, new_array, alpha, 0.0)
        morphed = np.transpose(morphed, (1, 0, 2))
        return pygame.surfarray.make_surface(morphed)

    async def pregenerate_image(self, prompt, mood_score):
        if not pipe:
            return None
        try:
            image = await asyncio.get_event_loop().run_in_executor(executor, lambda: pipe(
                prompt, num_inference_steps=14, guidance_scale=5.5
            ).images[0])
            img_array = np.array(image).astype(np.float32)
            img_array = np.clip(img_array, 0, 255)
            image = Image.fromarray(img_array.astype(np.uint8))
            img_buffer = BytesIO()
            image.save(img_buffer, format="PNG")
            img_buffer.seek(0)
            self.next_image_surface = pygame.image.load(img_buffer)
            return self.next_image_surface
        except Exception as e:
            logging.error(f"Ошибка предгенерации изображения: {e}")
            return None

# displacement effect
def improved_displacement(image_surface, strength=6.0):
    img_array = pygame.surfarray.array3d(image_surface)
    img_array = np.transpose(img_array, (1, 0, 2))
    height, width = img_array.shape[:2]
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    time = pygame.time.get_ticks() / 160.0
    displacement_x = np.sin(x * 1.2 + time) * np.cos(y * 1.2 + time)
    displacement_y = np.cos(x * 1.2 + time) * np.sin(y * 1.2 + time)
    displacement_x = cv2.GaussianBlur(displacement_x, (9, 9), 2) * strength
    displacement_y = cv2.GaussianBlur(displacement_y, (9, 9), 2) * strength

    map_x = (x + displacement_x).astype(np.float32)
    map_y = (y + displacement_y).astype(np.float32)
    displaced = cv2.remap(img_array, map_x, map_y, cv2.INTER_LINEAR)

    if len(displaced.shape) != 3:
        return image_surface

    displaced = np.transpose(displaced, (1, 0, 2))
    return pygame.surfarray.make_surface(displaced)

# Sharpen effect 
def improved_sharpen(image_surface, strength=0.6):
    img_array = pygame.surfarray.array3d(image_surface)
    img_array = np.transpose(img_array, (1, 0, 2)).astype(np.float32)

    sharpen_kernel = np.array([
        [0, -strength, 0],
        [-strength, 1 + 4 * strength, -strength],
        [0, -strength, 0]
    ])
    sharpened = cv2.filter2D(img_array, -1, sharpen_kernel)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    sharpened = np.transpose(sharpened, (1, 0, 2))
    return pygame.surfarray.make_surface(sharpened)

# Pulsating background
def improved_pulsating_background(screen, mood_score, generating):
    pulse = np.sin(pygame.time.get_ticks() / 1000.0) * 0.7 + 0.3
    intensity = 15 + pulse * 10 if generating else 10 + pulse * 5
    color = (
        int(20 + mood_score * 20),
        int(20 + mood_score * 20),
        int(30 + intensity)
    )
    screen.fill(color)

def generate_music_frequencies(
    duration=3.0,
    sample_rate=44100,
    mood_score=0.0,
    action_text="",
    story_progress=0,
    resonance=0.5,
    titan_relation=0.0,
    act_transition_steps=11
):
    # === living evolutionary music engine v2 ===
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # ultra-smooth attack + long release (anti-click)
    attack_time = max(0.015, duration * 0.05)
    release_time = max(0.8, duration * 0.45)

    attack_samples = int(sample_rate * attack_time)
    release_samples = int(sample_rate * release_time)

    env = np.ones_like(t)

    # attack (cosine fade-in)
    if attack_samples > 1:
        a = np.linspace(0, np.pi / 2, attack_samples)
        env[:attack_samples] = np.sin(a) ** 2

    # release (cosine fade-out, very long)
    if release_samples > 1 and release_samples < len(env):
        r = np.linspace(0, np.pi / 2, release_samples)
        env[-release_samples:] = np.cos(r) ** 2
    rng = np.random.rand()

    # global context energy
    context_energy = np.clip(abs(mood_score) + resonance + abs(titan_relation), 0.2, 2.5)

    # base pitch field
    base_freq = (
        90 if mood_score < -0.4 else
        180 if mood_score > 0.4 else
        130
    )
    base_freq *= 1.0 + 0.15 * np.sin(story_progress * 0.02)

    # --- macro modes ---
    if rng < 0.12:
        mode = "silence"
    elif rng < 0.30:
        mode = "anomaly"
    elif rng < 0.55:
        mode = "pulse"
    elif rng < 0.75:
        mode = "swell"
    else:
        mode = "drone"

    if mode == "silence":
        return np.zeros_like(t, dtype=np.float32)

    # === harmonic layer ===
    harmonics = np.array([1.0, 1.5, 2.01, 0.5])
    amps = np.array([1.0, 0.6, 0.4, 0.3])
    harmonic = np.zeros_like(t)

    for h, a in zip(harmonics, amps):
        drift = 1.0 + 0.01 * np.sin(2 * np.pi * 0.1 * t + np.random.rand() * 10)
        harmonic += a * np.sin(2 * np.pi * base_freq * h * drift * t)

    harmonic /= np.max(np.abs(harmonic) + 1e-8)
    # мягкий lowpass для всего harmonic
    b, a = scipy.signal.butter(4, 900 / (sample_rate * 0.5), btype="low")
    harmonic = scipy.signal.lfilter(b, a, harmonic)

    # === noise organism layer (dark sub noise) ===
    noise = np.random.normal(0, 1, len(t))
    b, a = scipy.signal.butter(4, 400 / (sample_rate * 0.5), btype="low")
    noise = scipy.signal.lfilter(b, a, noise)
    noise *= 0.22

    # === sub bass layer (deep mono energy) ===
    sub_freq = base_freq * 0.5
    sub = np.sin(2 * np.pi * sub_freq * t + np.random.rand() * 6.28)

    # very slow movement to avoid static bass
    sub *= 0.6 + 0.4 * np.sin(2 * np.pi * 0.03 * t)

    # hard lowpass (sub only)
    b, a = scipy.signal.butter(4, 120 / (sample_rate * 0.5), btype="low")
    sub = scipy.signal.lfilter(b, a, sub)

    sub *= 0.55

    # === granular texture layer ===
    grain_density = int(8 + context_energy * 18)
    grain_size = int(sample_rate * np.random.uniform(0.02, 0.06))  # 20–60 ms
    grains = np.zeros_like(t)

    for _ in range(grain_density):
        start = np.random.randint(0, max(1, len(t) - grain_size))
        window = np.hanning(grain_size)
        freq = base_freq * np.random.choice([0.5, 1.0, 1.5, 2.0])
        phase = np.random.rand() * 2 * np.pi
        grain = np.sin(2 * np.pi * freq * np.linspace(0, grain_size / sample_rate, grain_size) + phase)
        grains[start:start + grain_size] += grain * window

    if np.max(np.abs(grains)) > 0:
        grains /= np.max(np.abs(grains))

    grains *= 0.25 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.02 * t))

    # === ghost layer (appears rarely) ===
    ghost = np.zeros_like(t)
    if rng > 0.82:
        ghost_freq = base_freq * np.random.choice([0.25, 0.75, 1.33])
        ghost = np.sin(2 * np.pi * ghost_freq * t + np.random.rand() * 6.28)
        ghost *= 0.25 * np.sin(2 * np.pi * 0.03 * t)

    # === reverb harmonic mutation layer (very rare) ===
    reverb_harm = np.zeros_like(t)
    if rng > 0.88:
        rh_freqs = base_freq * np.random.choice([0.5, 1.25, 1.75], size=2, replace=False)
        for f in rh_freqs:
            ph = np.random.rand() * 6.28
            reverb_harm += np.sin(2 * np.pi * f * t + ph)

        # slow smear
        reverb_harm *= 0.2
        reverb_harm = scipy.signal.lfilter([1], [1, -0.995], reverb_harm)

    # === pulse / rhythm DNA ===
    pulse_rate = 0.5 + 2.5 * context_energy
    pulse = (np.sin(2 * np.pi * pulse_rate * t) > 0).astype(np.float32)
    pulse = scipy.signal.savgol_filter(pulse, 101, 3)

    # === anomaly layer ===
    anomaly = np.zeros_like(t)
    if mode == "anomaly":
        spikes = np.random.choice(len(t), size=len(t)//200, replace=False)
        anomaly[spikes] = np.random.uniform(-1, 1, size=len(spikes))
        anomaly = scipy.signal.lfilter([1, -0.97], [1], anomaly)
        anomaly *= 0.8

    # === morphing envelope ===
    morph = 0.5 + 0.5 * np.sin(2 * np.pi * 0.07 * t + story_progress * 0.01)

    # === layer fusion ===
    signal = (
        harmonic * morph +
        noise * (1 - morph) +
        anomaly +
        ghost +
        reverb_harm +
        grains +
        sub
    )

    if mode in ["pulse", "swell"]:
        signal *= pulse

    # === slow spectral mutation ===
    fft = np.fft.rfft(signal)
    mutation = 1.0 + 0.15 * np.sin(np.linspace(0, np.pi, len(fft)) + rng * 10)
    fft *= mutation
    signal = np.fft.irfft(fft, n=len(signal))

    # === dark spectral tilt (kills highs gently) ===
    fft = np.fft.rfft(signal)
    freqs = np.linspace(0, 1, len(fft))
    tilt = 1.0 / (1.0 + freqs * 6.0)
    fft *= tilt
    signal = np.fft.irfft(fft, n=len(signal))

    # === saturation + breathing dynamics ===
    breath = 0.6 + 0.4 * np.sin(2 * np.pi * 0.05 * t)
    signal *= breath
    signal = np.tanh(signal * (1.2 + context_energy * 0.3))

    # apply release envelope
    signal *= env

    # === occasional enhanced reverb + granular tail at sample end ===
    if rng > 0.7:
        tail_len = int(sample_rate * np.random.uniform(0.3, 0.7))
        tail = np.zeros(tail_len, dtype=np.float32)

        # granular smear
        grain_size = int(sample_rate * np.random.uniform(0.03, 0.08))
        for _ in range(np.random.randint(6, 14)):
            start = np.random.randint(0, max(1, len(signal) - grain_size))
            g = signal[start:start + grain_size]
            if len(g) == grain_size:
                window = np.hanning(grain_size)
                tail[:grain_size] += g * window * np.random.uniform(0.4, 0.8)

        # deep reverb feedback
        for i in range(1, len(tail)):
            tail[i] += tail[i - 1] * np.random.uniform(0.92, 0.97)

        # dark lowpass reverb tail
        b, a = scipy.signal.butter(2, 600 / (sample_rate * 0.5), btype="low")
        tail = scipy.signal.lfilter(b, a, tail)

        # gentle fade-out
        fade = np.linspace(1.0, 0.0, len(tail))
        tail *= fade

        signal = np.concatenate([signal, tail])

    # === FINAL ANTI-CLICK SANITIZATION (MANDATORY) ===

    # remove DC offset AFTER all nonlinearities
    signal -= np.mean(signal)

    # force zero-crossing at end (hard click killer)
    zc_len = min(512, len(signal))
    end_val = signal[-1]
    signal[-zc_len:] -= np.linspace(0, end_val, zc_len)

    # long linear fade (stable, predictable)
    fade_len = int(sample_rate * 0.02)  # 20 ms
    if fade_len * 2 < len(signal):
        fade = np.linspace(0.0, 1.0, fade_len)
        signal[:fade_len] *= fade
        signal[-fade_len:] *= fade[::-1]

    # normalize safely
    signal /= (np.max(np.abs(signal)) + 1e-8)

    return signal.astype(np.float32)

async def play_music(core, mood_score=0.0, action_text="", story_progress=0, resonance=0.5, titan_relation=0.0):
    """
    Асинхронный бесконечный музыкальный генератор с паузами, дилеем и ревером.
    WAV удаляются сразу после проигрывания.
    """
    try:
        sample_rate = 44100
        channel = pygame.mixer.Channel(0)

        local_mutation = 0.0

        while True:
            # --- сегмент ---
            segment_duration = np.random.uniform(2.0, 5.0)
            audio = generate_music_frequencies(
                duration=segment_duration,
                mood_score=mood_score,
                action_text=action_text,
                story_progress=story_progress,
                resonance=resonance,
                titan_relation=titan_relation
            )

            # evolutionary mutation drift
            local_mutation += np.random.normal(0, 0.05)
            local_mutation = np.clip(local_mutation, -1.0, 1.0)
            audio *= (1.0 + local_mutation * 0.3)

            # slow morphing mutation drift
            morph_lfo = 0.7 + 0.3 * np.sin(pygame.time.get_ticks() / 4000.0)
            audio *= morph_lfo

            # STRONG dub delay (clearly audible)
            delay_ms = 860
            feedback = 0.88
            wet = 0.82

            delay_samples = int(sample_rate * delay_ms / 1000)
            delayed = np.zeros_like(audio)

            # multi-tap feedback for dub feel
            for i in range(delay_samples, len(audio)):
                delayed[i] = (
                    audio[i - delay_samples] * feedback +
                    delayed[i - delay_samples] * feedback * 0.9
                )

            # slight pre-saturation
            delayed = np.tanh(delayed * 1.4)

            audio = (1 - wet) * audio + wet * delayed

            # dark reverb tail (lowpassed)
            rev = np.zeros_like(audio)
            rev_delay = int(sample_rate * 0.22)
            for i in range(rev_delay, len(audio)):
                rev[i] = audio[i - rev_delay] * 0.4 + rev[i - rev_delay] * 0.65
            b, a = scipy.signal.butter(2, 700 / (sample_rate * 0.5), btype="low")
            rev = scipy.signal.lfilter(b, a, rev)
            audio = audio + rev * 0.45

            # loudness compensation
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            audio *= 0.95

            # === tail padding to avoid hard cut ===
            tail = np.zeros(int(sample_rate * 0.6), dtype=np.float32)
            audio = np.concatenate([audio, tail])

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
                scipy.io.wavfile.write(fp.name, sample_rate, audio.astype(np.float32))
                segment_path = fp.name

            sound = pygame.mixer.Sound(segment_path)
            channel.play(sound, fade_ms=30)

            elapsed = 0.0
            dt = 0.1
            while elapsed < segment_duration:
                await asyncio.sleep(dt)
                elapsed += dt

            channel.fadeout(80)
            await asyncio.sleep(0.09)

            os.unlink(segment_path)

            # --- пауза ---
            pause_duration = np.random.uniform(1.0, 4.0)
            pause_audio = np.zeros(int(sample_rate * pause_duration), dtype=np.float32)

            reverb_samples_pause = int(0.3 * sample_rate)
            for i in range(len(pause_audio)):
                if i >= reverb_samples_pause:
                    pause_audio[i] += pause_audio[i - reverb_samples_pause] * 0.25

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
                scipy.io.wavfile.write(fp.name, sample_rate, pause_audio.astype(np.float32))
                pause_path = fp.name

            pause_sound = pygame.mixer.Sound(pause_path)
            channel.play(pause_sound, fade_ms=30)
            await asyncio.sleep(pause_duration)

            os.unlink(pause_path)

    except Exception as e:
        logging.error(f"Ошибка воспроизведения музыки: {e}")
        return 0.0


# Cache for intro video
_intro_video_cache = {
    "frames": None,
    "path": None,
    "size": None,
}

async def play_intro_video(core, path="intro.mp4"):
    """
    Асинхронно проигрывает видео с аудио на экране.
    Не блокирует event loop.
    """
    global _intro_video_cache


    if (_intro_video_cache["frames"] is None or
        _intro_video_cache["path"] != path or
        _intro_video_cache["size"] != (SCREEN_WIDTH, SCREEN_HEIGHT)):
        cap = cv2.VideoCapture(path)
        frames = []
        success, frame = cap.read()
        while success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (SCREEN_WIDTH, SCREEN_HEIGHT))
            surf = pygame.surfarray.make_surface(np.transpose(frame, (1,0,2)))
            frames.append(surf)
            success, frame = cap.read()
        cap.release()
        _intro_video_cache["frames"] = frames
        _intro_video_cache["path"] = path
        _intro_video_cache["size"] = (SCREEN_WIDTH, SCREEN_HEIGHT)
    else:
        frames = _intro_video_cache["frames"]

    if not frames:
        logging.warning("No intro frames loaded.")
        return

    try:
        clip = mp.VideoFileClip(path)
        if clip.audio:
            clip.audio.write_audiofile("intro_audio.wav", verbose=False, logger=None)
            sound = pygame.mixer.Sound("intro_audio.wav")
            channel = pygame.mixer.Channel(2)
            channel.play(sound)
    except Exception as e:
        logging.error(f"Ошибка воспроизведения аудио интро: {e}")


    idx = 0
    n = len(frames)
    frame_time = 1.0 / 24.0
    while True:
        if hasattr(core, "stop_intro") and core.stop_intro:
            break
        screen.blit(frames[idx], (0,0))
        pygame.display.flip()
        idx = (idx + 1) % n
        await asyncio.sleep(frame_time)


    if 'channel' in locals() and channel.get_busy():
        fade_duration = 2.0  
        steps = 20
        original_volume = channel.get_volume()
        for i in range(steps):
            channel.set_volume(original_volume * (1 - (i + 1)/steps))
            await asyncio.sleep(fade_duration / steps)
        channel.stop()

# Generate Image 
async def generate_image(core, prompt, mood_score=0.0):
    if not pipe:
        return None, 0.0

    if core.image_task and not core.image_task.done():
        return None, 0.0  

    last_event = core.world_state.get("last_event", "")
    arc = getattr(core.story, "arc", "awakening")
    image_prompt = f"cinematic sci-fi scene, {arc} arc, {last_event}, emotional, high contrast, no text"

    init_image = core.morphing_new_image
    if init_image:
        arr = pygame.surfarray.array3d(init_image)
        arr = np.transpose(arr, (1, 0, 2))
        init_image = Image.fromarray(arr)

    core.is_streaming = True
    # stream-diffusion feel: keep last latents alive visually
    core.last_stream_surface = core.morphing_new_image

    # Mini NN net
    class MiniUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
            self.conv2 = nn.Conv2d(8, 3, 3, padding=1)
        def forward(self, x):
            x = x.contiguous()
            x = torch.relu(self.conv1(x))
            x = torch.sigmoid(self.conv2(x))
            return x

    mini_nn = MiniUNet().to(device)

    def apply_micro_shader(surface):
        arr = pygame.surfarray.array3d(surface).astype(np.float32)
        arr = np.transpose(arr, (1,0,2))
        h, w, _ = arr.shape
        t = pygame.time.get_ticks() / 555.0
        warp_x = np.sin(np.linspace(0, np.pi*4, w) + t) * 2
        warp_y = np.cos(np.linspace(0, np.pi*4, h) + t) * 2
        map_x, map_y = np.meshgrid(np.arange(w)+warp_x, np.arange(h)+warp_y)
        displaced = cv2.remap(arr, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR)
        displaced = np.clip(displaced,0,255).astype(np.uint8)
        displaced = np.transpose(displaced,(1,0,2))
        return pygame.surfarray.make_surface(displaced)

    # --- Fractal Noise Shader ---
    def apply_fractal_noise_shader(surface, intensity=1.0, scale=0.1, octaves=4):
        """
        Applies fractal noise-based pixel displacement and blends with original.
        Time-dependent, using pygame.time.get_ticks().
        """
        arr = pygame.surfarray.array3d(surface).astype(np.float32)
        arr = np.transpose(arr, (1,0,2))
        h, w, c = arr.shape
        t = pygame.time.get_ticks() / 666.0
        # Generate base grid
        y, x = np.mgrid[0:h, 0:w]
        # Fractal noise function using summed sin/cos
        def fractal_noise(xx, yy, t, scale, octaves):
            n = np.zeros_like(xx, dtype=np.float32)
            freq = 1.0
            amp = 1.0
            total_amp = 0.0
            for i in range(octaves):
                n += (np.sin(xx * scale * freq + t * freq * 0.7 + i * 10.7) +
                      np.cos(yy * scale * freq + t * freq * 1.2 + i * 5.3)) * 0.5 * amp
                total_amp += amp
                freq *= 2.0
                amp *= 0.5
            return n / (total_amp if total_amp != 0 else 1.0)
        # Displacement fields
        noise_x = fractal_noise(x, y, t, scale, octaves)
        noise_y = fractal_noise(y, x, t + 31.4, scale, octaves)
        # Scale noise for displacement
        disp_strength = 4.0 * intensity
        map_x = (x + noise_x * disp_strength).astype(np.float32)
        map_y = (y + noise_y * disp_strength).astype(np.float32)
        displaced = cv2.remap(arr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        # Blend with original
        alpha = np.clip(intensity, 0.0, 1.0)
        blended = np.clip(arr * (1.0 - alpha) + displaced * alpha, 0, 255).astype(np.uint8)
        blended = np.transpose(blended, (1,0,2))
        return pygame.surfarray.make_surface(blended)

    def callback(step, timestep, latents):
        try:
            start_step = 8   
            full_step = 14  


            with torch.no_grad():
                latents_ = latents / 0.18215
                image = pipe.vae.decode(latents_).sample
                image = (image / 2 + 0.5).clamp(0,1)
                image = image.cpu().permute(0,2,3,1).numpy()[0]
                image = (image*255).astype(np.uint8)

            surface = pygame.surfarray.make_surface(np.transpose(image,(1,0,2)))

        
            surface = apply_micro_shader(surface)

            # --- Fractal noise shader (new micro effect) ---
            surface = apply_fractal_noise_shader(surface, intensity=0.6, scale=0.06, octaves=4)

            # alpha 
            alpha = min(max((step - start_step) / (full_step - start_step) * 0.25, 0.0), 0.25)

            if core.morphing_new_image and alpha > 0.0:
                base_arr = np.transpose(pygame.surfarray.array3d(core.morphing_new_image),(1,0,2))
                new_arr = np.transpose(pygame.surfarray.array3d(surface),(1,0,2))

                # low-res prediction mini-NN
                lr = cv2.resize(new_arr,(new_arr.shape[1]//4,new_arr.shape[0]//4))
                lr_tensor = torch.tensor(lr/255.0).permute(2,0,1).unsqueeze(0).float().to(device)
                with torch.no_grad():
                    pred = mini_nn(lr_tensor)[0].permute(1,2,0).cpu().numpy()*255.0
                pred = cv2.resize(pred,(new_arr.shape[1],new_arr.shape[0]))
                new_arr = np.clip((1-alpha)*new_arr + alpha*pred,new_arr.min(),new_arr.max()).astype(np.uint8)

                blended = cv2.addWeighted(base_arr.astype(np.float32),1.0-alpha,new_arr.astype(np.float32),alpha,0.0)
                blended = np.transpose(blended,(1,0,2))
                surface = pygame.surfarray.make_surface(blended)

            core.last_stream_surface = surface

        except Exception as e:
            logging.error(f"SD stream callback error: {e}")

    async def _run():
        
        try:
            def _gen():
                return pipe(
                    prompt=image_prompt,
                    init_image=core.morphing_new_image or init_image,
                    strength=0.4,
                    num_inference_steps=16,
                    guidance_scale=6.6,
                    callback=callback,
                    callback_steps=1
                ).images[0]

            image = await asyncio.get_event_loop().run_in_executor(executor, _gen)
            img_array = np.array(image).astype(np.uint8)
            surface = pygame.surfarray.make_surface(np.transpose(img_array, (1, 0, 2)))

            core.start_morphing(core.morphing_new_image or surface, surface)
            core.morphing_new_image = surface
            core.update_visual_features(surface)
            core.add_image_to_buffer(surface)

            # === Online learning MiniUNet step ===
            # Prepare a small image tensor from surface
            arr = pygame.surfarray.array3d(surface).astype(np.float32)
            arr = np.transpose(arr, (1, 0, 2))
            # Downscale for speed
            arr_lr = cv2.resize(arr, (arr.shape[1] // 4, arr.shape[0] // 4))
            input_tensor = torch.tensor(arr_lr / 255.0).permute(2, 0, 1).contiguous().unsqueeze(0).float().to(device)
            target_tensor = input_tensor.clone()
            # Add small noise for denoising task
            noisy_tensor = input_tensor + 0.04 * torch.randn_like(input_tensor)
            noisy_tensor = torch.clamp(noisy_tensor, 0.0, 1.0)
            # Setup optimizer and loss
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(mini_nn.parameters(), lr=1e-4)
            mini_nn.train()
            for _ in range(14):  # epochs
                optimizer.zero_grad()
                output = mini_nn(noisy_tensor)
                loss = criterion(output, target_tensor)
                loss.backward()
                optimizer.step()
            mini_nn.eval()
            # === End online learning step ===

            return surface
        finally:
            core.is_streaming = False

    core.image_task = asyncio.create_task(_run())
    return None, 0.8

async def speak(core, text):
    if not tts or not text.strip():
        return 0.0

    sentences = re.findall(r'[^.!?]+[.!?]?', text)
    channel = pygame.mixer.Channel(1)

    try:
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
                await asyncio.get_event_loop().run_in_executor(
                    executor,
                    lambda: tts.tts_to_file(
                        text=sentence,
                        file_path=fp.name,
                        split_sentences=False,
                        max_decoder_steps=20000
                    )
                )
                wav_path = fp.name

            # load wav and apply dub delay
            rate, data = scipy.io.wavfile.read(wav_path)

            if data.dtype != np.float32:
                data = data.astype(np.float32) / np.max(np.abs(data))

            delay_ms = 320
            feedback = 0.55
            wet = 0.20
            delay_samples = int(rate * delay_ms / 1000)

            delayed = np.zeros_like(data)
            for i in range(delay_samples, len(data)):
                delayed[i] = data[i - delay_samples] * feedback + delayed[i - delay_samples] * feedback

            effected = (1 - wet) * data + wet * delayed
            effected = effected / (np.max(np.abs(effected)) + 1e-8)

            scipy.io.wavfile.write(wav_path, rate, effected.astype(np.float32))
            sound = pygame.mixer.Sound(wav_path)

            # intentional pre-speech delay
            await asyncio.sleep(0.15)

            while channel.get_busy():
                await asyncio.sleep(0.02)

            channel.play(sound)
            await asyncio.sleep(sound.get_length() + 0.05)

            channel.stop()
            os.unlink(wav_path)

        return 0.8

    except asyncio.CancelledError:
        # корректная отмена (timeout / новая реплика)
        return 0.0

    except Exception as e:
        logging.error(f"TTS error: {e}")
        return 0.0


def interpret_action(action_text, core):
    action_text = action_text.lower()
    sentiment = "NEUTRAL"
    score = 0.0
    action_impact = 0.0
    focus = None

    if sentiment_analyzer:
        try:
            sentiment_result = sentiment_analyzer(action_text)[0]
            sentiment = sentiment_result["label"]
            score = sentiment_result["score"] if sentiment == "POSITIVE" else -sentiment_result["score"]
            action_impact = score * 0.3
        except Exception as e:
            logging.error(f"Ошибка анализа настроения: {e}")

    keywords = {
        "titan": {"weight": 0.5, "focus": "Megatitan"},
        "network": {"weight": 0.5, "focus": "IGI Node"},
        "colossi": {"weight": 0.5, "focus": "Echo of the Colossi"},
        "help": {"weight": 0.3, "impact": 0.2},
        "destroy": {"weight": 0.3, "impact": -0.2},
        "create": {"weight": 0.3, "impact": 0.3},
        "explore": {"weight": 0.3, "impact": 0.1},
        "solve": {"weight": 0.3, "impact": 0.2},
        "fight": {"weight": 0.3, "impact": -0.1},
        "light": {"weight": 0.2, "visual": "brightness"},
        "dark": {"weight": 0.2, "visual": "contrast"},
        "fracture": {"weight": 0.2, "impact": -0.15, "visual": "edges"},
        "pulse": {"weight": 0.2, "impact": 0.1, "visual": "brightness"},
    }
    total_weight = 0
    visual_modifiers = defaultdict(float)
    for word, data in keywords.items():
        if word in action_text:
            total_weight += data.get("weight", 0.0)
            if "focus" in data:
                focus = data["focus"]
            if "impact" in data:
                action_impact += data["impact"]
            if "visual" in data:
                visual_modifiers[data["visual"]] += data.get("weight", 0.0)

    visual = getattr(core, "visual_features", {})
    if visual:
        action_impact += (visual.get("contrast", 0) - 0.3) * 0.15
        action_impact += (visual.get("brightness", 0) - 0.5) * 0.1

        if "fracture" in action_text or "edges" in visual_modifiers:
            action_impact += (visual.get("edges", 0) - 0.2) * 0.2

    action_impact *= 1 + core.pain_level * 0.5

    action_impact = min(max(action_impact * (total_weight / 1.0 if total_weight > 0 else 1), -0.5), 0.5)
    return sentiment, score, action_impact, focus

# --- New: interpret_and_generate_signals ---
def interpret_and_generate_signals(action_text, core):
    """
    Interpret user action and generate control signals for visuals/music.
    Returns: sentiment, score, action_impact, focus, signals (dict)
    """
    sentiment, score, action_impact, focus = interpret_action(action_text, core)
    # Generate signals for visuals/music
    signals = {
        "brightness": 0.0,
        "contrast": 0.0,
        "edges": 0.0,
        "music_mood": 0.0,
        "music_energy": 0.0,
        "displacement_intensity": 0.0,
        "fractal_noise_intensity": 0.0,
        "thread_strength_factor": 1.0,
    }
    # Map keywords to signals
    text = action_text.lower()
    # Thread-visual-music mapping
    thread_map = {
        "titan": {"displacement": 0.6, "music_energy": 0.7, "music_mood": -0.4},
        "pulse": {"displacement": 0.3, "music_energy": 0.5, "music_mood": 0.3},
        "fracture": {"displacement": 0.7, "fractal_noise": 0.8, "music_energy": 0.6, "music_mood": -0.2},
        "network": {"displacement": 0.2, "fractal_noise": 0.3, "music_energy": 0.4, "music_mood": 0.1},
        "colossi": {"displacement": 0.4, "music_energy": 0.6, "music_mood": 0.0},
        "light": {"brightness": 0.25, "music_mood": 0.2},
        "dark": {"contrast": 0.25, "music_mood": -0.2},
    }
    # Accumulate thread strengths
    thread_strength = {}
    if hasattr(core, "memory") and "threads" in core.memory:
        for k, v in core.memory["threads"].items():
            if k in thread_map:
                thread_strength[k] = v
    # Find top thread(s)
    top_threads = sorted(thread_strength.items(), key=lambda x: -x[1])[:2]
    scaling_factor = 1.0
    for tk, tv in top_threads:
        tmap = thread_map.get(tk, {})
        for sig, val in tmap.items():
            if sig == "displacement":
                signals["displacement_intensity"] += val * tv
            elif sig == "fractal_noise":
                signals["fractal_noise_intensity"] += val * tv
            elif sig == "music_energy":
                signals["music_energy"] += val * tv
            elif sig == "music_mood":
                signals["music_mood"] += val * tv
            elif sig == "brightness":
                signals["brightness"] += val * tv
            elif sig == "contrast":
                signals["contrast"] += val * tv
    # Text-based cues (keep legacy)
    if "light" in text or "pulse" in text:
        signals["brightness"] += 0.2
    if "dark" in text:
        signals["contrast"] += 0.2
    if "fracture" in text or "edge" in text:
        signals["edges"] += 0.2
    # Music signals
    if sentiment == "POSITIVE":
        signals["music_mood"] += 0.5 + score * 0.5
    elif sentiment == "NEGATIVE":
        signals["music_mood"] += -0.5 + score * 0.5
    # Energy: based on action impact and keywords
    signals["music_energy"] += min(max(abs(action_impact), 0.0), 1.0)
    if "fight" in text or "destroy" in text:
        signals["music_energy"] += 0.2
    if "explore" in text or "create" in text:
        signals["music_energy"] += 0.1
    # Thread-based scaling: stronger threads amplify effects
    max_thread_strength = max([tv for _, tv in top_threads], default=0.0)
    scaling_factor = 1.0 + min(max_thread_strength, 2.0) * 0.7
    signals["thread_strength_factor"] = scaling_factor
    # Clamp all values
    for k in ["brightness", "contrast", "edges", "displacement_intensity", "fractal_noise_intensity"]:
        signals[k] = min(max(signals[k], 0.0), 2.0)
    signals["music_energy"] = min(max(signals["music_energy"], 0.0), 2.0)
    signals["music_mood"] = min(max(signals["music_mood"], -2.0), 2.0)
    return sentiment, score, action_impact, focus, signals

def generate_event(core, action_text, action_impact, focus):
    # === WORLD ACTS WITHOUT PLAYER ===
    core.world_state["titan_timer"] -= 1
    if core.world_state["titan_timer"] <= 0:
        core.world_state["titan_timer"] = random.randint(80, 140)
        core.world_state["collapse"] += 0.4
        return (
            "Megatitan",
            "The Megatitan acts first. A sector is erased.",
            "cosmic titan destroying a sector of a neural universe, brutal, cinematic"
        )
    """
    Улучшенная сюжетно-ориентированная генерация:
    действие пользователя + нити + визуальный контекст -> сюжетный ответ -> конкретное событие мира.
    """
    # === ACTION LOCKS ===
    if getattr(core.story, "arc", None) == "rupture" and "create" in action_text:
        core.world_state["locks"].add("creation")
        return (
            "System",
            "Creation is no longer possible. This path is sealed.",
            "collapsed void, broken light structures, no hope"
        )


    # --- Dynamic visual/audio synchronization based on key threads ---
    # (See main loop for actual modulation; here, just story logic.)
    core.story.advance_arc(
        action_impact,
        core.memory["mood_score"],
        threads=core.memory["threads"],
        visual=core.visual_features
    )
    core.story.next_beat(action_text, threads=core.memory["threads"])

    core.pain_level = min(1.0, (core.world_state["collapse"] + core.world_state["instability"]) / 2)


    narrative_reply = core.story.respond(
        action_text,
        core.memory["mood_score"],
        core.resonance,
        core
    )


    top_threads = sorted(core.memory["threads"].items(), key=lambda x: -x[1])
    thread_event = None
    if top_threads:
        strongest_thread = top_threads[0][0]
        thread_event = {
            "network": "A new network pattern emerges, shifting the balance.",
            "titan": "The Megatitan stirs, its presence warping reality.",
            "colossi": "Echoes of the Colossi ripple through the void.",
            "pulse": "A cosmic pulse resonates, awakening dormant codes.",
            "fracture": "A fracture spreads, splitting the neural landscape.",
            "light": "A surge of radiant light illuminates hidden paths.",
            "dark": "A shroud of darkness envelops a sector.",
        }.get(strongest_thread, None)

    story_arc = getattr(core.story, "arc", "awakening")
    world_event = thread_event or {
        "awakening": "A new node ignites within the network.",
        "convergence": "Factions draw closer as old boundaries dissolve.",
        "rupture": "A rupture occurs. One of the paths collapses.",
        "synthesis": "The world restructures into a more stable form."
    }[story_arc]


    core.memory["world_state"]["last_event"] = world_event  # только для изображения
    core.memory["world_state"]["arc"] = story_arc


    speaker = f"Pulse::{story_arc}"


    dialogue = narrative_reply  # игрок слышит только сюжетный диалог


    image_prompt = (
        f"cinematic sci-fi scene, {story_arc} arc, "
        f"{world_event}, "
        f"{' '.join([k for k, v in top_threads[:1]]) if top_threads else ''}, "
        f"emotional, high contrast, no text"
    )

    return speaker, dialogue, image_prompt

async def main():
    core = PulseCore()


    core.stop_intro = False
    intro_task = asyncio.create_task(play_intro_video(core, path="intro.mp4"))


    start_dialogue = "Welcome to the Pulsating Network. You are the Pulse, the bringer of freedom. What will you do?"
    core.set_dialogue(start_dialogue)

    start_image_prompt = "A cosmic void with golden and blue threads forming a pulsating network, radiant light at the center"
    start_image_surface = await core.pregenerate_image(start_image_prompt, core.memory["mood_score"])


    core.stop_intro = True
    await asyncio.sleep(0.1)  # дать таску завершиться
    if intro_task and not intro_task.done():
        try:
            await asyncio.wait_for(intro_task, timeout=1.0)
        except Exception:
            pass

    if start_image_surface:
        core.start_morphing(start_image_surface, start_image_surface)
        core.morphing_old_image = start_image_surface
        core.morphing_new_image = start_image_surface
        img = pygame.transform.scale(start_image_surface, (SCREEN_WIDTH, SCREEN_HEIGHT))
        screen.blit(img, (0, 0))
        pygame.display.flip()


    asyncio.create_task(speak(core, start_dialogue))
    speaker = "Pulse::awakening"

    image_surface = core.morphing_new_image
    generating = False
    hidden = None
    asyncio.create_task(play_music(core))

    input_text = ""
    dialogue_score = None

    # === TTL for image display ===
    last_image_time = pygame.time.get_ticks()
    IMAGE_TTL = 1200  # ms
    morph_finished = True

    running = True
    while running:
        current_time = pygame.time.get_ticks()
        # === AUTO CAPTURE ON IDLE OR HIGH RESONANCE ===
        if (
            core.resonance > 0.75 or
            current_time - core.last_input_time > core.idle_capture_ms
        ):
            if core.morphing_new_image is not None:
                core.capture_frame(core.morphing_new_image)
                core.last_input_time = current_time
        improved_pulsating_background(screen, core.memory["mood_score"], generating)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.display.toggle_fullscreen()

                elif event.key == pygame.K_RETURN and input_text.strip() and not generating:
                    core.last_input_time = pygame.time.get_ticks()
                    generating = True

                    sentiment, score, action_impact, focus, signals = interpret_and_generate_signals(input_text, core)
                    core.grow_network(action_impact)
                    # --- Dynamic visual/music synchronization ---
                    # Visuals
                    if hasattr(core, "visual_features"):
                        vf = core.visual_features
                        # Thread-based scaling
                        thread_factor = signals.get("thread_strength_factor", 1.0)
                        # Clamp to safe range
                        vf["brightness"] = min(max(vf.get("brightness", 0.0) + signals.get("brightness", 0.0) * thread_factor, 0.0), 1.0)
                        vf["contrast"] = min(max(vf.get("contrast", 0.0) + signals.get("contrast", 0.0) * thread_factor, 0.0), 1.0)
                        vf["edges"] = min(max(vf.get("edges", 0.0) + signals.get("edges", 0.0) * thread_factor, 0.0), 1.0)
                    # Music
                    if hasattr(core, "music_state"):
                        ms = core.music_state
                        thread_factor = signals.get("thread_strength_factor", 1.0)
                        ms["mood"] = max(min(signals.get("music_mood", 0.0) * thread_factor, 1.0), -1.0)
                        ms["energy"] = min(max(signals.get("music_energy", 0.0) * thread_factor, 0.0), 1.0)
                    # Visual displacement/fractal noise (for shaders)
                    if hasattr(core, "displacement_strength"):
                        disp_base = 10.0
                        disp_amt = signals.get("displacement_intensity", 0.0) * signals.get("thread_strength_factor", 1.0)
                        # Clamp and blend with base
                        core.displacement_strength = min(max(disp_base + disp_amt * 10.0, 2.0), 32.0)
                    if hasattr(core, "fractal_noise_intensity"):
                        # Optionally, set a core.fractal_noise_intensity if used by shaders
                        core.fractal_noise_intensity = min(max(signals.get("fractal_noise_intensity", 0.0) * signals.get("thread_strength_factor", 1.0), 0.0), 2.0)

                    # === HARD FAIL STATE ===
                    if core.world_state["collapse"] >= 1.0:
                        core.set_dialogue("The network collapses. There will be no continuation.")
                        asyncio.create_task(
                            speak(core, "The network collapses. There will be no continuation.")
                        )
                        await asyncio.sleep(3)
                        return

                    visual_vec = [
                        core.visual_features["brightness"],
                        core.visual_features["contrast"],
                        core.visual_features["edges"]
                    ]

                    input_vec = torch.tensor(
                        [[score] + visual_vec + [0] * 6],
                        dtype=torch.float32
                    ).unsqueeze(0).to(device)

                    with torch.no_grad():
                        pred, hidden = core.fractal_memory(input_vec, hidden)
                    predicted_score = pred[0, 0, 0].item()

                    qnn_input = torch.tensor(
                        [[score] + visual_vec + [0] * 6],
                        dtype=torch.float32
                    ).to(device)

                    with torch.no_grad():
                        qnn_pred = core.quantum_nn(qnn_input)
                    qnn_score = qnn_pred[0, 0].item()

                    predicted_sentiment = core.quantum_memory.predict(
                        sentiment, core.visual_features
                    )
                    core.quantum_memory.update(sentiment, predicted_sentiment, input_text)

                    core.memory["prompts"].append(input_text)
                    core.memory["sentiments"].append(sentiment)
                    core.memory["mood_score"] = min(
                        max(core.memory["mood_score"] + (predicted_score + qnn_score) * 0.1, -1.0),
                        1.0
                    )

                    try:
                        speaker, dialogue, image_prompt = generate_event(
                            core, input_text, action_impact, focus
                        )
                        core.set_dialogue(dialogue)
                    except Exception as e:
                        logging.error(f"Ошибка генерации события: {e}")
                        speaker = "Error Entity"
                        dialogue = "Something went wrong. Try again."
                        core.set_dialogue(dialogue)

                    core.record_event(input_text, action_impact, predicted_sentiment)

                    tts_text = sanitize_tts(dialogue.replace("*", ""))
                    asyncio.create_task(speak(core, tts_text))
                    text_quality = 0.8

                    _, image_quality = await generate_image(
                        core, image_prompt, core.memory["mood_score"]
                    )
                    morph_finished = False
                    audio_quality = 0.8

                    core.self_program(image_quality, audio_quality, text_quality)

                    input_text = ""
                    dialogue_score = None
                    generating = False

                elif event.key == pygame.K_BACKSPACE:
                    input_text = input_text[:-1]

                else:
                    input_text += event.unicode

        displayed_text = core.animate_text(current_time)
        current_image = image_surface  # держим последний валидный кадр

        # --- Streamed image: если есть last_stream_surface, показать его поверх image_surface ---
        if core.last_stream_surface is not None:
            image_surface = core.last_stream_surface

        if image_surface or core.morphing_old_image or core.morphing_new_image:
            img = core.apply_morphing(current_time)
            if img is not None:
                current_image = img
                image_surface = current_image
                last_image_time = current_time
            else:
                # морфинг завершён — старое изображение больше не показываем
                morph_finished = True
            current_image = improved_displacement(current_image, core.displacement_strength)
            current_image = core.apply_feedback_loop(current_image)
            current_image = core.apply_glow(current_image)
            current_image = improved_sharpen(current_image, strength=0.6)
        # --- TTL check for image_surface ---
        if morph_finished and current_time - last_image_time > IMAGE_TTL:
            current_image = None
            image_surface = None

        if current_image:
            img_width = SCREEN_WIDTH - int(SCREEN_WIDTH * 0.1)
            img_height = int(img_width / ASPECT_RATIO)
            scaled_img = pygame.transform.scale(current_image, (img_width, img_height))
            img_x = (SCREEN_WIDTH - img_width) // 2
            img_y = (SCREEN_HEIGHT - img_height) // 2
            screen.blit(scaled_img, (img_x, img_y))

        margin = SCREEN_WIDTH * 0.05
        text_y_start = SCREEN_HEIGHT * 0.6
        line_spacing = font_size * 1.2

        speaker_surface = font.render(f"{speaker}:", True, (200, 200, 255))
        screen.blit(speaker_surface, (margin, text_y_start))

        render_bold_wrapped(displayed_text, font, (255, 255, 255), screen, margin, text_y_start + line_spacing, line_spacing)

        txt_surface = font.render(">> " + input_text, True, (255, 255, 255))
        screen.blit(txt_surface, (margin, text_y_start + line_spacing * 3))

        world_status = f"Collapse: {core.world_state['collapse']:.1f} | Instability: {core.world_state['instability']:.1f} | Locks: {len(core.world_state['locks'])}"
        world_surface = font.render(world_status, True, (255, 150, 150) if core.world_state["collapse"] > 0.5 else (200, 200, 255))
        screen.blit(world_surface, (margin, text_y_start + line_spacing * 4))
        
        status = f"World: {core.current_world} | Nodes: {core.network_nodes} | Resonance: {core.resonance:.2f} | Titan: {core.titan_relation:.2f}"
        status_surface = font.render(status, True, (200, 200, 255))
        screen.blit(status_surface, (margin, text_y_start + line_spacing * 5))

        if core.conflicts:
            conflict_surface = font.render(f"Conflicts: {', '.join(core.conflicts)}", True, (255, 100, 100))
            screen.blit(conflict_surface, (margin, text_y_start + line_spacing * 6))
        if core.allies:
            ally_surface = font.render(f"Allies: {', '.join(core.allies)}", True, (100, 255, 100))
            screen.blit(ally_surface, (SCREEN_WIDTH * 0.5, text_y_start + line_spacing * 6))

        if dialogue_score is not None:
            score_surface = font.render(f"Dialogue Score: {dialogue_score:.1f}", True, (255, 255, 100))
            screen.blit(score_surface, (margin, text_y_start + line_spacing * 7))

        # === Fade-in mask с градиентом только для текущей строки ===
        if len(core.current_dialogue) > 0 and core.text_animation_index < len(core.current_dialogue):

            max_width = SCREEN_WIDTH - int(SCREEN_WIDTH * 0.1)

            words = core.current_dialogue.split(" ")
            lines = []
            current_line = ""

            for word in words:
                test_line = current_line + (" " if current_line else "") + word
                if font.size(test_line)[0] <= max_width:
                    current_line = test_line
                else:
                    lines.append(current_line)
                    current_line = word

            if current_line:
                lines.append(current_line)

            chars_counted = 0
            current_line = ""
            line_start_y = text_y_start
            line_chars_index = 0

            for line_idx, line in enumerate(lines):
                if core.text_animation_index <= chars_counted + len(line):
                    current_line = line
                    line_start_y = text_y_start + line_spacing * (line_idx + 1)
                    line_chars_index = core.text_animation_index - chars_counted
                    break
                chars_counted += len(line)

        pygame.display.flip()
        clock.tick(30)
        await asyncio.sleep(0)

    for f in core.audio_cache.values():
        if os.path.exists(f):
            try:
                os.unlink(f)
            except Exception as e:
                logging.error(f"Ошибка удаления файла: {e}")
    executor.shutdown(wait=True)
    pygame.mixer.quit()
    pygame.quit()

if __name__ == "__main__":
    asyncio.run(main())
