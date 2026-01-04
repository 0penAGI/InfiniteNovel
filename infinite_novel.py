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
# --- Рендер текста с поддержкой *bold* и динамическим переносом по ширине экрана ---
import cv2
import scipy.signal
import scipy.io.wavfile
import requests

#infinite_novel.py


# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Инициализация Pygame
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2)

# Соотношение сторон 2.35:1 и разрешение
ASPECT_RATIO = 4.2
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = int(SCREEN_WIDTH / ASPECT_RATIO)

# Проверка совместимости разрешения
display_info = pygame.display.Info()
max_width = display_info.current_w
if max_width < SCREEN_WIDTH:
    SCREEN_WIDTH = max_width
    SCREEN_HEIGHT = int(SCREEN_WIDTH / ASPECT_RATIO)

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN | pygame.SCALED)
pygame.display.set_caption("Infinite Novel")

# Динамическая настройка шрифта
font_size = int(SCREEN_HEIGHT / 25)
font = pygame.font.SysFont("arial", font_size)
clock = pygame.time.Clock()

# Определение устройства
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
logging.info(f"Используется устройство: {device}")

# Глобальный пул потоков
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

# Анализ настроения
try:
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device if device != "mps" else -1)
except Exception as e:
    logging.error(f"Ошибка загрузки sentiment-analyzer: {e}")
    sentiment_analyzer = None

def render_bold_wrapped(text, font, color, screen, x, y, line_spacing, max_width=None):
    """
    Рендер текста с *bold* выделением, перенос по ширине экрана.
    """
    pattern = r"\*(.*?)\*"
    parts = re.split(pattern, text)
    bold = False
    current_x = x
    current_y = y
    if max_width is None:
        max_width = screen.get_width() - x * 2
    for part in parts:
        if not part:
            bold = not bold
            continue
        f = pygame.font.SysFont("arial", font.get_height(), bold=bold)
        words = part.split(" ")
        for word in words:
            word_surface = f.render(word + " ", True, color)
            word_width = word_surface.get_width()
            # Если не помещается по ширине, переносим на новую строку
            if current_x + word_width > x + max_width:
                current_x = x
                current_y += line_spacing
            screen.blit(word_surface, (current_x, current_y))
            current_x += word_width
        bold = not bold

# Квантовая нейронная сеть
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

# Фрактальная память
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

# Квантовые кристаллы
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

# --- Gemma3:4b генерация через Ollama ---
def gemma3_generate(prompt: str, history=None, model="gemma3:1b"):
    """
    Вызывает локальный Ollama сервер, чтобы получить ответ от Gemma 3:1b.
    Склеивает все токены в один текст и возвращает результат.
    """
    import json
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": []
    }
    if history:
        payload["messages"].extend(history)
    payload["messages"].append({"role": "user", "content": prompt})
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        lines = response.text.strip().split("\n")
        full_text = ""
        for line in lines:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if "message" in data and "content" in data["message"]:
                    full_text += data["message"]["content"]
            except Exception:
                continue
        return full_text.strip()
    except Exception as e:
        logging.error(f"Gemma3 request failed: {e}")
        return ""


# --- Новый класс StoryDirector с памятью Gemma3 и профилем игрока ---
class StoryDirector:
    def __init__(self):
        self.arc = "awakening"
        self.beat = 0
        self.flags = set()
        self.history = []
        self.thread_influence = defaultdict(float)
        self.last_visual_context = {}
        self.conversation_memory = []  # последние реплики игрока и системы
        self.player_profile = []       # 30-50 реплик игрока для формирования «пульса»

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

        text = gemma3_generate(context, history=self.conversation_memory + self.player_profile)
        self.conversation_memory.append({"role": "assistant", "content": text})

        return text.strip()



# Ядро движка
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
        self.char_delay = 50  # мс на символ
        self.image_buffer = []
        self.max_buffer_size = 5
        self.feedback_opacity = 0.2
        self.displacement_strength = 10.0
        self.visual_features = {"brightness": 0.0, "contrast": 0.0, "edges": 0.0}
        self.morphing_state = {"active": False, "start_time": 0, "duration": 7000}
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
        
        # === WORLD FAILURE STATE (GAME LAYER) ===
        self.world_state = {
            "collapse": 0.0,
            "instability": 0.0,
            "locks": set(),
            "titan_timer": 120
        }

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
        alpha = min(elapsed / self.morphing_state["duration"], 1.0)
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
                prompt, num_inference_steps=10, guidance_scale=5.0
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

# Дисплейсмент-эффект
def apply_displacement(image_surface, strength=10.0):
    img_array = pygame.surfarray.array3d(image_surface)
    img_array = np.transpose(img_array, (1, 0, 2))
    height, width = img_array.shape[:2]
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    time = pygame.time.get_ticks() / 100.0
    displacement_x = np.sin(x * 1 + time) * np.cos(y * 1 + time)
    displacement_y = np.cos(x * 1 + time) * np.sin(y * 1 + time)
    displacement_x = cv2.GaussianBlur(displacement_x, (9, 9), 2) * strength
    displacement_y = cv2.GaussianBlur(displacement_y, (9, 9), 2) * strength
    
    map_x = (x + displacement_x).astype(np.float32)
    map_y = (y + displacement_y).astype(np.float32)
    displaced = cv2.remap(img_array, map_x, map_y, cv2.INTER_LINEAR)
    
    logging.info(f"Displaced array shape: {displaced.shape}")
    if len(displaced.shape) != 3:
        logging.error(f"Unexpected array shape: {displaced.shape}")
        return image_surface
    
    displaced = np.transpose(displaced, (1, 0, 2))
    return pygame.surfarray.make_surface(displaced)

# Эффект резкости (шарпен)
def apply_sharpen(image_surface, strength=1.0):
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

# Пульсирующий фон
def draw_pulsating_background(screen, mood_score, generating):
    pulse = np.sin(pygame.time.get_ticks() / 2000.0) * 0.5 + 0.5
    intensity = 20 + pulse * 10 if generating else 10 + pulse * 5
    color = (
        int(20 + mood_score * 20),
        int(20 + mood_score * 20),
        int(30 + intensity)
    )
    screen.fill(color)

# Музыка
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
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    act = 2 if story_progress > 100 else 1
    if act == 1 and story_progress > (100 - act_transition_steps * 10):
        transition_progress = (story_progress - (100 - act_transition_steps * 10)) / (act_transition_steps * 10)
    else:
        transition_progress = 0 if act == 1 else 1

    context_intensity = abs(mood_score) + resonance + abs(titan_relation)
    context_intensity = min(context_intensity, 2.0)
    is_aggressive = any(word in action_text.lower() for word in ["fight", "destroy", "battle"])
    is_explorative = any(word in action_text.lower() for word in ["explore", "create", "discover"])

    base_freq = (
        110 if mood_score < -0.3 else
        220 if mood_score > 0.3 else
        165
    )
    if is_aggressive:
        base_freq *= 1.2
    elif is_explorative:
        base_freq *= 0.9

    if act == 1 and transition_progress == 0:
        freqs = [base_freq, base_freq * 1.5, base_freq * 2]
        amplitudes = [1.0, 0.6, 0.3]
        texture = np.sin(2 * np.pi * 0.1 * t) * 0.2
    elif act == 2 and transition_progress == 1:
        freqs = [base_freq, base_freq * 1.3, base_freq * 1.8, base_freq * 2.5]
        amplitudes = [1.0, 0.8, 0.5, 0.3]
        texture = np.sin(2 * np.pi * 0.5 * t) * 0.3 + np.random.normal(0, 0.05, t.shape)
    else:
        freqs = [
            base_freq * (1 + 0.3 * transition_progress),
            base_freq * (1.5 + 0.3 * transition_progress),
            base_freq * (2 + 0.5 * transition_progress)
        ]
        amplitudes = [
            1.0,
            0.6 + 0.2 * transition_progress,
            0.3 + 0.2 * transition_progress
        ]
        texture = np.sin(2 * np.pi * (0.1 + 0.4 * transition_progress) * t) * (0.2 + 0.1 * transition_progress)

    freq_spectrum = np.zeros(int(sample_rate * duration // 2 + 1), dtype=np.complex128)
    for freq, amp in zip(freqs, amplitudes):
        bin_idx = int(freq * duration)
        if bin_idx < len(freq_spectrum):
            freq_spectrum[bin_idx] += amp * context_intensity

    noise = np.random.normal(0, 0.05 * context_intensity, freq_spectrum.shape)
    noise = scipy.signal.savgol_filter(noise, 51, 3)
    freq_spectrum += noise * (0.1 + transition_progress * 0.2)

    waveform = np.fft.irfft(freq_spectrum, n=int(sample_rate * duration))
    waveform = waveform[:int(sample_rate * duration)]

    waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
    waveform *= texture
    waveform *= 0.8

    if transition_progress > 0.5:
        waveform = scipy.signal.lfilter([1, -0.5], [1], waveform)

    return waveform.astype(np.float32)

async def play_music(core, mood_score=0.0, action_text="", story_progress=0, resonance=0.5, titan_relation=0.0):
    """
    Асинхронный бесконечный музыкальный генератор с паузами, дилеем и ревером.
    WAV удаляются сразу после проигрывания.
    """
    try:
        sample_rate = 44100
        channel = pygame.mixer.Channel(0)

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

            delay_samples = int(0.15 * sample_rate)
            reverb_samples = int(0.25 * sample_rate)
            decay = 0.6 + 0.3 * mood_score
            wet = 0.5

            delayed_audio = np.zeros_like(audio)
            reverb_audio = np.zeros_like(audio)

            for i in range(len(audio)):
                if i >= delay_samples:
                    delayed_audio[i] = audio[i - delay_samples] * decay
                if i >= reverb_samples:
                    reverb_audio[i] = audio[i - reverb_samples] * 0.35

            audio = (1 - wet) * audio + wet * delayed_audio + reverb_audio
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            audio *= 0.8

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
                scipy.io.wavfile.write(fp.name, sample_rate, audio.astype(np.float32))
                segment_path = fp.name

            sound = pygame.mixer.Sound(segment_path)
            channel.play(sound)

            elapsed = 0.0
            dt = 0.1
            while elapsed < segment_duration:
                await asyncio.sleep(dt)
                elapsed += dt

            channel.stop()
            os.unlink(segment_path)

            # --- пауза ---
            pause_duration = np.random.uniform(1.0, 3.0)
            pause_audio = np.zeros(int(sample_rate * pause_duration), dtype=np.float32)

            reverb_samples_pause = int(0.3 * sample_rate)
            for i in range(len(pause_audio)):
                if i >= reverb_samples_pause:
                    pause_audio[i] += pause_audio[i - reverb_samples_pause] * 0.25

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
                scipy.io.wavfile.write(fp.name, sample_rate, pause_audio.astype(np.float32))
                pause_path = fp.name

            pause_sound = pygame.mixer.Sound(pause_path)
            channel.play(pause_sound)
            await asyncio.sleep(pause_duration)

            channel.stop()
            os.unlink(pause_path)

    except Exception as e:
        logging.error(f"Ошибка воспроизведения музыки: {e}")
        return 0.0

# Изображение с микро-шейдером и low-res предсказанием
async def generate_image(core, prompt, mood_score=0.0):
    if not pipe:
        return None, 0.0

    if core.image_task and not core.image_task.done():
        return None, 0.0  # уже генерируется

    last_event = core.world_state.get("last_event", "")
    arc = getattr(core.story, "arc", "awakening")
    image_prompt = f"cinematic sci-fi scene, {arc} arc, {last_event}, emotional, high contrast, no text"

    init_image = core.morphing_new_image
    if init_image:
        arr = pygame.surfarray.array3d(init_image)
        arr = np.transpose(arr, (1, 0, 2))
        init_image = Image.fromarray(arr)

    core.is_streaming = True

    # Мини-NN для low-res предсказания
    class MiniUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
            self.conv2 = nn.Conv2d(8, 3, 3, padding=1)
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.sigmoid(self.conv2(x))
            return x

    mini_nn = MiniUNet().to(device)

    def apply_micro_shader(surface):
        arr = pygame.surfarray.array3d(surface).astype(np.float32)
        arr = np.transpose(arr, (1,0,2))
        h, w, _ = arr.shape
        t = pygame.time.get_ticks() / 500.0
        warp_x = np.sin(np.linspace(0, np.pi*4, w) + t) * 2
        warp_y = np.cos(np.linspace(0, np.pi*4, h) + t) * 2
        map_x, map_y = np.meshgrid(np.arange(w)+warp_x, np.arange(h)+warp_y)
        displaced = cv2.remap(arr, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR)
        displaced = np.clip(displaced,0,255).astype(np.uint8)
        displaced = np.transpose(displaced,(1,0,2))
        return pygame.surfarray.make_surface(displaced)

    def callback(step, timestep, latents):
        try:
            start_step = 8
            full_step = 16
            if step < start_step:
                return

            with torch.no_grad():
                latents_ = latents / 0.18215
                image = pipe.vae.decode(latents_).sample
                image = (image / 2 + 0.5).clamp(0,1)
                image = image.cpu().permute(0,2,3,1).numpy()[0]
                image = (image*255).astype(np.uint8)

            surface = pygame.surfarray.make_surface(np.transpose(image,(1,0,2)))

            # микро-шейдер поверх
            surface = apply_micro_shader(surface)

            # alpha растягивается от step 8 до 16
            alpha = min(max((step-start_step)/(full_step-start_step)*0.35,0),0.35)

            if core.morphing_new_image and alpha>0.0:
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
                if init_image is not None:
                    return pipe(
                        prompt=image_prompt,
                        init_image=init_image,
                        strength=0.55,
                        num_inference_steps=18,
                        guidance_scale=7.0,
                        callback=callback,
                        callback_steps=1
                    ).images[0]
                else:
                    return pipe(
                        prompt=image_prompt,
                        num_inference_steps=18,
                        guidance_scale=7.0,
                        callback=callback,
                        callback_steps=1
                    ).images[0]

            image = await asyncio.get_event_loop().run_in_executor(executor,_gen)
            img_array = np.array(image).astype(np.uint8)
            surface = pygame.surfarray.make_surface(np.transpose(img_array,(1,0,2)))

            core.start_morphing(core.morphing_new_image or surface, surface)
            core.morphing_new_image = surface
            core.update_visual_features(surface)
            core.add_image_to_buffer(surface)
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

            sound = pygame.mixer.Sound(wav_path)

            while channel.get_busy():
                await asyncio.sleep(0.02)

            channel.play(sound)
            await asyncio.sleep(sound.get_length() + 0.05)

            channel.stop()
            os.unlink(wav_path)

        return 0.8

    except Exception as e:
        logging.error(f"TTS error: {e}")
        return 0.0

# Улучшенная интерпретация действия с учетом визуальных модификаторов и ключевых слов
def interpret_action(action_text, core):
    action_text = action_text.lower()
    sentiment = "NEUTRAL"
    score = 0.0
    action_impact = 0.0
    focus = None

    # Анализируем настроение
    if sentiment_analyzer:
        try:
            sentiment_result = sentiment_analyzer(action_text)[0]
            sentiment = sentiment_result["label"]
            score = sentiment_result["score"] if sentiment == "POSITIVE" else -sentiment_result["score"]
            action_impact = score * 0.3
        except Exception as e:
            logging.error(f"Ошибка анализа настроения: {e}")

    # Ключевые слова и визуальные модификаторы
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

    # Усиливаем влияние визуальных особенностей мира
    visual = getattr(core, "visual_features", {})
    if visual:
        action_impact += (visual.get("contrast", 0) - 0.3) * 0.15
        action_impact += (visual.get("brightness", 0) - 0.5) * 0.1
        # edges усиливают эффект разрушения
        if "fracture" in action_text or "edges" in visual_modifiers:
            action_impact += (visual.get("edges", 0) - 0.2) * 0.2

    action_impact = min(max(action_impact * (total_weight / 1.0 if total_weight > 0 else 1), -0.5), 0.5)
    return sentiment, score, action_impact, focus

# Улучшенная генерация событий с усиленной связью нитей, визуального контекста и событий
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

    # Обновляем сюжетную дугу с учетом нитей и визуального контекста
    core.story.advance_arc(
        action_impact,
        core.memory["mood_score"],
        threads=core.memory["threads"],
        visual=core.visual_features
    )
    core.story.next_beat(action_text, threads=core.memory["threads"])

    # Сюжетная реплика (человеческая, не абстрактная), теперь с усиленной связью нитей и визуального контекста
    narrative_reply = core.story.respond(
        action_text,
        core.memory["mood_score"],
        core.resonance,
        core
    )

    # Конкретное событие мира, теперь с учетом самой сильной сюжетной нити
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
    # Если нет подходящей сюжетной нити, используем стандартное событие дуги
    story_arc = getattr(core.story, "arc", "awakening")
    world_event = thread_event or {
        "awakening": "A new node ignites within the network.",
        "convergence": "Factions draw closer as old boundaries dissolve.",
        "rupture": "A rupture occurs. One of the paths collapses.",
        "synthesis": "The world restructures into a more stable form."
    }[story_arc]

    # Фиксируем изменения мира
    core.memory["world_state"]["last_event"] = world_event  # только для изображения
    core.memory["world_state"]["arc"] = story_arc

    # Спикер теперь сюжетный
    speaker = f"Pulse::{story_arc}"

    # Итоговый диалог — только сюжетная реплика, без world_event и без системных строк
    dialogue = narrative_reply  # игрок слышит только сюжетный диалог

    # Промпт изображения теперь отражает событие и главную сюжетную нить
    image_prompt = (
        f"cinematic sci-fi scene, {story_arc} arc, "
        f"{world_event}, "
        f"{' '.join([k for k, v in top_threads[:1]]) if top_threads else ''}, "
        f"emotional, high contrast, no text"
    )

    return speaker, dialogue, image_prompt

# Основной цикл
# Основной цикл
async def main():
    core = PulseCore()
    
    # --- стартовая фраза и изображение ---
    start_dialogue = "Welcome to the Pulsating Network. You are the Pulse, the bringer of freedom. What will you do?"
    core.set_dialogue(start_dialogue)
    
    start_image_prompt = "A cosmic void with golden and blue threads forming a pulsating network, radiant light at the center"
    start_image_surface = await core.pregenerate_image(start_image_prompt, core.memory["mood_score"])
    if start_image_surface:
        core.start_morphing(start_image_surface, start_image_surface)
        core.morphing_old_image = start_image_surface
        core.morphing_new_image = start_image_surface
        img = pygame.transform.scale(start_image_surface, (SCREEN_WIDTH, SCREEN_HEIGHT))
        screen.blit(img, (0, 0))
        pygame.display.flip()
    
    # ❗ НЕ БЛОКИРУЕМ LOOP
    asyncio.create_task(speak(core, start_dialogue))
    speaker = "Pulse::awakening"
    
    image_surface = core.morphing_new_image
    generating = False
    hidden = None
    asyncio.create_task(play_music(core))

    input_text = ""
    dialogue_score = None

    running = True
    while running:
        current_time = pygame.time.get_ticks()
        draw_pulsating_background(screen, core.memory["mood_score"], generating)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.display.toggle_fullscreen()

                elif event.key == pygame.K_RETURN and input_text.strip() and not generating:
                    generating = True

                    sentiment, score, action_impact, focus = interpret_action(input_text, core)
                    core.grow_network(action_impact)

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

                    # ❗ TTS ПАРАЛЛЕЛЬНО
                    tts_text = dialogue.replace("*", "")
                    asyncio.create_task(speak(core, tts_text))
                    text_quality = 0.8

                    _, image_quality = await generate_image(
                        core, image_prompt, core.memory["mood_score"]
                    )
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

        # дальше код рендера — БЕЗ ИЗМЕНЕНИЙ

        # Показываем только морфинг и пост-обработку, не отображая напрямую кадры генерации SD
        if image_surface or core.morphing_old_image or core.morphing_new_image:
            img = core.apply_morphing(current_time)
            if img is not None:
                current_image = img
            current_image = apply_displacement(current_image, core.displacement_strength)
            current_image = core.apply_feedback_loop(current_image)
            current_image = core.apply_glow(current_image)
            current_image = apply_sharpen(current_image, strength=1.0)
        if current_image:
            image_surface = current_image
        else:
            current_image = None

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

        # --- Используем render_bold_wrapped для текста с *bold* ---
        render_bold_wrapped(displayed_text, font, (255, 255, 255), screen, margin, text_y_start + line_spacing, line_spacing)

        txt_surface = font.render(">> " + input_text, True, (255, 255, 255))
        screen.blit(txt_surface, (margin, text_y_start + line_spacing * 3))

        # Добавляем отображение состояния мира
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
