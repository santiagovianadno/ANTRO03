import sys
import os
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple, List, Dict
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd

# Pygame must be imported before OpenGL.GL on some platforms
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, FULLSCREEN, QUIT, KEYDOWN, VIDEORESIZE

import cv2
import mediapipe as mp
from OpenGL.GL import *
from OpenGL.GLU import *

try:
    from plyfile import PlyData
except ImportError:
    PlyData = None  # We handle this later

# =============================================================================
# AUDIO ENGINE ================================================================
# =============================================================================


class AudioEngine:
    """Real-time generative audio: cross-fade between crackling and water flow."""

    def __init__(self, sample_rate: int = 44100, buffer_size: int = 1024, volume: float = 0.15):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.volume = volume

        # State exposed to the host program (0 → solo agua, 1 → solo crujidos)
        self.crackle_level = 0.0

        # Internal filter memory for pinkish water noise
        self._water_prev = 0.0
        # Low-frequency oscillator phase for slow dynamics
        self._lfo_phase = 0.0
        self._lfo_freq = 0.4  # Hz
        # Keep track of total samples for phase accuracy
        self._samples_processed = 0

        try:
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=2,
                blocksize=self.buffer_size,
                dtype="float32",
                callback=self._callback,
            )
            self._running = False
        except Exception as e:
            print("[AudioEngine] Failed to init stream:", e)
            self._stream = None
            self._running = False

    # ---------------- PUBLIC API ----------------
    def start(self):
        if self._stream and not self._running:
            self._stream.start()
            self._running = True

    def stop(self):
        if self._stream and self._running:
            self._stream.stop()
            self._running = False

    def update(self, crackle_level: float):
        """Set current crackle intensity (0-1). Water intensity is 1-level."""
        self.crackle_level = float(max(0.0, min(1.0, crackle_level)))

    # ---------------- CALLBACK ------------------
    def _callback(self, outdata, frames, _time, status):
        if status:
            # Underrun/overflow messages
            print("Audio status:", status)

        # Generate white noise base
        white = np.random.uniform(-1.0, 1.0, frames).astype(np.float32)

        # --- Water texture: low-pass filtered noise (smooth) ---
        alpha = 0.02  # low-pass coefficient
        # Exponential low-pass filter
        water = np.empty(frames, dtype=np.float32)
        prev = self._water_prev
        for i in range(frames):
            prev = (1 - alpha) * prev + alpha * white[i]
            water[i] = prev
        self._water_prev = prev

        # --- Crackle texture: sparse random impulses ---
        crackle = np.zeros(frames, dtype=np.float32)
        # Density of impulses scales non-linearly for better control
        prob = 0.001 + 0.02 * (self.crackle_level ** 1.5)
        mask = np.random.rand(frames) < prob
        crackle[mask] = np.random.uniform(-1.0, 1.0, mask.sum())

        # Exponential decay to give body to clicks (deeper crack)
        decay = np.exp(-np.linspace(0, 3, frames))
        crackle = np.convolve(crackle, decay, mode="same")

        # Gentle low-pass (moving average) to soften high frequencies
        crackle = (crackle + np.roll(crackle, 1) + np.roll(crackle, 2)) / 3.0

        # Slow amplitude modulation for liveliness (LFO)
        phase = self._lfo_phase
        angles = phase + 2 * np.pi * self._lfo_freq * np.arange(frames) / self.sample_rate
        lfo = 0.6 + 0.4 * np.sin(angles)  # 0.2 range around 1.0
        self._lfo_phase = (phase + 2 * np.pi * self._lfo_freq * frames / self.sample_rate) % (2 * np.pi)

        crackle *= lfo

        # Mix
        mix_crackle = 0.5 * self.crackle_level * crackle  # reduce harshness
        mix_water = (1.0 - self.crackle_level) * water
        signal = (mix_crackle + mix_water) * self.volume

        # Output stereo
        outdata[:, 0] = signal
        outdata[:, 1] = signal

if TYPE_CHECKING:
    from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList

###############################################################################
# CONFIGURATION ################################################################
###############################################################################

WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080
FOVY = 45.0  # Field of view in the y direction
Z_NEAR = 0.1
Z_FAR = 1000.0

# Auto-rotation speed in degrees per frame
AUTO_ROT_SPEED = 0.2

# Camera distance limits (these are OpenGL units)
MIN_CAM_DIST = 5.0
MAX_CAM_DIST = 25.0

# Pinch distance thresholds in meters (MediaPipe is normalized ~[0,1] in image space)
PINCH_DIST_MIN = 0.02
PINCH_DIST_MAX = 0.20

# Pinch distance (in normalized image space) that ya cuenta como "máximo" para la interacción
PINCH_DIST_FULL = 0.08  # más permisivo que el mínimo absoluto

# Gesture detection parameters
PALM_TOUCH_THRESHOLD = 0.10  # distance in normalized image space between right index tip and left palm to toggle mode
STATE_COOLDOWN = 1.0  # seconds between state switches

# Derived constant for lerp smoothing (zoom)
ZOOM_LERP_T = 0.4

# Plant settings
PLANT_COUNT = 15
PLANT_RADIUS = 3.0   # distance from center in model space units
PLANT_BASE_SCALE = 0.3
PLANT_SHRINK_RATE = 0.4  # units per second when shrinking

# Biodiversity (animales/plantas peques) settings
BIO_COUNT = 300
BIO_RADIUS = 10.0
BIO_COLORS = [(0.0, 0.8, 0.0), (0.9, 0.9, 0.0), (1.0, 0.5, 0.0)]  # verde, amarillo, naranja
BIO_Y = -3.2  # posición vertical de la biodiversidad (más baja que -1.0)

# River settings
RIVER_RADIUS = BIO_RADIUS + 2.0  # radio máximo del río
RIVER_SEGMENTS = 120
RIVER_SPEED_DEG = 30.0  # grados por segundo para animar flujo
RIVER_COLOR = (0.0, 0.4, 1.0)
RIVER_BASE_WIDTH = 40  # grosor inicial del río (más grueso)
RIVER_ALPHA = 0.25  # opacidad reducida (0 = transparente, 1 = opaco)
RIVER_WIDTH_MIN_FACTOR = 0.2  # grosor final 20% del inicial
RIVER_NOISE_FREQ = 8.0  # mayor frecuencia de ondulación
RIVER_NOISE_AMPL = 1.0  # mayor amplitud de ondulación
RIVER_NOISE_SPEED = 1.0  # rad/s
RIVER_HEIGHT_AMPL = 0.5  # amplitud vertical del río
RIVER_HEIGHT_FREQ = 3.0  # frecuencia vertical

# Cracked ground effect settings
CRACK_COUNT = 120
CRACK_LENGTH = 1.5
CRACK_FREQUENCY = 3.0  # Hz de parpadeo
CRACK_THRESHOLD = 0.4  # Umbral para encendido
EFFECT_COLOR = (0.6, 0.05, 0.05)  # mismo rojo oscuro

# Effect parameters
FLOW_DURATION = 3.0  # seconds that the red "liquid" takes to recorrer de raíces a hojas

# Misc
DEFAULT_MODEL_PATH = os.environ.get("PLY_MODEL", "Espino.ply")

# Mirror (flip) camera feed horizontally so that the on-screen image acts like a mirror
MIRROR_CAMERA = True

###############################################################################
# HELPER FUNCTIONS #############################################################
###############################################################################

def lerp(a: float, b: float, t: float) -> float:
    """Linearly interpolate between a and b by t."""
    return a + (b - a) * t


def normalize(v: np.ndarray) -> np.ndarray:
    """Return a normalized copy of v or zero vector if norm is 0."""
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n

###############################################################################
# PLY LOADING ##################################################################
###############################################################################

@dataclass
class PlyModel:
    vertices: np.ndarray  # shape (N, 3)
    normals: np.ndarray   # shape (N, 3) or None
    colors: np.ndarray    # shape (N, 3) or None
    center: np.ndarray = field(init=False)
    scale: float = field(init=False)

    def __post_init__(self):
        # Center and scale the model so it fits roughly within [-1,1]^3
        mins = self.vertices.min(axis=0)
        maxs = self.vertices.max(axis=0)
        self.center = (mins + maxs) / 2.0
        bbox_size = np.linalg.norm(maxs - mins)
        self.scale = 2.0 / bbox_size if bbox_size > 0 else 1.0
        # Apply normalization in-place
        self.vertices = (self.vertices - self.center) * self.scale

        # Pre-compute normalized heights (0 = raíces, 1 = hojas) para efectos
        ys = self.vertices[:, 1]
        self.y_norm = (ys - ys.min()) / (ys.max() - ys.min() + 1e-6)

        # Asignar colores de biodiversidad si el PLY no los trae
        if self.colors is None:
            palette = np.array(BIO_COLORS, dtype=np.float32)
            idx = np.random.randint(0, len(palette), size=len(self.vertices))
            self.colors = palette[idx]

        # Guardar copia para efectos (flujo, etc.)
        self.base_colors = self.colors.copy()

    @classmethod
    def load(cls, path: str) -> "PlyModel":
        if PlyData is None:
            raise RuntimeError("plyfile is not installed. Please add it to requirements.txt")
        ply = PlyData.read(path)
        vertex_data = ply["vertex"]
        vertices = np.vstack((vertex_data["x"], vertex_data["y"], vertex_data["z"])).T.astype(np.float32)
        normals = None
        if {"nx", "ny", "nz"}.issubset(vertex_data.properties):
            normals = np.vstack((vertex_data["nx"], vertex_data["ny"], vertex_data["nz"]).T).astype(np.float32)
        colors = None
        if {"red", "green", "blue"}.issubset(vertex_data.properties):
            colors = np.vstack((vertex_data["red"], vertex_data["green"], vertex_data["blue"]).T).astype(np.float32) / 255.0
        return cls(vertices, normals, colors)

    def draw(self):
        """Render the model as GL_POINTS; extend to triangles if faces present."""
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointerf(self.vertices)
        if self.colors is not None:
            glEnableClientState(GL_COLOR_ARRAY)
            glColorPointerf(self.colors)
        glPointSize(1.0)
        glDrawArrays(GL_POINTS, 0, len(self.vertices))
        glDisableClientState(GL_VERTEX_ARRAY)
        if self.colors is not None:
            glDisableClientState(GL_COLOR_ARRAY)

    # ------------- EFFECT: FLOWING RED -----------------
    def _norm_from_dir(self, flow_dir: np.ndarray) -> np.ndarray:
        """Return normalized projection of vertices onto flow_dir (0→inicio,1→fin)."""
        d = flow_dir / (np.linalg.norm(flow_dir) + 1e-9)
        proj = self.vertices @ d  # scalar projection
        p_min = proj.min()
        p_max = proj.max()
        return (proj - p_min) / (p_max - p_min + 1e-9)

    def draw_flow(self, progress: float, flow_dir: Optional[np.ndarray] = None):
        """Render progression-based coloration along given direction (default usar eje Y)."""
        # Compute normalized values depending on chosen direction
        if flow_dir is None:
            norm_vals = self.y_norm
        else:
            norm_vals = self._norm_from_dir(flow_dir)

        # Clamp progress
        p = max(0.0, min(1.0, progress))

        # Colorize
        colors = self.base_colors.copy()
        mask = norm_vals <= p
        colors[mask] = [0.6, 0.05, 0.05]

        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointerf(self.vertices)
        glEnableClientState(GL_COLOR_ARRAY)
        glColorPointerf(colors)
        glPointSize(3.0)
        glDrawArrays(GL_POINTS, 0, len(self.vertices))
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

###############################################################################
# ENUMERATIONS #################################################################
###############################################################################

class ViewerState(Enum):
    NEUTRAL = auto()
    EFFECT_1 = auto()
    EFFECT_2 = auto()

###############################################################################
# VIEWER CLASS #################################################################
###############################################################################

class Viewer:
    """Encapsulates camera, state, and rendering logic."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        print(f"Loading model: {model_path}")
        self.model = PlyModel.load(model_path)

        # Camera parameters
        self.cam_distance_target = 15.0  # Desired distance (smoothed)
        self.cam_distance = self.cam_distance_target
        self.cam_rot_x = 0.0  # degrees
        self.cam_rot_y = 0.0  # degrees

        # Previous right hand center for rotation deltas
        self.prev_right_avg: Optional[Tuple[float, float]] = None

        # Flow intensity driven por pinch (0-1)
        self.effect1_progress: float = 0.0

        # Generate biodiversity points around base
        # Generate random positions uniformly inside a circle (radio BIO_RADIUS)
        self.bio_positions: List[Tuple[float, float]] = [
            (
                math.cos(theta) * r,
                math.sin(theta) * r,
            )
            for theta, r in (
                (random.uniform(0, 2 * math.pi), math.sqrt(random.uniform(0, 1)) * BIO_RADIUS)
                for _ in range(BIO_COUNT)
            )
        ]
        self.bio_scale: float = 1.0  # escala (1 visible, 0 invisible)

        # Generate cracked ground segments (líneas) alrededor del árbol
        self.cracks: List[Tuple[float, float, float, float, float]] = []  # x1,z1,x2,z2,phase
        for _ in range(CRACK_COUNT):
            theta = random.uniform(0, 2 * math.pi)
            r = random.uniform(0, BIO_RADIUS)
            x1 = math.cos(theta) * r
            z1 = math.sin(theta) * r
            angle = random.uniform(0, 2 * math.pi)
            x2 = x1 + math.cos(angle) * CRACK_LENGTH
            z2 = z1 + math.sin(angle) * CRACK_LENGTH
            phase = random.uniform(0, 2 * math.pi)
            self.cracks.append((x1, z1, x2, z2, phase))

        # River animation phase (degrees)
        self.river_phase: float = 0.0

        # Model transform parameters (for manual rotation / scaling)
        self.model_scale: float = 8.0
        self.model_rot_x: float = 270.0
        self.model_rot_y: float = 0.0
        self.model_rot_z: float = 90.0

        # State control
        self.state = ViewerState.NEUTRAL
        self.last_state_switch_time = 0.0

    ############# HAND PROCESSING ##############################################

    def process_right_hand(self, landmarks: Optional['NormalizedLandmarkList']):
        """Use right hand to control camera rotation and zoom via pinch."""
        if not landmarks:
            self.prev_right_avg = None
            return

        # Compute average position of landmarks for camera rotation
        xs = [lm.x for lm in landmarks.landmark]
        ys = [lm.y for lm in landmarks.landmark]
        avg_x = sum(xs) / len(xs)
        avg_y = sum(ys) / len(ys)

        if self.prev_right_avg is not None:
            dx = avg_x - self.prev_right_avg[0]
            # Suprimimos rotación en X vía mano (solo Y) y aumentamos sensibilidad
            self.cam_rot_y += dx * 300.0
        self.prev_right_avg = (avg_x, avg_y)

        # Zoom via pinch (thumb tip 4 and index tip 8)
        thumb_tip = landmarks.landmark[4]
        index_tip = landmarks.landmark[8]
        pinch_dist = math.dist((thumb_tip.x, thumb_tip.y), (index_tip.x, index_tip.y))
        pinch_dist = max(min(pinch_dist, PINCH_DIST_MAX), PINCH_DIST_MIN)
        # Map normalized pinch dist to camera distance
        interp = (pinch_dist - PINCH_DIST_MIN) / (PINCH_DIST_MAX - PINCH_DIST_MIN)
        # Invert mapping: pinch (small dist) -> zoom out (far), separate -> zoom in (near)
        target_dist = lerp(MIN_CAM_DIST, MAX_CAM_DIST, interp)
        self.cam_distance_target = target_dist

    def process_left_hand(self, landmarks: Optional['NormalizedLandmarkList'], dt: float):
        """Control del efecto de flujo con la mano izquierda según distancia de pinch."""
        if landmarks is None:
            # sin mano: desactiva efecto
            self.effect1_progress = 0.0
            if self.state == ViewerState.EFFECT_1:
                self.state = ViewerState.NEUTRAL
            return

        thumb_tip = landmarks.landmark[4]
        index_tip = landmarks.landmark[8]
        pinch_dist = math.dist((thumb_tip.x, thumb_tip.y), (index_tip.x, index_tip.y))
        pinch_dist = max(min(pinch_dist, PINCH_DIST_MAX), PINCH_DIST_MIN)

        # Map pinch a progreso (0→sin flujo, 1→flujo completo)
        self.effect1_progress = 1.0 - (
            (pinch_dist - PINCH_DIST_MIN) / (PINCH_DIST_MAX - PINCH_DIST_MIN)
        )

        # Activa o desactiva estado según progreso
        if self.effect1_progress > 0.05:
            self.state = ViewerState.EFFECT_1
        elif self.state == ViewerState.EFFECT_1:
            self.state = ViewerState.NEUTRAL

    def maybe_toggle_by_hands(self, right: Optional['NormalizedLandmarkList'],
                              left: Optional['NormalizedLandmarkList']):
        if right is None or left is None:
            return
        # Distance between right index tip and left palm center (approx wrist landmark 0)
        idx = right.landmark[8]
        palm = left.landmark[0]
        dist = math.dist((idx.x, idx.y), (palm.x, palm.y))
        if dist < PALM_TOUCH_THRESHOLD:
            self._cycle_state()

    def _toggle_state(self, target: ViewerState):
        now = time.time()
        if now - self.last_state_switch_time < STATE_COOLDOWN:
            return
        if self.state == target:
            self.state = ViewerState.NEUTRAL
            if target == ViewerState.EFFECT_1:
                self.effect1_progress = 0.0 # Reset progress when state changes
        else:
            self.state = target
            if target == ViewerState.EFFECT_1:
                self.effect1_progress = 0.0 # Reset progress when state changes
        self.last_state_switch_time = now

    def _cycle_state(self):
        now = time.time()
        if now - self.last_state_switch_time < STATE_COOLDOWN:
            return
        members = list(ViewerState)
        next_index = (members.index(self.state) + 1) % len(members)
        self.state = members[next_index]
        if self.state == ViewerState.EFFECT_1:
            self.effect1_progress = 0.0 # Reset progress when state changes
        else:
            self.effect1_progress = 0.0 # Reset progress when state changes
        self.last_state_switch_time = now

    ############# UPDATE & RENDER ##############################################

    def update(self, dt: float):
        # Smooth camera zoom
        self.cam_distance = lerp(self.cam_distance, self.cam_distance_target, ZOOM_LERP_T)

        # Clamp rotations to avoid numerical overflow
        self.cam_rot_x = (self.cam_rot_x + 360.0) % 360.0
        self.cam_rot_y = (self.cam_rot_y + 360.0) % 360.0

        # Auto rotation regardless of state
        self.cam_rot_y += AUTO_ROT_SPEED

        # Escala de biodiversidad controlada por pinch izquierda
        self.bio_scale = 1.0 - self.effect1_progress

        # Actualizar fase del río
        self.river_phase = (self.river_phase + RIVER_SPEED_DEG * dt) % 360.0

    def render(self):
        # Set up camera
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(FOVY, WINDOW_WIDTH / WINDOW_HEIGHT, Z_NEAR, Z_FAR)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Camera transform (translate backwards, then rotate)
        gluLookAt(0, 0, self.cam_distance,
                  0, 0, 0,
                  0, 1, 0)
        glRotatef(self.cam_rot_x, 1, 0, 0)
        glRotatef(self.cam_rot_y, 0, 1, 0)

        # Clear and render model
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        self._apply_state_visuals_pre()
        # Apply user-controlled model transforms
        glPushMatrix()
        glScalef(self.model_scale, self.model_scale, self.model_scale)
        glRotatef(self.model_rot_x, 1, 0, 0)
        glRotatef(self.model_rot_y, 0, 1, 0)
        glRotatef(self.model_rot_z, 0, 0, 1)

        # Choose draw method depending on state
        if self.state == ViewerState.EFFECT_1 and self.effect1_progress > 0.0:
            progress = self.effect1_progress
            # Build rotation matrix for current model orientation
            rx = math.radians(self.model_rot_x)
            ry = math.radians(self.model_rot_y)
            rz = math.radians(self.model_rot_z)
            cx, sx = math.cos(rx), math.sin(rx)
            cy, sy = math.cos(ry), math.sin(ry)
            cz, sz = math.cos(rz), math.sin(rz)

            R_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            R_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            R_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

            # OpenGL multiplies matrices right-to-left as they are issued, por lo que
            # tras glRotatef(X) luego Y luego Z el resultado es R = R_x * R_y * R_z
            R = R_x @ R_y @ R_z

            world_up = np.array([0.0, 1.0, 0.0])
            flow_dir_model = R.T @ world_up  # convert to model space

            self.model.draw_flow(progress, flow_dir_model)
        else:
            self.model.draw()

        glPopMatrix()
        # Dibujar río
        self._draw_river()

        # Dibujar biodiversidad
        self._draw_biodiversity()
        # Dibujar grietas del suelo
        self._draw_ground_cracks()
        self._apply_state_visuals_post()

    def _apply_state_visuals_pre(self):
        """Example effect modifications before drawing model."""
        if self.state == ViewerState.EFFECT_1:
            glPointSize(3.0)
            glColor3f(0.6, 0.05, 0.05)
        elif self.state == ViewerState.EFFECT_2:
            glColor3f(0.0, 0.0, 0.0)  # negro
        else:
            glColor3f(1.0, 1.0, 1.0)

    def _apply_state_visuals_post(self):
        glPointSize(1.0)

    # ------------------- BIODIVERSITY ----------------------
    def _draw_biodiversity(self):
        if self.bio_scale <= 0.01:
            return
        glPointSize(6 * self.bio_scale)
        glBegin(GL_POINTS)
        for (x, z) in self.bio_positions:
            color = random.choice(BIO_COLORS)
            glColor3f(*color)
            glVertex3f(x, BIO_Y, z)
        glEnd()

    # ------------------- CRACKED GROUND ----------------------
    def _draw_ground_cracks(self):
        # Mostrar grietas sólo si el progreso del pinch es significativo
        if self.effect1_progress < 0.05:
            return
        t = time.time()
        glLineWidth(2.0)
        glBegin(GL_LINES)
        for x1, z1, x2, z2, phase in self.cracks:
            val = (math.sin(2 * math.pi * CRACK_FREQUENCY * t + phase) + 1) / 2.0
            if val > CRACK_THRESHOLD:
                brightness = val * self.effect1_progress  # modula por pinch
                glColor3f(EFFECT_COLOR[0] * brightness, EFFECT_COLOR[1] * brightness, EFFECT_COLOR[2] * brightness)
                glVertex3f(x1, BIO_Y, z1)
                glVertex3f(x2, BIO_Y, z2)
        glEnd()

    # ------------------- RIVER ----------------------------
    def _draw_river(self):
        # Grosor depende del pinch
        width = RIVER_BASE_WIDTH * (1.0 - (1.0 - RIVER_WIDTH_MIN_FACTOR) * self.effect1_progress)
        if width < 1.0:
            return

        glLineWidth(width)
        # Habilitar transparencia y color con alpha
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glColor4f(RIVER_COLOR[0], RIVER_COLOR[1], RIVER_COLOR[2], RIVER_ALPHA)
        glBegin(GL_LINE_STRIP)
        phase_rad = math.radians(self.river_phase)
        t = time.time()
        for i in range(RIVER_SEGMENTS + 1):
            theta = 2 * math.pi * i / RIVER_SEGMENTS + phase_rad
            # Organic displacement via sinusoidal noise
            noise = math.sin(theta * RIVER_NOISE_FREQ + t * RIVER_NOISE_SPEED) * RIVER_NOISE_AMPL
            r = RIVER_RADIUS + noise
            x = math.cos(theta) * r
            z = math.sin(theta) * r
            y_offset = math.sin(theta * RIVER_HEIGHT_FREQ + t * RIVER_NOISE_SPEED) * RIVER_HEIGHT_AMPL
            glVertex3f(x, BIO_Y + 0.05 + y_offset, z)
        glEnd()
        glDisable(GL_BLEND)

###############################################################################
# HAND TRACKER WRAPPER #########################################################
###############################################################################

class HandTracker:
    """Wrapper around MediaPipe Hands solution for convenience."""

    def __init__(self, max_num_hands: int = 2, detection_confidence: float = 0.5, tracking_confidence: float = 0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         max_num_hands=max_num_hands,
                                         model_complexity=1,
                                         min_detection_confidence=detection_confidence,
                                         min_tracking_confidence=tracking_confidence)
        self.right_hand: Optional['NormalizedLandmarkList'] = None
        self.left_hand: Optional['NormalizedLandmarkList'] = None

    def process(self, frame_rgb: np.ndarray):
        self.right_hand = None
        self.left_hand = None
        result = self.hands.process(frame_rgb)
        if result.multi_hand_landmarks and result.multi_handedness:
            for landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                label = handedness.classification[0].label  # "Right" or "Left"
                if label == "Right":
                    self.right_hand = landmarks
                else:
                    self.left_hand = landmarks
        return result

###############################################################################
# CAMERA OVERLAY FUNCTIONS #####################################################
###############################################################################

def draw_landmarks_on_frame(frame: np.ndarray, result, mp_hands):
    mp_drawing = mp.solutions.drawing_utils
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return frame

###############################################################################
# MAIN LOOP ####################################################################
###############################################################################

def run():
    pygame.init()
    pygame.display.set_caption("Visualizador 3D ─ Control Bimanual con MediaPipe")
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), DOUBLEBUF | OPENGL)
    clock = pygame.time.Clock()

    # Initialize OpenGL context
    glEnable(GL_POINT_SMOOTH)
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.05, 0.05, 0.05, 1.0)

    viewer = Viewer(DEFAULT_MODEL_PATH)

    # ---------------- AUDIO ----------------
    # Try to start audio engine; if fails, continue silently.
    audio_engine = AudioEngine()
    audio_enabled = False
    if audio_engine._stream is not None:
        audio_engine.start()
        audio_enabled = True

    # Initialize camera toggle states
    show_camera_overlay = True
    show_camera_window = True  # Ventana externa para la cámara (OpenCV)

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        sys.exit(1)

    hand_tracker = HandTracker(max_num_hands=2)

    running = True
    while running:
        dt = clock.tick(60) / 1000.0  # delta time in seconds; 60 fps cap

        # ---------------- EVENT HANDLING ----------------
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_f:
                    pygame.display.toggle_fullscreen()
                elif event.key == pygame.K_c:
                    show_camera_overlay = not show_camera_overlay
                elif event.key == pygame.K_v:
                    show_camera_window = not show_camera_window
                # ----- Model rotation with arrow keys -----
                elif event.key == pygame.K_LEFT:
                    viewer.model_rot_y -= 10.0
                elif event.key == pygame.K_RIGHT:
                    viewer.model_rot_y += 10.0
                elif event.key == pygame.K_UP:
                    viewer.model_rot_x -= 10.0
                elif event.key == pygame.K_DOWN:
                    viewer.model_rot_x += 10.0
                # ----- Model scaling with +/- (or = and -) -----
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    viewer.model_scale *= 1.1
                elif event.key in (pygame.K_MINUS, pygame.K_UNDERSCORE):
                    viewer.model_scale = max(0.1, viewer.model_scale / 1.1)
                # ----- Reset transform -----
                elif event.key == pygame.K_r:
                    viewer.model_scale = 1.0
                    viewer.model_rot_x = viewer.model_rot_y = viewer.model_rot_z = 0.0
                # ----- Z-axis rotation (roll) -----
                elif event.key == pygame.K_q:  # rotate left
                    viewer.model_rot_z -= 10.0
                elif event.key == pygame.K_e:  # rotate right
                    viewer.model_rot_z += 10.0
                # ----- Toggle audio -----
                elif event.key == pygame.K_m:
                    audio_enabled = not audio_enabled
                    if audio_engine._stream is not None:
                        if audio_enabled:
                            audio_engine.start()
                        else:
                            audio_engine.stop()

        # ---------------- HAND TRACKING -----------------
        ret, frame = cap.read()
        if not ret:
            continue

        # Flip camera horizontally if enabled (mirror effect)
        if MIRROR_CAMERA:
            frame = cv2.flip(frame, 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hand_tracker.process(frame_rgb)

        # Dibujar landmarks sobre la imagen para las vistas overlay y ventana
        annotated_frame = draw_landmarks_on_frame(frame.copy(), result, hand_tracker.mp_hands)

        # Feed landmarks to viewer
        viewer.process_right_hand(hand_tracker.right_hand)
        viewer.process_left_hand(hand_tracker.left_hand, dt)
        viewer.maybe_toggle_by_hands(hand_tracker.right_hand, hand_tracker.left_hand)

        # ---------------- EFFECT PROGRESS (pinch mano izquierda) -------------
        def pinch_progress(landmarks):
            if landmarks is None:
                return 0.0
            thumb_tip = landmarks.landmark[4]
            index_tip = landmarks.landmark[8]
            dist = math.dist((thumb_tip.x, thumb_tip.y), (index_tip.x, index_tip.y))
            dist = max(min(dist, PINCH_DIST_MAX), PINCH_DIST_MIN)
            if dist <= PINCH_DIST_FULL:
                norm = 1.0
            else:
                norm = 1.0 - ((dist - PINCH_DIST_FULL) / (PINCH_DIST_MAX - PINCH_DIST_FULL))
                norm = max(0.0, norm)
            # Curva gamma (raíz) para facilitar llegar a 1 con distancias moderadas
            return math.sqrt(norm)

        progress = pinch_progress(hand_tracker.left_hand)
        viewer.effect1_progress = progress
        if progress > 0.05:
            viewer.state = ViewerState.EFFECT_1
        elif viewer.state == ViewerState.EFFECT_1:
            viewer.state = ViewerState.NEUTRAL

        # Update audio engine with crackle intensity (progress)
        if audio_engine._stream is not None and audio_enabled:
            audio_engine.update(crackle_level=progress)

        # Update logic
        viewer.update(dt)

        # Render 3D scene
        viewer.render()

        # Overlay camera feed if enabled
        if show_camera_overlay:
            thickness = 4  # grosor de línea para ambos indicadores

            # Draw pinch line entre pulgar e índice de la mano izquierda
            if hand_tracker.left_hand is not None:
                h, w = annotated_frame.shape[:2]
                tt = hand_tracker.left_hand.landmark[4]
                it = hand_tracker.left_hand.landmark[8]
                pt1 = (int(tt.x * w), int(tt.y * h))
                pt2 = (int(it.x * w), int(it.y * h))
                color = (0, 0, 150)  # rojo oscuro (BGR)
                cv2.line(annotated_frame, pt1, pt2, color, thickness)

            # Indicador para la mano derecha (solo visual)
            if hand_tracker.right_hand is not None:
                h, w = annotated_frame.shape[:2]
                tt_r = hand_tracker.right_hand.landmark[4]
                it_r = hand_tracker.right_hand.landmark[8]
                pr1 = (int(tt_r.x * w), int(tt_r.y * h))
                pr2 = (int(it_r.x * w), int(it_r.y * h))
                color_r = (0, 255, 0)  # verde
                cv2.line(annotated_frame, pr1, pr2, color_r, thickness)

            # Convert to pygame surface AFTER drawing line
            surf = pygame.image.frombuffer(annotated_frame.tobytes(), annotated_frame.shape[1::-1], "BGR")

            # Small picture-in-picture at bottom left
            pip_width = int(WINDOW_WIDTH * 0.25)
            pip_height = int(pip_width * annotated_frame.shape[0] / annotated_frame.shape[1])
            pip_surf = pygame.transform.smoothscale(surf, (pip_width, pip_height))
            screen.blit(pip_surf, (10, WINDOW_HEIGHT - pip_height - 10))

        # Ventana externa de cámara usando OpenCV
        if show_camera_window:
            cv2.imshow("Camera", annotated_frame)
        else:
            # Cerrar la ventana si existe y se desactiva
            cv2.destroyWindow("Camera")

        # Necesario para refrescar la ventana OpenCV (y permitir cierre con 'q')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False

        pygame.display.flip()

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    if audio_engine._stream is not None:
        audio_engine.stop()
    pygame.quit()

###############################################################################
# ENTRY POINT ##################################################################
###############################################################################

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualizador 3D ─ Control Bimanual con MediaPipe")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Ruta al archivo .ply a visualizar")
    args = parser.parse_args()

    # Update default model path globally so Viewer picks it up
    DEFAULT_MODEL_PATH = args.model

    try:
        run()
    except Exception as e:
        pygame.quit()
        raise e