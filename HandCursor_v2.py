

#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           ✦ HAND CURSOR v2.0 — ÉDITION ULTIME ✦                            ║
║      Contrôle de curseur par gestes · Filtre Kalman · Multi-thread          ║
║      Accélération dynamique · Machine à états · Profils JSON                ║
╚══════════════════════════════════════════════════════════════════════════════╝

GESTES :
  ☞  Index seul           → Déplacer  (vitesse adaptative)
  ✌  Index + Majeur       → Précision pixel (lent = ultra-précis)
  👌  Pince pouce+index   → Clic gauche  (jauge de confirmation)
  🤙  Pouce + auriculaire → Clic droit
  ✊  Poing fermé         → Glisser-déposer
  🖐  Main ouverte        → Scroll (vitesse ∝ inclinaison)
  👍  Pouce seul          → Double-clic
  ✌+annulaire             → Clic milieu

CLAVIER (fenêtre active) :
  [Q]/[Échap]  Quitter          [P]  Pause
  [D]          Debug landmarks  [C]  Calibration
  [+]/[-]      Sensibilité      [S]  Screenshot
  [1][2][3]    Profils rapides  [R]  Reset profil
"""

# ─────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import math
import threading
import sys
import os
import json
import copy
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from pathlib import Path

pyautogui.FAILSAFE = True
pyautogui.PAUSE    = 0.0

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTES ÉCRAN
# ─────────────────────────────────────────────────────────────────────────────
SCREEN_W, SCREEN_H = pyautogui.size()
PROFILE_PATH = Path.home() / ".handcursor_profiles.json"

# ─────────────────────────────────────────────────────────────────────────────
#  PALETTE CYBERPUNK DEEP
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "bg"       : (8,   8,   16 ),
    "panel"    : (16,  18,  30 ),
    "cyan"     : (0,   230, 185),
    "pink"     : (220, 50,  140),
    "gold"     : (240, 195, 0  ),
    "violet"   : (160, 80,  255),
    "white"    : (235, 235, 250),
    "grey"     : (70,  80,  100),
    "dgrey"    : (35,  40,  55 ),
    "red"      : (50,  40,  210),
    "green"    : (50,  210, 90 ),
    "orange"   : (20,  140, 255),
}

# ─────────────────────────────────────────────────────────────────────────────
#  PROFIL DE CONFIGURATION (serialisable JSON)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Profile:
    name             : str   = "Défaut"
    sensitivity      : float = 1.8
    smooth_alpha     : float = 0.30    # lissage Kalman (0=max lisse, 1=brut)
    smooth_frames    : int   = 6
    deadzone_px      : int   = 3
    click_threshold  : float = 0.040   # distance pince → clic
    click_hold_ms    : int   = 120     # durée maintien pince pour confirmer clic
    scroll_speed     : int   = 10
    scroll_accel     : float = 1.5     # accélération scroll selon vitesse
    accel_low        : float = 0.6     # vitesse basse → coefficient faible (précision)
    accel_high       : float = 2.8     # vitesse haute → grand déplacement
    accel_threshold  : float = 0.012   # seuil vitesse pour accélération (normalisé/frame)
    gesture_confirm  : int   = 3       # frames pour confirmer un geste (anti-bruit)
    roi_margin       : float = 0.06    # marge ROI (0..0.5)
    cam_index        : int   = 0
    cam_w            : int   = 1280
    cam_h            : int   = 720
    cam_fps          : int   = 60

# ─────────────────────────────────────────────────────────────────────────────
#  GESTIONNAIRE DE PROFILS JSON
# ─────────────────────────────────────────────────────────────────────────────
class ProfileManager:
    DEFAULTS = [
        Profile("Rapide",    sensitivity=2.4, smooth_alpha=0.45, accel_high=3.5),
        Profile("Équilibré", sensitivity=1.8, smooth_alpha=0.30, accel_high=2.8),
        Profile("Précision", sensitivity=1.0, smooth_alpha=0.18, deadzone_px=2,
                accel_low=0.4, accel_high=1.6, accel_threshold=0.008),
    ]

    def __init__(self):
        self.profiles: list[Profile] = []
        self.active   : int          = 1   # index actif (0-based)
        self._load()

    def _load(self):
        if PROFILE_PATH.exists():
            try:
                data = json.loads(PROFILE_PATH.read_text())
                self.profiles = [Profile(**p) for p in data.get("profiles", [])]
                self.active   = data.get("active", 1)
                return
            except Exception:
                pass
        self.profiles = copy.deepcopy(self.DEFAULTS)
        self._save()

    def _save(self):
        data = {
            "profiles": [asdict(p) for p in self.profiles],
            "active"  : self.active,
        }
        PROFILE_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    @property
    def current(self) -> Profile:
        return self.profiles[self.active]

    def switch(self, idx: int):
        idx = max(0, min(len(self.profiles)-1, idx))
        self.active = idx
        self._save()

    def update_sensitivity(self, delta: float):
        p = self.current
        p.sensitivity = round(max(0.3, min(5.0, p.sensitivity + delta)), 1)
        self._save()

# ─────────────────────────────────────────────────────────────────────────────
#  FILTRE DE KALMAN 2D (position curseur)
# ─────────────────────────────────────────────────────────────────────────────
class KalmanFilter2D:
    """
    Filtre Kalman simplifié pour position 2D.
    État : [x, y, vx, vy]   Mesure : [x, y]
    """
    def __init__(self, q=1e-3, r=1e-1):
        self.kf = cv2.KalmanFilter(4, 2)
        dt = 1.0
        self.kf.transitionMatrix    = np.array([[1,0,dt,0],[0,1,0,dt],
                                                 [0,0,1, 0],[0,0,0, 1]], np.float32)
        self.kf.measurementMatrix   = np.array([[1,0,0,0],[0,1,0,0]],    np.float32)
        self.kf.processNoiseCov     = np.eye(4, dtype=np.float32) * q
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * r
        self.kf.errorCovPost        = np.eye(4, dtype=np.float32)
        self.kf.statePost           = np.zeros((4,1), np.float32)
        self._init = False

    def update(self, x: float, y: float) -> tuple[float, float]:
        meas = np.array([[np.float32(x)],[np.float32(y)]])
        if not self._init:
            self.kf.statePost[:2] = meas
            self._init = True
        self.kf.predict()
        state = self.kf.correct(meas)
        return float(state[0][0]), float(state[1][0])

    def reset(self):
        self.kf.statePost = np.zeros((4,1), np.float32)
        self._init = False

    def set_noise(self, q: float, r: float):
        self.kf.processNoiseCov     = np.eye(4, dtype=np.float32) * q
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * r

# ─────────────────────────────────────────────────────────────────────────────
#  LISSEUR HYBRIDE  Kalman + moyenne glissante + deadzone
# ─────────────────────────────────────────────────────────────────────────────
class HybridSmoother:
    def __init__(self, profile: Profile):
        self.p      = profile
        self.kf     = KalmanFilter2D(q=5e-4, r=8e-2)
        self.buf    = deque(maxlen=profile.smooth_frames)
        self.last   = None
        self._prev_raw = None

    def update(self, sx: float, sy: float) -> tuple[int, int]:
        # 1. Kalman
        kx, ky = self.kf.update(sx, sy)
        # 2. Moyenne glissante
        self.buf.append((kx, ky))
        mx = sum(p[0] for p in self.buf) / len(self.buf)
        my = sum(p[1] for p in self.buf) / len(self.buf)
        # 3. EMA
        alpha = self.p.smooth_alpha
        if self.last is None:
            self.last = (mx, my)
        fx = alpha * mx + (1-alpha) * self.last[0]
        fy = alpha * my + (1-alpha) * self.last[1]
        # 4. Deadzone
        dz = self.p.deadzone_px
        if abs(fx - self.last[0]) < dz and abs(fy - self.last[1]) < dz:
            return int(self.last[0]), int(self.last[1])
        self.last = (fx, fy)
        return int(fx), int(fy)

    def reset(self):
        self.kf.reset()
        self.buf.clear()
        self.last = None
        self._prev_raw = None

    def reload(self, profile: Profile):
        self.p   = profile
        self.buf = deque(maxlen=profile.smooth_frames)

# ─────────────────────────────────────────────────────────────────────────────
#  ACCÉLÉRATION DYNAMIQUE DU CURSEUR
# ─────────────────────────────────────────────────────────────────────────────
class DynamicAccel:
    """
    Vitesse de déplacement de la main → coefficient multiplicateur.
    Lent = précis (coefficient proche de accel_low)
    Rapide = grande amplitude (coefficient proche de accel_high)
    Courbe sigmoïde pour transition douce.
    """
    def __init__(self, profile: Profile):
        self.p     = profile
        self._prev = None

    def coeff(self, nx: float, ny: float) -> float:
        if self._prev is None:
            self._prev = (nx, ny)
            return 1.0
        speed = math.hypot(nx - self._prev[0], ny - self._prev[1])
        self._prev = (nx, ny)
        t = self.p.accel_threshold
        if speed < t * 0.3:
            k = 0.0
        else:
            k = min(1.0, (speed - t*0.3) / (t * 2.5))
        # Sigmoïde : k ∈ [0,1] → coeff ∈ [accel_low, accel_high]
        sigma = 1.0 / (1.0 + math.exp(-10 * (k - 0.5)))
        lo, hi = self.p.accel_low, self.p.accel_high
        return lo + sigma * (hi - lo)

    def reset(self):
        self._prev = None

    def reload(self, p: Profile):
        self.p = p

# ─────────────────────────────────────────────────────────────────────────────
#  MACHINE À ÉTATS FINIS (FSM) — gestion des clics et états
# ─────────────────────────────────────────────────────────────────────────────
class GestureState(Enum):
    IDLE         = auto()
    MOVING       = auto()
    PRECISION    = auto()
    PINCH_HOLD   = auto()   # pince tenue → compte à rebours avant clic
    CLICKING     = auto()   # clic en cours (cooldown)
    DRAGGING     = auto()
    SCROLLING    = auto()
    RCLICK_HOLD  = auto()
    DCLICK_HOLD  = auto()

class FSM:
    COOLDOWN_MS  = 350   # ms entre deux clics distincts
    DCLICK_WIN   = 180   # fenêtre double-clic (ms)

    def __init__(self, profile: Profile):
        self.p          = profile
        self.state      = GestureState.IDLE
        self._since     = time.time()
        self._last_cl_t = 0.0
        self._pinch_t   = None   # timestamp début pince
        self._pending   = None   # action en attente

    def _elapsed_ms(self) -> float:
        return (time.time() - self._since) * 1000

    def transition(self, raw_gesture: str) -> list[str]:
        """
        Reçoit le geste brut, retourne une liste d'actions à exécuter :
        ex. ["MOVE", "CLICK_L", "DRAG_START", "DRAG_MOVE", "DRAG_END", ...]
        """
        now  = time.time()
        acts = []
        g    = raw_gesture
        st   = self.state

        # ── IDLE / MOVING ──────────────────────────────────────────────────
        if g == "MOVE":
            if st == GestureState.DRAGGING:
                acts.append("DRAG_END")
            self.state = GestureState.MOVING
            acts.append("MOVE")

        elif g == "MOVE_PRECISE":
            if st == GestureState.DRAGGING:
                acts.append("DRAG_END")
            self.state = GestureState.PRECISION
            acts.append("MOVE_PRECISE")

        # ── PINCE → CLIC GAUCHE ────────────────────────────────────────────
        elif g == "CLICK_L":
            if st not in (GestureState.PINCH_HOLD, GestureState.CLICKING):
                if self._pinch_t is None:
                    self._pinch_t = now
                held_ms = (now - self._pinch_t) * 1000
                # Confirmation : maintenir la pince N ms
                if held_ms >= self.p.click_hold_ms:
                    cooldown_ok = (now - self._last_cl_t) * 1000 > self.COOLDOWN_MS
                    if cooldown_ok:
                        acts.append("CLICK_L")
                        self._last_cl_t = now
                        self._pinch_t   = None
                        self.state      = GestureState.CLICKING
                else:
                    acts.append("PINCH_PROGRESS")   # pour la jauge visuelle
            elif st == GestureState.CLICKING:
                if self._elapsed_ms() > self.COOLDOWN_MS:
                    self.state = GestureState.MOVING
        else:
            self._pinch_t = None

        # ── POING → DRAG ───────────────────────────────────────────────────
        if g == "DRAG":
            if st != GestureState.DRAGGING:
                acts.append("DRAG_START")
                self.state = GestureState.DRAGGING
                self._since = now
            acts.append("DRAG_MOVE")

        # ── SCROLL ────────────────────────────────────────────────────────
        elif g == "SCROLL":
            if st == GestureState.DRAGGING:
                acts.append("DRAG_END")
            self.state = GestureState.SCROLLING
            acts.append("SCROLL")

        # ── CLIC DROIT ────────────────────────────────────────────────────
        elif g == "CLICK_R":
            if st != GestureState.RCLICK_HOLD:
                self.state = GestureState.RCLICK_HOLD
                self._since = now
            elif self._elapsed_ms() > 200:
                acts.append("CLICK_R")
                self.state  = GestureState.IDLE
                self._since = now

        # ── DOUBLE-CLIC ───────────────────────────────────────────────────
        elif g == "DCLICK":
            if st != GestureState.DCLICK_HOLD:
                self.state  = GestureState.DCLICK_HOLD
                self._since = now
            elif self._elapsed_ms() > 150:
                acts.append("DCLICK")
                self.state  = GestureState.IDLE
                self._since = now

        # ── CLIC MILIEU ───────────────────────────────────────────────────
        elif g == "CLICK_M":
            acts.append("CLICK_M")
            self.state = GestureState.IDLE

        # ── IDLE ──────────────────────────────────────────────────────────
        elif g == "IDLE":
            if st == GestureState.DRAGGING:
                acts.append("DRAG_END")
            self.state = GestureState.IDLE

        return acts

    def pinch_progress(self) -> float:
        """Retourne 0..1 selon la progression de la confirmation pince."""
        if self._pinch_t is None:
            return 0.0
        held = (time.time() - self._pinch_t) * 1000
        return min(1.0, held / max(1, self.p.click_hold_ms))

    def reload(self, p: Profile):
        self.p = p

# ─────────────────────────────────────────────────────────────────────────────
#  DÉTECTEUR DE GESTES
# ─────────────────────────────────────────────────────────────────────────────
class GestureDetector:
    W=0;T4=4;I8=8;M12=12;R16=16;P20=20
    T3=3;I6=6;M10=10;R14=14;P18=18
    M9=9

    def __init__(self, lm):
        self.lm = lm

    def _xy(self, i):
        return self.lm[i].x, self.lm[i].y

    def _dist(self, a, b):
        ax,ay=self._xy(a); bx,by=self._xy(b)
        return math.hypot(ax-bx, ay-by)

    def _up(self, tip, pip):
        return self.lm[tip].y < self.lm[pip].y

    def fingers(self):
        lm = self.lm
        th = lm[self.T4].x < lm[self.T3].x   # pouce (main droite, miroir)
        return [
            th,
            self._up(self.I8,  self.I6),
            self._up(self.M12, self.M10),
            self._up(self.R16, self.R14),
            self._up(self.P20, self.P18),
        ]

    def pinch(self):
        return self._dist(self.T4, self.I8)

    def detect(self):
        f  = self.fingers()
        pd = self.pinch()
        th, ix, md, rg, pk = f
        ixx, ixy = self._xy(self.I8)
        mxx, mxy = self._xy(self.M12)

        if not any(f):
            return "DRAG", (ixx, ixy)

        if pd < 0.040 and not md and not rg and not pk:
            return "CLICK_L", (ixx, ixy)

        if th and not ix and not md and not rg and not pk:
            return "DCLICK", (ixx, ixy)

        if th and not ix and not md and not rg and pk:
            return "CLICK_R", (ixx, ixy)

        if sum(f) >= 4:
            return "SCROLL", (ixx, ixy)

        if ix and md and not rg and not pk:
            return "MOVE_PRECISE", ((ixx+mxx)/2, (ixy+mxy)/2)

        if ix and md and rg and not pk:
            return "CLICK_M", (ixx, ixy)

        if ix and not md and not rg and not pk:
            return "MOVE", (ixx, ixy)

        return "IDLE", (ixx, ixy)

    def wrist_tilt_speed(self) -> float:
        """Vitesse angulaire poignet → intensité scroll."""
        wx, wy = self._xy(self.W)
        mx, my = self._xy(self.M9)
        angle = math.degrees(math.atan2(wy - my, mx - wx))
        # Normalise l'angle en vitesse scroll
        return (angle - 90.0) / 45.0   # ±2 environ

    def index_tip(self):
        return self._xy(self.I8)

# ─────────────────────────────────────────────────────────────────────────────
#  HUD ARTISTIQUE v2
# ─────────────────────────────────────────────────────────────────────────────
GESTURE_META = {
    "MOVE"         : ("☞  DÉPLACER",        C["cyan"],   "Vitesse adaptative"),
    "MOVE_PRECISE" : ("✌  PRÉCISION",        C["gold"],   "Ultra-précis lent"),
    "CLICK_L"      : ("👌  CLIC GAUCHE",     C["pink"],   "Maintenir la pince…"),
    "CLICK_R"      : ("🤙  CLIC DROIT",      C["violet"], "Pouce+auriculaire"),
    "DRAG"         : ("✊  GLISSER",          C["orange"], "Poing fermé"),
    "SCROLL"       : ("🖐  DÉFILEMENT",      C["cyan"],   "Inclinaison main"),
    "DCLICK"       : ("👍  DOUBLE-CLIC",     C["pink"],   "Pouce seul"),
    "CLICK_M"      : ("✌+  CLIC MILIEU",    C["violet"], "3 doigts"),
    "IDLE"         : ("—  ATTENTE",           C["grey"],   ""),
}

class HUD:
    def __init__(self):
        self._t0      = time.time()
        self._fpsbuf  = deque(maxlen=30)
        self._lt      = time.time()
        self._pulse   = 0.0
        self._trail   = deque(maxlen=28)   # traînée du curseur
        self._last_g  = "IDLE"
        self._g_alpha = 0.0   # opacité transition geste
        self._font    = cv2.FONT_HERSHEY_DUPLEX
        self._fonts   = cv2.FONT_HERSHEY_SIMPLEX

    # ── Primitives visuelles ─────────────────────────────────────────────────

    def _blend_rect(self, img, x1,y1,x2,y2, color, alpha=0.60, radius=8):
        ov = img.copy()
        r  = radius
        cv2.rectangle(ov, (x1+r,y1),(x2-r,y2), color, -1)
        cv2.rectangle(ov, (x1,y1+r),(x2,y2-r), color, -1)
        for cx,cy in [(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
            cv2.circle(ov,(cx,cy),r,color,-1)
        cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)

    def _glow(self, img, pts_or_fn, color, thickness=2, layers=4):
        for i in range(layers,0,-1):
            ov = img.copy()
            if callable(pts_or_fn):
                pts_or_fn(ov, color, thickness+i*2)
            cv2.addWeighted(ov, 0.12*i, img, 1-0.12*i, 0, img)
        if callable(pts_or_fn):
            pts_or_fn(img, color, thickness)

    def _glow_circle(self, img, cx,cy, r, color, t=2):
        def draw(surface, col, th):
            cv2.circle(surface,(cx,cy),r,col,th)
        self._glow(img, draw, color, t)

    def _glow_line(self, img, p1, p2, color, t=2):
        def draw(surface, col, th):
            cv2.line(surface, p1, p2, col, th)
        self._glow(img, draw, color, t)

    def _gauge(self, img, x,y,w,h, val, color, bg=None):
        """Barre de progression avec glow."""
        bg = bg or C["dgrey"]
        cv2.rectangle(img,(x,y),(x+w,y+h),bg,-1)
        if val > 0:
            fw = int(w * val)
            cv2.rectangle(img,(x,y),(x+fw,y+h),color,-1)
            # glow sur le bord droit
            ov = img.copy()
            cv2.line(ov,(x+fw,y),(x+fw,y+h),color,3)
            cv2.addWeighted(ov,0.5,img,0.5,0,img)
        cv2.rectangle(img,(x,y),(x+w,y+h),C["grey"],1)

    # ── Squelette lumineux ───────────────────────────────────────────────────
    def _skeleton(self, img, lm, W, H, gesture):
        color = GESTURE_META.get(gesture, GESTURE_META["IDLE"])[1]
        CONN  = mp.solutions.hands.HAND_CONNECTIONS
        pts   = [(int(l.x*W), int(l.y*H)) for l in lm]

        # Connexions
        for a,b in CONN:
            self._glow_line(img, pts[a], pts[b], color, 2)

        # Joints
        TIPS = {4,8,12,16,20}
        for i,(px,py) in enumerate(pts):
            r = 7 if i in TIPS else 4
            cv2.circle(img,(px,py),r,color,-1)
            cv2.circle(img,(px,py),r+2,tuple(int(c*0.3) for c in color),1)

        # Anneau animé sur l'index
        ix,iy = pts[8]
        R = 14 + int(3*math.sin(self._pulse*2))
        self._glow_circle(img, ix,iy, R, C["white"], 1)

        # Traînée de l'index
        self._trail.append((ix,iy))
        for j in range(1,len(self._trail)):
            a_pt = self._trail[j-1]
            b_pt = self._trail[j]
            alpha = j / len(self._trail)
            trail_color = tuple(int(c*alpha*0.6) for c in color)
            cv2.line(img, a_pt, b_pt, trail_color, max(1,int(alpha*3)))

    # ── Dessin principal ─────────────────────────────────────────────────────
    def draw(self, frame, gesture, raw_gesture, hand_lm, fsm, profile,
             paused, debug, ctrl_screen, pinch_prog, fps):
        H, W = frame.shape[:2]
        self._pulse = (self._pulse + 0.09) % (2*math.pi)
        pv = 0.5 + 0.5*math.sin(self._pulse)

        # ── Fond bas ────────────────────────────────────────────────────────
        self._blend_rect(frame, 0,H-130,W,H, C["bg"], 0.80, 0)

        # ── Fond haut ───────────────────────────────────────────────────────
        self._blend_rect(frame, 0,0,W,56, C["bg"], 0.75, 0)

        # ── Titre pulsé ─────────────────────────────────────────────────────
        title = "✦ HAND CURSOR v2 ✦"
        tc = tuple(int(C["cyan"][i]*pv + C["grey"][i]*(1-pv)) for i in range(3))
        cv2.putText(frame, title, (W//2-150, 37), self._font, 0.9, tc, 2, cv2.LINE_AA)
        cv2.line(frame,(0,56),(W,56),C["dgrey"],1)

        # ── FPS ─────────────────────────────────────────────────────────────
        fc = C["green"] if fps>28 else C["gold"] if fps>18 else C["red"]
        cv2.putText(frame, f"{fps:3.0f} FPS", (W-110,37), self._fonts, 0.7, fc, 1, cv2.LINE_AA)

        # ── Profil actif ────────────────────────────────────────────────────
        prof_txt = f"[ {profile.name}  ·  ×{profile.sensitivity:.1f} ]"
        cv2.putText(frame, prof_txt, (20,37), self._fonts, 0.60, C["gold"], 1, cv2.LINE_AA)

        # ── Geste principal ──────────────────────────────────────────────────
        label, color, hint = GESTURE_META.get(raw_gesture, GESTURE_META["IDLE"])
        # transition alpha
        if raw_gesture != self._last_g:
            self._g_alpha = 0.0
            self._last_g  = raw_gesture
        self._g_alpha = min(1.0, self._g_alpha + 0.12)
        ga = self._g_alpha
        gc = tuple(int(color[i]*ga + C["grey"][i]*(1-ga)) for i in range(3))

        # Glow text
        cv2.putText(frame, label, (22,H-82), self._font, 0.95,
                    tuple(int(c*0.3) for c in gc), 5, cv2.LINE_AA)
        cv2.putText(frame, label, (22,H-82), self._font, 0.95, gc, 2, cv2.LINE_AA)

        # Hint
        if hint:
            cv2.putText(frame, hint, (22,H-58), self._fonts, 0.50, C["grey"], 1, cv2.LINE_AA)

        # ── Jauge pince (confirmation clic) ─────────────────────────────────
        if pinch_prog > 0:
            gx, gy, gw, gh = W-180, H-90, 160, 10
            self._gauge(frame, gx,gy,gw,gh, pinch_prog, C["pink"])
            cv2.putText(frame,"CLIC …",(gx,gy-6),self._fonts,0.40,C["pink"],1,cv2.LINE_AA)

        # ── Indicateur état FSM ──────────────────────────────────────────────
        state_colors = {
            GestureState.DRAGGING  : C["orange"],
            GestureState.SCROLLING : C["cyan"],
            GestureState.CLICKING  : C["pink"],
            GestureState.PRECISION : C["gold"],
        }
        sc = state_colors.get(fsm.state)
        if sc:
            self._glow_circle(frame, W-30, 30, 8, sc, -1)

        # ── Commandes bas ────────────────────────────────────────────────────
        hints2 = "[Q] Quitter  [P] Pause  [D] Debug  [+/-] Sens.  [1][2][3] Profils  [C] Calibration"
        cv2.putText(frame, hints2, (22,H-20), self._fonts, 0.38, C["grey"], 1, cv2.LINE_AA)

        # ── Indicateur drag actif ─────────────────────────────────────────
        if fsm.state == GestureState.DRAGGING:
            self._blend_rect(frame, W-160,H-130,W-10,H-95, C["panel"], 0.85)
            cv2.putText(frame,"✊ DRAG ACTIF",(W-155,H-106),
                        self._fonts,0.50,C["orange"],1,cv2.LINE_AA)

        # ── Squelette ────────────────────────────────────────────────────────
        if hand_lm:
            self._skeleton(frame, hand_lm, W, H, raw_gesture)

        # ── Croix curseur projetée ───────────────────────────────────────────
        if ctrl_screen:
            cx, cy = ctrl_screen
            wx = int(cx/SCREEN_W*W)
            wy = int(cy/SCREEN_H*H)
            size = 16 + int(4*pv)
            self._glow_circle(frame, wx,wy, size, C["pink"], 2)
            cv2.line(frame,(wx-size-4,wy),(wx+size+4,wy),C["pink"],1)
            cv2.line(frame,(wx,wy-size-4),(wx,wy+size+4),C["pink"],1)
            cv2.circle(frame,(wx,wy),3,C["white"],-1)

        # ── Debug landmarks ──────────────────────────────────────────────────
        if debug and hand_lm:
            for i,lm in enumerate(hand_lm):
                px,py=int(lm.x*W),int(lm.y*H)
                cv2.circle(frame,(px,py),3,C["gold"],-1)
                cv2.putText(frame,str(i),(px+4,py-3),cv2.FONT_HERSHEY_PLAIN,0.55,C["grey"],1)

        # ── PAUSE overlay ────────────────────────────────────────────────────
        if paused:
            ov = frame.copy()
            cv2.rectangle(ov,(0,0),(W,H),(8,8,16),-1)
            cv2.addWeighted(ov,0.55,frame,0.45,0,frame)
            cv2.putText(frame,"⏸  PAUSE",(W//2-110,H//2),
                        self._font, 1.6, C["gold"], 3, cv2.LINE_AA)

        return frame

# ─────────────────────────────────────────────────────────────────────────────
#  THREAD CONTRÔLE SOURIS  (séparé du thread caméra)
# ─────────────────────────────────────────────────────────────────────────────
class MouseThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._lock    = threading.Lock()
        self._actions = []
        self._pos     = None
        self._running = True

    def push(self, actions: list[str], pos: tuple[int,int] | None):
        with self._lock:
            self._actions = actions
            self._pos     = pos

    def run(self):
        dragging = False
        while self._running:
            with self._lock:
                acts = self._actions[:]
                pos  = self._pos
                self._actions.clear()

            for act in acts:
                try:
                    x, y = pos if pos else (None, None)
                    if act == "MOVE" and x:
                        if dragging: pyautogui.mouseUp(button='left'); dragging=False
                        pyautogui.moveTo(x, y)
                    elif act == "MOVE_PRECISE" and x:
                        if dragging: pyautogui.mouseUp(button='left'); dragging=False
                        pyautogui.moveTo(x, y)
                    elif act == "DRAG_START" and x:
                        pyautogui.moveTo(x, y)
                        pyautogui.mouseDown(button='left')
                        dragging = True
                    elif act == "DRAG_MOVE" and x:
                        pyautogui.moveTo(x, y)
                    elif act == "DRAG_END":
                        if dragging:
                            pyautogui.mouseUp(button='left')
                            dragging = False
                    elif act == "CLICK_L" and x:
                        if dragging: pyautogui.mouseUp(button='left'); dragging=False
                        pyautogui.click(x, y, button='left')
                    elif act == "CLICK_R" and x:
                        if dragging: pyautogui.mouseUp(button='left'); dragging=False
                        pyautogui.click(x, y, button='right')
                    elif act == "DCLICK" and x:
                        if dragging: pyautogui.mouseUp(button='left'); dragging=False
                        pyautogui.doubleClick(x, y)
                    elif act == "CLICK_M" and x:
                        pyautogui.click(x, y, button='middle')
                    elif act == "SCROLL_UP":
                        pyautogui.scroll(1)
                    elif act == "SCROLL_DOWN":
                        pyautogui.scroll(-1)
                except Exception:
                    pass

            time.sleep(0.004)   # ~250 Hz

    def stop(self):
        self._running = False

# ─────────────────────────────────────────────────────────────────────────────
#  CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────
class Calibrator:
    """Guide l'utilisateur à balayer les coins pour calibrer la ROI."""
    STEPS = [
        ("COIN HAUT-GAUCHE",   "Pointe l'index vers le coin supérieur gauche",  (0.1,0.1)),
        ("COIN HAUT-DROIT",    "Pointe l'index vers le coin supérieur droit",   (0.9,0.1)),
        ("COIN BAS-DROIT",     "Pointe vers le coin inférieur droit",           (0.9,0.9)),
        ("COIN BAS-GAUCHE",    "Pointe vers le coin inférieur gauche",          (0.1,0.9)),
    ]
    HOLD_FRAMES = 30

    def __init__(self):
        self.active  = False
        self.step    = 0
        self.samples : list[tuple] = []
        self._held   = 0
        self._result = None   # (x1,y1,x2,y2) ROI normalisée

    def start(self):
        self.active  = True
        self.step    = 0
        self.samples = []
        self._held   = 0
        self._result = None

    def feed(self, nx, ny) -> bool:
        """Retourne True quand la calibration est terminée."""
        if not self.active:
            return False
        sx,sy = self.STEPS[self.step][2]
        if math.hypot(nx-sx, ny-sy) < 0.15:
            self._held += 1
        else:
            self._held = 0
        if self._held >= self.HOLD_FRAMES:
            self.samples.append((nx, ny))
            self._held = 0
            self.step += 1
            if self.step >= len(self.STEPS):
                xs = [p[0] for p in self.samples]
                ys = [p[1] for p in self.samples]
                margin = 0.04
                self._result = (
                    max(0, min(xs)-margin),
                    max(0, min(ys)-margin),
                    min(1, max(xs)+margin),
                    min(1, max(ys)+margin),
                )
                self.active = False
                return True
        return False

    def draw_overlay(self, frame):
        if not self.active:
            return
        H, W = frame.shape[:2]
        ov = frame.copy()
        cv2.rectangle(ov,(0,0),(W,H),(8,8,16),-1)
        cv2.addWeighted(ov,0.6,frame,0.4,0,frame)

        title, instr, (tx,ty) = self.STEPS[self.step]
        cv2.putText(frame, "CALIBRATION", (W//2-140,H//3-50),
                    cv2.FONT_HERSHEY_DUPLEX,1.1,C["gold"],2,cv2.LINE_AA)
        cv2.putText(frame, title, (W//2-180,H//3),
                    cv2.FONT_HERSHEY_DUPLEX,0.9,C["cyan"],2,cv2.LINE_AA)
        cv2.putText(frame, instr, (W//2-220,H//3+45),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,C["white"],1,cv2.LINE_AA)

        # Cible
        cx,cy = int(tx*W), int(ty*H)
        cv2.circle(frame,(cx,cy),30,C["pink"],2)
        cv2.circle(frame,(cx,cy),8,C["pink"],-1)
        cv2.line(frame,(cx-45,cy),(cx+45,cy),C["pink"],1)
        cv2.line(frame,(cx,cy-45),(cx,cy+45),C["pink"],1)

        # Barre de maintien
        if self._held > 0:
            prog = self._held / self.HOLD_FRAMES
            bx = W//2-120; by=H//3+80; bw=240; bh=12
            cv2.rectangle(frame,(bx,by),(bx+bw,by+bh),C["dgrey"],-1)
            cv2.rectangle(frame,(bx,by),(bx+int(bw*prog),by+bh),C["gold"],-1)
            cv2.rectangle(frame,(bx,by),(bx+bw,by+bh),C["grey"],1)

        # Étapes
        for i,(s,_,_) in enumerate(self.STEPS):
            col = C["gold"] if i < self.step else (C["cyan"] if i==self.step else C["grey"])
            cv2.putText(frame,f"{'✓' if i<self.step else str(i+1)}. {s}",
                        (W//2-180, H//3+130+i*26),
                        cv2.FONT_HERSHEY_SIMPLEX,0.48,col,1,cv2.LINE_AA)

# ─────────────────────────────────────────────────────────────────────────────
#  CONTRÔLEUR PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
class HandCursorController:

    def __init__(self):
        self.pm   = ProfileManager()
        self.prof = self.pm.current

        self.mp_hands = mp.solutions.hands
        self.hands    = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75,
        )

        self.smoother = HybridSmoother(self.prof)
        self.accel    = DynamicAccel(self.prof)
        self.fsm      = FSM(self.prof)
        self.hud      = HUD()
        self.mouse    = MouseThread()
        self.calib    = Calibrator()

        self.paused      = False
        self.debug       = False
        self._ctrl_scr   = None
        self._pinch_prog = 0.0
        self._cur_gest   = "IDLE"
        self._fps        = 0.0
        self._fps_buf    = deque(maxlen=30)
        self._last_ft    = time.time()

        # ROI (peut être recalibrée)
        self.roi = (self.prof.roi_margin, self.prof.roi_margin,
                    1-self.prof.roi_margin, 1-self.prof.roi_margin)

        # Scroll accumulateur
        self._scroll_acc = 0.0

    def _reload_profile(self):
        self.prof = self.pm.current
        self.smoother.reload(self.prof)
        self.accel.reload(self.prof)
        self.fsm.reload(self.prof)

    def _cam_to_screen(self, nx, ny) -> tuple[int,int]:
        nx = 1.0 - nx   # miroir
        x1,y1,x2,y2 = self.roi
        rx = (nx - x1) / max(1e-6, x2 - x1)
        ry = (ny - y1) / max(1e-6, y2 - y1)
        rx = max(0.0, min(1.0, rx))
        ry = max(0.0, min(1.0, ry))
        # Amplification centrée + accélération (déjà appliquée sur raw avant)
        sens = self.prof.sensitivity
        cx = 0.5 + (rx - 0.5) * sens
        cy = 0.5 + (ry - 0.5) * sens
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        return int(cx*SCREEN_W), int(cy*SCREEN_H)

    def _fps_update(self):
        now = time.time()
        dt  = now - self._last_ft
        self._last_ft = now
        self._fps_buf.append(1.0/max(dt, 1e-6))
        self._fps = sum(self._fps_buf)/len(self._fps_buf)

    def run(self):
        p   = self.prof
        cap = cv2.VideoCapture(p.cam_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  p.cam_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, p.cam_h)
        cap.set(cv2.CAP_PROP_FPS,          p.cam_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        if not cap.isOpened():
            print("❌  Caméra introuvable (index 0).")
            sys.exit(1)

        win = "✦ Hand Cursor v2"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1024, 576)

        self.mouse.start()

        print(__doc__)
        print(f"  Écran : {SCREEN_W}×{SCREEN_H}  ·  Profil : {p.name}\n")

        raw_gest = "IDLE"

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            H, W  = frame.shape[:2]
            self._fps_update()

            # ── Calibration en cours ────────────────────────────────────
            if self.calib.active:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                res = self.hands.process(rgb)
                if res.multi_hand_landmarks:
                    lm  = res.multi_hand_landmarks[0].landmark
                    gd  = GestureDetector(lm)
                    ix, iy = gd.index_tip()
                    done = self.calib.feed(1-ix, iy)  # miroir
                    if done and self.calib._result:
                        self.roi = self.calib._result
                        print(f"  ✅  Calibration : ROI = {self.roi}")
                self.calib.draw_overlay(frame)
                cv2.imshow(win, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'),27): break
                continue

            # ── Traitement principal ────────────────────────────────────
            if not self.paused:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                res  = self.hands.process(rgb)
                rgb.flags.writeable = True

                hand_lm  = None
                raw_gest = "IDLE"

                if res.multi_hand_landmarks:
                    hand_lm = res.multi_hand_landmarks[0].landmark
                    gd      = GestureDetector(hand_lm)
                    raw_gest, (nx, ny) = gd.detect()

                    # Accélération dynamique sur la position brute
                    k     = self.accel.coeff(nx, ny)
                    # Mapping → coordonnées écran brutes
                    sx_r, sy_r = self._cam_to_screen(nx, ny)
                    # Amplification par accélération (depuis centre écran)
                    cx = SCREEN_W//2 + (sx_r - SCREEN_W//2) * k
                    cy = SCREEN_H//2 + (sy_r - SCREEN_H//2) * k
                    sx_r = int(max(0,min(SCREEN_W-1, cx)))
                    sy_r = int(max(0,min(SCREEN_H-1, cy)))

                    # Lissage hybride
                    sx, sy = self.smoother.update(sx_r, sy_r)
                    self._ctrl_scr = (sx, sy)

                    # FSM
                    actions = self.fsm.transition(raw_gest)
                    self._pinch_prog = self.fsm.pinch_progress()

                    # Scroll : accumuler selon vitesse d'inclinaison
                    if "SCROLL" in actions:
                        speed = gd.wrist_tilt_speed()
                        self._scroll_acc += speed * self.prof.scroll_speed * 0.1
                        if abs(self._scroll_acc) >= 1.0:
                            ticks = int(self._scroll_acc)
                            for _ in range(abs(ticks)):
                                actions.append("SCROLL_UP" if ticks>0 else "SCROLL_DOWN")
                            self._scroll_acc -= ticks
                        actions = [a for a in actions if a != "SCROLL"]

                    self.mouse.push(actions, (sx, sy))
                    self._cur_gest = raw_gest

                else:
                    self.smoother.reset()
                    self.accel.reset()
                    actions = self.fsm.transition("IDLE")
                    self.mouse.push(actions, None)
                    self._ctrl_scr   = None
                    self._pinch_prog = 0.0
                    raw_gest         = "IDLE"
            else:
                hand_lm = None

            # ── HUD ─────────────────────────────────────────────────────
            frame = self.hud.draw(
                frame, self._cur_gest, raw_gest, hand_lm,
                self.fsm, self.prof,
                self.paused, self.debug,
                self._ctrl_scr, self._pinch_prog, self._fps
            )

            # ── ROI rectangle ────────────────────────────────────────────
            x1,y1,x2,y2 = self.roi
            # Miroir : x est déjà flippé à l'affichage, on flip la ROI
            rx1 = int((1-x2)*W); rx2 = int((1-x1)*W)
            ry1 = int(y1*H);     ry2 = int(y2*H)
            cv2.rectangle(frame,(rx1,ry1),(rx2,ry2),C["dgrey"],1)

            cv2.imshow(win, frame)

            # ── Clavier ─────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'),ord('Q'),27):
                break
            elif key in (ord('p'),ord('P')):
                self.paused = not self.paused
            elif key in (ord('d'),ord('D')):
                self.debug = not self.debug
            elif key in (ord('c'),ord('C')):
                self.calib.start()
                print("  🎯  Calibration lancée")
            elif key in (ord('+'),ord('=')):
                self.pm.update_sensitivity(+0.1)
                self._reload_profile()
                print(f"  Sensibilité : {self.prof.sensitivity:.1f}x")
            elif key in (ord('-'),ord('_')):
                self.pm.update_sensitivity(-0.1)
                self._reload_profile()
                print(f"  Sensibilité : {self.prof.sensitivity:.1f}x")
            elif key == ord('1'):
                self.pm.switch(0); self._reload_profile()
                print(f"  Profil : {self.prof.name}")
            elif key == ord('2'):
                self.pm.switch(1); self._reload_profile()
                print(f"  Profil : {self.prof.name}")
            elif key == ord('3'):
                self.pm.switch(2); self._reload_profile()
                print(f"  Profil : {self.prof.name}")
            elif key in (ord('r'),ord('R')):
                self.pm.profiles = copy.deepcopy(ProfileManager.DEFAULTS)
                self.pm._save()
                self._reload_profile()
                print("  ♻️  Profils réinitialisés")
            elif key in (ord('s'),ord('S')):
                fn = f"HandCursor_{datetime.now():%Y%m%d_%H%M%S}.png"
                cv2.imwrite(fn, frame)
                print(f"  📸  {fn}")

        # ── Nettoyage ────────────────────────────────────────────────────
        self.mouse.push(["DRAG_END"], None)
        time.sleep(0.05)
        self.mouse.stop()
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        print("\n  👋  Hand Cursor v2 — terminé.\n")


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        HandCursorController().run()
    except KeyboardInterrupt:
        print("\n  Arrêt clavier.")
    except Exception as e:
        import traceback
        print(f"\n❌  {e}")
        traceback.print_exc()
