import sys
import math
import random
import pygame

SKY_TOP      = (120, 185, 240)
SKY_BOT      = (200, 230, 255)
PARK_DARK    = (28,  75,  28)
PARK_MID     = (42,  100, 42)
PARK_LIGHT   = (68,  130, 52)
PARK_ACCENT  = (90,  160, 60)
BUFFER_DARK  = (148, 128, 52)
BUFFER_MID   = (178, 158, 72)
FARM_DARK    = (108, 62,  24)
FARM_MID     = (148, 90,  34)
FARM_CROP    = (192, 170, 52)
BROWN        = (90,  58,  28)
WATER_DEEP   = (38,  128, 196)
WATER_LIGHT  = (100, 180, 235)
FENCE_COL    = (130, 82,  38)
NIGHT_TINT   = (0,   10,  50,  100)
STAR_COL     = (255, 248, 220)

ACTION_MAP = {
    0: ("NO ALERT",            "●", (55,  135, 55)),
    1: ("LOW ALERT",           "▲", (205, 170, 35)),
    2: ("HIGH ALERT",          "⚠",  (215, 65,  45)),
    3: ("DEPLOY RANGER",       "★", (150, 35,  35)),
    4: ("SEND SMS TO FARMERS", "✉",  (45,  105, 195)),
    5: ("ACTIVATE DETERRENT",  "⚡", (185, 105, 25)),
}


# ── Drawing helpers ────────────────────────────────────────────────────────────

def _gradient_rect(surface, top_col, bot_col, rect):
    """Vertical gradient fill inside a pygame.Rect."""
    x, y, w, h = rect
    for i in range(h):
        t   = i / max(h - 1, 1)
        col = tuple(int(top_col[c] + (bot_col[c] - top_col[c]) * t) for c in range(3))
        pygame.draw.line(surface, col, (x, y + i), (x + w, y + i))


def _draw_sky(surface, width, height):
    ground_y = int(height * 0.08)
    _gradient_rect(surface, SKY_TOP, SKY_BOT, (0, 0, width, ground_y))


def _draw_stars(surface, width, height, seed=42):
    """Faint stars in the sky strip — visible at night."""
    rng = random.Random(seed)
    ground_y = int(height * 0.08)
    for _ in range(60):
        sx = rng.randint(0, width)
        sy = rng.randint(0, ground_y - 2)
        pygame.draw.circle(surface, STAR_COL, (sx, sy), 1)


def _draw_trees(surface, x_start, x_end, height, seed=7):
    rng = random.Random(seed)
    ground_y = int(height * 0.08)
    for _ in range(22):
        tx       = rng.randint(x_start + 12, x_end - 12)
        ty       = rng.randint(int(height * 0.18), int(height * 0.80))
        trunk_h  = rng.randint(20, 38)
        canopy_r = rng.randint(14, 26)
        # trunk
        pygame.draw.rect(surface, BROWN,
                         pygame.Rect(tx - 4, ty, 8, trunk_h), border_radius=2)
        # canopy layers
        shade = rng.randint(0, 22)
        col   = (
            max(0,   PARK_DARK[0] - shade),
            min(255, PARK_DARK[1] + shade),
            max(0,   PARK_DARK[2] - shade),
        )
        pygame.draw.circle(surface, col,        (tx, ty),              canopy_r)
        pygame.draw.circle(surface, PARK_ACCENT,(tx - 3, ty - 4),      canopy_r - 4)
        pygame.draw.circle(surface, PARK_DARK,  (tx, ty),              canopy_r, 2)


def _draw_crops(surface, x_start, x_end, height):
    row_spacing = 20
    crop_stalk  = (130, 110, 28)
    crop_head   = (192, 170, 52)
    ground_y    = int(height * 0.10)
    for row_y in range(ground_y, int(height * 0.86), row_spacing):
        pygame.draw.line(surface, FARM_DARK, (x_start, row_y), (x_end, row_y), 1)
        for cx in range(x_start + 6, x_end - 4, 13):
            pygame.draw.line(surface, crop_stalk, (cx, row_y), (cx, row_y - 11), 2)
            pygame.draw.circle(surface, crop_head, (cx, row_y - 13), 4)


def _draw_fence(surface, x, height):
    pygame.draw.line(surface, FENCE_COL, (x, 8), (x, height - 8), 3)
    for py in range(8, height - 8, 28):
        pygame.draw.rect(surface, FENCE_COL,
                         pygame.Rect(x - 5, py, 10, 16), border_radius=3)
        pygame.draw.rect(surface, (160, 110, 60),
                         pygame.Rect(x - 5, py, 10, 16), 1, border_radius=3)


def _draw_water_hole(surface, x, y, radius=24):
    glow = pygame.Surface((radius * 4, radius * 3), pygame.SRCALPHA)
    pygame.draw.ellipse(glow, (*WATER_LIGHT, 40),
                        pygame.Rect(0, 0, radius * 4, radius * 3))
    surface.blit(glow, (x - radius * 2, y - radius))

    pygame.draw.ellipse(surface, WATER_DEEP,
                        pygame.Rect(x - radius, y - radius // 2, radius * 2, radius))
    pygame.draw.ellipse(surface, WATER_LIGHT,
                        pygame.Rect(x - radius, y - radius // 2, radius * 2, radius), 2)
    # highlight
    pygame.draw.line(surface, (200, 235, 255),
                     (x - radius // 3, y - 4), (x + radius // 3, y - 4), 2)


def _draw_buffalo(surface, x, y, size=30, direction=1, leg_phase=0.0):
    s = size

    sh = pygame.Surface((s * 3, s // 2), pygame.SRCALPHA)
    pygame.draw.ellipse(sh, (0, 0, 0, 50), sh.get_rect())
    surface.blit(sh, (x - s * 3 // 2, y + int(s * 0.65)))

    # Legs
    leg_w      = max(4, int(s * 0.20))
    leg_h      = int(s * 0.58)
    leg_cols   = [(38, 28, 16), (50, 38, 22), (38, 28, 16), (50, 38, 22)]
    leg_x_offs = [-int(s * 0.62), -int(s * 0.20), int(s * 0.20), int(s * 0.62)]
    for i, ox in enumerate(leg_x_offs):
        swing = int(math.sin(leg_phase + i * math.pi * 0.5) * s * 0.16)
        lx    = x + ox - leg_w // 2
        ly    = y + int(s * 0.44) + swing
        pygame.draw.rect(surface, leg_cols[i],
                         pygame.Rect(lx, ly, leg_w, leg_h), border_radius=3)
        pygame.draw.rect(surface, (18, 12, 6),
                         pygame.Rect(lx, ly + leg_h - 5, leg_w, 6), border_radius=2)
    bw, bh = s * 2, int(s * 1.12)
    pygame.draw.ellipse(surface, (42, 32, 20),
                        pygame.Rect(x - bw // 2, y - bh // 2, bw, bh))
    pygame.draw.ellipse(surface, (58, 44, 26),
                        pygame.Rect(x - bw // 2 + 4, y - bh // 2 + 4,
                                    bw - 8, bh - 8))
    pygame.draw.ellipse(surface, (22, 16, 10),
                        pygame.Rect(x - bw // 2, y - bh // 2, bw, bh), 2)

    # Shoulder hump
    hump_x = x - direction * int(s * 0.18)
    pygame.draw.ellipse(surface, (52, 40, 24),
                        pygame.Rect(hump_x - int(s * 0.52), y - int(s * 0.92),
                                    int(s * 1.04), int(s * 0.72)))

    # Head
    hx      = x + direction * int(s * 0.98)
    hy      = y - int(s * 0.16)
    head_w  = int(s * 0.88)
    head_h  = int(s * 0.76)
    pygame.draw.ellipse(surface, (52, 40, 26),
                        pygame.Rect(hx - head_w // 2, hy - head_h // 2, head_w, head_h))
    pygame.draw.ellipse(surface, (22, 16, 10),
                        pygame.Rect(hx - head_w // 2, hy - head_h // 2, head_w, head_h), 2)

    sx = hx + direction * int(s * 0.40)
    sy = hy + int(s * 0.14)
    pygame.draw.ellipse(surface, (68, 52, 36),
                        pygame.Rect(sx - int(s * 0.24), sy - int(s * 0.16),
                                    int(s * 0.46), int(s * 0.32)))
    for side in (-1, 1):
        pygame.draw.circle(surface, (18, 12, 6),
                           (sx + side * int(s * 0.09), sy + int(s * 0.06)),
                           max(2, int(s * 0.07)))

    
    hbx = hx + direction * int(s * 0.06)
    hby = hy - int(s * 0.34)
    for side in (-1, 1):
        pts = [
            (hbx + side * int(s * 0.16), hby),
            (hbx + side * int(s * 0.44), hby - int(s * 0.60)),
            (hbx + side * int(s * 0.60), hby - int(s * 0.38)),
            (hbx + side * int(s * 0.38), hby - int(s * 0.10)),
        ]
        pygame.draw.polygon(surface, (26, 16, 6),  pts)
        pygame.draw.polygon(surface, (14,  9, 4),  pts, 1)

   
    ex = hx + direction * int(s * 0.24)
    ey = hy - int(s * 0.08)
    pygame.draw.circle(surface, (215, 195, 155), (ex, ey), max(3, int(s * 0.12)))
    pygame.draw.circle(surface, (8,    6,    4), (ex, ey), max(2, int(s * 0.07)))
    pygame.draw.circle(surface, (255, 255, 255), (ex + 1, ey - 1), max(1, int(s * 0.03)))

  
    tx = x - direction * int(s * 0.98)
    ty = y - int(s * 0.08)
    tail_pts = [
        (tx, ty),
        (tx - direction * int(s * 0.26), ty - int(s * 0.36)),
        (tx - direction * int(s * 0.16), ty - int(s * 0.54)),
    ]
    pygame.draw.lines(surface, (32, 24, 14), False, tail_pts, 3)
    pygame.draw.circle(surface, (18, 12, 6),
                       (tx - direction * int(s * 0.16), ty - int(s * 0.58)), 4)



class Renderer:

    def __init__(self, width: int = 1060, height: int = 640):
        pygame.init()
        self.W = width
        self.H = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("ULINZI  ·  Wildlife Conflict Early Warning")
        self.clock = pygame.time.Clock()

        self.park_end   = 310
        self.buffer_end = 670
        self.MAX_DIST   = 10_000.0
        self.fnt_tiny   = pygame.font.SysFont("consolas",  13)
        self.fnt_small  = pygame.font.SysFont("consolas",  15)
        self.fnt_label  = pygame.font.SysFont("arial",     14, bold=True)
        self.fnt_medium = pygame.font.SysFont("arial",     20, bold=True)
        self.fnt_large  = pygame.font.SysFont("arial",     27, bold=True)
        self.fnt_title  = pygame.font.SysFont("arial",     13, bold=True)

        self.is_open        = True
        self._leg_phase     = 0.0
        self._last_distance = self.MAX_DIST
        self._star_surf     = None   # cached star layer

        self._bg      = self._build_background()
        self._star_surf = self._build_stars()

    def _build_background(self) -> pygame.Surface:
        bg       = pygame.Surface((self.W, self.H))
        ground_y = int(self.H * 0.08)

        # Sky gradient
        _gradient_rect(bg, SKY_TOP, SKY_BOT, (0, 0, self.W, ground_y + 4))

        _gradient_rect(bg, PARK_MID,   PARK_DARK,
                       (0, ground_y, self.park_end, self.H - ground_y))
        _gradient_rect(bg, BUFFER_MID, BUFFER_DARK,
                       (self.park_end, ground_y,
                        self.buffer_end - self.park_end, self.H - ground_y))
        _gradient_rect(bg, FARM_MID,   FARM_DARK,
                       (self.buffer_end, ground_y,
                        self.W - self.buffer_end, self.H - ground_y))

        for i in range(0, self.park_end, 38):
            pygame.draw.rect(bg, PARK_LIGHT,
                             pygame.Rect(i, ground_y, 18, self.H - ground_y))
        for i in range(self.park_end, self.buffer_end, 34):
            pygame.draw.rect(bg, BUFFER_DARK,
                             pygame.Rect(i, ground_y, 13, self.H - ground_y))
        for i in range(self.buffer_end, self.W, 26):
            pygame.draw.rect(bg, FARM_DARK,
                             pygame.Rect(i, ground_y, 12, self.H - ground_y))

        # Horizon line
        pygame.draw.line(bg, (80, 60, 30),
                         (0, ground_y), (self.W, ground_y), 2)

        _draw_trees(bg, 0, self.park_end, self.H, seed=7)
        _draw_crops(bg, self.buffer_end + 10, self.W - 5, self.H)
        _draw_water_hole(bg, int(self.park_end * 0.36), int(self.H * 0.66))
        _draw_fence(bg, self.park_end, self.H)

        
        _draw_fence(bg, self.buffer_end, self.H)

        # Zone labels
        for text, cx, col in [
            ("NATIONAL PARK",
             self.park_end // 2,                              PARK_LIGHT),
            ("BUFFER ZONE",
             (self.park_end + self.buffer_end) // 2,          (225, 205, 105)),
            ("FARMLAND",
             (self.buffer_end + self.W) // 2,                 FARM_CROP),
        ]:
            surf = self.fnt_title.render(text, True, col)
            bg.blit(surf, (cx - surf.get_width() // 2, 14))

        return bg

    def _build_stars(self) -> pygame.Surface:
        surf = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        _draw_stars(surf, self.W, self.H, seed=99)
        return surf

    def draw(self, state, step: int, action: int = None) -> None:
        if not self.is_open:
            return

        (step_length, speed, turning_angle,
         distance, ndvi, time_of_day,
         season, dist_water, conflict_hist) = state

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit()

        # Leg animation
        moved = self._last_distance - distance
        if moved > 0:
            self._leg_phase += moved * 0.020
        self._last_distance = distance

        self.screen.blit(self._bg, (0, 0))

        is_night = (time_of_day >= 18) or (time_of_day <= 6)

        if is_night:
            # Stars appear
            self.screen.blit(self._star_surf, (0, 0))
            # Night tint overlay
            night = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
            night.fill(NIGHT_TINT)
            self.screen.blit(night, (0, 0))

        ratio = 1.0 - (distance / self.MAX_DIST)
        ratio = max(0.0, min(1.0, ratio))
        x_pos = int(ratio * self.W)
        x_pos = max(46, min(self.W - 46, x_pos))
        y_pos = int(self.H * 0.52)

        direction = 1 if x_pos < self.buffer_end else -1

        if distance <= 1200:
            pulse = abs(math.sin(pygame.time.get_ticks() * 0.004)) * 0.5 + 0.5
            if distance <= 200:
                ring_col = (255, 60, 20)
                rings    = 3
            elif distance <= 1000:
                ring_col = (255, 190, 0)
                rings    = 2
            else:
                ring_col = (255, 230, 80)
                rings    = 1

            for r in range(rings):
                radius  = 52 + r * 20 + int(pulse * 8)
                alpha   = int(200 - r * 50)
                ring_s  = pygame.Surface((radius * 2 + 4, radius * 2 + 4),
                                         pygame.SRCALPHA)
                pygame.draw.circle(ring_s, (*ring_col, alpha),
                                   (radius + 2, radius + 2), radius, 3)
                self.screen.blit(ring_s,
                                 (x_pos - radius - 2, y_pos - radius - 2))

        _draw_buffalo(self.screen, x_pos, y_pos,
                      size=30, direction=direction,
                      leg_phase=self._leg_phase)

        self._draw_hud(distance, dist_water, speed, ndvi,
                       time_of_day, season, conflict_hist,
                       step, is_night, turning_angle, step_length)

        self._draw_distance_bar(ratio)

        if action is not None:
            self._draw_action_banner(action)

        pygame.display.flip()
        self.clock.tick(10)

    def _draw_hud(self, dist_farm, dist_water, speed, ndvi,
                  time_of_day, season, conflict_hist,
                  step, is_night, turning_angle, step_length):

        PW, PH = 262, 222
        PX, PY = 8, int(self.H * 0.08) + 6

        # Panel background — dark glass
        panel = pygame.Surface((PW, PH), pygame.SRCALPHA)
        panel.fill((8, 14, 10, 175))
        pygame.draw.rect(panel, (80, 160, 80, 120),
                         pygame.Rect(0, 0, PW, PH), 1, border_radius=6)
        self.screen.blit(panel, (PX, PY))

        # Title bar
        title_bg = pygame.Surface((PW, 22), pygame.SRCALPHA)
        title_bg.fill((30, 80, 30, 200))
        self.screen.blit(title_bg, (PX, PY))
        t = self.fnt_label.render("ULINZI  ·  LIVE TELEMETRY", True, (140, 220, 140))
        self.screen.blit(t, (PX + PW // 2 - t.get_width() // 2, PY + 4))

        season_str = "DRY  " if season == 1 else "WET  "
        night_str  = "NIGHT " if is_night else "DAY "
        hour       = int(time_of_day)
        minute     = int((time_of_day % 1) * 60)

        # Colour-code distance
        if dist_farm <= 200:
            dist_col = (255, 80,  60)
        elif dist_farm <= 1000:
            dist_col = (255, 200, 40)
        elif dist_farm <= 3000:
            dist_col = (200, 220, 80)
        else:
            dist_col = (140, 220, 140)

        rows = [
            ("STEP",      f"{step:03d}",                    (160, 220, 160)),
            ("DIST FARM", f"{dist_farm:7.1f} m",            dist_col),
            ("DIST WATER",f"{dist_water:7.1f} m",           (100, 180, 230)),
            ("SPEED",     f"{speed:.2f} m/s",               (160, 220, 160)),
            ("NDVI",      f"{ndvi:.3f}",                    (120, 200, 100)),
            ("TURN ANG",  f"{turning_angle:.1f}°",          (160, 220, 160)),
            ("SEASON",    season_str,                        (220, 200, 100)),
            ("TIME",      f"{hour:02d}:{minute:02d}  {night_str}", (160, 220, 200)),
            ("CONFLICTS", f"{int(conflict_hist):02d}",
             (255, 80, 60) if conflict_hist > 0 else (140, 220, 140)),
        ]

        for i, (key, val, col) in enumerate(rows):
            ky = PY + 28 + i * 21
            k_surf = self.fnt_tiny.render(f"{key:<10}", True, (100, 150, 100))
            v_surf = self.fnt_small.render(val,         True, col)
            self.screen.blit(k_surf, (PX + 8,  ky))
            self.screen.blit(v_surf, (PX + 118, ky))

    def _draw_distance_bar(self, ratio: float):
        BAR_H  = 16
        BAR_W  = self.W - 44
        BAR_X  = 22
        BAR_Y  = self.H - 72

        # Track
        pygame.draw.rect(self.screen, (30, 30, 30),
                         (BAR_X, BAR_Y, BAR_W, BAR_H), border_radius=8)

        # Zone colour segments
        park_w   = int(BAR_W * 0.30)
        buffer_w = int(BAR_W * 0.35)
        farm_w   = BAR_W - park_w - buffer_w

        pygame.draw.rect(self.screen, (42, 100, 42),
                         (BAR_X,                     BAR_Y, park_w,   BAR_H))
        pygame.draw.rect(self.screen, (178, 158, 72),
                         (BAR_X + park_w,            BAR_Y, buffer_w, BAR_H))
        pygame.draw.rect(self.screen, (148, 90, 34),
                         (BAR_X + park_w + buffer_w, BAR_Y, farm_w,   BAR_H))


        marker_x = BAR_X + int(ratio * BAR_W)
        marker_x = max(BAR_X + 6, min(BAR_X + BAR_W - 6, marker_x))

        pygame.draw.polygon(self.screen, (255, 255, 255), [
            (marker_x,     BAR_Y - 8),
            (marker_x - 6, BAR_Y - 14),
            (marker_x + 6, BAR_Y - 14),
        ])
        pygame.draw.circle(self.screen, (240, 220, 160),
                           (marker_x, BAR_Y + BAR_H // 2), 7)
        pygame.draw.circle(self.screen, (20, 14, 8),
                           (marker_x, BAR_Y + BAR_H // 2), 7, 2)

        # Border
        pygame.draw.rect(self.screen, (160, 160, 160),
                         (BAR_X, BAR_Y, BAR_W, BAR_H), 1, border_radius=8)

        # Label
        lbl = self.fnt_tiny.render(
            "◀  PARK          BUFFER ZONE          FARMLAND  ▶",
            True, (190, 190, 190))
        self.screen.blit(lbl,
                         (BAR_X + BAR_W // 2 - lbl.get_width() // 2,
                          BAR_Y - 18))


    def _draw_action_banner(self, action: int):
        label, icon, colour = ACTION_MAP.get(action, (str(action), "?", (60, 60, 60)))

        BANNER_H = 50
        banner   = pygame.Surface((self.W, BANNER_H), pygame.SRCALPHA)

        # Gradient banner
        for i in range(BANNER_H):
            t   = i / (BANNER_H - 1)
            col = tuple(int(colour[c] * (1 - t * 0.35)) for c in range(3))
            pygame.draw.line(banner, (*col, 235),
                             (0, i), (self.W, i))

        # Top edge highlight
        pygame.draw.line(banner, (255, 255, 255, 80), (0, 0), (self.W, 0), 2)

        self.screen.blit(banner, (0, self.H - BANNER_H))

        text  = f"  {icon}   AGENT ACTION :  {label}"
        t_sur = self.fnt_large.render(text, True, (255, 255, 255))
        # Subtle shadow
        sh    = self.fnt_large.render(text, True, (0, 0, 0))
        ty    = self.H - BANNER_H + (BANNER_H - t_sur.get_height()) // 2
        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            self.screen.blit(sh, (18 + dx, ty + dy))
        self.screen.blit(t_sur, (18, ty))


    def close(self):
        if self.is_open:
            self.is_open = False
            pygame.quit()


def demo():
    """
    Self-contained demo: simulates a buffalo moving from park to farmland
    with rule-based actions. No RL model required.
    """
    renderer      = Renderer()
    distance      = 9800.0
    dist_water    = 3200.0
    time_of_day   = 6.5
    conflict_hist = 0
    step          = 0

    clock = pygame.time.Clock()

    while renderer.is_open:
        speed         = 2.8 + math.sin(step * 0.05) * 0.8
        step_length   = speed * 60.0
        turning_angle = max(5.0, 90.0 - (1.0 - distance / 9800.0) * 85.0)
        distance      = max(0.0, distance - speed * 14)
        dist_water    = max(0.0, dist_water - speed * 2.5)
        time_of_day   = (time_of_day + 0.05) % 24
        ndvi          = 0.35 + math.sin(step * 0.02) * 0.10
        season        = 1   

        if distance <= 200:
            conflict_hist = min(10, conflict_hist + 1)

        state = (step_length, speed, turning_angle,
                 distance, ndvi, time_of_day,
                 season, dist_water, float(conflict_hist))

        if distance > 5000:
            action = 0
        elif distance > 3000:
            action = 1
        elif distance > 1500:
            action = 2
        elif distance > 800:
            action = 3
        elif distance > 300:
            action = 4
        else:
            action = 5

        renderer.draw(state, step, action)
        step += 1
        clock.tick(10)

        if distance <= 0:
            distance      = 9800.0
            dist_water    = 3200.0
            conflict_hist = 0
            step          = 0

    renderer.close()


if __name__ == "__main__":
    demo()