import pygame
import pygame.gfxdraw
import numpy as np
import math
import sys
from typing import Optional, Dict, Any, List


WIDTH, HEIGHT = 1100, 700
FPS = 4

PARK_GREEN = (34, 85, 34)
FARMLAND_TAN = (194, 160, 105)
BOUNDARY_RED = (220, 50, 47)
SKY_DAWN = (255, 200, 120)
SKY_DAY = (135, 206, 235)
SKY_DUSK = (255, 120, 60)
SKY_NIGHT = (15, 20, 50)

BUFFALO_BODY = (60, 40, 20)
BUFFALO_HORN = (200, 180, 100)
BUFFALO_EYE = (255, 80, 80)

ALERT_COLORS = {
    0: (80, 80, 80),
    1: (30, 144, 255),
    2: (255, 165, 0),
    3: (148, 0, 211),
    4: (255, 215, 0),
    5: (220, 20, 60),
}

ALERT_LABELS = [
    "NO ALERT",
    "COMMUNITY SMS",
    "RANGER DISPATCH",
    "SCARE DEVICE",
    "ELEVATED WATCH",
    "EMERGENCY BROADCAST",
]

FEATURE_NAMES = [
    "Boundary Proximity",
    "Speed",
    "Heading Change",
    "Displacement X",
    "Displacement Y",
    "Time of Day",
    "Vegetation Density",
    "Herd Cohesion",
    "Crossing History",
]


def _lerp_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def _get_sky_color(time_norm):
    t = (time_norm + 1) / 2.0
    if t < 0.25:
        return _lerp_color(SKY_NIGHT, SKY_DAWN, t / 0.25)
    elif t < 0.5:
        return _lerp_color(SKY_DAWN, SKY_DAY, (t - 0.25) / 0.25)
    elif t < 0.75:
        return _lerp_color(SKY_DAY, SKY_DUSK, (t - 0.5) / 0.25)
    else:
        return _lerp_color(SKY_DUSK, SKY_NIGHT, (t - 0.75) / 0.25)


class UlinziRenderer:
    def __init__(self, surface: Optional[pygame.Surface] = None):
        if not pygame.get_init():
            pygame.init()

        if surface is not None:
            self.screen = surface
            self._owns_display = False
        else:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("ULINZI — Wildlife Alert System | Volcanoes National Park, Rwanda")
            self._owns_display = True

        self.clock = pygame.time.Clock()
        self._load_fonts()

        self.buffalo_x = 260
        self.buffalo_y = HEIGHT // 2
        self.trail: List[tuple] = []
        self.reward_particles: List[dict] = []
        self.alert_flash = 0
        self.last_action = 0
        self.step_count = 0
        self.total_reward = 0.0

    def _load_fonts(self):
        try:
            self.font_title = pygame.font.SysFont("dejavusans", 22, bold=True)
            self.font_label = pygame.font.SysFont("dejavusans", 14)
            self.font_small = pygame.font.SysFont("dejavusans", 12)
            self.font_alert = pygame.font.SysFont("dejavusans", 18, bold=True)
            self.font_stat = pygame.font.SysFont("dejavumono", 13)
        except Exception:
            self.font_title = pygame.font.Font(None, 26)
            self.font_label = pygame.font.Font(None, 18)
            self.font_small = pygame.font.Font(None, 15)
            self.font_alert = pygame.font.Font(None, 22)
            self.font_stat = pygame.font.Font(None, 16)

    def reset(self, obs: np.ndarray):
        self.buffalo_x = 260
        self.buffalo_y = HEIGHT // 2
        self.trail = []
        self.reward_particles = []
        self.alert_flash = 0
        self.step_count = 0
        self.total_reward = 0.0

    def _draw_background(self, time_norm: float, boundary_proximity: float):
        sky = _get_sky_color(time_norm)
        self.screen.fill(sky, rect=(0, 0, WIDTH, 160))

        park_rect = pygame.Rect(0, 160, 520, HEIGHT - 160)
        pygame.draw.rect(self.screen, PARK_GREEN, park_rect)

        for _ in range(40):
            rng = np.random.default_rng(abs(hash(str(boundary_proximity * 100))) % (2**32))
            gx = int(rng.uniform(10, 510))
            gy = int(rng.uniform(180, HEIGHT - 20))
            gr = int(rng.uniform(6, 22))
            col = _lerp_color((20, 70, 20), (50, 110, 50), rng.random())
            pygame.gfxdraw.filled_ellipse(self.screen, gx, gy, gr, int(gr * 0.6), col)

        farm_rect = pygame.Rect(520, 160, WIDTH - 520, HEIGHT - 160)
        pygame.draw.rect(self.screen, FARMLAND_TAN, farm_rect)

        row_color = _lerp_color(FARMLAND_TAN, (160, 120, 60), 0.3)
        for ry in range(180, HEIGHT, 28):
            pygame.draw.line(self.screen, row_color, (520, ry), (WIDTH - 220, ry), 2)

        hut_positions = [(680, 320), (740, 420), (790, 280), (650, 500)]
        for hx, hy in hut_positions:
            pygame.draw.circle(self.screen, (180, 140, 80), (hx, hy), 14)
            roof_pts = [(hx, hy - 22), (hx - 18, hy - 2), (hx + 18, hy - 2)]
            pygame.draw.polygon(self.screen, (139, 69, 19), roof_pts)

    def _draw_boundary(self, boundary_proximity: float):
        pulse = abs(math.sin(pygame.time.get_ticks() * 0.003))
        alpha = int(120 + 135 * boundary_proximity * pulse)
        alpha = min(255, alpha)
        bx = 520
        color = _lerp_color((200, 200, 50), BOUNDARY_RED, boundary_proximity)
        for i in range(3):
            lw = 3 - i
            pygame.draw.line(self.screen, (*color, max(0, alpha - i * 40)), (bx, 160), (bx, HEIGHT), lw)

        label = self.font_small.render("PARK BOUNDARY", True, BOUNDARY_RED)
        self.screen.blit(label, (bx - label.get_width() // 2, 165))

        pygame.draw.rect(self.screen, (40, 60, 40), (0, 155, 520, 10))
        pygame.draw.rect(self.screen, FARMLAND_TAN, (520, 155, WIDTH - 520, 10))

        park_lbl = self.font_small.render("VOLCANOES NATIONAL PARK", True, (220, 255, 220))
        self.screen.blit(park_lbl, (20, 168))
        farm_lbl = self.font_small.render("KINIGI FARMLAND — MUSANZE DISTRICT", True, (80, 50, 10))
        self.screen.blit(farm_lbl, (530, 168))

    def _compute_buffalo_position(self, obs: np.ndarray) -> tuple:
        bp = float(obs[0])
        dx = float(obs[3])
        dy = float(obs[4])
        target_x = int(40 + bp * 460)
        target_y = int(HEIGHT // 2 + dy * 120)
        target_y = max(190, min(HEIGHT - 40, target_y))
        self.buffalo_x = int(self.buffalo_x * 0.7 + target_x * 0.3)
        self.buffalo_y = int(self.buffalo_y * 0.7 + target_y * 0.3)
        return self.buffalo_x, self.buffalo_y

    def _draw_trail(self):
        if len(self.trail) < 2:
            return
        for i in range(1, len(self.trail)):
            alpha = int(255 * i / len(self.trail))
            col = _lerp_color((20, 20, 20), BUFFALO_BODY, i / len(self.trail))
            pygame.draw.line(self.screen, col, self.trail[i - 1], self.trail[i], 2)

    def _draw_buffalo(self, bx: int, by: int, speed: float, action: int):
        body_r = 22
        pygame.gfxdraw.filled_ellipse(self.screen, bx, by, body_r, int(body_r * 0.65), BUFFALO_BODY)
        pygame.gfxdraw.filled_ellipse(self.screen, bx + body_r - 6, by - 4, 14, 11, BUFFALO_BODY)

        pygame.draw.line(self.screen, BUFFALO_HORN, (bx + body_r + 4, by - 12), (bx + body_r + 16, by - 22), 3)
        pygame.draw.line(self.screen, BUFFALO_HORN, (bx + body_r + 4, by - 10), (bx + body_r + 14, by - 5), 3)

        pygame.draw.circle(self.screen, BUFFALO_EYE, (bx + body_r + 4, by - 6), 3)
        pygame.draw.circle(self.screen, (0, 0, 0), (bx + body_r + 5, by - 6), 1)

        if action > 0:
            alert_col = ALERT_COLORS[action]
            t = (pygame.time.get_ticks() % 1000) / 1000.0
            ring_r = int(body_r + 10 + 15 * t)
            ring_alpha = int(200 * (1 - t))
            pygame.gfxdraw.circle(self.screen, bx, by, ring_r, (*alert_col, ring_alpha))

        if speed > 0.5:
            for i in range(3):
                mx = bx - 30 - i * 12
                my = by + np.random.randint(-4, 5)
                pygame.draw.circle(self.screen, (150, 110, 60), (mx, my), 3)

    def _draw_herd(self, bx: int, by: int, cohesion: float):
        count = int(3 + cohesion * 5)
        rng = np.random.default_rng(42)
        spread = int(80 * (1 - cohesion) + 20)
        for i in range(count):
            ox = int(rng.uniform(-spread, spread))
            oy = int(rng.uniform(-spread // 2, spread // 2))
            hx, hy = bx + ox, by + oy
            if 0 < hx < 510 and 190 < hy < HEIGHT - 10:
                pygame.gfxdraw.filled_ellipse(self.screen, hx, hy, 12, 8, _lerp_color(BUFFALO_BODY, (80, 60, 30), 0.4))

    def _draw_risk_meter(self, risk: float):
        mx, my, mw, mh = 30, 30, 200, 22
        pygame.draw.rect(self.screen, (30, 30, 30), (mx, my, mw, mh), border_radius=4)
        bar_color = _lerp_color((50, 200, 50), BOUNDARY_RED, risk)
        bar_w = int(mw * risk)
        if bar_w > 0:
            pygame.draw.rect(self.screen, bar_color, (mx, my, bar_w, mh), border_radius=4)
        pygame.draw.rect(self.screen, (200, 200, 200), (mx, my, mw, mh), 1, border_radius=4)
        lbl = self.font_small.render(f"RISK  {risk:.2f}", True, (240, 240, 240))
        self.screen.blit(lbl, (mx + 4, my + 4))

    def _draw_alert_panel(self, action: int, reward: float):
        px, py, pw, ph = WIDTH - 210, 10, 200, HEIGHT - 20
        panel_surf = pygame.Surface((pw, ph), pygame.SRCALPHA)
        panel_surf.fill((10, 10, 20, 210))
        self.screen.blit(panel_surf, (px, py))
        pygame.draw.rect(self.screen, (80, 80, 120), (px, py, pw, ph), 1)

        title = self.font_title.render("ULINZI", True, (255, 200, 50))
        self.screen.blit(title, (px + pw // 2 - title.get_width() // 2, py + 8))
        sub = self.font_small.render("Wildlife Alert System", True, (180, 180, 200))
        self.screen.blit(sub, (px + pw // 2 - sub.get_width() // 2, py + 32))

        pygame.draw.line(self.screen, (80, 80, 120), (px + 10, py + 50), (px + pw - 10, py + 50))

        al_title = self.font_label.render("ACTIVE ALERT", True, (180, 180, 200))
        self.screen.blit(al_title, (px + 10, py + 58))

        ac = ALERT_COLORS[action]
        pygame.draw.rect(self.screen, ac, (px + 10, py + 76, pw - 20, 36), border_radius=6)
        al_lbl = self.font_alert.render(ALERT_LABELS[action], True, (255, 255, 255))
        self.screen.blit(al_lbl, (px + pw // 2 - al_lbl.get_width() // 2, py + 84))

        pygame.draw.line(self.screen, (80, 80, 120), (px + 10, py + 120), (px + pw - 10, py + 120))

        rw_title = self.font_label.render("STEP REWARD", True, (180, 180, 200))
        self.screen.blit(rw_title, (px + 10, py + 128))
        rw_col = (80, 220, 80) if reward >= 0 else (220, 80, 80)
        rw_val = self.font_title.render(f"{reward:+.1f}", True, rw_col)
        self.screen.blit(rw_val, (px + 10, py + 146))

        cum_lbl = self.font_label.render("CUMULATIVE", True, (180, 180, 200))
        self.screen.blit(cum_lbl, (px + 10, py + 172))
        cum_col = (80, 220, 80) if self.total_reward >= 0 else (220, 80, 80)
        cum_val = self.font_title.render(f"{self.total_reward:+.1f}", True, cum_col)
        self.screen.blit(cum_val, (px + 10, py + 190))

        pygame.draw.line(self.screen, (80, 80, 120), (px + 10, py + 218), (px + pw - 10, py + 218))

        step_lbl = self.font_label.render(f"Step: {self.step_count} / 48", True, (180, 180, 200))
        self.screen.blit(step_lbl, (px + 10, py + 226))

    def _draw_obs_panel(self, obs: np.ndarray):
        px, py, pw = 10, HEIGHT - 185, 500
        panel_surf = pygame.Surface((pw, 178), pygame.SRCALPHA)
        panel_surf.fill((10, 10, 20, 200))
        self.screen.blit(panel_surf, (px, py))
        pygame.draw.rect(self.screen, (80, 80, 120), (px, py, pw, 178), 1)

        title = self.font_label.render("OBSERVATION VECTOR  (30-min GPS collar snapshot)", True, (200, 200, 255))
        self.screen.blit(title, (px + 8, py + 6))

        for i, (name, val) in enumerate(zip(FEATURE_NAMES, obs)):
            row_y = py + 26 + i * 17
            name_surf = self.font_stat.render(f"{name:<22}", True, (160, 200, 160))
            self.screen.blit(name_surf, (px + 8, row_y))

            bar_x = px + 195
            bar_w = 200
            bar_h = 10
            pygame.draw.rect(self.screen, (40, 40, 60), (bar_x, row_y + 2, bar_w, bar_h))
            norm_val = np.clip((float(val) + 1) / 2.0, 0, 1)
            bar_fill = int(bar_w * norm_val)
            if bar_fill > 0:
                col = _lerp_color((50, 180, 255), (255, 80, 80), norm_val)
                pygame.draw.rect(self.screen, col, (bar_x, row_y + 2, bar_fill, bar_h))

            val_surf = self.font_stat.render(f"{val:+.3f}", True, (220, 220, 150))
            self.screen.blit(val_surf, (bar_x + bar_w + 8, row_y))

    def _draw_alert_legend(self):
        lx, ly = 530, HEIGHT - 108
        lbl = self.font_small.render("ALERT LEVELS", True, (200, 200, 200))
        self.screen.blit(lbl, (lx, ly))
        for i, (label, color) in enumerate(zip(ALERT_LABELS, ALERT_COLORS.values())):
            ry = ly + 16 + i * 16
            pygame.draw.rect(self.screen, color, (lx, ry, 12, 12), border_radius=2)
            txt = self.font_small.render(label, True, (200, 200, 200))
            self.screen.blit(txt, (lx + 16, ry))

    def draw_frame(self, obs: np.ndarray, action: int, reward: float, info: Dict[str, Any]):
        time_norm = float(obs[5]) if len(obs) > 5 else 0.0
        bp = float(obs[0])
        speed = float(obs[1])
        cohesion = float(obs[7]) if len(obs) > 7 else 0.5
        risk = float(info.get("risk_score", 0.5 * bp + 0.3 * speed))

        self._draw_background(time_norm, bp)
        self._draw_boundary(bp)

        bx, by = self._compute_buffalo_position(obs)
        self.trail.append((bx, by))
        if len(self.trail) > 30:
            self.trail.pop(0)

        self._draw_trail()
        self._draw_herd(bx, by, cohesion)
        self._draw_buffalo(bx, by, speed, action)
        self._draw_risk_meter(risk)
        self._draw_obs_panel(obs)
        self._draw_alert_legend()
        self._draw_alert_panel(action, reward)

    def render(self, obs: np.ndarray, action: int, reward: float, info: Dict[str, Any]):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.close()
                sys.exit()

        self.last_action = action
        self.total_reward += reward
        self.step_count += 1

        self.draw_frame(obs, action, reward, info)

        if self._owns_display:
            pygame.display.flip()
            self.clock.tick(FPS)

    def close(self):
        if self._owns_display and pygame.get_init():
            pygame.quit()


def run_random_demo():
    import sys
    sys.path.insert(0, ".")
    from environment.custom_env import UlinziEnv

    env = UlinziEnv(render_mode="human", max_steps=48)
    renderer = UlinziRenderer()
    env.renderer = renderer

    obs, info = env.reset(seed=42)
    renderer.reset(obs)

    print("\n" + "=" * 60)
    print("  ULINZI — Random Agent Demo (No Model)")
    print("  Volcanoes National Park, Rwanda")
    print("=" * 60)

    done = False
    step = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        renderer.render(obs, action, reward, info)

        print(
            f"  Step {step+1:02d} | Action: {info['action_label']:<22} "
            f"| Risk: {info['risk_score']:.3f} | Reward: {reward:+.2f} "
            f"| Cumulative: {info['total_reward']:+.2f}"
        )
        step += 1

    env.close()
    print("\n  Demo complete.")


if __name__ == "__main__":
    run_random_demo()