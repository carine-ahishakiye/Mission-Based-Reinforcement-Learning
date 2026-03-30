import sys
import pygame


WHITE      = (245, 245, 245)
BLACK      = (40,  40,  40)
GREEN      = (60,  120, 60)    
YELLOW     = (200, 180, 80)    
RED        = (160, 60,  60)   
BLUE       = (70,  130, 180)   
ORANGE     = (220, 120, 40)  
DARK_GREEN = (30,  80,  30)     

ACTION_MAP = {
    0: "NO ALERT",
    1: "LOW ALERT",
    2: "HIGH ALERT",
    3: "DEPLOY RANGER",
    4: "SEND SMS TO FARMERS",
    5: "ACTIVATE DETERRENT",
}

ACTION_COLOURS = {
    0: (100, 160, 100),  
    1: (200, 180, 60),  
    2: (200, 80,  60),   
    3: (180, 60,  60),  
    4: (60,  120, 200), 
    5: (180, 120, 40),    
}


class Renderer:
  

    def __init__(self, width: int = 900, height: int = 600):
        pygame.init()
        self.width  = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("ULINZI – Wildlife Conflict Early Warning")
        self.clock = pygame.time.Clock()

        self.park_end   = 250  
        self.buffer_end = 600   

        
        self.MAX_DISTANCE = 10000.0

        # Fonts
        self.font_small  = pygame.font.SysFont("monospace", 18)
        self.font_medium = pygame.font.SysFont("monospace", 22, bold=True)
        self.font_large  = pygame.font.SysFont("monospace", 28, bold=True)

        self.is_open = True

    def draw(self, state, step: int, action: int = None):
       
        if not self.is_open:
            return

        (step_length, speed, turning_angle,
         distance, ndvi, time_of_day,
         season, dist_water, conflict_hist) = state

       
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit()
        self.screen.fill(WHITE)

        pygame.draw.rect(self.screen, GREEN,
                         (0, 0, self.park_end, self.height))
        pygame.draw.rect(self.screen, YELLOW,
                         (self.park_end, 0,
                          self.buffer_end - self.park_end, self.height))
        pygame.draw.rect(self.screen, RED,
                         (self.buffer_end, 0,
                          self.width - self.buffer_end, self.height))

        # Zone labels
        self._label(self.font_small, "NATIONAL PARK", DARK_GREEN, 10,  10)
        self._label(self.font_small, "BUFFER ZONE",   BLACK,      270, 10)
        self._label(self.font_small, "FARMLAND",      WHITE,      620, 10)

        # Zone boundary lines
        pygame.draw.line(self.screen, BLACK,
                         (self.park_end,   0), (self.park_end,   self.height), 2)
        pygame.draw.line(self.screen, BLACK,
                         (self.buffer_end, 0), (self.buffer_end, self.height), 2)

        ratio = 1.0 - (distance / self.MAX_DISTANCE)
        ratio = max(0.0, min(1.0, ratio))      
        x_pos = int(ratio * self.width)
        x_pos = max(14, min(self.width - 14, x_pos)) 
        y_pos = self.height // 2

        pygame.draw.circle(self.screen, BLUE,  (x_pos, y_pos), 14)
        pygame.draw.circle(self.screen, BLACK, (x_pos, y_pos), 14, 2)
        self._label(self.font_small, "Buffalo", BLUE, x_pos - 22, y_pos + 18)

        season_str = "DRY"   if season == 1                          else "WET"
        night_str  = "NIGHT" if (time_of_day >= 18 or time_of_day <= 6) else "DAY"
        hour       = int(time_of_day)
        minute     = int((time_of_day % 1) * 60)

        lines = [
            f"Step          : {step:03d}",
            f"Distance Farm : {distance:7.1f} m",
            f"Distance Water: {dist_water:7.1f} m",
            f"Speed         : {speed:.2f} m/s",
            f"NDVI          : {ndvi:.3f}",
            f"Season        : {season_str}",
            f"Time          : {hour:02d}:{minute:02d}  ({night_str})",
            f"Conflicts     : {int(conflict_hist)}",
        ]
        panel_x, panel_y = 10, 40
        for i, line in enumerate(lines):
            surf = self.font_small.render(line, True, BLACK)
            self.screen.blit(surf, (panel_x, panel_y + i * 22))

    
        if action is not None:
            label       = ACTION_MAP.get(action, str(action))
            colour      = ACTION_COLOURS.get(action, BLACK)
            banner_rect = pygame.Rect(0, self.height - 50, self.width, 50)
            pygame.draw.rect(self.screen, colour, banner_rect)
            action_surf = self.font_large.render(
                f"AGENT ACTION:  {label}", True, WHITE
            )
            self.screen.blit(action_surf, (20, self.height - 42))

        bar_y  = self.height - 70
        bar_h  = 12
        bar_w  = self.width - 40
        bar_x  = 20
        fill_w = int(bar_w * ratio)

        pygame.draw.rect(self.screen, (180, 180, 180), (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(self.screen, BLUE,            (bar_x, bar_y, fill_w, bar_h))
        pygame.draw.rect(self.screen, BLACK,           (bar_x, bar_y, bar_w, bar_h), 1)
        self._label(
            self.font_small,
            "← PARK           DISTANCE TO FARMLAND           FARM →",
            BLACK, bar_x, bar_y - 18,
        )

        pygame.display.flip()
        self.clock.tick(10)   

    def _label(self, font, text: str, colour, x: int, y: int):
        surf = font.render(text, True, colour)
        self.screen.blit(surf, (x, y))

    def close(self):
        if self.is_open:
            self.is_open = False
            pygame.quit()