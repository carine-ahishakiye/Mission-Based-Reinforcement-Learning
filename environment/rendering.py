import pygame

# colors
WHITE   = (245, 245, 245) 
BLACK   = (40, 40, 40)     
GREEN   = (60, 120, 60)     # Park
YELLOW  = (200, 180, 80)    # Buffer zone)
RED     = (160, 60, 60)     # farmlan
BLUE    = (70, 130, 180)    # Buffalo



class Renderer:
    def __init__(self, width=800, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Wildlife Conflict Simulation")
        self.clock = pygame.time.Clock()

        # Map boundaries
        self.park_boundary = 200   
        self.farm_boundary = 600   

    def draw(self, state, step, action=None):
        step_length, speed, turning_angle, distance, ndvi, time_of_day, season = state

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
        self.screen.fill(WHITE)
        pygame.draw.rect(self.screen, GREEN, (0, 0, self.park_boundary, self.height))   
        pygame.draw.rect(self.screen, YELLOW, (self.park_boundary, 0,
                                               self.farm_boundary - self.park_boundary, self.height))  
        pygame.draw.rect(self.screen, RED, (self.farm_boundary, 0,
                                            self.width - self.farm_boundary, self.height))  
        x_pos = max(0, min(self.width, self.width - distance/20)) 
        y_pos = self.height // 2
        pygame.draw.circle(self.screen, BLUE, (int(x_pos), int(y_pos)), 10)

        # Info text
        font = pygame.font.SysFont(None, 24)
        season_str = "DRY" if season == 1 else "WET"
        info_text = f"Step {step} | Dist: {distance:.1f}m | Speed: {speed:.2f}m/s | Season: {season_str} | Time: {int(time_of_day)}:00"
        text_surface = font.render(info_text, True, BLACK)
        self.screen.blit(text_surface, (20, 20))

        # Show action text
        if action is not None:
            action_map = {0: "NO ALERT", 1: "LOW ALERT", 2: "HIGH ALERT"}
            action_text = font.render(f"Action: {action_map.get(action, action)}", True, BLACK)
            self.screen.blit(action_text, (20, 50))

        # Update display
        pygame.display.flip()
        self.clock.tick(30)  #

    def close(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
        pygame.quit()
