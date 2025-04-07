from pygame.locals import *
import pygame
import sys
import time


class Keyboard:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((600, 50))
        self.font = pygame.font.Font(None, 50)
        pygame.display.set_caption("keyboard event") 
        
    def update(self, action):
        self.screen.fill((0, 0, 0))
        action_txt = self.font.render(f"pressed key: {action}", True, (255, 255, 255))
        self.screen.blit(action_txt, [0, 0])
        
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit()
                    sys.exit()
            pygame.display.update()
            
    def control(self):
        dx, dy, dz, da, grip = 0, 0, 0, 0, 0
        dv = 0.003
        
        pressed_key = pygame.key.get_pressed()
        if any(pressed_key):
            if pressed_key[K_d]:
                dx -= dv
            if pressed_key[K_a]:
                dx += dv
            if pressed_key[K_w]:
                dy -= dv
            if pressed_key[K_s]:
                dy += dv
            if pressed_key[K_DOWN]:
                dz -= dv
            if pressed_key[K_UP]:
                dz += dv
            if pressed_key[K_RIGHT] and pressed_key[K_LSHIFT]:
                da -= 0.10
            if pressed_key[K_LEFT] and pressed_key[K_LSHIFT]:
                da += 0.10
            if pressed_key[K_SPACE]:
                grip += 1
                
        return dx, dy, dz, da, grip


def main():
    keyboard = Keyboard()
    while True:
        dx, dy, dz, da, grip = keyboard.control()
        keyboard.update([dx, dy, dz, da, grip])
        
        
if __name__ == "__main__":
    main()