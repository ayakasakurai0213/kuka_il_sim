from pygame.locals import *
import pygame
import sys

class Keyboard:
    def __init__(self):
        self.width = 400
        self.height = 50
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("keyboard event")
        
        self.font = pygame.font.Font(None, 50)
        
    def update(self, text):
        self.screen.fill((0, 0, 0))
        if text is not None:
            self.screen.blit(text, [0, 0])
        
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit()
                    sys.exit()  
        pygame.display.update()
    
    def get_pressed_key(self):
        # action_dict = {
        #     0: "stay", 1: "right", 2: "left", 3: "front", 4: "back", 5: "down", 
        #     6: "up", 7: "gripper CW", 8: "gripper semi-CW", 9: "gripper close"
        #     }
        action = []
        pressed_key = pygame.key.get_pressed()
        if any(pressed_key):
            if pressed_key[K_d]:
                action.append(1)
            if pressed_key[K_a]:
                action.append(2)
            if pressed_key[K_w]:
                action.append(3)
            if pressed_key[K_s]:
                action.append(4)
            if pressed_key[K_DOWN]:
                action.append(5)
            if pressed_key[K_UP]:
                action.append(6)
            if pressed_key[K_RIGHT] and pressed_key[K_LSHIFT]:
                action.append(7)
            if pressed_key[K_LEFT] and pressed_key[K_LSHIFT]:
                action.append(8)
            if pressed_key[K_SPACE]:
                action.append(9)
        else:
            action.append(0)
        text = self.font.render(f"pressed key: {action}", True, (255, 255, 255))
        return action, text
            
def main():
    keyboard = Keyboard()
    while True:
        action, text = keyboard.get_pressed_key()
        keyboard.update(text)
    
    
if __name__ == "__main__":
    main()