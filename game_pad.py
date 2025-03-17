import pygame
import time

pygame.init()

j = pygame.joystick.Joystick(0)
j.init()

# pygame.init()
# pygame.joystick.init()

# joystick_count = pygame.joystick.get_count()
# print(f"Number of joysticks: {joystick_count}")

print("press button of controller")
try:
    while True:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.JOYBUTTONDOWN: #ボタンが押された場合
                if j.get_button(0):
                    print("四角ボタンが押されました")
                    time.sleep(0.1)
                elif j.get_button(1):
                    print("バツボタンが押されました")
                    time.sleep(0.1)
                elif j.get_button(2):
                    print("丸ボタンが押されました")
                    time.sleep(0.1)
                elif j.get_button(3):
                    print("三角ボタンが押されました")
                    time.sleep(0.1)
                elif j.get_button(4):
                    print("L1が押されました")
                    time.sleep(0.1)
                elif j.get_button(5):
                    print("R1が押されました")
                    time.sleep(0.1)
                elif j.get_button(6):
                    print("L2が押されました")
                    print("L2の押し込み量")
                    while j.get_axis(5) > -1:
                        events = pygame.event.get()
                        print(j.get_axis(5))
                        time.sleep(0.1)
                elif j.get_button(7):
                    print("R2が押されました")
                    print("R2の押し込み量")
                    while j.get_axis(4) > -1:
                        events = pygame.event.get()
                        print(j.get_axis(4))
                        time.sleep(0.1)
                elif j.get_button(8):
                    print("シェアボタンが押されました")
                    time.sleep(0.1)
                elif j.get_button(9):
                    print("オプションボタンが押されました")
                    time.sleep(0.1)
                elif j.get_button(10):
                    print("左スティックが押し込まれました")
                    time.sleep(0.1)
                elif j.get_button(11):
                    print("右スティックが押し込まれました")
                    time.sleep(0.1)
                elif j.get_button(12):
                    print("psボタンが押されました")
                    time.sleep(0.1)
                elif j.get_button(13):
                    print("タッチパッドが押されました")
                    time.sleep(0.1)
            elif event.type == pygame.JOYHATMOTION:
                print("十字キー座標")
                print("("+str((j.get_hat(0))[0])+","+str((j.get_hat(0))[1])+")")
            elif event.type == pygame.JOYAXISMOTION:
                if abs((j.get_axis(0))) >= 0.5 or abs((j.get_axis(1))) >= 0.5:
                    print("左スティック座標")
                    print("("+str(j.get_axis(0))+","+ str(j.get_axis(1))+")")
                    time.sleep(0.1)
                elif abs((j.get_axis(2))) >= 0.5 or abs((j.get_axis(3))) >= 0.5:
                    print("右スティック座標")
                    print("("+str(j.get_axis(2))+","+ str(j.get_axis(3))+")")
                    time.sleep(0.1)
except KeyboardInterrupt:
    print("プログラムを終了します")
    j.quit()
    


