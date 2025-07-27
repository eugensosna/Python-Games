# Dino

# Author : Prajjwal Pathak (pyguru)
# Date : Sunday, 17 October, 2021

import random
import pygame
import threading
import queue  # For thread-safe communication

from objects import Ground, Dino, Cactus, Cloud, Ptera, Star
from handmove import DetectMoving
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(threadName)s - %(message)s"
)

pygame.init()
SCREEN = WIDTH, HEIGHT = (600, 200)
win = pygame.display.set_mode(SCREEN, pygame.RESIZABLE)

clock = pygame.time.Clock()
FPS = 15
base_speed = 5

half_speed = False
half_speed_cactus = dict()


# COLORS *********************************************************************

WHITE = (225, 225, 225)
BLACK = (0, 0, 0)
GRAY = (32, 33, 36)
RED = (255, 0, 0)  # (Red, Green, Blue)

# IMAGES *********************************************************************

start_img = pygame.image.load("Assets/start_img.png")
start_img = pygame.transform.scale(start_img, (60, 64))

game_over_img = pygame.image.load("Assets/game_over.png")
game_over_img = pygame.transform.scale(game_over_img, (200, 36))

replay_img = pygame.image.load("Assets/replay.png")
replay_img = pygame.transform.scale(replay_img, (40, 36))
replay_rect = replay_img.get_rect()
replay_rect.x = WIDTH // 2 - 20
replay_rect.y = 100

numbers_img = pygame.image.load("Assets/numbers.png")
numbers_img = pygame.transform.scale(numbers_img, (120, 12))

# SOUNDS *********************************************************************

jump_fx = pygame.mixer.Sound("Sounds/jump.wav")
die_fx = pygame.mixer.Sound("Sounds/die.wav")
checkpoint_fx = pygame.mixer.Sound("Sounds/checkPoint.wav")

# OBJECTS & GROUPS ***********************************************************

ground = Ground()
dino = Dino(50, 160)

cactus_group = pygame.sprite.Group()
ptera_group = pygame.sprite.Group()
cloud_group = pygame.sprite.Group()
stars_group = pygame.sprite.Group()

# FUNCTIONS ******************************************************************


def reset():
    global counter, SPEED, score, high_score

    if score and score >= high_score:
        high_score = score

    counter = 0
    SPEED = 5
    prev_speed = SPEED
    half_speed = False
    score = 0

    cactus_group.empty()
    ptera_group.empty()
    cloud_group.empty()
    stars_group.empty()

    dino.reset()


# CHEATCODES *****************************************************************

# GODMODE -> immortal jutsu ( can't die )
# DAYMODE -> Swap between day and night
# LYAGAMI -> automatic jump and duck
# IAMRICH -> add 10,000 to score
# HISCORE -> highscore is 99999
# SPEEDUP -> increase speed by 2
# HANDUP

keys = []
GODMODE = False
DAYMODE = False
LYAGAMI = False
HANDUP = True

# VARIABLES ******************************************************************

counter = 0
enemy_time = 100
cloud_time = 500
stars_time = 175
distance = 0

SPEED = 5
prev_speed = SPEED
jump = False
duck = False
half_speed_que = []
half_speed_min_distance = 150
score = 0
high_score = 0

start_page = True
mouse_pos = (-1, -1)

# THREADING SETUP ***********************************************************
detector = DetectMoving()
event_queue = queue.Queue()  # Queue to pass events to the detector thread
action_queue = queue.Queue()  # Queue to receive actions from the detector thread
stop_detector = threading.Event()  # Event to signal the thread to stop


def detector_thread_function():

    while True:
        if stop_detector.is_set():
            detector.stop()
            break

        try:

            current_events = event_queue.get_nowait(
                # timeout=0.01
            )  # Get events from the main thread
            # action_queue
            if detector.updateEventsWhile(
                stop_detector=stop_detector, action=action_queue
            ):
                action_queue.put(
                    stop_detector, True
                )  # Signal that an action should be taken
        except queue.Empty:
            # print("No events to process in detector thread.")
            pass  # No events to process yet
        except Exception as e:
            print(f"Detector thread error: {e}")

    return None


# Start the detector thread
detector_thread = threading.Thread(target=detector_thread_function)
detector_thread.daemon = (
    True  # Allow the program to exit even if this thread is running
)
detector_thread.start()

# MAIN GAME LOOP *************************************************************
running = True
while running:
    jump = False
    # SPEED = prev_speed
    if DAYMODE:
        win.fill(WHITE)
    else:
        win.fill(GRAY)

    events = pygame.event.get()

    # Pass current events to the detector thread if dino is not jumping
    # if not dino.isJumping:
    try:
        event_queue.put_nowait(events)
    except queue.Full:
        pass  # Queue is full, skip this frame or handle as needed

    # Check for actions from the detector thread
    try:
        if action_queue.get_nowait():
            # If the detector returns true, simulate a jump event
            # You might want to be more specific here (e.g., add a custom event type)
            # For demonstration, we'll simulate a K_SPACE down event
            jump_event = pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_PERIOD})
            logging.info(" Get event Hand lifted up! Pressing 'up arrow' once.")
            events.append(jump_event)
            # event_queue.put_nowait([jump_event])  # Pass the jump event to the queue

            # jump = True # Set jump flag directly as well
            # jump_fx.play() # Play jump sound if triggered by detector
    except queue.Empty:
        pass  # No action from detector yet

    for event in events:
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                running = False

            if event.key == pygame.K_SPACE:
                if start_page:
                    start_page = False
                elif dino.alive:
                    jump = True
                    jump_fx.play()
                else:
                    reset()

            if event.key == pygame.K_UP:
                jump = True
                jump_fx.play()

            if event.key == pygame.K_DOWN:
                duck = True

            key = pygame.key.name(event.key)
            keys.append(key)
            keys = keys[-7:]
            if "".join(keys).upper() == "GODMODE":
                GODMODE = not GODMODE

            if "".join(keys).upper() == "DAYMODE":
                DAYMODE = not DAYMODE

            if "".join(keys).upper() == "LYAGAMI":
                LYAGAMI = not LYAGAMI

            if "".join(keys).upper() == "SPEEDUP":
                logging.info(f"1 speed up: {SPEED} ")
                SPEED += 2

            if "".join(keys).upper() == "IAMRICH":
                score += 10000

            if "".join(keys).upper() == "HISCORE":
                high_score = 99999

            if event.key == pygame.K_PERIOD:
                jump = True
                HANDUP = True
                if half_speed:
                    logging.info(f"Hand speed {SPEED} prev speed {prev_speed}")
                    SPEED = prev_speed
                    logging.info(f"2 Hand speed {SPEED} prev speed {prev_speed}")
                    half_speed = False
                logging.info(f"Jump activated: {jump} | Speed: {SPEED}")
                # dx = cactus.rect.x - dino.rect.x
                # if 0 <= dx <= (70 + (score//100)):
                # 	 jump = True

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_SPACE or event.key == pygame.K_UP:
                jump = False

            if event.key == pygame.K_DOWN:
                duck = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = event.pos

        if event.type == pygame.MOUSEBUTTONUP:
            mouse_pos = (-1, -1)

    if start_page:
        win.blit(start_img, (50, 100))
    else:
        if dino.alive:
            counter += 1
            if counter % int(enemy_time) == 0:
                if random.randint(1, 10) == 5:
                    y = random.choice([85, 130])
                    ptera = Ptera(WIDTH, y)  # Uncommented Ptera
                    # ptera_group.add(ptera)
                else:
                    type = random.randint(1, 4)
                    cactus = Cactus(type)
                    cactus_group.add(cactus)

            if counter % cloud_time == 0:
                y = random.randint(40, 100)
                cloud = Cloud(WIDTH, y)
                cloud_group.add(cloud)

            if counter % stars_time == 0:
                type = random.randint(1, 3)
                y = random.randint(40, 100)
                star = Star(WIDTH, y, type)
                stars_group.add(star)

            if counter % 100 == 0:
                # SPEED += 0.1
                # enemy_time -= 0.5
                pass

            if counter % 5 == 0:
                score += 1

            if score and score % 100 == 0:
                checkpoint_fx.play()

            if not GODMODE:
                distance = 0
                min_cactus = None
                min_distance = 1000

                for cactus in cactus_group:
                    dx = cactus.rect.x - dino.rect.x
                    if dx > 0:
                        if dx <= min_distance:
                            min_distance = dx
                            min_cactus = cactus

                if (
                    min_cactus is not None
                    and half_speed_cactus.get(min_cactus, None) is None
                ):

                    if min_distance <= half_speed_min_distance:
                        half_speed_cactus[min_cactus] = base_speed / 2
                        prev_speed = base_speed
                        half_speed
                        logging.debug(
                            f"2.1Half speed activated: {SPEED} cactus {min_cactus} jump status {jump}"
                        )

                for cactus in cactus_group:
                    if jump and half_speed_cactus.get(cactus, None) is not None:
                        half_speed_cactus[cactus] = base_speed
                    dx = cactus.rect.x - dino.rect.x

                    # if 0 <= dx <= 120 and half_speed == False:
                    # 	 half_speed = True
                    # 	 prev_speed = SPEED
                    # 	 SPEED = SPEED / 1.5
                    # 	 logging.info(f"3 New speed {SPEED} prev speed {prev_speed}")

                    distance = dx if distance == 0 else min(distance, dx)
                    if (
                        3 < dx <= half_speed_min_distance
                        and half_speed_cactus.get(cactus, None) is None
                    ):
                        half_speed_cactus[cactus] = base_speed / 2
                        prev_speed = SPEED
                        half_speed = True

                        # logging.info(f"Distance: {distance} | Speed: {SPEED}")
                        # prev_speed = SPEED
                        # SPEED = SPEED / 2
                        # half_speed = True
                        # half_speed_que.append(cactus)
                        logging.debug(
                            f"4 Half speed activated: {half_speed_cactus[cactus]} jump status {jump}, distance:{dx}"
                        )
                    if half_speed_cactus.get(cactus, None) is not None and jump:
                        logging.debug("5 return base speed")
                        half_speed_cactus[cactus] = base_speed

                    if half_speed_cactus.get(cactus, None) is not None:
                        SPEED = half_speed_cactus[cactus]
                        half_speed = False
                        logging.debug(
                            f"6. Set Speed: {SPEED} | distance: {distance} prev_speed: {prev_speed}"
                        )

                    # if HANDUP:
                    # 	 logging.debug(f"Hand speed {SPEED} prev speed {prev_speed}")
                    # 	 jump = True
                    # 	 SPEED = prev_speed
                    # 	 HANDUP = False  # Reset HANDUP after using it
                    # 	 dx = cactus.rect.x - dino.rect.x
                    # 	 if 0 <= dx <= (30 + (score // 100)):
                    # 		 pass

                    if LYAGAMI:
                        dx = cactus.rect.x - dino.rect.x
                        if 0 <= dx <= (70 + (score // 100)):
                            jump = True

                    if pygame.sprite.collide_mask(dino, cactus):
                        SPEED = 0
                        logging.debug(f"7 Collision with cactus at distance {distance}")
                        dino.alive = False
                        die_fx.play()

                for (
                    ptera
                ) in ptera_group:  # Corrected variable name from cactus to ptera
                    dx = ptera.rect.x - dino.rect.x
                    distance = dx if distance == 0 else min(distance, dx)
                    if LYAGAMI:
                        dx = ptera.rect.x - dino.rect.x

                        if 0 <= dx <= 70:
                            if dino.rect.top <= ptera.rect.top:
                                jump = True
                            else:
                                duck = True
                        else:
                            duck = False

                    if pygame.sprite.collide_mask(dino, ptera):
                        SPEED = 0
                        dino.alive = False
                        die_fx.play()
        if SPEED != base_speed:
            logging.debug(f"8. set grund speed:{SPEED} ")
        ground.update(SPEED)
        ground.draw(win)
        cloud_group.update(SPEED - 3, dino)
        cloud_group.draw(win)
        stars_group.update(SPEED - 3, dino)
        stars_group.draw(win)
        cactus_group.update(SPEED, dino)
        cactus_group.draw(win)
        ptera_group.update(SPEED - 1, dino)
        ptera_group.draw(win)
        dino.update(jump, duck)
        dino.draw(win)

        string_score = str(score).zfill(5)
        for i, num in enumerate(string_score):
            win.blit(numbers_img, (520 + 11 * i, 10), (10 * int(num), 0, 10, 12))

        if high_score:
            win.blit(numbers_img, (425, 10), (100, 0, 20, 12))
            string_score = f"{high_score}".zfill(5)
            for i, num in enumerate(string_score):
                win.blit(numbers_img, (455 + 11 * i, 10), (10 * int(num), 0, 10, 12))

        if 0 < abs(distance) <= half_speed_min_distance:
            if abs(distance) % 2 == 0:
                pygame.draw.rect(win, RED, (5, 5, 10, 10))
            else:
                pygame.draw.rect(win, WHITE, (5, 5, 10, 10))

        string_distance = str(abs(int(distance))).zfill(5)
        for i, num in enumerate(string_distance):
            win.blit(numbers_img, (320 + 11 * i, 10), (10 * int(num), 0, 10, 12))
        if 0 < abs(distance) <= half_speed_min_distance:
            if abs(distance) % 2 == 0:
                pygame.draw.rect(win, RED, (320, 325, 10, 10))
            else:
                pygame.draw.rect(win, WHITE, (5, 5, 10, 10))

        if not dino.alive:
            win.blit(game_over_img, (WIDTH // 2 - 100, 55))
            win.blit(replay_img, replay_rect)

            if replay_rect.collidepoint(mouse_pos):
                reset()

    pygame.draw.rect(win, WHITE, (0, 0, WIDTH, HEIGHT), 4)
    clock.tick(FPS)
    pygame.display.update()

# Clean up: Signal the detector thread to stop before quitting Pygame
stop_detector.set()
detector_thread.join()  # Wait for the thread to finish
pygame.quit()
