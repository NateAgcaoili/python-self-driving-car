from __future__ import nested_scopes
import pygame
import os
import math
import sys
import neat

SCREEN_WIDTH = 1080
SCREEN_HEIGHT = 900
FPS = 100
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
ASSETS = os.path.join(os.path.dirname(__file__), 'assets')
TRACK = pygame.image.load(os.path.join(ASSETS, 'track1.png'))
print(os.path.join(ASSETS, 'track.png'))

blue_check = pygame.Color(102, 113, 255)
yellow_check = pygame.Color(255, 244, 102)
red_check = pygame.Color(255, 0, 79)
check_points = [blue_check, yellow_check, red_check]
clock = pygame.time.Clock()


class Car(pygame.sprite.Sprite):
    def __init__(self) -> None:
        super().__init__()
        self.original_image = pygame.image.load(
            os.path.join(ASSETS, 'car.png'))
        self.image = self.original_image
        self.rect = self.image.get_rect(center=(560, 800))
        #self.drive_state = False
        self.vel_vector = pygame.math.Vector2(0.8, 0)
        self.angle = 0
        self.speed = 8
        self.rotation_vel = 5
        self.direction = 0
        self.radar_angles = [-60, -30, 0, 30, 60]
        self.alive = True
        self.lap_cd = True
        self.cd_time = 350
        self.check_index = 0
        self.radars = []  # added for ai
        self.lap_count = 0
        self.fitness_bonus = 1000

    def update(self):
        self.radars.clear()  # added for ai
        self.drive()
        self.rotate()
        for ra in self.radar_angles:
            self.radar(ra)
        self.collision()
        self.fitness_bonus -= 2
        self.data()  # added for ai

    def drive(self):
        """
        if self.drive_state:
            self.rect.center += self.vel_vector * self.speed
        """
        self.rect.center += self.vel_vector * self.speed

    def rotate(self):
        if self.direction == 1:
            self.angle -= self.rotation_vel
            self.vel_vector.rotate_ip(self.rotation_vel)
        if self.direction == -1:
            self.angle += self.rotation_vel
            self.vel_vector.rotate_ip(-self.rotation_vel)

        self.image = pygame.transform.rotozoom(
            self.original_image, self.angle, 0.1)
        self.rect = self.image.get_rect(center=self.rect.center)

    def radar(self, radar_angle):
        length = 0
        x = int(self.rect.center[0])
        y = int(self.rect.center[1])
        try:
            while not SCREEN.get_at((x, y)) == pygame.Color(0, 104, 56, 255) and length < 200:
                length += 1
                x = int(self.rect.center[0] +
                        math.cos(math.radians(self.angle + radar_angle)) * length)
                y = int(self.rect.center[1] -
                        math.sin(math.radians(self.angle + radar_angle)) * length)

            # Draw Radar
            pygame.draw.line(SCREEN, (255, 255, 255, 125),
                             self.rect.center, (x, y), 1)
            pygame.draw.circle(SCREEN, (255, 0, 0, 0), (x, y), 3)

            dist = int(math.sqrt(  # added for ai
                math.pow(self.rect.center[0] - x, 2) + math.pow(self.rect.center[1] - y, 2)))
            self.radars.append([radar_angle, dist])
        except IndexError:
            pass

    def collision(self):
        length = 45
        collision_point_right = [int(self.rect.center[0] + math.cos(math.radians(self.angle + 18)) * length),
                                 int(self.rect.center[1] - math.sin(math.radians(self.angle + 18)) * length)]
        collision_point_left = [int(self.rect.center[0] + math.cos(math.radians(self.angle - 18)) * length),
                                int(self.rect.center[1] - math.sin(math.radians(self.angle - 18)) * length)]

        # Die on Collision
        try:
            self.death_check(collision_point_right, collision_point_left)
            self.direction_check(collision_point_right, collision_point_left)
            self.lap_check(collision_point_right, collision_point_left)
        except IndexError:
            print("got one")
            self.alive = False

        # Draw Collision Points
        pygame.draw.circle(SCREEN, (229, 218, 21, 0), collision_point_right, 4)
        pygame.draw.circle(SCREEN, (229, 218, 21, 0), collision_point_left, 4)

    # Die on Collision
    def death_check(self, cpr, cpl):
        if SCREEN.get_at(cpr) == pygame.Color(0, 104, 56, 255) \
                or SCREEN.get_at(cpl) == pygame.Color(0, 104, 56, 255):
            self.alive = False

    def direction_check(self, cpr, cpl):
        if SCREEN.get_at(cpr) == check_points[self.check_index % 3] \
                or SCREEN.get_at(cpl) == check_points[self.check_index % 3]:
            self.check_index += 1
        if SCREEN.get_at(cpr) == check_points[(self.check_index + 1) % 3] \
                or SCREEN.get_at(cpl) == check_points[(self.check_index + 1) % 3]:
            self.alive = False

    def lap_check(self, cpr, cpl):
        if not self.lap_cd:
            if SCREEN.get_at(cpr) == pygame.Color(255, 255, 255, 255) \
                or SCREEN.get_at(cpl) == pygame.Color(255, 255, 255, 255) \
                    or SCREEN.get_at(cpr) == pygame.Color(53, 53, 53, 255) \
                    or SCREEN.get_at(cpl) == pygame.Color(53, 53, 53, 255):
                self.lap_cd = True
                self.lap_count += 1
                print(self.lap_count)

    def data(self):  # added for ai
        input = [0, 0, 0, 0, 0]
        for i, radar in enumerate(self.radars):
            input[i] = int(radar[1])
        return input

#car = pygame.sprite.GroupSingle(Car())


def remove(index):
    cars.pop(index)  # remove car
    ge.pop(index)  # remove genome
    nets.pop(index)  # remove neural net


def eval_genomes(genomes, config):
    global cars, ge, nets

    cars = []
    ge = []
    nets = []

    for genome_id, genome in genomes:
        cars.append(pygame.sprite.GroupSingle(Car()))
        ge.append(genome)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        genome.fitness = 0

    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        SCREEN.blit(TRACK, (0, 0))

        if len(cars) == 0:
            break

        for i, car in enumerate(cars):
            ge[i].fitness += 1
            if car.sprite.lap_count == 1:
                print(car.sprite.fitness_bonus)
                ge[i].fitness += car.sprite.fitness_bonus
                car.sprite.alive = False
            if not car.sprite.alive:
                remove(i)

        for i, car in enumerate(cars):
            output = nets[i].activate(car.sprite.data())
            if output[0] > 0.7:
                car.sprite.direction = 1
            if output[1] > 0.7:
                car.sprite.direction = -1
            if output[0] <= 0.7 and output[1] <= 0.7:
                car.sprite.direction = 0
        """
        # User Input
        user_input = pygame.key.get_pressed()
        if sum(pygame.key.get_pressed()) <= 1:  # no keys are pressed
            car.sprite.drive_state = False
            car.sprite.direction = 0
        # Drive
        if user_input[pygame.K_UP] or user_input[pygame.K_w]:
            car.sprite.drive_state = True
        # Steer
        if user_input[pygame.K_RIGHT] or user_input[pygame.K_d]:
            car.sprite.direction = 1
        if user_input[pygame.K_LEFT] or user_input[pygame.K_a]:
            car.sprite.direction = -1
        """
        # Update
        """        
        car.draw(SCREEN)
        car.update()
        """
        for car in cars:
            if car.sprite.lap_cd:
                car.sprite.cd_time -= 1
                if car.sprite.cd_time < 0:
                    car.sprite.lap_cd = False
                    car.sprite.cd_time = 350
            car.draw(SCREEN)
            car.update()
        pygame.display.update()


# eval_genomes()

# Setup NEAT Neural Network
def run(config_path):
    global pop
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    pop = neat.Population(config)

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    pop.run(eval_genomes, 5)
    clock.tick(FPS)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run(config_path)
