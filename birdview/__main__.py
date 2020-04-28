import carla
import math
import random

from cv2 import cv2 as cv
import birdview
from birdview import BirdViewProducer, BirdView
from birdview.mask import PixelDimensions


MPS_TO_KMH = 3.6


def get_speed(actor: carla.Actor) -> float:
    """in km/h"""
    vector: carla.Vector3D = actor.get_velocity()
    return MPS_TO_KMH * math.sqrt(vector.x ** 2 + vector.y ** 2 + vector.z ** 2)


if __name__ == "__main__":
    client = carla.Client("localhost", 2000)
    client.set_timeout(3.0)
    world = client.get_world()
    map = world.get_map()
    spawn_points = map.get_spawn_points()
    blueprints = world.get_blueprint_library()

    # settings = world.get_settings()
    # settings.synchronous_mode = True
    # settings.no_rendering_mode = True
    # settings.fixed_delta_seconds = 1 / 10.0
    # world.apply_settings(settings)

    hero_bp = random.choice(blueprints.filter("vehicle.ford.mustang"))
    hero_bp.set_attribute("role_name", "hero")
    transform = random.choice(map.get_spawn_points())
    agent = world.spawn_actor(hero_bp, transform)
    agent.set_autopilot(True)

    birdview_producer = BirdViewProducer(
        client,
        PixelDimensions(width=birdview.DEFAULT_WIDTH, height=birdview.DEFAULT_HEIGHT),
        pixels_per_meter=4,
    )
    stuck_frames_count = 0

    while True:
        # world.tick()
        # imshow interprets data as BGR...
        birdview: BirdView = birdview_producer.produce(agent_vehicle=agent)
        bgr = cv.cvtColor(BirdViewProducer.as_rgb(birdview), cv.COLOR_BGR2RGB)
        cv.imshow("BirdView RGB", bgr)

        # Teleport when stuck
        if get_speed(agent) < 3:
            stuck_frames_count += 1
        else:
            stuck_frames_count = 0

        if stuck_frames_count == 20:
            agent.set_autopilot(False)
            agent.set_transform(random.choice(map.get_spawn_points()))
            agent.set_autopilot(True)

        key = cv.waitKey(10) & 0xFF
        if key == 27:  # ESC
            break
    cv.destroyAllWindows()