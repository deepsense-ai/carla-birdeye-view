import carla
from typing import NamedTuple, List

is_vehicle = lambda actor: "vehicle" in actor.type_id
is_pedestrian = lambda actor: "walker" in actor.type_id
is_traffic_light = lambda actor: "traffic_light" in actor.type_id


class SegregatedActors(NamedTuple):
    vehicles: List[carla.Actor]
    pedestrians: List[carla.Actor]
    traffic_lights: List[carla.Actor]


def segregate_by_type(actors: List[carla.Actor]) -> SegregatedActors:
    vehicles = []
    pedestrians = []
    traffic_lights = []
    for actor in actors:
        if is_vehicle(actor):
            vehicles.append(actor)
        elif is_pedestrian(actor):
            pedestrians.append(actor)
        elif is_traffic_light(actor):
            traffic_lights.append(actor)
    return SegregatedActors(vehicles, pedestrians, traffic_lights)


def query_all(world: carla.World) -> List[carla.Actor]:
    snapshot: carla.WorldSnapshot = world.get_snapshot()
    all_actors = []
    for actor_snapshot in snapshot:
        actor = world.get_actor(actor_snapshot.id)
        if actor is not None:
            all_actors.append(actor)
    return all_actors
