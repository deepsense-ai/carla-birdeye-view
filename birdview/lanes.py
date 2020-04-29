import carla
import numpy as np

from cv2 import cv2 as cv
from enum import IntEnum

"""
This file is mostly a copy&paste from "Learning by Cheating" code.
It requires refactor.
"""


class LaneSide(IntEnum):
    LEFT = -1
    RIGHT = 1


def lateral_shift(transform, shift):
    """Makes a lateral shift of the forward vector of a transform"""
    transform.rotation.yaw += 90
    return transform.location + shift * transform.get_forward_vector()


def draw_solid_line(canvas, color, closed, points, width):
    """Draws solid lines in a surface given a set of points, width and color"""
    if len(points) >= 2:
        cv.polylines(
            img=canvas,
            pts=np.int32([points]),
            isClosed=closed,
            color=color,
            thickness=width,
        )


def draw_broken_line(canvas, color, closed, points, width):
    """Draws broken lines in a surface given a set of points, width and color"""
    # Select which lines are going to be rendered from the set of lines
    broken_lines = [x for n, x in enumerate(zip(*(iter(points),) * 20)) if n % 3 == 0]

    # Draw selected lines
    for line in broken_lines:
        cv.polylines(
            img=canvas,
            pts=np.int32([line]),
            isClosed=closed,
            color=color,
            thickness=width,
        )


def get_lane_markings(
    lane_marking_type,
    lane_marking_color,
    waypoints,
    side: LaneSide,
    location_to_pixel_func,
):
    """For multiple lane marking types (SolidSolid, BrokenSolid, SolidBroken and BrokenBroken),
    it converts them as a combination of Broken and Solid lines.
    """
    margin = 0.25
    sign = side.value
    marking_1 = [
        location_to_pixel_func(lateral_shift(w.transform, sign * w.lane_width * 0.5))
        for w in waypoints
    ]
    if lane_marking_type == carla.LaneMarkingType.Broken or (
        lane_marking_type == carla.LaneMarkingType.Solid
    ):
        return [(lane_marking_type, lane_marking_color, marking_1)]
    else:
        marking_2 = [
            location_to_pixel_func(
                lateral_shift(w.transform, sign * (w.lane_width * 0.5 + margin * 2))
            )
            for w in waypoints
        ]
        if lane_marking_type == carla.LaneMarkingType.SolidBroken:
            return [
                (carla.LaneMarkingType.Broken, lane_marking_color, marking_1),
                (carla.LaneMarkingType.Solid, lane_marking_color, marking_2),
            ]
        elif lane_marking_type == carla.LaneMarkingType.BrokenSolid:
            return [
                (carla.LaneMarkingType.Solid, lane_marking_color, marking_1),
                (carla.LaneMarkingType.Broken, lane_marking_color, marking_2),
            ]
        elif lane_marking_type == carla.LaneMarkingType.BrokenBroken:
            return [
                (carla.LaneMarkingType.Broken, lane_marking_color, marking_1),
                (carla.LaneMarkingType.Broken, lane_marking_color, marking_2),
            ]
        elif lane_marking_type == carla.LaneMarkingType.SolidSolid:
            return [
                (carla.LaneMarkingType.Solid, lane_marking_color, marking_1),
                (carla.LaneMarkingType.Solid, lane_marking_color, marking_2),
            ]
    return [(carla.LaneMarkingType.NONE, carla.LaneMarkingColor.Other, [])]


def draw_lane_marking_single_side(
    surface, waypoints, side: LaneSide, location_to_pixel_func, color
):
    """Draws the lane marking given a set of waypoints and decides
    whether drawing the right or left side of the waypoint based on the sign parameter
    """
    previous_marking_type = carla.LaneMarkingType.NONE
    markings_list = []
    temp_waypoints = []
    current_lane_marking = carla.LaneMarkingType.NONE
    for sample in waypoints:
        lane_marking = (
            sample.left_lane_marking
            if side is LaneSide.LEFT
            else sample.right_lane_marking
        )

        if lane_marking is None:
            continue

        marking_type = lane_marking.type
        marking_color = lane_marking.color

        if current_lane_marking != marking_type:
            # Get the list of lane markings to draw
            markings = get_lane_markings(
                previous_marking_type,
                color,  # lane_marking_color_to_tango(previous_marking_color),
                temp_waypoints,
                side,
                location_to_pixel_func,
            )
            current_lane_marking = marking_type

            # Append each lane marking in the list
            for marking in markings:
                markings_list.append(marking)

            temp_waypoints = temp_waypoints[-1:]

        else:
            temp_waypoints.append((sample))
            previous_marking_type = marking_type

    # Add last marking
    last_markings = get_lane_markings(
        previous_marking_type,
        color,  # lane_marking_color_to_tango(previous_marking_color),
        temp_waypoints,
        side,
        location_to_pixel_func,
    )

    for marking in last_markings:
        markings_list.append(marking)

    # Once the lane markings have been simplified to Solid or Broken lines, we draw them
    for markings in markings_list:
        if markings[0] == carla.LaneMarkingType.Solid:
            draw_solid_line(surface, markings[1], False, markings[2], 1)
        elif markings[0] == carla.LaneMarkingType.Broken:
            draw_broken_line(surface, markings[1], False, markings[2], 1)
