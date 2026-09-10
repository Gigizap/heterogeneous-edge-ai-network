#!/usr/bin/env python3
"""
mock_skills.py

Hardware-free sensing preset. Every skill returns a plausible canned answer
instead of touching a camera, a sensor or a model, so this preset runs on any
machine and is meant for testing dispatch, tool selection and the leader's
summarisation end to end.

- detect_people  : how many people are in view
- measure_co2    : CO2 concentration in the room, in ppm
- identify_face  : who is in front of the camera
"""

import random
import logging

# Child of the "SensingLogic" logger sensing_agent configures, so every line here is
# stamped with this agent's ID (see SensingLogic/sensing_agent.py).
log = logging.getLogger(__name__)

KNOWN_FACES = ["Luigi", "Marco", "Anna", "Sara"]


def detect_people(**kwargs):
    """Return a mock head count for the people currently in view."""
    n = random.randint(0, 4)
    log.info("detect_people: mock count = %d", n)
    status = "room occupied" if n > 0 else "room free"
    return f"{status}: {n} people in view"


def measure_co2(**kwargs):
    """Return a mock CO2 reading in ppm together with an air-quality verdict."""
    ppm = random.randint(400, 1600)
    log.info("measure_co2: mock reading = %d ppm", ppm)
    if ppm < 800:
        verdict = "air quality good"
    elif ppm < 1200:
        verdict = "air quality acceptable, ventilation advised"
    else:
        verdict = "air quality poor, the room needs airing out"
    return f"{ppm} ppm CO2 - {verdict}"


def identify_face(**kwargs):
    """Return a mock identification of the person in front of the camera."""
    name = random.choice(KNOWN_FACES + [None])
    log.info("identify_face: mock recognition = %s", name or "unknown")
    if name is None:
        return "no known face in front of the camera"
    return f"recognized {name} in front of the camera"
