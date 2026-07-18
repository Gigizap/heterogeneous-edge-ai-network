import logging

log = logging.getLogger(__name__)


def use_benchmark(**kwargs):
    log.info("use_benchmark executed with %s", kwargs)
    return "benchmark says: BENCHMARK EXECUTED!" + str(kwargs)

def camera_status(**kwargs):
    log.info("camera_status executed with %s", kwargs)
    return "camera status: CAMERA IS ON!" + str(kwargs)
