import hashlib
import carla


def generate_opendrive_content_hash(map: carla.Map) -> str:
    opendrive_content = map.to_opendrive()
    hash_func = hashlib.sha1()
    hash_func.update(opendrive_content.encode("UTF-8"))
    opendrive_hash = str(hash_func.hexdigest())
    return opendrive_hash
