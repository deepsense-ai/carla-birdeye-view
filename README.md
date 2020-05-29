![](https://img.shields.io/badge/contributions%20welcome-forking&gt;copying-orange.svg?style=popout-square)
![](https://img.shields.io/badge/release-v1.1.1-brightgreen.svg?style=popout-square)
![](https://img.shields.io/badge/pypi-v1.1.1-brightgreen.svg?style=popout-square)
![](https://img.shields.io/badge/CARLA-0.9.6+-blue.svg?style=popout-square)
![](https://img.shields.io/badge/python-3.6%20|%203.7%20|3.8-blue.svg?style=popout-square)
![](https://img.shields.io/badge/license-MIT-blue.svg?style=popout-square)

## Bird-eye's view for CARLA


Freeway-oriented shape | *[Learning By Cheating](https://arxiv.org/abs/1912.12294)*-like shape | Centerlines layer
:-------------------------:|:-------------------------:|:---------------------------------------:
![cruising](https://user-images.githubusercontent.com/64484917/80508193-04edde00-8978-11ea-956d-721e50a6a3c9.gif) | ![square-cruising](https://user-images.githubusercontent.com/64484917/80508095-e4258880-8977-11ea-8601-0e88942711ff.gif) | ![centerlines](https://user-images.githubusercontent.com/64484917/83240703-0fc7a880-a19a-11ea-9df0-74e340da8c3d.gif)


(GIFs above present feature maps after applying `as_rgb()` function)

### Motivation

During our [research](https://arxiv.org/abs/1911.12905) we found a very inspiring paper called [Learning By Cheating]( https://arxiv.org/abs/1912.12294). **Bird-eye's view** is made specifically to **learn faster thanks to much simpler, 2D world representation** (cheating oracle) which we think fits well in **Reinforcement Learning** setup.

This repository is an almost complete reimplementation that gives better performance and compatibility with most recent versions of CARLA. You can use it out-of-the-box as input for your model, and if necessary convert and visualize into RGB.


### Features
- **one-hot 3D feature map** (8x2D layers, each representing other entities, e.g. road layer, pedestrians layer) - made specifically to feed your CNN
- feature map **can be converted to an RGB** image
- layers can be easily removed
- caching mechanism for static layers like: roads and lanes
- using **OpenCV rendering** (efficient, multi-threading friendly) instead of slow Pygame method
- huge **FPS speedup** thanks to restricted rendering (only agent's surroundings, not whole map)
- all CARLA maps are supported out-of-the-box, custom maps with valid OpenDrive file made in RoadRunner are also supported
- current implementation is specifically  adjusted for highway scenarios (prolonged shape), but other shapes and crops are easy to implement 

### Installation
```bash
pip install carla-birdeye-view
```

### How to run
Make sure that `PYTHONPATH` env variable contains CARLA distribution egg, so that `carla` package can be imported.
```bash
# Launch server instance
./CarlaUE4.sh

# (optional) For CARLA 0.9.8+ you may get additional performance improvement with this
python PythonAPI/util/config.py --no-rendering

# Preview while cruising on autopilot (birdview/__main__.py)
python -m carla_birdeye_view
```

### Basic code usage

```python
from carla_birdeye_view import BirdViewProducer, BirdViewCropType, PixelDimensions

birdview_producer = BirdViewProducer(
    client,  # carla.Client
    target_size=PixelDimensions(width=150, height=336),
    pixels_per_meter=4,
    crop_type=BirdViewCropType.FRONT_AND_REAR_AREA
)

# Input for your model - call it every simulation step
# returned result is np.ndarray with ones and zeros of shape (8, height, width)
birdview = birdview_producer.produce(
    agent_vehicle=agent  # carla.Actor (spawned vehicle)
)

# Use only if you want to visualize
# produces np.ndarray of shape (height, width, 3)
rgb = BirdViewProducer.as_rgb(birdview)
```

### Contribution and feedback
We'd :heart: to collct any feedback, issues and pull requests!

### Credits

Project born at [deepsense.ai](deepsense.ai), made by:

![](https://avatars2.githubusercontent.com/u/12485656?s=22&v=4) [Michał Martyniak (@micmarty)](https://micmarty.github.io)


