![](https://img.shields.io/badge/release-v1.0-brightgreen.svg?style=popout-square)
![](https://img.shields.io/badge/pypi-v1.0-brightgreen.svg?style=popout-square)
![](https://img.shields.io/badge/CARLA-0.9.6+-blue.svg?style=popout-square)
![](https://img.shields.io/badge/python-3.6%20|%203.7-blue.svg?style=popout-square)
![](https://img.shields.io/badge/license-MIT-blue.svg?style=popout-square)

## Bird eye's view for CARLA

(GIFs below present feature maps after applying RGB conversion)
Freeway-oriented shape | *Learning By Cheating*-like shape
:-------------------------:|:-------------------------:
![cruising](https://user-images.githubusercontent.com/64484917/80508193-04edde00-8978-11ea-956d-721e50a6a3c9.gif) | ![square-cruising](https://user-images.githubusercontent.com/64484917/80508095-e4258880-8977-11ea-8601-0e88942711ff.gif)

### Features
- **one-hot 3D feature map** (8x2D layers, each representing other entities, e.g. road layer, pedestrians layer) - made specifically to feed your CNN
- road and lane layers caching mechanism
- feature map **can be converted to an RGB** image
- uses **OpenCV rendering** (efficient, multi-threading friendly) instead of slow Pygame method
- huge **FPS speedup** thanks to restricted rendering (only agent's surroundings, not whole map)
- all CARLA maps are supported out-of-the-box, custom maps with valid OpenDrive file made in RoadRunner are also supported
- current implementation is specifically  adjusted for highway scenarios (prolonged shape), but other shapes and crops are easy to implement 

### Installation
```bash
pip install carla-birdview
```

### How to run

Make sure that PYTHONPATH env variable contains CARLA dist egg
```bash
# Launch server instance
./CarlaUE4.sh

# Preview while cruising on autopilot
python -m birdview
```

### Contribution and feedback
We'd :heart: to collct any feedback, issues and pull requests!

### Credits

Project born at [deepsense.ai](deepsense.ai), made by:

![](https://avatars2.githubusercontent.com/u/12485656?s=22&v=4) [Michał Martyniak (@micmarty)](https://micmarty.github.io)


