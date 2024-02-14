# CursorMover
Simple 2D env and an agent with brain-inspired architecture

## Features

The project contains codes implementing a visual environment that displays shapes and cursors, and an agent that controls the gaze and a cursor based on the visual information.

### Environment

The environment (`CursorMoverEnv.py`) gives an observation of visual objects (card game suites) and cursors.  The environment receives saccade commands to move the gaze and cursor control commands including dragging an object.  It uses  to render the observation.  
<p align="center">
<img src="/CursorMoverFig1.png" width="250px"/><br><strong>Fig.1</strong><br>
Observation Size = Scene Image<br> Size = Grid Size × Scene Size<br>
Stage Image Size = ½ Scene Image Size<br>
Visual objects are displayed in the Stage.</p>

### Agent
The agent (`CursorMover.py`) directs its gaze to the most salient object in the observation and moves a cursor/drags an object by sending  commands to the environment.

#### Gaze control
The gaze is directed to the most salient part in the visual field.  The mechanism is imported from another repository [Vision1](https://github.com/rondelion/Vision1), where `Vision1.py` contains modules `Periphery2Saliency` to calculate saliency and `PriorityMap2Gaze` to calculate gaze shift (saccade).

#### Cursor control
The cursor is controlled by a [corico-BG-loop](https://en.wikipedia.org/wiki/Cortico-basal_ganglia-thalamo-cortical_loop) model ([MinimalCtxBGA](https://github.com/rondelion/MinimalCtxBGA)).   When the cursor overlaps an object, the object may be dragged.

#### Agent architecture
<p align="center">
<img src="/CursorMoverArchitecture.png" width="500px"/><br><strong>Fig.2</strong><br>


## How to Install
* Clone this repository, [Vision1](https://github.com/rondelion/Vision1), [AEPredictor](https://github.com/rondelion/AEPredictor), and [MinimalCtxBGA](https://github.com/rondelion/MinimalCtxBGA).

* Install [BriCA](https://github.com/wbap/BriCA1) and [BriCAL](https://github.com/wbap/BriCAL).

* Install [Gymnasium](https://gymnasium.farama.org), cv2 (OpenCV for Vision1), ~~TensorForce,~~ and [PyGame](https://www.pygame.org/news).

* Add python paths to the installed modules.

* Register the environment to Gym
    * Place `CursorMoverEnv.py` file in `gymnasium/envs/myenv`  
    (wherever Gym to be used is installed)
    * Add to `__init__.py` (located in the same folder)  
      `from gymnasium.envs.myenv.CursorMoverEnv import CursorMoverEnv`
    * Add to `gymnasium/envs/__init__.py`  
```
register(
    id="CursorMoverEnv-v0",
    entry_point="gymnasium.envs.myenv:CursorMoverEnv",
    max_episode_steps=1000,
)
```

## Usage

### Command options

      --dump: dump file path
      --episode_count: Number of training episodes (default: 1)
      --max_steps: Max steps in an episode (default: 20)
      --config: Model configuration (default: CBT1CA.json)
      --brical: BriCA Language file 
      --dump_flags: f:fovea, c:cursor action, b:bg, o:obs, p:predictor
      --no_render: No env image rendering

### Sample usage

```
$ python CursorMover.py --episode_count 1000 --dump_flags "c" --config "CursorMoverDQN.json" --dump "dump/dump.txt" --no_render

```


### Required files:
* `CursorMover.py`: main program
* `CursorMoverDQN.json`: config file
* `CursorMover.brical.json` : architecture description for BriCAL

### Other files:
* A dump file is created with --dump option.
* ML modules use model files -- see the code and configuration files.
