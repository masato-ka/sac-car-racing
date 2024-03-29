# Configuration
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
N_CHANNEL = 3
INPUT_DIM = (IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNEL)

# Raw camera input
CAMERA_HEIGHT = 120
CAMERA_WIDTH = 160
CAMERA_RESOLUTION = (CAMERA_WIDTH, CAMERA_HEIGHT)
MARGIN_TOP = CAMERA_HEIGHT // 3
# Region Of Interest
# r = [margin_left, margin_top, width, height]
ROI = [0, MARGIN_TOP, CAMERA_WIDTH, CAMERA_HEIGHT - MARGIN_TOP]

N_COMMAND_HISTORY = 20

MAX_THROTTLE = 0.7#0.6
MIN_THROTTLE = 0.5

MAX_STEERING_DIFF = 0.2#0.15
MAX_STEERING = 1
MIN_STEERING = -1

CTE_ERROR = 2.0

REWARD_CRASH = -10 #negative number.

CRASH_REWARD_WEIGHT = 5
THROTTLE_REWARD_WEIGHT = 0.1
JERK_REWARD_WEIGHT = 0.00
