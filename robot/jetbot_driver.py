#from jetbot import Camera, bgr8_to_jpeg, Robot
import threading
import time

width = 160
height = 120

class JetbotDriver:

    observe = None

    def __init__(self):
        self.jetbot = Robot()
        self.controller = MobileController(10, self.jetbot)
        self.camera = Camera(width=width,height=height)
        self.controller.run()
        self.camera.observe(self._image_proc, names='value')
        self.camera.start()

    def _image_proc(self, change):
        image_value = change['new']
        self.observe = self.bridge.cv2_to_imgmsg(image_value, 'bgr8')


    def step(self,action):
        self.controller.radius = action[0]
        self.controller.speed = action[1]
        reward = 0.0
        done = False
        return self.observe, reward, done, None

    def reset(self):
        self.controller.radius = 0.0
        self.controller.speed = 0.0

        return self.observe

    def close(self):

        pass

    def render(self):
        pass


class MobileController:

    loop = True
    controll_thread = None
    left_v = 0.0
    right_v = 0.0
    max_radius = 60.0
    gradient = 30

    def __init__(self, wheel_distance, robot: Robot):
        self.wheel_distance = wheel_distance
        self.robot = robot
        self.radius = 0.0
        self.speed = 0.0

    def _controll_loop(self):

        while self.loop:
            self.robot.set_motors(self.left_v, self.right_v)
            time.sleep(0.1)

    def run(self):
        self.controll_thread = threading.Thread(target=self._controll_loop)
        self.controll_thread.start()
        pass

    def stop(self):
        self.loop = False

    def controll(self):

        if self.radius < 0e-10:
            radius = (30 * self.radius) + (self.max_radius)
            self.left_v = ((radius-self.wheel_distance)*self.speed)/(self.max_radius-self.wheel_distance) if radius != 0.0 else self.speed
            self.right_v = ((radius+self.wheel_distance)*self.speed)/(self.max_radius+self.wheel_distance) if radius != 0.0 else self.speed
        elif self.radius > 0e-10:
            radius = (-30 * self.radius) + (self.max_radius)
            self.left_v = (radius+self.wheel_distance)*self.speed/(self.max_radius+self.wheel_distance)  if radius != 0.0 else self.speed
            self.right_v = (radius-self.wheel_distance)*self.speed/(self.max_radius-self.wheel_distance) if radius != 0.0 else self.speed
        else:
            self.right_v = self.speed
            self.left_v = self.speed


