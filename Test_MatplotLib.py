import matplotlib
matplotlib.use('module://kivy.garden.matplotlib.backend_kivy')
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
canvas = fig.canvas


class MyApp(App):
    def build(self):
        box = BoxLayout()
        self.i = 0
        self.line = [self.i]
        box.add_widget(canvas)
        plt.show()
        Clock.schedule_interval(self.update, 1)
        return box

    def update(self, *args):
        plt.plot(self.line, self.line)
        self.i += 1
        self.line.append(self.i)
        canvas.draw_idle()


MyApp().run()
