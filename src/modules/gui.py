import numpy as np
import matplotlib.pyplot as plt

class DraggablePoint:
    def __init__(self, point, index, scatter, update_callback, fig, ax, control_points):
        self.point = point
        self.index = index
        self.scatter = scatter
        self.update_callback = update_callback
        self.press = None
        self.control_points = control_points
        self.fig = fig
        self.ax = ax


    def connect(self):
        self.cidpress = self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.cidrelease = self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.cidmotion = self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)


    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        contains, _ = self.scatter.contains(event)
        if contains:
            # Find the nearest point to click
            dists = np.linalg.norm(self.control_points - np.array([event.xdata, event.ydata]), axis=1)
            idx = np.argmin(dists)
            if np.allclose(self.control_points[idx], self.point):
                self.press = (self.point[0], self.point[1], event.xdata, event.ydata)

    def on_release(self, event):
        self.press = None


    def on_motion(self, event):
        if self.press is None or event.inaxes != self.ax:
            return
        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.point[0] = x0 + dx
        self.point[1] = y0 + dy
        self.scatter.set_offsets(self.control_points)
        # Update the corresponding label position
        # point_labels[self.index].set_position((self.point[0] + 0.03, self.point[1] + 0.03))
        self.update_callback()

class CursorController:
    def __init__(self, cursor, scatter, update_callback, fig, ax, control_points):
        self.cursor = cursor
        self.scatter = scatter
        self.update_callback = update_callback
        self.press = None
        self.fig = fig
        self.ax = ax
        self.control_points = control_points

    def connect(self):
        self.cidpress = self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.cidrelease = self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.cidmotion = self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        contains, _ = self.scatter.contains(event)
        if contains:
            self.press = (self.cursor[0], self.cursor[1], event.xdata, event.ydata)

    def on_release(self, event):
        self.press = None

    def on_motion(self, event):
        if self.press is None or event.inaxes != self.ax:
            return
        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.cursor[0] = x0 + dx
        self.cursor[1] = y0 + dy
        self.scatter.set_offsets([self.cursor])
        self.update_callback(isCursor=True)