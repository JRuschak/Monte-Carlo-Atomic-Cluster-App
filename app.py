from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLineEdit, QLabel, QHBoxLayout
from PyQt6.QtCore import QThread, pyqtSignal
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import numpy as np
import sys
import argon
import time

class SimulationWorker(QThread):
    update_signal = pyqtSignal(np.ndarray, float)  # single numpy array for positions

    def __init__(self, num_atoms, temp, iterations):
        super().__init__()
        self.num_atoms = num_atoms
        self.temp = temp
        self.iterations = iterations

    def run(self):
        cluster = argon.initialize_cluster(self.num_atoms)
        
        for i in range(self.iterations):
            argon.translational_move(cluster, self.temp, 0.25)
            argon.COM(cluster)

            if i % 200 == 0:  # update less frequently for performance
                coords = np.array(argon.graph(cluster)).T  # shape (N,3)
                energy = argon.calc_energy(cluster)
                self.update_signal.emit(coords, energy)
                self.msleep(2)  # give GUI breathing room

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Monte Carlo PyQtGraph 3D")
        self.resize(1200, 800)

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Graph takes most of the space
        self.view = gl.GLViewWidget()
        self.view.opts['distance'] = 40
        main_layout.addWidget(self.view, stretch=8)  # stretch factor high for graph

        grid = gl.GLGridItem()
        grid.scale(2,2,1)
        self.view.addItem(grid)

        self.scatter = gl.GLScatterPlotItem(size=1, color=(1,0,0,1), pxMode=False)
        self.view.addItem(self.scatter)

        # Compact horizontal layout for inputs
        controls_layout = QHBoxLayout()
        main_layout.addLayout(controls_layout, stretch=1)  # less space than graph

        controls_layout.addWidget(QLabel("Atoms:"))
        self.atom_input = QLineEdit()
        self.atom_input.setFixedWidth(60)
        controls_layout.addWidget(self.atom_input)

        controls_layout.addWidget(QLabel("steps:"))
        self.steps_input = QLineEdit()
        self.steps_input.setFixedWidth(60)
        controls_layout.addWidget(self.steps_input)

        controls_layout.addWidget(QLabel("Temp:"))
        self.temp_input = QLineEdit()
        self.temp_input.setFixedWidth(60)
        controls_layout.addWidget(self.temp_input)

        self.run_button = QPushButton("Run Simulation")
        controls_layout.addWidget(self.run_button)
        self.run_button.clicked.connect(self.run_sim)

        # Energy display
        self.energy_label = QLabel("Energy: 0.0")
        controls_layout.addWidget(self.energy_label)

    def run_sim(self):
        try:
            num_atoms = int(self.atom_input.text())
            temp = int(self.temp_input.text())
            steps = int(self.steps_input.text())
        except ValueError:
            print("Invalid input")
            return

        self.worker = SimulationWorker(num_atoms, temp, steps)
        self.worker.update_signal.connect(self.update_plot)
        self.worker.start()

    def update_plot(self, coords, energy):
        self.scatter.setData(pos=coords)
        self.energy_label.setText(f"Energy: {energy:.3f}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())