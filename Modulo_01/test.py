import sys

# Flexible import supporting either PyQt6 or PySide6
try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QApplication,
        QGridLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QWidget,
    )
except ImportError:
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (
            QApplication,
            QGridLayout,
            QLabel,
            QLineEdit,
            QPushButton,
            QWidget,
        )
    except ImportError:
        print("Error: Neither PyQt6 nor PySide6 is installed.")
        print("Install one using:")
        print("    uv add PyQt6")
        print("  or:")
        print("    pip install PyQt6")
        sys.exit(1)


class SimpleQtApp(QWidget):
    def __init__(self):
        super().__init__()
        self.counter = 0
        self.init_ui()

    def init_ui(self):
        # Set window properties
        self.setWindowTitle("Qt Grid Layout Example")
        self.resize(400, 250)

        # Create Grid Layout
        grid = QGridLayout()
        grid.setSpacing(10)

        # 1. Header / Title Text (spans 1 row, 2 columns)
        title_label = QLabel("Qt Demo: Buttons, Texts & Grid")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        grid.addWidget(title_label, 0, 0, 1, 2)

        # 2. Text label and LineEdit input
        name_label = QLabel("Enter your name:")
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Type something here...")
        grid.addWidget(name_label, 1, 0)
        grid.addWidget(self.name_input, 1, 1)

        # 3. Action Buttons
        self.greet_btn = QPushButton("Submit")
        self.greet_btn.clicked.connect(self.on_submit)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.on_clear)

        grid.addWidget(self.greet_btn, 2, 0)
        grid.addWidget(self.clear_btn, 2, 1)

        # 4. Click Counter Button and Display
        self.counter_btn = QPushButton("Click Me!")
        self.counter_btn.clicked.connect(self.on_increment_counter)

        self.counter_label = QLabel("Clicks: 0")
        self.counter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        grid.addWidget(self.counter_btn, 3, 0)
        grid.addWidget(self.counter_label, 3, 1)

        # 5. Output / Status Text (spans 1 row, 2 columns)
        self.output_label = QLabel("Welcome! Enter your name and click Submit.")
        self.output_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.output_label.setStyleSheet("color: #2b579a; font-style: italic; margin-top: 10px;")
        grid.addWidget(self.output_label, 4, 0, 1, 2)

        # Set the grid as main layout
        self.setLayout(grid)

    def on_submit(self):
        text = self.name_input.text().strip()
        if text:
            self.output_label.setText(f"Hello, {text}! Nice to meet you.")
        else:
            self.output_label.setText("Please enter a name first!")

    def on_clear(self):
        self.name_input.clear()
        self.counter = 0
        self.counter_label.setText("Clicks: 0")
        self.output_label.setText("Cleared! Enter your name and click Submit.")

    def on_increment_counter(self):
        self.counter += 1
        self.counter_label.setText(f"Clicks: {self.counter}")


def main():
    app = QApplication(sys.argv)
    window = SimpleQtApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
