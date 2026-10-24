"""PyQt5 tool used to annotate expert masks (arrhythmia regions) on pre-processed ECG windows.

Usage: python tools/labelling_app.py --data-dir data/CPSC_final
Masks are saved as <data-dir>/mask/<record>_mask.npz, the layout expected by train.py.
"""
import argparse
import sys
import os
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QComboBox, QSlider, QLabel, QMessageBox
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class ECGLabelingApp(QMainWindow):
    def __init__(self, data_dir):
        super().__init__()
        self.signal = None
        self.mask = None
        self.current_lead = 0  # Default to lead 1 (index 0)
        self.input_file_name = None
        self.input_file_dir = None
        self.default_dir = data_dir
        self.mask_dir = os.path.join(data_dir, 'mask')
        self.metadata = None
        self.label_file_map = {}
        self.initUI()

    def initUI(self):
        self.setWindowTitle('ECG Arrhythmia Labeling App')

        # Main widget
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)

        # Main horizontal layout
        main_layout = QHBoxLayout(main_widget)

        # Left vertical layout for buttons and selectors
        button_layout = QVBoxLayout()
        button_layout.addStretch()  # Push buttons to top


        # Label selection
        self.label_selector = QComboBox(self)
        self.label_selector.addItem("Select Label")
        self.label_selector.currentIndexChanged.connect(self.update_record_selector)
        button_layout.addWidget(QLabel('Select Label:'))
        button_layout.addWidget(self.label_selector)


        # Record selection
        self.record_selector = QComboBox(self)
        self.record_selector.addItem("Select Record")
        self.record_selector.currentIndexChanged.connect(self.load_npz)
        button_layout.addWidget(QLabel('Select Record:'))
        button_layout.addWidget(self.record_selector)


        self.load_csv()


        # Load NPZ button (fallback for no CSV)
        self.load_button = QPushButton('Load NPZ File', self)
        self.load_button.clicked.connect(self.load_npz_fallback)
        button_layout.addWidget(self.load_button)

        # Create empty mask button
        self.create_empty_button = QPushButton('Create Empty Mask', self)
        self.create_empty_button.clicked.connect(self.create_empty_mask)
        button_layout.addWidget(self.create_empty_button)

        # Save mask button
        self.save_button = QPushButton('Save Mask', self)
        self.save_button.clicked.connect(self.save_mask)
        button_layout.addWidget(self.save_button)
        button_layout.addStretch()  # Keep buttons compact
        main_layout.addLayout(button_layout)

        # Right vertical layout for plot and controls
        controls_layout = QVBoxLayout()

        # Matplotlib canvas
        self.figure = Figure(figsize=(8, 8))
        self.canvas = FigureCanvas(self.figure)
        controls_layout.addWidget(self.canvas)

        # Plot setup
        self.ax_lead1 = self.figure.add_subplot(411)
        self.ax_lead2 = self.figure.add_subplot(412)
        self.ax_lead3 = self.figure.add_subplot(413)
        self.ax_mask = self.figure.add_subplot(414)
        self.ax_lead1.set_title('ECG Lead (No Signal Loaded)')
        self.ax_lead1.set_xlabel('Sample')
        self.ax_lead1.set_ylabel('Amplitude')
        self.ax_lead2.set_xlabel('Sample')
        self.ax_lead2.set_ylabel('Amplitude')
        self.ax_lead3.set_xlabel('Sample')
        self.ax_lead3.set_ylabel('Amplitude')
        self.ax_mask.set_title('Mask Preview')
        self.ax_mask.set_xlabel('Sample')
        self.ax_mask.set_ylabel('Mask Value')
        self.figure.tight_layout()

        # Lead selection
        self.lead_selector = QComboBox(self)
        self.lead_selector.addItems([f'Lead {i+1}' for i in range(12)])
        controls_layout.addWidget(QLabel('Select Lead:'))
        controls_layout.addWidget(self.lead_selector)

        # Slider for masking
        self.start_slider = QSlider(Qt.Horizontal, self)
        self.start_slider.setMinimum(0)
        self.start_slider.valueChanged.connect(self.update_mask)
        controls_layout.addWidget(QLabel('Mask Start:'))
        controls_layout.addWidget(self.start_slider)

        self.end_slider = QSlider(Qt.Horizontal, self)
        self.end_slider.setMinimum(0)
        self.end_slider.valueChanged.connect(self.update_mask)
        controls_layout.addWidget(QLabel('Mask End:'))
        controls_layout.addWidget(self.end_slider)

        main_layout.addLayout(controls_layout)

        # Connect lead selector after UI setup
        self.lead_selector.currentIndexChanged.connect(self.change_lead)

    def create_empty_mask(self):
        if self.signal is None:
            print("No signal loaded. Load an NPZ file first.")
            QMessageBox.warning(self, "Error", "No signal loaded. Please load an NPZ file first.")
            return
        # Reset mask to zeros
        self.mask = np.zeros(self.signal.shape[1], dtype=np.int8)
        # Reset sliders to initial values
        signal_length = self.signal.shape[1]
        self.start_slider.setValue(500 if signal_length > 500 else 0)
        self.end_slider.setValue(1000 if signal_length > 1000 else signal_length - 1)
        # Update plots
        self.plot_signal_and_mask()

        
    def load_csv(self):
        file_name = os.path.join(self.default_dir, 'meta_data.csv')  # Default to metadata.csv in default directory
        if file_name:
            try:
                self.metadata = pd.read_csv(file_name)
                if 'record' in self.metadata.columns and 'label' in self.metadata.columns:
                    # Populate label selector with unique labels (as strings)
                    labels = sorted(self.metadata['label'].astype(str).unique())
                    self.label_selector.clear()
                    self.label_selector.addItem("Select Label")
                    self.label_selector.addItems(labels)
                    # Map labels to record names
                    self.label_file_map = {}
                    for label in labels:
                        self.label_file_map[label] = self.metadata[self.metadata['label'].astype(str) == label]['record'].tolist()
                    
                    self.input_file_dir = os.path.dirname(file_name)  # Use CSV directory as base
                    # Clear record selector
                    self.record_selector.clear()
                    self.record_selector.addItem("Select Record")
                else:
                    print("Error: CSV must contain 'record' and 'label' columns")
                    QMessageBox.warning(self, "Error", "CSV must contain 'record' and 'label' columns")
            except Exception as e:
                print(f"Error loading CSV: {e}")
                QMessageBox.warning(self, "Error", f"Failed to load CSV: {e}")

    def update_record_selector(self, index):
        # Clear record selector
        self.record_selector.clear()
        self.record_selector.addItem("Select Record")
        # Populate with records for selected label
        selected_label = self.label_selector.currentText()
        if selected_label != "Select Label" and self.metadata is not None:
            record_names = self.label_file_map.get(selected_label, [])
            if not record_names:
                print(f"No records found for label: {selected_label}")
                QMessageBox.warning(self, "Error", f"No records found for label: {selected_label}")
                return
            # Filter out records with existing mask files
            unmasked_records = []
            for record in record_names:
                mask_file = os.path.join(self.mask_dir, f"{record}_mask.npz")
                if not os.path.exists(mask_file):
                    unmasked_records.append(record)
            if unmasked_records:
                self.record_selector.addItems(unmasked_records)
            else:
                print(f"All records for label {selected_label} have been masked")
                QMessageBox.warning(self, "Error", f"All records for label {selected_label} have been masked")

    def load_npz(self, index):
        if self.metadata is None or self.record_selector.currentText() == "Select Record":
            return  # Do nothing if no CSV or no record selected

        selected_record = self.record_selector.currentText()
        selected_label = self.label_selector.currentText()

        print(f"Selected Record: {selected_record}, Selected Label: {selected_label}")
        if selected_label == "Select Label" or selected_record == "Select Record" or selected_record not in self.label_file_map.get(selected_label, []):
            print("Please select a valid label")
            QMessageBox.warning(self, "Error", "Please select a valid label")
            return

        # Construct file path
        file_name = os.path.join(self.input_file_dir, f"{selected_record}.npz")
        if not os.path.exists(file_name):
            print(f"File not found: {file_name}")
            QMessageBox.warning(self, "Error", f"File not found: {file_name}")
            return

        try:
            data = np.load(file_name)
            if 'x' in data:
                self.signal = data['x']
                if self.signal.shape[0] == 12:
                    self.mask = np.zeros(self.signal.shape[1], dtype=np.int8)
                    signal_length = self.signal.shape[1]
                    self.start_slider.setMaximum(signal_length - 1)
                    self.end_slider.setMaximum(signal_length - 1)
                    # Set initial slider values
                    self.start_slider.setValue(500 if signal_length > 500 else 0)
                    self.end_slider.setValue(1000 if signal_length > 1000 else signal_length - 1)
                    # Store input file name and directory
                    self.input_file_name = selected_record
                    self.update_mask()  # Update mask with initial slider values
                    self.plot_signal_and_mask()
                else:
                    print("Error: Expected 12 leads in 'x' field")
                    QMessageBox.warning(self, "Error", "Expected 12 leads in 'x' field")
            else:
                print("Error: 'x' field not found in NPZ file")
                QMessageBox.warning(self, "Error", "'x' field not found in NPZ file")
        except Exception as e:
            print(f"Error loading NPZ file: {e}")
            QMessageBox.warning(self, "Error", f"Failed to load NPZ file: {e}")

    def load_npz_fallback(self):
        # Fallback to original file dialog if no CSV is loaded
        file_name, _ = QFileDialog.getOpenFileName(self, "Open NPZ File", self.default_dir, "NPZ Files (*.npz)")
        if file_name:
            try:
                data = np.load(file_name)
                if 'x' in data:
                    self.signal = data['x']
                    if self.signal.shape[0] == 12:
                        self.mask = np.zeros(self.signal.shape[1], dtype=np.int8)
                        signal_length = self.signal.shape[1]
                        self.start_slider.setMaximum(signal_length - 1)
                        self.end_slider.setMaximum(signal_length - 1)
                        # Set initial slider values
                        self.start_slider.setValue(500 if signal_length > 500 else 0)
                        self.end_slider.setValue(1000 if signal_length > 1000 else signal_length - 1)
                        # Store input file name and directory
                        self.input_file_dir = os.path.dirname(file_name)
                        self.input_file_name = os.path.splitext(os.path.basename(file_name))[0]
                        self.update_mask()  # Update mask with initial slider values
                        self.plot_signal_and_mask()
                    else:
                        print("Error: Expected 12 leads in 'x' field")
                        QMessageBox.warning(self, "Error", "Expected 12 leads in 'x' field")
                else:
                    print("Error: 'x' field not found in NPZ file")
                    QMessageBox.warning(self, "Error", "'x' field not found in NPZ file")
            except Exception as e:
                print(f"Error loading NPZ file: {e}")
                QMessageBox.warning(self, "Error", f"Failed to load NPZ file: {e}")

    def change_lead(self, index):
        self.current_lead = index
        self.plot_signal_and_mask()

    def update_mask(self):
        if self.signal is None:
            return
        start_idx = self.start_slider.value()
        end_idx = self.end_slider.value()
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
        self.mask = np.zeros(self.signal.shape[1], dtype=np.int8)
        self.mask[start_idx:end_idx + 1] = 1
        self.plot_signal_and_mask()

    def plot_signal_and_mask(self):
        self.ax_lead1.clear()
        self.ax_lead2.clear()
        self.ax_lead3.clear()
        self.ax_mask.clear()

        if self.signal is None:
            self.ax_lead1.set_title('ECG Lead (No Signal Loaded)')
            self.ax_lead1.set_xlabel('Sample')
            self.ax_lead1.set_ylabel('Amplitude')
            self.ax_lead2.set_xlabel('Sample')
            self.ax_lead2.set_ylabel('Amplitude')
            self.ax_lead3.set_xlabel('Sample')
            self.ax_lead3.set_ylabel('Amplitude')
            self.ax_mask.set_title('Mask Preview')
            self.ax_mask.set_xlabel('Sample')
            self.ax_mask.set_ylabel('Mask Value')
            self.canvas.draw()
            return

        # Determine leads to display
        lead1_idx = self.current_lead
        lead2_idx = (self.current_lead + 1) % 12  # Next lead, wrap around
        lead3_idx = (self.current_lead + 2) % 12  # Next lead, wrap around

        # Plot ECG signals for three leads
        self.ax_lead1.plot(self.signal[lead1_idx, :], label=f'Lead {lead1_idx + 1}', color='blue')
        self.ax_lead2.plot(self.signal[lead2_idx, :], label=f'Lead {lead2_idx + 1}', color='green')
        self.ax_lead3.plot(self.signal[lead3_idx, :], label=f'Lead {lead3_idx + 1}', color='purple')

        # Highlight masked regions with axvspan
        mask_indices = np.where(self.mask == 1)[0]
        if len(mask_indices) > 0:
            # Find continuous segments in the mask
            start_indices = [mask_indices[0]]
            end_indices = []
            for i in range(1, len(mask_indices)):
                if mask_indices[i] != mask_indices[i-1] + 1:
                    end_indices.append(mask_indices[i-1])
                    start_indices.append(mask_indices[i])
            end_indices.append(mask_indices[-1])
            # Draw axvspan for each continuous segment
            for start, end in zip(start_indices, end_indices):
                self.ax_lead1.axvspan(start, end + 1, color='red', alpha=0.3, label='Arrhythmia' if start == start_indices[0] else None)
                self.ax_lead2.axvspan(start, end + 1, color='red', alpha=0.3, label='Arrhythmia' if start == start_indices[0] else None)
                self.ax_lead3.axvspan(start, end + 1, color='red', alpha=0.3, label='Arrhythmia' if start == start_indices[0] else None)

        self.ax_lead1.set_title(f'ECG Lead {lead1_idx + 1}')
        self.ax_lead1.set_xlabel('Sample')
        self.ax_lead1.set_ylabel('Amplitude')
        self.ax_lead1.grid(True)
        self.ax_lead1.legend()

        self.ax_lead2.set_title(f'ECG Lead {lead2_idx + 1}')
        self.ax_lead2.set_xlabel('Sample')
        self.ax_lead2.set_ylabel('Amplitude')
        self.ax_lead2.grid(True)
        self.ax_lead2.legend()

        self.ax_lead3.set_title(f'ECG Lead {lead3_idx + 1}')
        self.ax_lead3.set_xlabel('Sample')
        self.ax_lead3.set_ylabel('Amplitude')
        self.ax_lead3.grid(True)
        self.ax_lead3.legend()


        # Plot mask preview
        self.ax_mask.step(range(len(self.mask)), self.mask, where='mid', color='black', label='Mask')
        self.ax_mask.set_ylim(-0.1, 1.1)
        self.ax_mask.set_title('Mask Preview')
        self.ax_mask.set_xlabel('Sample')
        self.ax_mask.set_ylabel('Mask Value')
        self.ax_mask.grid(True)
        self.ax_mask.legend()

        self.canvas.draw()

    def save_mask(self):
        if self.mask is None:
            print("No mask to save. Load a signal first.")
            return
        if self.input_file_name is None or self.input_file_dir is None:
            print("No input file loaded. Please load an NPZ file first.")
            return
        # Construct output filename
        os.makedirs(self.mask_dir, exist_ok=True)
        output_file = os.path.join(self.mask_dir, f"{self.input_file_name}_mask.npz")
        np.savez(output_file, mask=self.mask)
        print(f"Mask saved to {output_file}")
        # Show success message
        QMessageBox.information(self, "Success", f"Mask successfully saved")
        # Refresh record selector to exclude the newly masked record
        self.update_record_selector(self.label_selector.currentIndex())

def main():
    parser = argparse.ArgumentParser(description="ECG expert-mask labelling tool")
    parser.add_argument('--data-dir', default='data/CPSC_final', help="Directory with meta_data.csv and <record>.npz files")
    args, qt_args = parser.parse_known_args()

    app = QApplication(sys.argv[:1] + qt_args)
    ex = ECGLabelingApp(args.data_dir)
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()