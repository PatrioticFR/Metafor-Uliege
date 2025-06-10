# POST-ANALYSIS GRAPHICAL VISUALIZATION TOOL
# Run this script AFTER all simulations are completed to generate comprehensive graphs

import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from datetime import datetime


class PostAnalysisVisualizer:
    """Generate comprehensive graphs from all simulation results"""

    def __init__(self, results_base_dir=None):
        if results_base_dir is None:
            results_base_dir = r"C:\Users\adpie\Desktop\Stage 4A\Metafor\AutomatedResults"

        self.results_dir = Path(results_base_dir)
        self.output_dir = self.results_dir / "GraphicalAnalysis"
        self.output_dir.mkdir(exist_ok=True)

        print(f"Post-analysis visualizer initialized")
        print(f"Results directory: {self.results_dir}")
        print(f"Output directory: {self.output_dir}")

        # Configure matplotlib for better plots
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10

        self.simulation_data = {}

    def load_simulation_data(self):
        """Load data from all simulation directories"""
        print("\n=== LOADING SIMULATION DATA ===")

        # Find all simulation directories
        sim_dirs = [d for d in self.results_dir.iterdir()
                    if d.is_dir() and d.name != 'GraphicalAnalysis']

        print(f"Found {len(sim_dirs)} simulation directories:")

        for sim_dir in sim_dirs:
            sim_name = sim_dir.name
            print(f"  Loading: {sim_name}")

            # Load all .ascii files from this simulation
            ascii_files = list(sim_dir.glob("*.ascii"))

            sim_data = {
                'directory': sim_dir,
                'files': {},
                'config_info': self._extract_config_info(sim_name)
            }

            # Load each ascii file
            for ascii_file in ascii_files:
                try:
                    with open(ascii_file, 'r') as f:
                        values = [float(line.strip()) for line in f if line.strip()]

                    file_key = ascii_file.stem  # filename without extension
                    sim_data['files'][file_key] = values

                except Exception as e:
                    print(f"    Warning: Could not load {ascii_file.name}: {e}")

            print(f"    Loaded {len(sim_data['files'])} data files")
            self.simulation_data[sim_name] = sim_data

        print(f"\nTotal simulations loaded: {len(self.simulation_data)}")

    def _extract_config_info(self, sim_name):
        """Extract configuration info from simulation name"""
        config = {
            'material': 'BE_CU',  # default
            'thermal': False,
            'variation': None
        }

        if 'INVAR' in sim_name:
            config['material'] = 'INVAR'
        elif 'STEEL' in sim_name:
            config['material'] = 'STEEL'

        if 'Thermal' in sim_name:
            config['thermal'] = True

        if 'Thick' in sim_name:
            config['variation'] = 'thickness'
        elif 'Width' in sim_name:
            config['variation'] = 'width'
        elif 'Mechanical_Only' in sim_name:
            config['variation'] = 'mechanical_only'

        return config

    def plot_displacement_comparison(self):
        """Compare mass displacements across all simulations"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Mass Displacement Comparison Across All Simulations', fontsize=16, fontweight='bold')

        # Define displacement variables to plot
        displacement_vars = [
            ('dispY_Bottom_right_mass', 'Mass Bottom Right Y'),
            ('dispY_Top_right_mass', 'Mass Top Right Y'),
            ('displacement_rod_end_Y', 'Rod End Y'),
            ('dispX_rod_end', 'Rod End X (Thermal Effect)')
        ]

        for idx, (var_name, plot_title) in enumerate(displacement_vars):
            ax = axes[idx // 2, idx % 2]

            for sim_name, sim_data in self.simulation_data.items():
                if var_name in sim_data['files'] and 'time' in sim_data['files']:
                    time_data = sim_data['files']['time']
                    disp_data = sim_data['files'][var_name]

                    if len(time_data) == len(disp_data):
                        # Color and style based on configuration
                        config = sim_data['config_info']
                        color = self._get_color_for_config(config)
                        linestyle = '--' if config['thermal'] else '-'

                        ax.plot(time_data, disp_data,
                                label=self._get_label_for_sim(sim_name, config),
                                color=color, linestyle=linestyle, linewidth=2)

            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Displacement [mm]')
            ax.set_title(plot_title)
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / 'displacement_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_thermal_effects(self):
        """Plot thermal-specific effects for configurations with thermal loading"""
        thermal_sims = {name: data for name, data in self.simulation_data.items()
                        if data['config_info']['thermal']}

        if not thermal_sims:
            print("No thermal simulations found - skipping thermal plots")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Thermal Effects Analysis', fontsize=16, fontweight='bold')

        # Temperature evolution
        ax1 = axes[0, 0]
        for sim_name, sim_data in thermal_sims.items():
            if 'temp_mean_blade_K' in sim_data['files'] and 'time' in sim_data['files']:
                time_data = sim_data['files']['time']
                temp_data = [t - 273.15 for t in sim_data['files']['temp_mean_blade_K']]  # Convert to Celsius

                color = self._get_color_for_config(sim_data['config_info'])
                ax1.plot(time_data, temp_data, label=sim_name, color=color, linewidth=2)

        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Temperature [°C]')
        ax1.set_title('Blade Temperature Evolution')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Thermal strain
        ax2 = axes[0, 1]
        for sim_name, sim_data in thermal_sims.items():
            if 'thermal_strain_max_blade' in sim_data['files'] and 'time' in sim_data['files']:
                time_data = sim_data['files']['time']
                strain_data = sim_data['files']['thermal_strain_max_blade']

                color = self._get_color_for_config(sim_data['config_info'])
                ax2.plot(time_data, strain_data, label=sim_name, color=color, linewidth=2)

        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Thermal Strain [-]')
        ax2.set_title('Maximum Thermal Strain Evolution')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Thermal displacement X
        ax3 = axes[1, 0]
        for sim_name, sim_data in thermal_sims.items():
            if 'thermal_dispX_mass_top_right' in sim_data['files'] and 'time' in sim_data['files']:
                time_data = sim_data['files']['time']
                disp_data = sim_data['files']['thermal_dispX_mass_top_right']

                color = self._get_color_for_config(sim_data['config_info'])
                ax3.plot(time_data, disp_data, label=sim_name, color=color, linewidth=2)

        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel('Thermal Displacement X [mm]')
        ax3.set_title('Mass Thermal Displacement (X)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # 4. Thermal displacement Y
        ax4 = axes[1, 1]
        for sim_name, sim_data in thermal_sims.items():
            if 'thermal_dispY_mass_top_right' in sim_data['files'] and 'time' in sim_data['files']:
                ax4.plot(sim_data['files']['time'], sim_data['files']['thermal_dispY_mass_top_right'],
                         label=sim_name,
                         color=self._get_color_for_config(sim_data['config_info']), linewidth=2)
        ax4.set_xlabel('Time [s]')
        ax4.set_ylabel('Thermal Displacement Y [mm]')
        ax4.set_title('Mass Thermal Displacement (Y)')
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / 'thermal_effects_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def run_all(self):
        """Run the full analysis workflow"""
        self.load_simulation_data()
        self.plot_displacement_comparison()
        self.plot_thermal_effects()
        print("\nAll plots generated successfully.")

# Entry point
if __name__ == '__main__':
    visualizer = PostAnalysisVisualizer(
            results_base_dir=r"C:\Users\adpie\Desktop\Stage 4A\Metafor\AutomatedResults"
        )
    visualizer.run_all()