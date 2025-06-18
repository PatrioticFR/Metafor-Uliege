import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
import matplotlib.patches as patches

# Geometric parameters of the system (extracted from Metafor code)
# Spring blade
e = 0.24  # thickness
L = 105.25  # length

# Rod
l = 79.2  # total length
H = 3.875  # thickness
r = 7  # distance between mass and end of rod
R = H
enc = 57.32  # Position of the clamping point on the rod

# Mass
D = 39.99  # height
d = 13.96  # length
y = D / 2  # mass offset (centered on the rod)
h = l - r - d

# Geometry points (from Metafor code)
points = {
    # Spring blade
    'p1': (enc, H / 2),
    'p2': (enc + e, H / 2),
    'p3': (enc + e, L),
    'p4': (enc, L),

    # Rod
    'p5': (0.0, H / 2),
    'p6': (0.0, -H / 2),
    'p7': (h, -H / 2),
    'p8': (h, H / 2),

    # Mass
    'p9': (h, D - y),
    'p10': (h, -y),
    'p11': (h + d, -y),
    'p12': (h + d, D - y),

    # End of rod
    'p13': (h + d, R / 2),
    'p14': (h + d, -R / 2),
    'p15': (h + d + r, -R / 2),
    'p16': (h + d + r, R / 2),

    # Geometric compatibility points
    'p17': (0.0, 0.0),
    'p18': (h, 0.0),
    'p19': (h + d, 0.0),
    'p20': (h + d + r, 0.0),
    'p21': (enc, -H / 2),
    'p22': (enc + e, -H / 2),

    # Ground
    'p25': (h + d + r, -y),
    'p26': (0, -y),

    # Opposite clamping point
    'p27': (-67.5227 + enc, 0.0),

    # Median plane
    'p28': (enc, 0.0),
    'p29': (e + enc, 0.0),

    # Hinge spring
    'p30': (0.0, -H / 2),

    # New spring blade in Invar (side by side with the Be-Cu blade)
    'p31': (enc - e - 0.1, H / 2),  # Bottom left point Invar blade
    'p32': (enc - 0.1, H / 2),  # Bottom right point Invar blade
    'p33': (enc - 0.1, L - e - 0.1),  # Top right point Invar blade
    'p34': (enc - e - 0.1, L - e - 0.1),  # Top left point Invar blade

    # Compatibility planes for Invar blade
    'p35': (enc - e - 0.1, -H / 2),  # Left median plane point Invar blade
    'p36': (enc - 0.1, -H / 2),  # Right median plane point Invar blade

    # Median plane
    'p37': (enc - e - 0.1, 0.0),
    'p38': (enc - 0.1, 0.0),
}

def create_visualization():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

    # Overview (ax1)
    ax1.set_title('Overview of the muVINS system\n(Inertial sensor for gravitational waves)',
                  fontsize=14, fontweight='bold')

    # Draw the spring blade
    blade_points = np.array([points['p1'], points['p2'], points['p3'], points['p4'], points['p1']])
    blade = Polygon(blade_points[:-1], facecolor='lightblue', edgecolor='blue', linewidth=2, alpha=0.7)
    ax1.add_patch(blade)
    ax1.text(enc + e / 2, L / 2, 'Spring Blade\n(Be-Cu)', ha='center', va='center', fontsize=10, fontweight='bold')

    # Draw the Invar spring blade
    invar_blade_points = np.array([points['p31'], points['p32'], points['p33'], points['p34'], points['p31']])
    invar_blade = Polygon(invar_blade_points[:-1], facecolor='lightgreen', edgecolor='green', linewidth=2, alpha=0.7)
    ax1.add_patch(invar_blade)
    ax1.text(enc - e / 2 - 0.1, L / 2, 'Spring Blade\n(Invar)', ha='center', va='center', fontsize=10, fontweight='bold')

    # Draw the main rod
    rod_points = np.array([points['p5'], points['p6'], points['p21'], points['p22'],
                           points['p7'], points['p8'], points['p5']])
    rod = Polygon(rod_points[:-1], facecolor='lightgray', edgecolor='black', linewidth=2, alpha=0.7)
    ax1.add_patch(rod)

    # Rod between mass and end
    rod2_points = np.array([points['p8'], points['p7'], points['p10'], points['p9'], points['p8']])
    rod2 = Polygon(rod2_points[:-1], facecolor='lightgray', edgecolor='black', linewidth=2, alpha=0.7)
    ax1.add_patch(rod2)

    # Draw the inertial mass
    mass_points = np.array([points['p9'], points['p10'], points['p11'], points['p12'], points['p9']])
    mass = Polygon(mass_points[:-1], facecolor='darkgray', edgecolor='black', linewidth=2, alpha=0.8)
    ax1.add_patch(mass)
    ax1.text(h + d / 2, 0, 'Inertial\nMass\n(Steel)', ha='center', va='center',
             fontsize=11, fontweight='bold', color='white')

    # End of rod
    end_rod_points = np.array([points['p13'], points['p14'], points['p15'], points['p16'], points['p13']])
    end_rod = Polygon(end_rod_points[:-1], facecolor='lightgray', edgecolor='black', linewidth=2, alpha=0.7)
    ax1.add_patch(end_rod)

    # Hinge
    hinge = plt.Circle(points['p17'], 2, facecolor='red', edgecolor='darkred', linewidth=2)
    ax1.add_patch(hinge)
    ax1.text(points['p17'][0] - 8, points['p17'][1], 'Hinge', ha='center', va='center', fontsize=10)

    # Hinge spring
    ax1.plot([points['p6'][0], points['p30'][0]], [points['p6'][1], points['p30'][1]],
             'r-', linewidth=3, label='Hinge Spring')

    # Clamping point
    clamp = plt.Circle(points['p28'], 1.5, facecolor='orange', edgecolor='darkorange', linewidth=2)
    ax1.add_patch(clamp)
    ax1.text(points['p28'][0], points['p28'][1] + 8, 'Clamping\nPoint', ha='center', va='center', fontsize=10)

    # Ground
    ground_x = np.linspace(-10, h + d + r + 10, 100)
    ground_y = np.full_like(ground_x, -y - 2)
    ax1.fill_between(ground_x, ground_y - 5, ground_y, color='brown', alpha=0.5, label='Ground')
    ax1.plot(ground_x, ground_y, 'k-', linewidth=2)

    # Important points
    ax1.plot(points['p20'][0], points['p20'][1], 'ro', markersize=8, label='Measurement Point')

    # Annotations with dimensions
    ax1.annotate('', xy=(0, H / 2 + 10), xytext=(h, H / 2 + 10),
                 arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax1.text(h / 2, H / 2 + 15, f'l = {l} mm', ha='center', va='bottom', fontsize=10, color='green')

    ax1.annotate('', xy=(enc + e + 5, 0), xytext=(enc + e + 5, L),
                 arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
    ax1.text(enc + e + 10, L / 2, f'L = {L} mm', ha='left', va='center', fontsize=10, color='blue')

    ax1.annotate('', xy=(h - 5, -y), xytext=(h - 5, D - y),
                 arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
    ax1.text(h - 10, 0, f'D = {D} mm', ha='right', va='center', fontsize=10, color='purple')

    ax1.set_xlim(-20, h + d + r + 20)
    ax1.set_ylim(-y - 10, L + 10)
    ax1.set_xlabel('X Position (mm)')
    ax1.set_ylabel('Y Position (mm)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    ax1.set_aspect('equal')

    # Detailed view of the hinge and clamping point (ax2)
    ax2.set_title('Detail of the hinge and clamping point', fontsize=14, fontweight='bold')

    # Zoom on the hinge area
    zoom_xlim = (-15, enc + e + 10)
    zoom_ylim = (-15, 25)

    # Redraw elements in the zoom area
    blade_zoom = Polygon(blade_points[:-1], facecolor='lightblue', edgecolor='blue', linewidth=2, alpha=0.7)
    ax2.add_patch(blade_zoom)

    invar_blade_zoom = Polygon(invar_blade_points[:-1], facecolor='lightgreen', edgecolor='green', linewidth=2, alpha=0.7)
    ax2.add_patch(invar_blade_zoom)

    rod_zoom = Polygon(rod_points[:-1], facecolor='lightgray', edgecolor='black', linewidth=2, alpha=0.7)
    ax2.add_patch(rod_zoom)

    hinge_zoom = plt.Circle(points['p17'], 2, facecolor='red', edgecolor='darkred', linewidth=2)
    ax2.add_patch(hinge_zoom)

    clamp_zoom = plt.Circle(points['p28'], 1.5, facecolor='orange', edgecolor='darkorange', linewidth=2)
    ax2.add_patch(clamp_zoom)

    # Hinge spring with more details
    spring_x = np.linspace(points['p6'][0], points['p30'][0], 20)
    spring_y = points['p6'][1] + 0.5 * np.sin(10 * np.pi * np.linspace(0, 1, 20))
    ax2.plot(spring_x, spring_y, 'r-', linewidth=2, label='Helical Spring')

    # Detailed annotations
    ax2.text(points['p17'][0], points['p17'][1] - 8, f'Hinge\n(0, 0)', ha='center', va='top', fontsize=10)
    ax2.text(points['p28'][0], points['p28'][1] + 8, f'Clamping\n({enc}, 0)', ha='center', va='bottom', fontsize=10)

    # Blade dimensions
    ax2.annotate('', xy=(enc, H / 2 + 5), xytext=(enc + e, H / 2 + 5),
                 arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
    ax2.text(enc + e / 2, H / 2 + 7, f'e = {e} mm', ha='center', va='bottom', fontsize=10, color='blue')

    ax2.annotate('', xy=(enc - 5, -H / 2), xytext=(enc - 5, H / 2),
                 arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax2.text(enc - 7, 0, f'H = {H} mm', ha='right', va='center', fontsize=10)

    ax2.set_xlim(zoom_xlim)
    ax2.set_ylim(zoom_ylim)
    ax2.set_xlabel('X Position (mm)')
    ax2.set_ylabel('Y Position (mm)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_aspect('equal')

    plt.tight_layout()
    return fig

# Function to display the coordinates of key points
def print_key_points():
    print("Coordinates of key points in the muVINS system:")
    print("=" * 50)
    key_points = ['p17', 'p28', 'p20', 'p3', 'p12', 'p31', 'p34']
    descriptions = ['Hinge (origin)', 'Clamping Point', 'Measurement Point',
                    'End of Spring Blade', 'Top Corner of Mass', 'End of Invar Blade (bottom)', 'End of Invar Blade (top)']

    for point, desc in zip(key_points, descriptions):
        x, y = points[point]
        print(f"{point} - {desc}: ({x:.2f}, {y:.2f}) mm")

    print("\nGeometric parameters:")
    print(f"- Spring blade length: {L} mm")
    print(f"- Spring blade thickness: {e} mm")
    print(f"- Total rod length: {l} mm")
    print(f"- Clamping point position: {enc} mm")
    print(f"- Mass dimensions: {d} × {D} mm")

# Execution
if __name__ == "__main__":
    print_key_points()
    fig = create_visualization()
    plt.show()

# To save the figure
# fig.savefig('muvins_system.png', dpi=300, bbox_inches='tight')
