import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

# Paramètres géométriques du système (extraits du code Metafor)
# Lame ressort
e = 0.24  # épaisseur
L = 105.25  # longueur
enc = 57.32  # Position du point de serrage sur la tige
# Tige
l = 79.2  # longueur totale
H = 3.875  # épaisseur
r = 7  # distance entre masse et bout de tige
R = H
# Masse
D = 39.99  # hauteur
d = 13.96  # longueur
y = D / 2  # décalage masse (centrée sur la tige)
h = l - r - d
Dx = -67.0042

# Paramètres de maillage
ne = 8       # lame ressort - éléments à travers l'épaisseur
nL = int(L*20)  # lame ressort - éléments le long de la longueur
nd = 10
nr = 1
n56 = 2    # tige verticale (/2)
n7 = 5     # tige horizontale
n9 = 1     # tige horiz 2
n14 = 3    # masse verticale 1
n15 = 17   # masse verticale 2
prog = 5   # progression pour certains éléments

# Points de la géométrie (d’après le code Metafor)
points = {
    # Lame ressort
    'p1': (enc, H / 2),
    'p2': (enc + e, H / 2),
    'p3': (enc + e, L),
    'p4': (enc, L),
    # Tige
    'p5': (0.0, H / 2),
    'p6': (0.0, -H / 2),
    'p7': (h, -H / 2),
    'p8': (h, H / 2),
    # Masse
    'p9': (h, D - y),
    'p10': (h, -y),
    'p11': (h + d, -y),
    'p12': (h + d, D - y),
    # Bout de tige
    'p13': (h + d, R / 2),
    'p14': (h + d, -R / 2),
    'p15': (h + d + r, -R / 2),
    'p16': (h + d + r, R / 2),
    # Compatibilité géométrique
    'p17': (0.0, 0.0),
    'p18': (h, 0.0),
    'p19': (h + d, 0.0),
    'p20': (h + d + r, 0.0),
    'p21': (enc, -H / 2),
    'p22': (enc + e, -H / 2),
    # Sol
    'p25': (h + d + r, -y),
    'p26': (0, -y),
    # Point de serrage opposé
    'p27': (Dx + enc, 0.0),
    # Plan médian
    'p28': (enc, 0.0),
    'p29': (e + enc, 0.0),
    # Ressort de charnière
    'p30': (0.0, -H / 2),
}

# Courbes
curves = {
    # Lame ressort
    'c1': ['p1', 'p2'],
    'c2': ['p29', 'p3'],
    'c3': ['p3', 'p4'],
    'c4': ['p4', 'p28'],
    # Tige
    'c5': ['p5', 'p17'],
    'c6': ['p17', 'p6'],
    'c7': ['p6', 'p21'],
    'c8': ['p21', 'p22'],
    'c9': ['p22', 'p7'],
    'c10': ['p7', 'p18'],
    'c11': ['p18', 'p8'],
    'c12': ['p8', 'p2'],
    'c13': ['p1', 'p5'],
    # Masse
    'c14': ['p9', 'p8'],
    'c15': ['p7', 'p10'],
    'c16': ['p10', 'p11'],
    'c17': ['p11', 'p14'],
    'c18': ['p14', 'p19'],
    'c19': ['p19', 'p13'],
    'c20': ['p13', 'p12'],
    'c21': ['p12', 'p9'],
    # Bout de tige
    'c22': ['p14', 'p15'],
    'c23': ['p15', 'p20'],
    'c24': ['p20', 'p16'],
    'c25': ['p16', 'p13'],
    # Sol
    'c26': ['p25', 'p26'],
    # Plan médian
    'c27': ['p17', 'p28'],
    'c28': ['p28', 'p29'],
    'c29': ['p29', 'p18'],
}

# Wires
wires = {
    'w1': ['c28', 'c2', 'c3', 'c4'],
    'w2': ['c5', 'c27', 'c28', 'c29', 'c11', 'c12', 'c1', 'c13'],
    'w3': ['c6', 'c7', 'c8', 'c9', 'c10', 'c29', 'c28', 'c27'],
    'w4': ['c14', 'c11', 'c10', 'c15', 'c16', 'c17', 'c18', 'c19', 'c20', 'c21'],
    'w5': ['c19', 'c18', 'c22', 'c23', 'c24', 'c25'],
    'w6': ['c26']
}

def mesh_curve(curve_key, num_elements):
    curve = curves[curve_key]
    start_point = points[curve[0]]
    end_point = points[curve[1]]

    # Calculer les points intermédiaires
    x_start, y_start = start_point
    x_end, y_end = end_point

    x_values = np.linspace(x_start, x_end, num_elements + 1)
    y_values = np.linspace(y_start, y_end, num_elements + 1)

    # Retourner une liste de segments (chaque segment est une paire de points)
    segments = []
    for i in range(len(x_values) - 1):
        segments.append(((x_values[i], y_values[i]), (x_values[i+1], y_values[i+1])))
    return segments

def generate_mesh():
    mesh_segments = []

    # Blade
    mesh_segments.extend(mesh_curve('c1', ne))
    mesh_segments.extend(mesh_curve('c2', nL))
    mesh_segments.extend(mesh_curve('c3', ne))
    mesh_segments.extend(mesh_curve('c4', nL))

    # Rod
    mesh_segments.extend(mesh_curve('c5', n56))
    mesh_segments.extend(mesh_curve('c6', n56))
    mesh_segments.extend(mesh_curve('c7', n7))
    mesh_segments.extend(mesh_curve('c8', ne))
    mesh_segments.extend(mesh_curve('c9', n9))
    mesh_segments.extend(mesh_curve('c10', n56))
    mesh_segments.extend(mesh_curve('c11', n56))
    mesh_segments.extend(mesh_curve('c12', n9))
    mesh_segments.extend(mesh_curve('c13', n7))

    # Mass
    mesh_segments.extend(mesh_curve('c14', n14))
    mesh_segments.extend(mesh_curve('c15', n15))
    mesh_segments.extend(mesh_curve('c16', nd))
    mesh_segments.extend(mesh_curve('c17', n15))
    mesh_segments.extend(mesh_curve('c18', n56))
    mesh_segments.extend(mesh_curve('c19', n56))
    mesh_segments.extend(mesh_curve('c20', n14))
    mesh_segments.extend(mesh_curve('c21', nd))

    # End rod
    mesh_segments.extend(mesh_curve('c22', nr))
    mesh_segments.extend(mesh_curve('c23', n56))
    mesh_segments.extend(mesh_curve('c24', n56))
    mesh_segments.extend(mesh_curve('c25', nr))

    # Middle plane
    mesh_segments.extend(mesh_curve('c27', n7))
    mesh_segments.extend(mesh_curve('c28', ne))
    mesh_segments.extend(mesh_curve('c29', n9))

    return mesh_segments

def plot_points():
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    for key, (x, y) in points.items():
        ax.plot(x, y, 'ro', markersize=8)
        ax.text(x, y, key, fontsize=10, ha='right', va='bottom')
    ax.set_title('Points Numérotés')
    ax.set_xlim(-20, h + d + r + 20)
    ax.set_ylim(-30, L + 10)
    ax.set_xlabel('Position X (mm)')
    ax.set_ylabel('Position Y (mm)')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    return fig

def plot_curves():
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    for key, curve in curves.items():
        start_point = points[curve[0]]
        end_point = points[curve[1]]
        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'b-', linewidth=2)
        ax.text(np.mean([start_point[0], end_point[0]]), np.mean([start_point[1], end_point[1]]), key, fontsize=10, ha='center', va='bottom', color='blue')
    ax.set_title('Courbes Numérotées')
    ax.set_xlim(-20, h + d + r + 20)
    ax.set_ylim(-y - 10, L + 10)
    ax.set_xlabel('Position X (mm)')
    ax.set_ylabel('Position Y (mm)')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    return fig

def plot_wires():
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    for key, wire in wires.items():
        for curve in wire:
            start_point = points[curves[curve][0]]
            end_point = points[curves[curve][1]]
            ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'g-', linewidth=2)
        # Centroid of the wire for text placement
        centroid_x = np.mean(
            [points[curves[curve][0]][0] for curve in wire] +
            [points[curves[curve][1]][0] for curve in wire]
        )
        centroid_y = np.mean(
            [points[curves[curve][0]][1] for curve in wire] +
            [points[curves[curve][1]][1] for curve in wire]
        )
        ax.text(centroid_x, centroid_y, key, fontsize=10, ha='center', va='bottom', color='green')
    ax.set_title('Wires Numérotés')
    ax.set_xlim(-20, h + d + r + 20)
    ax.set_ylim(-y - 10, L + 10)
    ax.set_xlabel('Position X (mm)')
    ax.set_ylabel('Position Y (mm)')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    return fig

def plot_all():
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    for key, (x, y) in points.items():
        ax.plot(x, y, 'ro', markersize=8)
        ax.text(x, y, key, fontsize=10, ha='right', va='bottom')
    for key, curve in curves.items():
        start_point = points[curve[0]]
        end_point = points[curve[1]]
        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'b-', linewidth=2)
        ax.text(np.mean([start_point[0], end_point[0]]), np.mean([start_point[1], end_point[1]]), key, fontsize=10, ha='center', va='bottom', color='blue')
    for key, wire in wires.items():
        for curve in wire:
            start_point = points[curves[curve][0]]
            end_point = points[curves[curve][1]]
            ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'g-', linewidth=2)
        centroid_x = np.mean(
            [points[curves[curve][0]][0] for curve in wire] +
            [points[curves[curve][1]][0] for curve in wire]
        )
        centroid_y = np.mean(
            [points[curves[curve][0]][1] for curve in wire] +
            [points[curves[curve][1]][1] for curve in wire]
        )
        ax.text(centroid_x, centroid_y, key, fontsize=10, ha='center', va='bottom', color='green')
    ax.set_title('Points, Courbes et Wires Numérotés')
    ax.set_xlim(-20, h + d + r + 20)
    ax.set_ylim(-30, L + 10)
    ax.set_xlabel('Position X (mm)')
    ax.set_ylabel('Position Y (mm)')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    return fig

def plot_all_with_mesh():
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)

    # Tracer les points
    for key, (x, y) in points.items():
        ax.plot(x, y, 'ro', markersize=8)
        ax.text(x, y, key, fontsize=10, ha='right', va='bottom')

    # Tracer les courbes
    for key, curve in curves.items():
        start_point = points[curve[0]]
        end_point = points[curve[1]]
        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'b-', linewidth=2)
        ax.text(np.mean([start_point[0], end_point[0]]), np.mean([start_point[1], end_point[1]]), key, fontsize=10, ha='center', va='bottom', color='blue')

    # Tracer les wires
    for key, wire in wires.items():
        for curve in wire:
            start_point = points[curves[curve][0]]
            end_point = points[curves[curve][1]]
            ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'g-', linewidth=2)
        centroid_x = np.mean(
            [points[curves[curve][0]][0] for curve in wire] +
            [points[curves[curve][1]][0] for curve in wire]
        )
        centroid_y = np.mean(
            [points[curves[curve][0]][1] for curve in wire] +
            [points[curves[curve][1]][1] for curve in wire]
        )
        ax.text(centroid_x, centroid_y, key, fontsize=10, ha='center', va='bottom', color='green')

    # Générer et tracer le maillage
    mesh_segments = generate_mesh()
    for segment in mesh_segments:
        start_point, end_point = segment
        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'k-', linewidth=1, alpha=0.5)

    ax.set_title('Points, Courbes, Wires et Maillage')
    ax.set_xlim(-20, h + d + r + 20)
    ax.set_ylim(-30, L + 10)
    ax.set_xlabel('Position X (mm)')
    ax.set_ylabel('Position Y (mm)')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    fig_points = plot_points()
    fig_curves = plot_curves()
    fig_wires = plot_wires()
    fig_all = plot_all()
    fig_mesh = plot_all_with_mesh()
    plt.show()
