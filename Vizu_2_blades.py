import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

# Paramètres géométriques du système (extraits du code Metafor)
# Lame ressort
e = 0.24  # épaisseur
L = 105.25  # longueur

enc = 57.32  # Position du point de serrage sur la tige

# Invar blade
# Invar blade
ei = 0.07  # invar thickness
# Position relative des lames (décalage horizontal de 0.1mm)
decalage = 0.1  # décalage entre les lames
rayon_interne_BeCu = enc / 2
R_beCu = rayon_interne_BeCu + e / 2  # Rayon médian de la lame Be-Cu
R_invar = rayon_interne_BeCu - decalage - ei / 2  # Rayon médian de la lame Invar
ratio = R_invar / R_beCu
Li = L * ratio  # La longueur doit être proportionnelle au rayon pour un même angle de pliage

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

# Points de la géométrie (d'après le code Metafor)
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

    # Points de compatibilité géométrique
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
    'p27': (-67.5227 + enc, 0.0),

    # Plan médian
    'p28': (enc, 0.0),
    'p29': (e + enc, 0.0),

    # Ressort de charnière
    'p30': (0.0, -H / 2),

    # Nouvelle lame ressort en Invar (côte à côte avec la lame Be-Cu)
    'p31': (enc - ei - decalage, H / 2),      # Point bas gauche lame Invar
    'p32': (enc - decalage, H / 2),  # Point bas droit lame Invar
    'p33': (enc - decalage, Li),  # Point haut droit lame Invar
    'p34': (enc - ei - decalage, Li),  # Point haut gauche lame Invar

    # Plans de compatibilité pour lame Invar
    'p35': (enc - e - decalage, -H / 2),  # Point plan médian gauche lame Invar
    'p36': (enc - decalage, -H / 2),  # Point plan médian droit lame Invar

    # Plan médian
    'p37': (enc - e - decalage, 0.0),
    'p38': (enc - decalage, 0.0)
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
    'c71': ['p6', 'p35'],
    'c72': ['p35', 'p36'],
    'c73': ['p36', 'p21'],
    'c8': ['p21', 'p22'],
    'c9': ['p22', 'p7'],
    'c10': ['p7', 'p18'],
    'c11': ['p18', 'p8'],
    'c12': ['p8', 'p2'],
    'c131': ['p1', 'p32'],
    'c132': ['p31', 'p5'],

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
    'c271': ['p17', 'p37'],
    'c272': ['p38', 'p28'],
    'c28': ['p28', 'p29'],
    'c29': ['p29', 'p18'],

    # Lame ressort Invar
    'c30': ['p31', 'p32'],
    'c31': ['p38', 'p33'],
    'c32': ['p33', 'p34'],
    'c33': ['p34', 'p37'],

    # Plan médian lame Invar
    'c34': ['p37', 'p38']
}

# Wires
wires = {
    'w1': ['c28', 'c2', 'c3', 'c4'],
    'w2': ['c5', 'c271', 'c34', 'c272', 'c28', 'c29', 'c11', 'c12', 'c1', 'c131', 'c132'],
    'w3': ['c6', 'c71', 'c72', 'c73', 'c8', 'c9', 'c10', 'c29', 'c28', 'c272', 'c34', 'c271'],
    'w4': ['c14', 'c11', 'c10', 'c15', 'c16', 'c17', 'c18', 'c19', 'c20', 'c21'],
    'w5': ['c19', 'c18', 'c22', 'c23', 'c24', 'c25'],
    'w6': ['c26'],
    'w7': ['c34', 'c31', 'c32', 'c33']
}

def plot_points():
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    for key, (x, y) in points.items():
        ax.plot(x, y, 'ro', markersize=8)
        ax.text(x, y, key, fontsize=10, ha='right', va='bottom')
    ax.set_title('Points Numérotés')
    ax.set_xlim(-20, h + d + r + 20)
    ax.set_ylim(-y - 10, L + 10)
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
        centroid_x = np.mean([points[curves[curve][0]][0] for curve in wire] + [points[curves[curve][1]][0] for curve in wire])
        centroid_y = np.mean([points[curves[curve][0]][1] for curve in wire] + [points[curves[curve][1]][1] for curve in wire])
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
        # Centroid of the wire for text placement
        centroid_x = np.mean([points[curves[curve][0]][0] for curve in wire] + [points[curves[curve][1]][0] for curve in wire])
        centroid_y = np.mean([points[curves[curve][0]][1] for curve in wire] + [points[curves[curve][1]][1] for curve in wire])
        ax.text(centroid_x, centroid_y, key, fontsize=10, ha='center', va='bottom', color='green')
    ax.set_title('Points, Courbes et Wires Numérotés')
    ax.set_xlim(-20, h + d + r + 20)
    ax.set_ylim(-y - 10, L + 10)
    ax.set_xlabel('Position X (mm)')
    ax.set_ylabel('Position Y (mm)')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    return fig

# Exécution
if __name__ == "__main__":
    fig_points = plot_points()
    fig_curves = plot_curves()
    fig_wires = plot_wires()
    fig_all = plot_all()

    plt.show()
