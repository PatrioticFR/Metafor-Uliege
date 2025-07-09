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

# Points de la géométrie (d’après le code Metafor)
points = {
    # Lame ressort
     'p1': (enc +L , e / 2),
     'p2': (enc + L, -e/ 2),


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
    'p171': (0.0, e/2),
    'p172': (0.0, -e/2),
    'p181': (h, e/2),
    'p182': (h, -e/2),
    'p191': (h + d, e/2),
    'p192': (h + d, -e/2),
    'p201': (h + d + r, e/2),
    'p202': (h + d + r, -e/2),

    #Plan médiant
    'p21': (enc, H/2),
    'p22': (enc ,-H/2),

    # Sol
    'p25': (h + d + r, -y),
    'p26': (0, -y),

    # Point de serrage opposé
    'p27': (Dx + enc, 0.0),

    #Base lame
    'p28': (enc, e/2),
    'p29': (enc, -e/2),

    # Ressort de charnière
    'p30': (0.0, -H / 2),
}


# Courbes
curves = {
    # Lame ressort
    'c1': ['p1', 'p2'], #extremité droite lame
    'c2': ['p29', 'p2'], #partie inférieur lame
    'c3': ['p28', 'p1'], #partie supérieur lame
    'c4': ['p28', 'p29'], #extremité gauche lame

    # Tige
    'c51': ['p5', 'p171'], #extremité gauche supérieur
    'c52': ['p171', 'p172'], #extremité gauche milieu
    'c6': ['p172', 'p6'], #extremité gauche inferieur
    'c71': ['p6', 'p22'], #partie inférieur gauche
    'c72': ['p22', 'p7'], #partie inférieur droite
    'c101': ['p7', 'p182'], #extremité droite inférieru
    'c102': ['p182', 'p181'], #extremité droite milieu
    'c11': ['p181', 'p8'], #extreminité droite supérieur
    'c121': ['p8', 'p21'], #partie supérieur droite
    'c122': ['p21', 'p5'], #partie supérieur gauche

    # Masse
    'c14': ['p9', 'p8'], #extremité gauche supérieur  ( entre haut de la tige et haut gauche de la masse)
    'c15': ['p7', 'p10'], #extremité gauche inférieur  ( entre bas de la tige et bas gauche de la masse)
    'c16': ['p10', 'p11'], #partie inférieur
    'c17': ['p11', 'p14'], #extremité drote inférieur (entre bas droit de la masse et bas de la tige)
    'c181': ['p14', 'p192'], # extremité droite milieu bas (entre bas tige et point milieu inférieur tige)
    'c182': ['p192', 'p191'], # extremité doite milieu (entre les deux points milieux)
    'c19': ['p191', 'p13'], #extremité droite milieu haut (entre milieu supérieur tige et haut tige)
    'c20': ['p13', 'p12'], #extremité droite supérieur (entre haut tige et haut droit de la masse)
    'c21': ['p12', 'p9'], #partie supérieur

    # Bout de tige
    'c22': ['p14', 'p15'], #partie inférieur
    'c231': ['p15', 'p202'], #extremitée droite milieu bas (entre bas tige et point milieu inférieur tige)
    'c232': ['p202', 'p201'], # extremité doite milieu (entre les deux points milieux)
    'c24': ['p201', 'p16'], #extremité droite milieu haut (entre milieu supérieur tige et haut tige)
    'c25': ['p16', 'p13'], #partie supérieur

    # Sol
    'c26': ['p25', 'p26'],

    # Plan médian sup (tige gauche)
    #'c27': ['p171', 'p28'],
    'c28': ['p28', 'p181'],

    # Plan médian inf (tige froite)
    #'c29': ['p172', 'p29'],
    'c30': ['p29', 'p182'],

    #Plan coupe (vertical)
    'c31': ['p21', 'p28'],
    'c32': ['p29', 'p22'],
}


# Wires
wires = {
    # Lame ressort
    'w1': ['c1', 'c2', 'c4', 'c3'],

    # Tige partie gauche
    'w2': ['c51','c52','c6','c71','c31', 'c4', 'c32', 'c122'],

    # Tige partie droite
    'w3': ['c31', 'c4', 'c32', 'c121', 'c11', 'c102','c101','c72'],

    # Masse
    'w4': ['c14', 'c11', 'c102', 'c101', 'c15', 'c16', 'c17', 'c181', 'c182', 'c19', 'c20', 'c21'],

    # Bout de tige
    'w5': ['c19', 'c182', 'c181', 'c22', 'c231', 'c232', 'c24', 'c25'],

    # Sol
    'w6': ['c26']
}


# Fonctions d'affichage (inchangées)
def plot_points():
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    for key, (x, y) in points.items():
        ax.plot(x, y, 'ro', markersize=8)
        ax.text(x, y, key, fontsize=10, ha='right', va='bottom')
    ax.set_title('Points Numérotés')
    ax.set_xlim(-20, enc + L +10)
    #ax.set_ylim(-y - 10, y + 10)
    ax.set_ylim(-30,+30)
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
    ax.set_xlim(-20, enc + L +10)
    ax.set_ylim(-y - 10, y + 10)
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
    ax.set_xlim(-20, enc + L +10)
    ax.set_ylim(-y - 10, y + 10)
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
    ax.set_xlim(-20, enc + L +10)
    #ax.set_ylim(-y - 10, y + 10)
    ax.set_ylim(-30,+30)
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
