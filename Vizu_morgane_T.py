import matplotlib.pyplot as plt
import numpy as np

# Paramètres géométriques du système
# Lame ressort
e = 0.24  # épaisseur
L = 107.0  # longueur

# Tige
l = 89.6753  # longueur totale
H = 4.5  # épaisseur

# Trou
r = 28.8871
R = 23.332
a = 1.574

# Masse
D = 31.1092  # hauteur
d = 39.3506  # longueur

h = l - d
y_enc = -14.3
x_enc = 39.88

# Hinge blade
l_hinge = 0.5
t_hinge = 0.05
w_hinge = 8*2  #2 blades

Dx = -72.04  # Param to set the reference horizontal location of the clamping pt (dist btw the 2 clamping pt)
Dx1 = 0.0  # set horizontal shift (>0: shift closer to hinge)
Dy = -3.0  # set vertical shift (>0: shift upward)
angleClamp = 0.0  # set angle (>0: anti-clockwise)

# Points de la géométrie
points = {
    # Lame ressort
    'p1': (x_enc - e, y_enc),
    'p2': (x_enc, y_enc),
    'p3': (x_enc, y_enc + L),
    'p4': (x_enc - e, y_enc + L),
    # Tige
    'p5': (0.0, H / 2),
    'p6': (0.0, -H / 2),
    'p7': (h, -H / 2),
    'p8': (h, H / 2),
    # Masse pièce gauche
    'p9': (h, D / 2),
    'p10': (h, -D / 2),
    'p11': (h + a, -D / 2),
    'p12': (h + a, D / 2),
    # Masse pièce inférieure
    'p13': (h + d, -D / 2),
    'p14': (h + d, -R / 2),
    'p15': (h + a, -R / 2),
    # Masse pièce supérieure
    'p16': (h + a, R / 2),
    'p17': (h + d, R / 2),
    'p18': (h + d, D / 2),
    # Masse pièce droite
    'p19': (h + a + r, R / 2),
    'p20': (h + a + r, -R / 2),
    # Compatibilité géométrique
    'pO': (0.0, 0.0),
    'p22': (h, 0.0),
    'p23': (h + a, 0.0),
    'p28': (h, R / 2),
    'p29': (h, -R / 2),
    'p30': (h + a, -H / 2),
    'p31': (h + a, H / 2),
    'p32': (h + a + r, -D / 2),
    'p33': (h + a + r, D / 2),
    # Serrage
    'p24': (x_enc - e, -H / 2),
    'p25': (x_enc, -H / 2),
    'p26': (x_enc, H / 2),
    'p27': (x_enc - e, H / 2),
    # Sol
    'pGR': (h + d + 1, -D / 2),
    'pGL': (0, -D / 2),
    # Référence
    'pref': (Dx + x_enc, y_enc + Dy),
    # Lame de charnière
    'p39': (-l_hinge, t_hinge / 2),
    'p40': (-l_hinge, -t_hinge / 2),
    'p41': (0, -t_hinge / 2),
    'p42': (0, t_hinge / 2),
    'p43': (h, -t_hinge / 2),
    'p44': (h, t_hinge / 2),
    'p45': (h + a, -t_hinge / 2),
    'p46': (h + a, t_hinge / 2),
    'p47': (-l_hinge, 0)
}

# Courbes
curves = {
    # Lame ressort
    'c1': ['p1', 'p2'],
    'c2': ['p2', 'p3'],
    'c3': ['p3', 'p4'],
    'c4': ['p4', 'p1'],
    # Tige
    'c5': ['p5', 'p42'],
    'c6': ['p41', 'p6'],
    'c7': ['p6', 'p24'],
    'c8': ['p24', 'p25'],
    'c9': ['p25', 'p7'],
    'c10': ['p7', 'p43'],
    'c11': ['p44', 'p8'],
    'c12': ['p8', 'p26'],
    'c13': ['p26', 'p27'],
    'c14': ['p27', 'p5'],
    # Masse pièce gauche
    'c15': ['p9', 'p28'],
    'c16': ['p28', 'p8'],
    'c17': ['p7', 'p29'],
    'c18': ['p29', 'p10'],
    'c19': ['p10', 'p11'],
    'c20': ['p11', 'p15'],
    'c21': ['p15', 'p30'],
    'c22': ['p30', 'p45'],
    'c23': ['p46', 'p31'],
    'c24': ['p31', 'p16'],
    'c25': ['p16', 'p12'],
    'c26': ['p12', 'p9'],
    # Masse pièce inférieure
    'c27': ['p11', 'p32'],
    'c28': ['p32', 'p13'],
    'c29': ['p13', 'p14'],
    'c30': ['p14', 'p20'],
    'c31': ['p20', 'p15'],
    # Masse pièce supérieure
    'c32': ['p16', 'p19'],
    'c33': ['p19', 'p17'],
    'c34': ['p17', 'p18'],
    'c35': ['p18', 'p33'],
    'c36': ['p33', 'p12'],
    # Masse pièce droite
    'c37': ['p19', 'p20'],
    'c38': ['p14', 'p17'],
    # Serrage
    'c39': ['p24', 'p1'],
    'c40': ['p2', 'p25'],
    # Sol
    'c41': ['pGR', 'pGL'],
    # Charnière et compatibilité de charnière
    'c42': ['p42', 'pO'],
    'c43': ['pO', 'p41'],
    'c44': ['p43', 'p22'],
    'c45': ['p22', 'p44'],
    'c46': ['p45', 'p23'],
    'c47': ['p23', 'p46'],
    'c48': ['p42', 'p39'],
    'c49': ['p39', 'p47'],
    'c50': ['p47', 'p40'],
    'c51': ['p40', 'p41']
}

# Wires
wires = {
    'w1': ['c1', 'c2', 'c3', 'c4'],  # lame ressort
    'w2': ['c5', 'c42', 'c43', 'c6', 'c7', 'c8', 'c9', 'c10', 'c44', 'c45', 'c11', 'c12', 'c13', 'c14'],  # tige
    'w3': ['c15', 'c16', 'c11', 'c45', 'c44', 'c10', 'c17', 'c18', 'c19', 'c20', 'c21', 'c22', 'c46', 'c47', 'c23', 'c24', 'c25', 'c26'],  # masse pièce gauche
    'w4': ['c20', 'c27', 'c28', 'c29', 'c30', 'c31'],  # masse pièce inférieure
    'w5': ['c25', 'c32', 'c33', 'c34', 'c35', 'c36'],  # masse pièce supérieure
    'w6': ['c37', 'c30', 'c38', 'c33'],  # masse pièce droite
    'w7': ['c1', 'c40', 'c8', 'c39'],  # serrage
    'w8': ['c48', 'c49', 'c50', 'c51', 'c43', 'c42'],  # charnière
    'w9': ['c41']  # sol
}

def plot_points():
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    for key, (x, y) in points.items():
        ax.plot(x, y, 'ro', markersize=8)
        ax.text(x, y, key, fontsize=10, ha='right', va='bottom')
    ax.set_title('Points Numérotés')
    ax.set_xlim(-20, h + d + r + 20)
    ax.set_ylim(-D/2 - 10, L + 10)  # Ajustement ici
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
    ax.set_ylim(-D/2 - 10, L + 10)  # Ajustement ici
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
    ax.set_ylim(-D/2 - 10, L + 10)  # Ajustement ici
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
    ax.set_ylim(-D/2 - 10, L + 10)  # Ajustement ici
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


