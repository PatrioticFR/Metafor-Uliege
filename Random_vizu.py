import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
import matplotlib.patches as patches

# Paramètres géométriques du système (extraits du code Metafor)
# Lame ressort
e = 0.24  # épaisseur
L = 105.25  # longueur

# Tige
l = 79.2  # longueur totale
H = 3.875  # épaisseur
r = 7  # distance entre masse et bout de tige
R = H
enc = 57.32  # Position du point de serrage sur la tige

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
    'p31': (enc - e - 0.1, H / 2),  # Point bas gauche lame Invar
    'p32': (enc - 0.1, H / 2),  # Point bas droit lame Invar
    'p33': (enc - 0.1, L - e - 0.1),  # Point haut droit lame Invar
    'p34': (enc - e - 0.1, L - e - 0.1),  # Point haut gauche lame Invar

    # Plans de compatibilité pour lame Invar
    'p35': (enc - e - 0.1, -H / 2),  # Point plan médian gauche lame Invar
    'p36': (enc - 0.1, -H / 2),  # Point plan médian droit lame Invar

    # Plan médian
    'p37': (enc - e - 0.1, 0.0),
    'p38': (enc - 0.1, 0.0),
}

def create_visualization():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

    # Vue d'ensemble (ax1)
    ax1.set_title('Vue d\'ensemble du système muVINS\n(Capteur inertiel pour ondes gravitationnelles)',
                  fontsize=14, fontweight='bold')

    # Dessiner la lame ressort (leaf spring)
    blade_points = np.array([points['p1'], points['p2'], points['p3'], points['p4'], points['p1']])
    blade = Polygon(blade_points[:-1], facecolor='lightblue', edgecolor='blue', linewidth=2, alpha=0.7)
    ax1.add_patch(blade)
    ax1.text(enc + e / 2, L / 2, 'Lame ressort\n(Be-Cu)', ha='center', va='center', fontsize=10, fontweight='bold')

    # Dessiner la lame ressort en Invar
    invar_blade_points = np.array([points['p31'], points['p32'], points['p33'], points['p34'], points['p31']])
    invar_blade = Polygon(invar_blade_points[:-1], facecolor='lightgreen', edgecolor='green', linewidth=2, alpha=0.7)
    ax1.add_patch(invar_blade)
    ax1.text(enc - e / 2 - 0.1, L / 2, 'Lame ressort\n(Invar)', ha='center', va='center', fontsize=10, fontweight='bold')

    # Dessiner la tige principale
    rod_points = np.array([points['p5'], points['p6'], points['p21'], points['p22'],
                           points['p7'], points['p8'], points['p5']])
    rod = Polygon(rod_points[:-1], facecolor='lightgray', edgecolor='black', linewidth=2, alpha=0.7)
    ax1.add_patch(rod)

    # Tige entre masse et bout
    rod2_points = np.array([points['p8'], points['p7'], points['p10'], points['p9'], points['p8']])
    rod2 = Polygon(rod2_points[:-1], facecolor='lightgray', edgecolor='black', linewidth=2, alpha=0.7)
    ax1.add_patch(rod2)

    # Dessiner la masse inertielle
    mass_points = np.array([points['p9'], points['p10'], points['p11'], points['p12'], points['p9']])
    mass = Polygon(mass_points[:-1], facecolor='darkgray', edgecolor='black', linewidth=2, alpha=0.8)
    ax1.add_patch(mass)
    ax1.text(h + d / 2, 0, 'Masse\ninertielle\n(Acier)', ha='center', va='center',
             fontsize=11, fontweight='bold', color='white')

    # Bout de tige
    end_rod_points = np.array([points['p13'], points['p14'], points['p15'], points['p16'], points['p13']])
    end_rod = Polygon(end_rod_points[:-1], facecolor='lightgray', edgecolor='black', linewidth=2, alpha=0.7)
    ax1.add_patch(end_rod)

    # Charnière (hinge)
    hinge = plt.Circle(points['p17'], 2, facecolor='red', edgecolor='darkred', linewidth=2)
    ax1.add_patch(hinge)
    ax1.text(points['p17'][0] - 8, points['p17'][1], 'Charnière', ha='center', va='center', fontsize=10)

    # Ressort de charnière
    ax1.plot([points['p6'][0], points['p30'][0]], [points['p6'][1], points['p30'][1]],
             'r-', linewidth=3, label='Ressort de charnière')

    # Point de serrage
    clamp = plt.Circle(points['p28'], 1.5, facecolor='orange', edgecolor='darkorange', linewidth=2)
    ax1.add_patch(clamp)
    ax1.text(points['p28'][0], points['p28'][1] + 8, 'Point de\nserrage', ha='center', va='center', fontsize=10)

    # Sol
    ground_x = np.linspace(-10, h + d + r + 10, 100)
    ground_y = np.full_like(ground_x, -y - 2)
    ax1.fill_between(ground_x, ground_y - 5, ground_y, color='brown', alpha=0.5, label='Sol')
    ax1.plot(ground_x, ground_y, 'k-', linewidth=2)

    # Points importants
    ax1.plot(points['p20'][0], points['p20'][1], 'ro', markersize=8, label='Point de mesure')

    # Annotations avec dimensions
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
    ax1.set_xlabel('Position X (mm)')
    ax1.set_ylabel('Position Y (mm)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    ax1.set_aspect('equal')

    # Vue de détail de la charnière et du point de serrage (ax2)
    ax2.set_title('Détail de la charnière et du point de serrage', fontsize=14, fontweight='bold')

    # Zoom sur la zone de charnière
    zoom_xlim = (-15, enc + e + 10)
    zoom_ylim = (-15, 25)

    # Redessiner les éléments dans la zone de zoom
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

    # Ressort de charnière avec plus de détails
    spring_x = np.linspace(points['p6'][0], points['p30'][0], 20)
    spring_y = points['p6'][1] + 0.5 * np.sin(10 * np.pi * np.linspace(0, 1, 20))
    ax2.plot(spring_x, spring_y, 'r-', linewidth=2, label='Ressort hélicoïdal')

    # Annotations détaillées
    ax2.text(points['p17'][0], points['p17'][1] - 8, f'Charnière\n(0, 0)', ha='center', va='top', fontsize=10)
    ax2.text(points['p28'][0], points['p28'][1] + 8, f'Serrage\n({enc}, 0)', ha='center', va='bottom', fontsize=10)

    # Dimensions de la lame
    ax2.annotate('', xy=(enc, H / 2 + 5), xytext=(enc + e, H / 2 + 5),
                 arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
    ax2.text(enc + e / 2, H / 2 + 7, f'e = {e} mm', ha='center', va='bottom', fontsize=10, color='blue')

    ax2.annotate('', xy=(enc - 5, -H / 2), xytext=(enc - 5, H / 2),
                 arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax2.text(enc - 7, 0, f'H = {H} mm', ha='right', va='center', fontsize=10)

    ax2.set_xlim(zoom_xlim)
    ax2.set_ylim(zoom_ylim)
    ax2.set_xlabel('Position X (mm)')
    ax2.set_ylabel('Position Y (mm)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_aspect('equal')

    plt.tight_layout()
    return fig

# Fonction pour afficher les coordonnées des points importants
def print_key_points():
    print("Coordonnées des points clés du système muVINS:")
    print("=" * 50)
    key_points = ['p17', 'p28', 'p20', 'p3', 'p12', 'p31', 'p34']
    descriptions = ['Charnière (origine)', 'Point de serrage', 'Point de mesure',
                    'Extrémité lame ressort', 'Coin supérieur masse', 'Extrémité lame Invar (bas)', 'Extrémité lame Invar (haut)']

    for point, desc in zip(key_points, descriptions):
        x, y = points[point]
        print(f"{point} - {desc}: ({x:.2f}, {y:.2f}) mm")

    print("\nParamètres géométriques:")
    print(f"- Longueur lame ressort: {L} mm")
    print(f"- Épaisseur lame ressort: {e} mm")
    print(f"- Longueur tige totale: {l} mm")
    print(f"- Position point serrage: {enc} mm")
    print(f"- Dimensions masse: {d} × {D} mm")

# Exécution
if __name__ == "__main__":
    print_key_points()
    fig = create_visualization()
    plt.show()

# Pour sauvegarder la figure
# fig.savefig('muvins_system.png', dpi=300, bbox_inches='tight')
