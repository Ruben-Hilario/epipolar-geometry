import cv2
import numpy as np
import matplotlib.pyplot as plt

def normalize_points(points):
    """
    Normaliza un conjunto de puntos 2D para que su centro esté en el origen
    y la distancia media al origen sea sqrt(2).

    Parámetros:
      points: array de tamaño (N, 2)
      
    Retorna:
      normalized_points: puntos normalizados en coordenadas homogéneas (N, 3)
      T: matriz de transformación (3x3)
    """
    N = points.shape[0]
    centroid = np.mean(points, axis=0)
    shifted_points = points - centroid
    scale = np.sqrt(2) / np.mean(np.linalg.norm(shifted_points, axis=1))
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ])
    points_hom = np.hstack((points, np.ones((N, 1))))
    normalized_points = (T @ points_hom.T).T
    return normalized_points, T

def compute_fundamental_matrix(pts1, pts2):
    """
    Calcula la matriz fundamental F usando el algoritmo de 8 puntos
    con normalización, a partir de correspondencias entre dos imágenes.

    Parámetros:
      pts1: array de tamaño (N, 2) con puntos de la imagen 1.
      pts2: array de tamaño (N, 2) con puntos correspondientes de la imagen 2.
      
    Retorna:
      F: la matriz fundamental (3x3)
    """
    N = pts1.shape[0]
    pts1_norm, T1 = normalize_points(pts1)
    pts2_norm, T2 = normalize_points(pts2)
    
    A = np.zeros((N, 9))
    for i in range(N):
        x1, y1, _ = pts1_norm[i]
        x2, y2, _ = pts2_norm[i]
        A[i] = [x2 * x1, x2 * y1, x2,
                y2 * x1, y2 * y1, y2,
                x1,       y1,      1]
    
    # Resolver A f = 0 con SVD
    U, S, Vt = np.linalg.svd(A)
    f = Vt[-1]
    F_est = f.reshape(3, 3)
    
    # Forzar que F tenga rango 2 (propiedad teórica)
    U_f, S_f, Vt_f = np.linalg.svd(F_est)
    S_f[-1] = 0
    F_rank2 = U_f @ np.diag(S_f) @ Vt_f
    
    # Desnormalizar
    F = T2.T @ F_rank2 @ T1
    # (Opcional) Normalizar F para que F[2,2] sea 1 si es distinto de cero
    if np.abs(F[2, 2]) > 1e-6:
        F = F / F[2, 2]
    return F

def draw_epilines_and_points(img, lines, pts, line_thickness=3, circle_radius=8):
    """
    Dibuja en la imagen las líneas epipolares (definidas por a*x + b*y + c = 0)
    y los puntos correspondientes, con un grosor y tamaño mayor.

    Parámetros:
      img: imagen sobre la cual se dibuja.
      lines: líneas epipolares (array de tamaño (N,3)).
      pts: puntos correspondientes (array de tamaño (N,2)).
      line_thickness: grosor de las líneas.
      circle_radius: radio de los círculos que indican los puntos.
      
    Retorna:
      img_copy: imagen con líneas y puntos dibujados.
    """
    img_copy = img.copy()
    for line, pt in zip(lines, pts):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        a, b, c_line = line
        # Calcular dos puntos extremos de la línea
        if np.abs(b) > 1e-6:
            x0 = 0
            y0 = int(-c_line / b)
            x1 = img.shape[1]
            y1 = int(-(c_line + a * x1) / b)
        else:
            x0 = int(-c_line / a)
            y0 = 0
            x1 = int(-c_line / a)
            y1 = img.shape[0]
        img_copy = cv2.line(img_copy, (x0, y0), (x1, y1), color, line_thickness)
        img_copy = cv2.circle(img_copy, tuple(np.int32(pt)), circle_radius, color, -1)
    return img_copy

# ------------------ Código Principal ------------------

# Cargar las dos imágenes (ajusta las rutas a tus imágenes)
img1 = cv2.imread('imagen1.jpg')  # Imagen tomada desde un ángulo
img2 = cv2.imread('imagen2.jpg')  # Imagen tomada desde otro ángulo

if img1 is None or img2 is None:
    raise IOError("No se pudieron cargar las imágenes. Verifica las rutas.")

# Detectar características y encontrar correspondencias usando ORB
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

# Emparejar usando BFMatcher con distancia Hamming
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# Seleccionar un número suficiente de correspondencias (mínimo 8)
num_matches = 100
matches = matches[:num_matches]
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

# Estimar la matriz fundamental con el algoritmo de 8 puntos
F = compute_fundamental_matrix(pts1, pts2)
print("Matriz Fundamental (Algoritmo de 8 puntos):")
print(F)

# Calcular las líneas epipolares en cada imagen:
# En la imagen 1, las líneas epipolares se derivan de los puntos de la imagen 2.
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
# En la imagen 2, las líneas epipolares se derivan de los puntos de la imagen 1.
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)

# Dibujar en ambas imágenes las líneas epipolares y los puntos (con mayor tamaño)
img1_epilines = draw_epilines_and_points(img1, lines1, pts1, line_thickness=3, circle_radius=8)
img2_epilines = draw_epilines_and_points(img2, lines2, pts2, line_thickness=3, circle_radius=8)

# Visualizar resultados
plt.figure(figsize=(15, 7))
plt.subplot(121)
plt.title("Imagen 1: Líneas epipolares y puntos")
plt.imshow(cv2.cvtColor(img1_epilines, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.subplot(122)
plt.title("Imagen 2: Líneas epipolares y puntos")
plt.imshow(cv2.cvtColor(img2_epilines, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
