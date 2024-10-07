from flask import Flask, request, jsonify, render_template_string
import numpy as np
import cv2
import heapq
import tempfile
import os
import gc
import matplotlib.pyplot as plt

app = Flask(__name__)

# Cargar y procesar la imagen del laberinto con reducción de resolución
def load_maze_image(image_data):
    # Crear un archivo temporal para guardar la imagen y reducir la resolución
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    temp_file.write(image_data)
    temp_file.flush()
    temp_file.seek(0)

    # Leer la imagen desde el archivo temporal y reducir su resolución
    image = cv2.imread(temp_file.name, cv2.IMREAD_GRAYSCALE)
    
    # Reducir resolución para evitar uso excesivo de memoria
    max_size = 500  # Establece un tamaño máximo
    if image.shape[0] > max_size or image.shape[1] > max_size:
        scaling_factor = max_size / max(image.shape[0], image.shape[1])
        image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    _, binary_maze = cv2.threshold(image, 128, 1, cv2.THRESH_BINARY_INV)
    
    # Limpiar archivo temporal
    os.remove(temp_file.name)
    
    return binary_maze

# Algoritmo A* para encontrar el camino más corto
def a_star_search(maze, start, end):
    rows, cols = maze.shape
    open_list = []
    heapq.heappush(open_list, (0 + heuristic(start, end), 0, start))
    came_from = {}
    g_score = {(r, c): float('inf') for r in range(rows) for c in range(cols)}
    g_score[start] = 0
    f_score = {(r, c): float('inf') for r in range(rows) for c in range(cols)}
    f_score[start] = heuristic(start, end)

    while open_list:
        _, current_cost, current = heapq.heappop(open_list)
        
        if current == end:
            return reconstruct_path(came_from, current)
        
        for neighbor in get_neighbors(current, maze):
            tentative_g_score = g_score[current] + 1
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, end)
                heapq.heappush(open_list, (f_score[neighbor], tentative_g_score, neighbor))
    
    return None

# Heurística Manhattan
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Obtener los vecinos válidos (movimientos) en el laberinto
def get_neighbors(node, maze):
    neighbors = []
    x, y = node
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx, ny] == 0:
            neighbors.append((nx, ny))
    return neighbors

# Reconstruir el camino una vez encontrado
def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]

# Encontrar la entrada y salida en las paredes exteriores
def find_entrance_exit(maze):
    rows, cols = maze.shape
    start = None
    end = None
    
    # Buscar en la primera y última fila
    for j in range(cols):
        if maze[0, j] == 0:  # La primera fila
            start = (0, j)
        if maze[rows-1, j] == 0:  # La última fila
            end = (rows-1, j)
    
    # Buscar en la primera y última columna
    for i in range(rows):
        if maze[i, 0] == 0:  # La primera columna
            start = (i, 0)
        if maze[i, cols-1] == 0:  # La última columna
            end = (i, cols-1)
    
    return start, end

# Convertir el camino en comandos
def convert_path_to_commands(path):
    commands = []
    directions = {
        (-1, 0): "Arriba",
        (1, 0): "Abajo",
        (0, -1): "Izquierda",
        (0, 1): "Derecha"
    }
    
    for i in range(1, len(path)):
        dx = path[i][0] - path[i-1][0]
        dy = path[i][1] - path[i-1][1]
        if (dx, dy) in directions:
            commands.append(directions[(dx, dy)])
    
    return commands

# Dibujar el camino en el laberinto y liberar memoria
def draw_path_on_maze(maze, path):
    maze_with_path = np.copy(maze)
    for (x, y) in path:
        maze_with_path[x, y] = 2  # Marca el camino con un valor diferente

    plt.figure(figsize=(8, 8))
    plt.imshow(maze_with_path, cmap='gray')
    plt.title('Laberinto con el camino')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Forzar la recolección de basura para liberar memoria
    gc.collect()

@app.route('/')
def index():
    return render_template_string('''
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Upload and Crop Image</title>
    </head>
    <body>
        <h1>Upload and Crop Image</h1>
        <form id="upload-form" method="POST" action="/solve_maze" enctype="multipart/form-data">
            <input type="file" id="image-file" name="image" accept="image/*" required>
            <button type="submit">Submit</button>
        </form>
    </body>
    </html>
    ''')

@app.route('/solve_maze', methods=['POST'])
def solve_maze():
    if 'image' not in request.files:
        return jsonify({"error": "No se envió ninguna imagen"}), 400
    
    file = request.files['image']
    image_data = file.read()
    
    # Procesar la imagen y convertirla en un laberinto binario
    maze = load_maze_image(image_data)
    start, end = find_entrance_exit(maze)
    
    # Si no se encuentra inicio o fin, error
    if not start or not end:
        return jsonify({"error": "No se pudo encontrar la entrada o salida en el laberinto"}), 400
    
    # Ejecutar el algoritmo A* para encontrar el camino
    path = a_star_search(maze, start, end)
    
    if path:
        draw_path_on_maze(maze, path)
        commands = convert_path_to_commands(path)
        return jsonify({"commands": commands})
    else:
        return jsonify({"error": "No se encontró un camino"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
