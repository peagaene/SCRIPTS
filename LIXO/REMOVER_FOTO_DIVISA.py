import os
import shutil
from concurrent.futures import ThreadPoolExecutor

def move_file(file, folder1, destination_folder):
    file_path1 = os.path.join(folder1, file)
    dest_path = os.path.join(destination_folder, file)
    
    if os.path.isfile(file_path1):
        shutil.move(file_path1, dest_path)
        print(f"Movido: {file_path1} -> {dest_path}")

def move_duplicate_images(folder1, folder2, destination_folder):
    """
    Move apenas os arquivos TIFF duplicados da pasta 1 para outra pasta (com base no nome do arquivo), mantendo as imagens da pasta 2.
    """
    
    # Cria a pasta de destino se não existir
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # Obtém a lista de arquivos em ambas as pastas
    files1 = {f for f in os.listdir(folder1) if f.lower().endswith('.tif') or f.lower().endswith('.tiff')}
    files2 = {f for f in os.listdir(folder2) if f.lower().endswith('.tif') or f.lower().endswith('.tiff')}
    
    # Encontra os arquivos que existem em ambas as pastas
    duplicate_files = files1.intersection(files2)
    
    if not duplicate_files:
        print("Nenhum arquivo TIFF duplicado encontrado.")
        return
    
    # Usa multithreading para mover os arquivos duplicados mais rapidamente
    with ThreadPoolExecutor() as executor:
        executor.map(lambda file: move_file(file, folder1, destination_folder), duplicate_files)
    
    print("Processo concluído.")

# Exemplo de uso
folder1 = r"G:\SI_09"
folder2 = r"G:\DIVISA\SI13_SI14"
destination_folder = r"K:\SP22_BE_13_03052024_HD03\SI_13\RGB\14"
move_duplicate_images(folder1, folder2, destination_folder)
