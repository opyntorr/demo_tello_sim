#!/usr/bin/env python3
import os
import csv

def tune_ekf():
    print("--- Tello EKF Auto-Tuner ---")
    
    # 1. Leer CSV
    csv_path = 'src/tello_control_pos/tello_control_pos/ruido_odometria.csv'
    if not os.path.exists(csv_path):
        print(f"Error: No se encontró el archivo de ruido en: {csv_path}")
        print("Asegúrate de ejecutar este script desde la raíz del workspace (demo_tello_sim)")
        return

    x_vals, y_vals, z_vals = [], [], []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        try:
            next(reader) # Saltar cabecera
        except StopIteration:
            pass
            
        for row in reader:
            if len(row) >= 4:
                try:
                    x_vals.append(float(row[1]))
                    y_vals.append(float(row[2]))
                    z_vals.append(float(row[3]))
                except ValueError:
                    pass
                
    if len(x_vals) < 2:
        print("Datos insuficientes en el archivo CSV.")
        return
        
    # Calcular Varianza
    def get_var(vals):
        mean = sum(vals) / len(vals)
        var = sum((x - mean)**2 for x in vals) / len(vals)
        return var

    var_x = get_var(x_vals)
    var_y = get_var(y_vals)
    var_z = get_var(z_vals)
    
    print(f"Varianza calculada X: {var_x:.6f}")
    print(f"Varianza calculada Y: {var_y:.6f}")
    print(f"Varianza calculada Z: {var_z:.6f}")
    
    # Heurística de ajuste:
    # Measurement Noise (R) = Varianza observada real (con un piso mínimo)
    cov_px = max(var_x, 0.001)
    cov_py = max(var_y, 0.001)
    cov_pz = max(var_z, 0.001)
    
    # Process Noise (Q) = Ajustado dinámicamente
    proc_x = max(var_x * 0.5, 0.01)
    proc_y = max(var_y * 0.5, 0.01)
    proc_z = max(var_z * 0.5, 0.01)

    # 2. Actualizar YAML preservando los comentarios
    yaml_path = 'src/tello_control_pos/config/ekf.yaml'
    if not os.path.exists(yaml_path):
        print(f"Error: No se encontró el archivo de configuración en {yaml_path}")
        return
        
    with open(yaml_path, 'r') as f:
        lines = f.readlines()
        
    # Reemplazar valores de covariance_injector
    for i, line in enumerate(lines):
        if 'cov_pose_x:' in line:
            lines[i] = f"        cov_pose_x: {cov_px:.5f}\n"
        elif 'cov_pose_y:' in line:
            lines[i] = f"        cov_pose_y: {cov_py:.5f}\n"
        elif 'cov_pose_z:' in line:
            lines[i] = f"        cov_pose_z: {cov_pz:.5f}\n"
            
    # Reemplazar la diagonal de process_noise_covariance
    for i, line in enumerate(lines):
        if 'process_noise_covariance:' in line:
            lines[i+1] = f"          {proc_x:.5f}, 0.0,    0.0,    0.0,    0.0,    0.0,    0.0,     0.0,     0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,\n"
            lines[i+2] = f"          0.0,  {proc_y:.5f},   0.0,    0.0,    0.0,    0.0,    0.0,     0.0,     0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,\n"
            lines[i+3] = f"          0.0,  0.0,    {proc_z:.5f},   0.0,    0.0,    0.0,    0.0,     0.0,     0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,\n"
            break
            
    with open(yaml_path, 'w') as f:
        f.writelines(lines)
        
    print("\n¡Actualización completada!")
    print(f"-> Measurement Noise actualizado en ekf.yaml")
    print(f"-> Process Noise actualizado en ekf.yaml")
    print("\nRecuerda recompilar con 'colcon build' si este es tu primer ajuste para asegurarte de que ROS copie el yaml.")

if __name__ == '__main__':
    tune_ekf()
