import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print(os.listdir())

def load_base_model_data(file_path='atari-dreamerv2.json', task_filter='atari_ms_pacman'):
    """Carga los datos del modelo base desde el archivo JSON"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Filtrar para la tarea específica
        filtered_data = [item for item in data if item['task'] == task_filter]
        
        if not filtered_data:
            print(f"No se encontraron datos para {task_filter} en el archivo base")
            return None
        
        # Convertir a DataFrame
        base_results = []
        for item in filtered_data:
            method = item['method']
            seed = item['seed']
            for step, score in zip(item['xs'], item['ys']):
                base_results.append({
                    'step': step,
                    'score': score,
                    'method': method,
                    'seed': seed,
                    'model': 'base'
                })
        
        return pd.DataFrame(base_results)
    except Exception as e:
        print(f"Error al cargar el archivo base: {str(e)}")
        return None

def load_jsonl(file_path):
    """Carga los datos del modelo modificado desde un archivo JSONL"""
    data = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return data
    except Exception as e:
        print(f"Error al cargar el archivo JSONL: {str(e)}")
        return []

def debug_step_ranges():
    """Analiza los rangos de pasos disponibles en ambos conjuntos de datos"""
    print("Iniciando depuración de rangos de pasos...")
    
    # 1. Cargar datos del modelo base
    base_file = 'atari-dreamerv2.json'
    base_df = load_base_model_data(base_file)
    
    if base_df is not None:
        print(f"\nModelo base ({base_file}):")
        print(f"Total de registros: {len(base_df)}")
        print(f"Rango de pasos: {base_df['step'].min():,} - {base_df['step'].max():,}")
        
        # Ver cuántos registros hay por debajo de diferentes umbrales
        for threshold in [100000, 200000, 500000, 1000000]:
            count = len(base_df[base_df['step'] <= threshold])
            print(f"Registros con paso <= {threshold:,}: {count}")
        
        # Analizar los primeros pasos disponibles
        print("\nPrimeros pasos registrados en el modelo base:")
        first_steps = sorted(base_df['step'].unique())[:10]
        print(first_steps)
    else:
        print("No se pudieron cargar los datos del modelo base")
    
    # 2. Cargar datos del modelo modificado
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except:
        current_dir = '.'
        
    metrics_path = os.path.join(current_dir, 'metrics.jsonl')
    
    data = load_jsonl(metrics_path)
    if data:
        df = pd.DataFrame(data)
        train_df = df[df['train_return'].notna()].copy()
        
        if not train_df.empty:
            train_df = train_df.sort_values('step')
            modified_df = train_df.copy()
            modified_df = modified_df.rename(columns={'train_return': 'score'})
            
            print(f"\nModelo modificado ({metrics_path}):")
            print(f"Total de registros: {len(modified_df)}")
            print(f"Rango de pasos: {modified_df['step'].min():,} - {modified_df['step'].max():,}")
            
            # Ver primeros pasos del modelo modificado
            print("\nPrimeros pasos registrados en el modelo modificado:")
            first_steps = sorted(modified_df['step'].unique())[:10]
            print(first_steps)
        else:
            print("No hay datos de entrenamiento en el modelo modificado")
    else:
        print("No se pudieron cargar los datos del modelo modificado")
    
    # 3. Ejecutar con pasos ajustados
    print("\n--- Ejecutando con pasos ajustados ---")
    
    # Determinar el rango común de pasos
    if base_df is not None and 'modified_df' in locals() and not modified_df.empty:
        base_min = base_df['step'].min()
        base_max = base_df['step'].max()
        mod_min = modified_df['step'].min()
        mod_max = modified_df['step'].max()
        
        print(f"Rango modelo base: {base_min:,} - {base_max:,}")
        print(f"Rango modelo modificado: {mod_min:,} - {mod_max:,}")
        
        common_min = max(base_min, mod_min)
        common_max = min(base_max, mod_max)
        
        print(f"Rango común: {common_min:,} - {common_max:,}")
        
        # Filtrar ambos DataFrames al rango común
        base_filtered = base_df[(base_df['step'] >= common_min) & (base_df['step'] <= common_max)]
        mod_filtered = modified_df[(modified_df['step'] >= common_min) & (modified_df['step'] <= common_max)]
        
        print(f"Registros en rango común:")
        print(f"  Modelo base: {len(base_filtered)}")
        print(f"  Modelo modificado: {len(mod_filtered)}")
        
        # Mostrar gráfico comparativo básico si hay datos
        if not base_filtered.empty and not mod_filtered.empty:
            plt.figure(figsize=(10, 6))
            
            # Agrupar y promediar puntuaciones por paso para el modelo base
            base_grouped = base_filtered.groupby('step')['score'].mean()
            
            plt.plot(base_grouped.index, base_grouped.values, 'b-', label='Modelo Base')
            plt.plot(mod_filtered['step'], mod_filtered['score'], 'r-', label='Modelo Modificado')
            
            plt.title('Comparativa en Rango Común')
            plt.xlabel('Pasos')
            plt.ylabel('Puntuación')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.savefig('debug_comparison.png')
            print(f"Gráfico guardado como 'debug_comparison.png'")
        
        # Proponer un rango de comparación viable
        viable_range = min(mod_max, base_max)
        print(f"\nRango de comparación viable sugerido: 0 - {viable_range:,}")
        
        # Contar registros en ese rango
        print(f"Registros en rango viable:")
        base_viable = base_df[base_df['step'] <= viable_range]
        mod_viable = modified_df[modified_df['step'] <= viable_range]
        print(f"  Modelo base: {len(base_viable)}")
        print(f"  Modelo modificado: {len(mod_viable)}")
        
        # Calcular primera aparición de ciertos pasos
        print("\nPrimera aparición de pasos clave:")
        for step_threshold in [100000, 200000, 500000, 1000000]:
            base_first = base_df[base_df['step'] >= step_threshold]['step'].min() if not base_df[base_df['step'] >= step_threshold].empty else "N/A"
            mod_first = modified_df[modified_df['step'] >= step_threshold]['step'].min() if not modified_df[modified_df['step'] >= step_threshold].empty else "N/A"
            
            print(f"  {step_threshold:,}: Base={base_first}, Modificado={mod_first}")

if __name__ == "__main__":
    debug_step_ranges()
