"""
Uso:
python aaa.py <caminho_para_imagem.nii.gz>

Exemplo (estando no diretório 'src/'):
python aaa.py ../database/axl/OAS2_0001_MR1_axl.nii.gz
"""

import os
import sys
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from skimage.morphology import remove_small_objects, binary_opening, disk
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from skimage.filters import threshold_otsu, gaussian
from skimage.exposure import rescale_intensity, equalize_adapthist
from sklearn.cluster import KMeans

# --- Constantes e Caminhos (relativos ao diretório 'src/') ---

# CSV de Metadados (usado para buscar 'Age' e 'Group')
OASIS_CSV_PATH = '../database/oasis_longitudinal_demographic.csv'

# Caminhos dos Modelos Salvos
MODEL_LR_DEMENCIA_PATH = 'models/modelo_lr_demencia.joblib'
MODEL_XGB_DEMENCIA_PATH = 'models/modelo_xgb_demencia.joblib'
MODEL_LR_IDADE_PATH = 'models/modelo_lr_idade.joblib'
MODEL_XGB_IDADE_PATH = 'models/modelo_xgb_idade.joblib'

# Diretório de Saída para Imagens
OUTPUT_DIR = 'aaa'

# Parâmetros de Segmentação
N_CLUSTERS = 4
MIN_AREA_VENTRICLE = 100
MAX_AREA_VENTRICLE = 15000
MIN_AREA_BRAIN = 5000
CENTER_TOLERANCE_RATIO = 0.2

# Colunas que os modelos esperam
COLS_DEMENCIA = ['Age', 'Ventricle_Area', 'Ventricle_Perimeter', 'Ventricle_Circularity', 
                 'Ventricle_Eccentricity', 'Ventricle_Solidity', 'Ventricle_MajorAxisLength']

COLS_IDADE = ['Group_num', 'Ventricle_Area', 'Ventricle_Perimeter', 'Ventricle_Circularity', 
              'Ventricle_Eccentricity', 'Ventricle_Solidity', 'Ventricle_MajorAxisLength']

def apply_roi_crop(image_slice, center_ratio, width_ratio, height_ratio):
    rows, cols = image_slice.shape
    crop_rows = int(rows * height_ratio)
    crop_cols = int(cols * width_ratio)
    start_row = int(rows * center_ratio - crop_rows / 2)
    start_col = int(cols * center_ratio - crop_cols / 2)
    start_row = max(0, start_row)
    start_col = max(0, start_col)
    end_row = min(rows, start_row + crop_rows)
    end_col = min(cols, start_col + crop_cols)
    cropped_image = image_slice[start_row:end_row, start_col:end_col]
    return cropped_image, (start_row, end_row, start_col, end_col)

def segmentar_ventriculos(image_slice, n_clusters, min_area_brain, min_area_ventricle, max_area_ventricle, apply_crop=False, roi_params=None, center_tolerance_ratio=0.2):
    original_shape = image_slice.shape
    
    # 1. Pré-processamento
    img_norm = rescale_intensity(image_slice, out_range=(0, 1))
    img_clahe = equalize_adapthist(img_norm)
    img_smooth = gaussian(img_clahe, sigma=1)

    # 2. Remoção do Crânio (Skull Stripping)
    t = threshold_otsu(img_smooth)
    mask_cerebro = img_smooth > t
    mask_cerebro = binary_opening(mask_cerebro, disk(3))
    mask_cerebro = remove_small_objects(mask_cerebro, min_size=min_area_brain) 
    mask_cerebro = binary_fill_holes(mask_cerebro)
    
    labels_cerebro = label(mask_cerebro)
    if labels_cerebro.max() == 0:
        return {'ventriculos': np.zeros(original_shape, dtype=bool), 'pre_processamento': img_smooth, 'cropped_image': None}
        
    maior_comp_label = np.argmax([region.area for region in regionprops(labels_cerebro)]) + 1
    mask_cerebro = (labels_cerebro == maior_comp_label)
    img_sem_cranio = img_smooth * mask_cerebro
    
    # 3. Aplica o recorte (se solicitado)
    if apply_crop and roi_params:
        img_proc_cropped, crop_coords = apply_roi_crop(img_sem_cranio, 
                                                       roi_params['center'], 
                                                       roi_params['width'], 
                                                       roi_params['height'])
        mask_cerebro_cropped, _ = apply_roi_crop(mask_cerebro, 
                                                 roi_params['center'], 
                                                 roi_params['width'], 
                                                 roi_params['height'])
        image_for_kmeans = img_proc_cropped
        mask_for_kmeans = mask_cerebro_cropped
        shape_for_kmeans = img_proc_cropped.shape
    else:
        img_proc_cropped = None
        crop_coords = None
        image_for_kmeans = img_sem_cranio
        mask_for_kmeans = mask_cerebro
        shape_for_kmeans = original_shape
        
    # 4. K-Means
    pixels_cerebro = image_for_kmeans[mask_for_kmeans].reshape(-1, 1)
    if pixels_cerebro.shape[0] < n_clusters:
        return {'ventriculos': np.zeros(original_shape, dtype=bool), 'pre_processamento': img_sem_cranio, 'cropped_image': img_proc_cropped}

    kmeans = KMeans(n_clusters=n_clusters, random_state=64, n_init=10).fit(pixels_cerebro)
    centers = kmeans.cluster_centers_.flatten()
    labels_flat = kmeans.labels_

    # 5. Identificação do LCR
    sorted_indices = np.argsort(centers)
    indice_lcr = sorted_indices[0] 
    labels_kmeans = np.zeros(shape_for_kmeans, dtype=int)
    labels_kmeans[mask_for_kmeans] = labels_flat + 1
    mask_lcr_total = (labels_kmeans == (indice_lcr + 1))

    # 6. Isolamento dos Ventrículos 
    dist_transform = distance_transform_edt(mask_lcr_total)
    labels_lcr = label(dist_transform)
    regioes_lcr = regionprops(labels_lcr)
    mask_ventriculos_proc = np.zeros(shape_for_kmeans, dtype=bool)
    
    if not regioes_lcr:
        return {'ventriculos': np.zeros(original_shape, dtype=bool), 'pre_processamento': img_sem_cranio, 'cropped_image': img_proc_cropped}

    center_r, center_c = np.array(shape_for_kmeans) / 2
    max_dist_r = shape_for_kmeans[0] * center_tolerance_ratio
    max_dist_c = shape_for_kmeans[1] * center_tolerance_ratio
    
    for r in regioes_lcr:
        # 6.1. Filtro de Área
        is_correct_size = r.area > min_area_ventricle and r.area < max_area_ventricle
        # 6.2. Filtro de Posição Central
        centroid_r, centroid_c = r.centroid
        is_central_r = abs(centroid_r - center_r) < max_dist_r
        is_central_c = abs(centroid_c - center_c) < max_dist_c
        is_central = is_central_r and is_central_c
        
        if is_correct_size and is_central:
            mask_ventriculos_proc[labels_lcr == r.label] = True

    mask_ventriculos_proc = binary_fill_holes(mask_ventriculos_proc)
    
    # 7. Reverte o recorte
    mask_ventriculos_final = np.zeros(original_shape, dtype=bool)
    if apply_crop and crop_coords:
        sr, er, sc, ec = crop_coords
        mask_ventriculos_final[sr:er, sc:ec] = mask_ventriculos_proc
    else:
        mask_ventriculos_final = mask_ventriculos_proc
    
    return {
        'ventriculos': mask_ventriculos_final,
        'pre_processamento': img_sem_cranio,
        'cropped_image': img_proc_cropped
    }

def extract_features(ventricle_mask): 
    default_features = {
        'Ventricle_Area': 0, 'Ventricle_Perimeter': 0, 'Ventricle_Circularity': 0,
        'Ventricle_Eccentricity': 0, 'Ventricle_Solidity': 0, 'Ventricle_MajorAxisLength': 0
    }

    labels = label(ventricle_mask)
    props = regionprops(labels)
    
    if not props:
        return default_features # Retorna 0 se nenhum ventrículo for encontrado

    total_area = 0
    total_perimeter = 0
    metrics_list = {'Circularity': [], 'Eccentricity': [], 'Solidity': [], 'MajorAxisLength': []}

    for region in props:
        total_area += region.area
        total_perimeter += region.perimeter
        
        if region.perimeter > 0:
            circularity = (4 * np.pi * region.area) / (region.perimeter ** 2)
        else:
            circularity = 0
            
        metrics_list['Circularity'].append(circularity)
        metrics_list['Eccentricity'].append(region.eccentricity)
        metrics_list['Solidity'].append(region.solidity)
        metrics_list['MajorAxisLength'].append(region.major_axis_length)

    # Calcula os valores finais
    features = {
        'Ventricle_Area': total_area,
        'Ventricle_Perimeter': total_perimeter,
        'Ventricle_Circularity': np.mean(metrics_list['Circularity']),
        'Ventricle_Eccentricity': np.mean(metrics_list['Eccentricity']),
        'Ventricle_Solidity': np.mean(metrics_list['Solidity']),
        'Ventricle_MajorAxisLength': np.mean(metrics_list['MajorAxisLength'])
    }
    
    return features

def get_img_id_from_path(nii_path):
    filename = os.path.basename(nii_path)
    # Remove a(s) extensão(ões) (ex: 'OAS2_0001_MR1_axl')
    mri_id_base = filename.split('.')[0]
    
    # Remove sufixos comuns (como _axl) para corresponder ao CSV
    if mri_id_base.endswith('_axl'):
        mri_id = mri_id_base.removesuffix('_axl')
    # Adicione outros 'elif' se necessário, ex:
    # elif mri_id_base.endswith('_sag'):
    #     mri_id = mri_id_base.removesuffix('_sag')
    else:
        mri_id = mri_id_base
        
    return mri_id

def process_single_image(nii_path, oasis_df):
    try:
        # --- 1. Carregar Imagem e Obter Slice Central ---
        print(f"Processando: {nii_path}")
        mri_id = get_img_id_from_path(nii_path)
        print(f"Buscando ID: {mri_id}") 
        
        nii_img = nib.load(nii_path)
        data = nii_img.get_fdata()
        
        if data.ndim == 3:
            slice_z = data.shape[2] // 2
            image_slice = data[:, :, slice_z]
        elif data.ndim == 2:
            image_slice = data
        else:
            print(f"Erro: Dimensionalidade inesperada da imagem ({data.ndim}D).")
            return

        image_slice = np.rot90(image_slice)

        # --- 2. Segmentação ---
        print("Segmentando ventrículos...")
        segmentation_results = segmentar_ventriculos(
            image_slice,
            n_clusters=N_CLUSTERS,
            min_area_brain=MIN_AREA_BRAIN,
            min_area_ventricle=MIN_AREA_VENTRICLE,
            max_area_ventricle=MAX_AREA_VENTRICLE,
            center_tolerance_ratio=CENTER_TOLERANCE_RATIO
        )
        
        ventricle_mask = segmentation_results['ventriculos']
        preprocessed_img = segmentation_results['pre_processamento'] 

        # --- 3. Salvar Imagens de Saída ---
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        orig_path = os.path.join(OUTPUT_DIR, f"{mri_id}_original.png")
        prep_path = os.path.join(OUTPUT_DIR, f"{mri_id}_preprocessed.png")
        seg_path = os.path.join(OUTPUT_DIR, f"{mri_id}_segmented.png")
        
        # Salva original e pré-processada
        plt.imsave(orig_path, image_slice, cmap='gray')
        plt.imsave(prep_path, preprocessed_img, cmap='gray')

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image_slice, cmap='gray') 
        
        # Desenha apenas o contorno da máscara em amarelo
        ax.contour(ventricle_mask, levels=[0.5], colors='yellow', linewidths=1)
        
        ax.set_axis_off() 
        fig.savefig(seg_path, bbox_inches='tight', pad_inches=0, dpi=100) 
        plt.close(fig) 
        
        print(f"Imagens salvas em '/{OUTPUT_DIR}'")

        # --- 4. Extração de Features ---
        print("Extraindo features")
        features = extract_features(ventricle_mask)
        
        if features['Ventricle_Area'] == 0:
            print(f"AVISO: Nenhum ventrículo encontrado para {mri_id}.")

        # --- 5. Preparo de Dados ---
        print("Buscando metadados")
        try:
            metadata = oasis_df[oasis_df['MRI ID'] == mri_id].iloc[0]
            subject_id = metadata['Subject ID']
            age = metadata['Age']
            group = metadata['Group'] 
        except IndexError:
            print(f"AVISO: MRI ID {mri_id} não encontrado em {OASIS_CSV_PATH}. Usando NaN/Unknown.")
            subject_id, age, group = get_img_id_from_path(nii_path), np.nan, "Unknown"

        # Criar o DataFrame de uma linha para entrada no modelo
        data_dict = {
            'Subject ID': subject_id,
            'MRI ID': mri_id,
            'Group': group,
            'Age': age,
            'Ventricle_Area': features['Ventricle_Area'],
            'Ventricle_Perimeter': features['Ventricle_Perimeter'],
            'Ventricle_Circularity': features['Ventricle_Circularity'],
            'Ventricle_Eccentricity': features['Ventricle_Eccentricity'],
            'Ventricle_Solidity': features['Ventricle_Solidity'],
            'Ventricle_MajorAxisLength': features['Ventricle_MajorAxisLength']
        }
        df_single = pd.DataFrame([data_dict])
        
        # Mapear 'Group' para 'Group_num'
        df_single['Group_num'] = df_single['Group'].map({'NonDemented': 0, 'Demented': 1})

        # Imprimir a linha de features no formato CSV solicitado
        print("\n--- Features Extraídas (Formato CSV) ---")
        csv_string = df_single[data_dict.keys()].to_csv(sep=';', decimal=',', index=False, header=True, lineterminator='\n')
        print(csv_string.split('\n')[0]) # Header
        print(csv_string.split('\n')[1]) # Data

        # --- 6. Classificação ---
        print("\n--- Carregando Modelos e Classificando ---")
        
        X_demencia = df_single[COLS_DEMENCIA]
        X_idade = df_single[COLS_IDADE] 

        # --- Classificação Demência ---
        try:
            model_lr_dem = joblib.load(MODEL_LR_DEMENCIA_PATH)
            model_xgb_dem = joblib.load(MODEL_XGB_DEMENCIA_PATH)
            
            pred_lr_dem_val = model_lr_dem.predict(X_demencia)[0]
            pred_xgb_dem_val = model_xgb_dem.predict(X_demencia)[0]
            
            pred_lr_dem_label = 'Demented' if pred_lr_dem_val == 1 else 'NonDemented'
            pred_xgb_dem_label = 'Demented' if pred_xgb_dem_val == 1 else 'NonDemented'

            print("--- Resultados da Classificação (Demência) ---")
            if pd.isna(age):
                 print(f"Grupo Real:                  {group} (Idade não disponível para os modelos)")
            else:
                print(f"Grupo Real:                   {group}")
            print(f"Predição (Regressão Linear):      {pred_lr_dem_label}")
            print(f"Predição (XGBoost):               {pred_xgb_dem_label}")

        except FileNotFoundError:
            print("Erro: Modelos de demência não encontrados. Pulei a classificação de demência.")
        except Exception as e:
            print(f"Erro na classificação de demência: {e}")

        # --- Regressão Idade ---
        try:
            model_lr_age = joblib.load(MODEL_LR_IDADE_PATH)
            model_xgb_age = joblib.load(MODEL_XGB_IDADE_PATH)

            pred_lr_age = model_lr_age.predict(X_idade)[0]
            pred_xgb_age = model_xgb_age.predict(X_idade)[0]

            print("\n--- Resultados da Regressão (Idade) ---")
            print(f"Idade Real:                       {age}")
            print(f"Idade Predita (Regressão Linear): {pred_lr_age:.2f}")
            print(f"Idade Predita (XGBoost):          {pred_xgb_age:.2f}")

        except FileNotFoundError:
            print("Erro: Modelos de idade não encontrados. Pulei a regressão de idade.")
        except Exception as e:
            print(f"Erro na regressão de idade: {e}")

    except FileNotFoundError:
        print(f"Erro: Arquivo de imagem NIfTI não encontrado em '{nii_path}'")
    except Exception as e:
        print(f"Ocorreu um erro inesperado durante o processamento: {e}")

if __name__ == "__main__":   
    if len(sys.argv) < 2:
        print("Erro: Forneça o caminho para a imagem .nii.gz como argumento.")
        print("Uso: python classificar_imagem.py <caminho_para_imagem.nii.gz>")
        print("Exemplo (do diretório 'src/'): python classificar_imagem.py ../database/axl/OAS2_0001_MR1_axl.nii.gz")
        sys.exit(1)
        
    image_path_arg = sys.argv[1]

    # Carregar e pré-processar o CSV de metadados
    try:
        print(f"Carregando metadados de {OASIS_CSV_PATH}...")
        df_oasis = pd.read_csv(OASIS_CSV_PATH, sep=';', decimal=',')
        
        # Aplicar a "limpeza" do db_split (mapear 'Converted' para 'Demented')
        df_oasis['Group'] = df_oasis['Group'].map({
            'Nondemented': 'NonDemented', 
            'Demented': 'Demented', 
            'Converted': 'Demented'
        })
        # Remover linhas que não são Demented ou NonDemented (após o mapeamento)
        df_oasis = df_oasis.dropna(subset=['Group'])
        print("Metadados carregados e limpos.")
        
    except FileNotFoundError:
        print(f"Erro Crítico: CSV de metadados {OASIS_CSV_PATH} não encontrado.")
        print("Verifique se o script está sendo executado do diretório 'src/'.")
        sys.exit(1)
    except Exception as e:
        print(f"Erro ao carregar CSV de metadados: {e}")
        sys.exit(1)

    # Executar o pipeline completo para a imagem fornecida
    process_single_image(image_path_arg, df_oasis)