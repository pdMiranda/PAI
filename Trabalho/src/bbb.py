# --- IMPORTS NECESSÁRIOS (GUI E BACKEND) ---
import sys
import os
import pathlib
import io
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
import cv2

# Imports da PySide6
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFrame, QMenu, QMessageBox, QFileDialog,
    QGroupBox, QTextEdit, QGridLayout, QDialog,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
)
from PySide6.QtGui import (
    QPixmap, QAction, QFont, QImage, QTextCursor, 
    QPainter
)
from PySide6.QtCore import Qt, QObject, Signal, Slot

# Imports do Skimage
from skimage.morphology import remove_small_objects, binary_opening, disk
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from skimage.filters import threshold_otsu, gaussian
from skimage.exposure import rescale_intensity, equalize_adapthist

# Imports do Sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, mean_absolute_error, r2_score

# --- CAMINHOS GLOBAIS ---
BASE_DIR = pathlib.Path(__file__).parent.resolve()
MODELS_DIR = BASE_DIR / "models"
OUT_DIR = BASE_DIR / "out"
DATABASE_DIR = BASE_DIR.parent / "database"
OASIS_CSV_PATH = DATABASE_DIR / "oasis_longitudinal_demographic.csv"

# --- PARÂMETROS GLOBAIS ---
N_CLUSTERS = 4
MIN_AREA_VENTRICLE = 100
MAX_AREA_VENTRICLE = 15000
MIN_AREA_BRAIN = 5000
CENTER_TOLERANCE_RATIO = 0.2

# Colunas esperadas pelos pipelines de predição
# Pipeline de Demência (LR e XGB)
COLS_DEMENCIA = [
    'Age', 
    'Ventricle_Area', 'Ventricle_Perimeter', 'Ventricle_Circularity', 
    'Ventricle_Eccentricity', 'Ventricle_Solidity', 'Ventricle_MajorAxisLength'
]
# Pipeline de Idade (LR e XGB)
COLS_IDADE = [
    'Group_num', 
    'Ventricle_Area', 'Ventricle_Perimeter', 'Ventricle_Circularity', 
    'Ventricle_Eccentricity', 'Ventricle_Solidity', 'Ventricle_MajorAxisLength'
]

# --- CLASSE PARA REDIRECIONAR O TERMINAL ---
class EmittingStream(QObject):
    """
    Classe para redirecionar 'print' (stdout/stderr) para um QObject
    """
    textWritten = Signal(str)
    def write(self, text):
        self.textWritten.emit(str(text))
    def flush(self):
        pass

# --- INÍCIO: LÓGICA DE BACKEND ---

def backend_segmentar_ventriculos(image_slice):
    """
    Lógica de segmentação
    Recebe um slice 2D da imagem e retorna um dict com a máscara e a img pré-processada.
    """
    original_shape = image_slice.shape
    
    # 1. Pré-processamento
    img_norm = rescale_intensity(image_slice, out_range=(0, 1))
    img_clahe = equalize_adapthist(img_norm)
    img_smooth = gaussian(img_clahe, sigma=1)

    # 2. Remoção do Crânio (Skull Stripping)
    t = threshold_otsu(img_smooth)
    mask_cerebro = img_smooth > t
    mask_cerebro = binary_opening(mask_cerebro, disk(3))
    mask_cerebro = remove_small_objects(mask_cerebro, min_size=MIN_AREA_BRAIN) 
    mask_cerebro = binary_fill_holes(mask_cerebro)
    
    labels_cerebro = label(mask_cerebro)
    if labels_cerebro.max() == 0:
        return {'ventriculos': np.zeros(original_shape, dtype=bool), 'pre_processamento': img_smooth}
        
    maior_comp_label = np.argmax([region.area for region in regionprops(labels_cerebro)]) + 1
    mask_cerebro = (labels_cerebro == maior_comp_label)
    img_sem_cranio = img_smooth * mask_cerebro
    
    image_for_kmeans = img_sem_cranio
    mask_for_kmeans = mask_cerebro
    shape_for_kmeans = original_shape
        
    # 4. K-Means
    pixels_cerebro = image_for_kmeans[mask_for_kmeans].reshape(-1, 1)
    if pixels_cerebro.shape[0] < N_CLUSTERS:
        return {'ventriculos': np.zeros(original_shape, dtype=bool), 'pre_processamento': img_sem_cranio}

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=64, n_init=10).fit(pixels_cerebro)
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
        return {'ventriculos': np.zeros(original_shape, dtype=bool), 'pre_processamento': img_sem_cranio}

    center_r, center_c = np.array(shape_for_kmeans) / 2
    max_dist_r = shape_for_kmeans[0] * CENTER_TOLERANCE_RATIO
    max_dist_c = shape_for_kmeans[1] * CENTER_TOLERANCE_RATIO
    
    for r in regioes_lcr:
        is_correct_size = r.area > MIN_AREA_VENTRICLE and r.area < MAX_AREA_VENTRICLE
        centroid_r, centroid_c = r.centroid
        is_central_r = abs(centroid_r - center_r) < max_dist_r
        is_central_c = abs(centroid_c - center_c) < max_dist_c
        is_central = is_central_r and is_central_c
        
        if is_correct_size and is_central:
            mask_ventriculos_proc[labels_lcr == r.label] = True

    mask_ventriculos_proc = binary_fill_holes(mask_ventriculos_proc)
    
    return {
        'ventriculos': mask_ventriculos_proc,
        'pre_processamento': img_sem_cranio,
    }

def backend_extract_features(ventricle_mask): 
    """
    Lógica de extração de features copiada de 'aaa.py'
    Recebe a máscara 2D e retorna um DICIONÁRIO com as 6 features.
    """
    default_features = {
        'Ventricle_Area': 0, 'Ventricle_Perimeter': 0, 'Ventricle_Circularity': 0,
        'Ventricle_Eccentricity': 0, 'Ventricle_Solidity': 0, 'Ventricle_MajorAxisLength': 0
    }

    labels = label(ventricle_mask)
    props = regionprops(labels)
    
    if not props:
        return default_features

    total_area = 0
    total_perimeter = 0
    
    weighted_ecc = 0
    weighted_solidity = 0
    weighted_major_axis = 0

    for prop in props:
        total_area += prop.area
        total_perimeter += prop.perimeter
        
        weighted_ecc += prop.eccentricity * prop.area
        weighted_solidity += prop.solidity * prop.area
        weighted_major_axis += prop.major_axis_length * prop.area

    if total_area == 0:
        return default_features

    if total_perimeter == 0:
        circularity = 0
    else:
        circularity = (4 * np.pi * total_area) / (total_perimeter ** 2)
        
    avg_ecc = weighted_ecc / total_area
    avg_solidity = weighted_solidity / total_area
    avg_major_axis = weighted_major_axis / total_area

    features = {
        'Ventricle_Area': total_area,
        'Ventricle_Perimeter': total_perimeter,
        'Ventricle_Circularity': circularity,
        'Ventricle_Eccentricity': avg_ecc,
        'Ventricle_Solidity': avg_solidity,
        'Ventricle_MajorAxisLength': avg_major_axis
    }
    
    return features

def backend_load_nii_slice(nii_path):
    """
    Carrega um arquivo .nii.gz ou imagem, extrai o slice central e o ID.
    """
    try:
        filename = os.path.basename(nii_path)
        mri_id_base = filename.split('.')[0]
        mri_id = mri_id_base.replace('_axl', '').strip() # Limpa o ID
        
        nii_img = nib.load(nii_path)
        data = nii_img.get_fdata()
        
        if data.ndim == 3:
            slice_z = data.shape[2] // 2
            image_slice = data[:, :, slice_z]
        elif data.ndim == 2:
            image_slice = data
        else:
            raise Exception(f"Dimensionalidade inesperada: {data.ndim}D")

        # Rotaciona para a orientação correta
        image_slice_rotacionada = np.rot90(image_slice)
        return image_slice_rotacionada, mri_id

    except Exception as e:
        print(f"Erro ao carregar NIfTI: {e}")
        # Tenta carregar como imagem comum (png, jpg)
        try:
            image_slice = cv2.imread(nii_path, cv2.IMREAD_GRAYSCALE)
            if image_slice is None:
                raise Exception("Não foi possível ler o arquivo como imagem.")
            mri_id = os.path.basename(nii_path).split('.')[0]
            # Imagem já é 2D, não precisa de slice ou rotação
            return image_slice, mri_id
        except Exception as img_e:
            print(f"Erro ao carregar como imagem: {img_e}")
            raise Exception(f"Falha ao carregar {nii_path}.")

def backend_get_metadata(mri_id):
    """
    Busca 'Age' e 'Group' no CSV da OASIS.
    """
    try:
        df_oasis = pd.read_csv(OASIS_CSV_PATH, sep=';', decimal=',')
        
        # Limpeza do 'Group'
        df_oasis['Group'] = df_oasis['Group'].map({
            'Nondemented': 'NonDemented', 
            'Demented': 'Demented', 
            'Converted': 'Demented' # Mapeia Converted para Demented
        })
        df_oasis = df_oasis.dropna(subset=['Group'])
        
        metadata = df_oasis[df_oasis['MRI ID'] == mri_id].iloc[0]
        
        age = metadata['Age']
        group = metadata['Group']
        group_num = 1 if group == 'Demented' else 0
        
        return {'Age': age, 'Group': group, 'Group_num': group_num, 'MRI ID': mri_id}
        
    except Exception as e:
        print(f"AVISO: Metadados não encontrados para {mri_id}. Erro: {e}")
        # Retorna NaN/Unknown para que a predição ainda possa tentar
        return {'Age': np.nan, 'Group': 'Unknown', 'Group_num': np.nan, 'MRI ID': mri_id}

def backend_run_prediction(features_dict, metadata_dict):
    """
    Carrega os 4 pipelines (modelos+scalers) e faz a predição.
    """
    try:
        # 1. Criar o DataFrame de 1 linha com todas as colunas
        data = {}
        data.update(features_dict)
        data.update(metadata_dict)
        df_predict = pd.DataFrame([data])
        
        # 2. Separar os dataframes para cada tipo de modelo
        df_demencia = df_predict[COLS_DEMENCIA]
        df_idade = df_predict[COLS_IDADE]
        
        # 3. Carregar Modelos e Predizer
        
        # --- Demência ---
        model_lr_dem = joblib.load(MODELS_DIR / 'modelo_lr_demencia.joblib')
        model_xgb_dem = joblib.load(MODELS_DIR / 'modelo_xgb_demencia.joblib')
        
        pred_lr_dem_val = model_lr_dem.predict(df_demencia)[0]
        pred_xgb_dem_val = model_xgb_dem.predict(df_demencia)[0]
        
        pred_lr_dem_label = 'Demente' if pred_lr_dem_val == 1 else 'Não Demente'
        pred_xgb_dem_label = 'Demente' if pred_xgb_dem_val == 1 else 'Não Demente'

        # --- Idade ---
        model_lr_age = joblib.load(MODELS_DIR / 'modelo_lr_idade.joblib')
        model_xgb_age = joblib.load(MODELS_DIR / 'modelo_xgb_idade.joblib')
        
        pred_lr_age = model_lr_age.predict(df_idade)[0]
        pred_xgb_age = model_xgb_age.predict(df_idade)[0]
        
        return {
            'lr_dem': pred_lr_dem_label,
            'xgb_dem': pred_xgb_dem_label,
            'lr_age': f"{pred_lr_age:.1f}",
            'xgb_age': f"{pred_xgb_age:.1f}"
        }

    except Exception as e:
        print(f"Erro na predição: {e}")
        QMessageBox.critical(None, "Erro de Predição", f"Não foi possível carregar os modelos ou fazer a predição.\nVerifique se os arquivos .joblib existem em {MODELS_DIR}\n\nErro: {e}")
        return None

def backend_create_contour_overlay(slice_img, mask_img):
    """
    Usa Matplotlib para criar a imagem de contorno
    """
    try:
        # Normaliza a imagem original para exibição
        slice_norm = cv2.normalize(slice_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(slice_norm, cmap='gray') 
        
        # Desenha apenas o contorno da máscara em amarelo
        ax.contour(mask_img, levels=[0.5], colors='yellow', linewidths=1)
        
        ax.set_axis_off()
        
        # Salva a figura em um buffer de bytes em memória
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close(fig) # Fecha a figura para liberar memória
        
        buf.seek(0)
        return buf.getvalue() # Retorna os bytes da imagem PNG
    
    except Exception as e:
        print(f"Erro ao criar overlay do matplotlib: {e}")
        return None


# --- Funções de Treinamento Raso ---

def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.ravel().shape[0] == 4:
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
    else: # Caso só preveja uma classe
        specificity = 0 
    return specificity

def plot_confusion_matrix_backend(y_true, y_pred, title, filename):
    plt.figure(figsize=(7, 5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['NonDemented', 'Demented'],
                yticklabels=['NonDemented', 'Demented'])
    plt.title(title)
    plt.ylabel('Verdadeiro (Actual)')
    plt.xlabel('Predito (Predicted)')
    plt.savefig(OUT_DIR / f"{filename}.png")
    plt.close()
    print(f"Matriz de confusão salva em {OUT_DIR}/{filename}.png")

def plot_age_scatterplot_backend(y_true, y_pred, title, filename):
    plt.figure(figsize=(7, 7))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, s=100)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.title(title)
    plt.xlabel('Idade Real (Actual Age)')
    plt.ylabel('Idade Predita (Predicted Age)')
    plt.grid(True)
    plt.savefig(OUT_DIR / f"{filename}.png")
    plt.close()
    print(f"Gráfico de predição de idade salvo em {OUT_DIR}/{filename}.png")

def load_data_set_backend(set_name):
    """ Carrega os dados de treino/validação/teste """
    base_dir = DATABASE_DIR / set_name
    try:
        data = pd.read_csv(base_dir / 'features_full.csv', sep=';', decimal=',')
        data['Group_num'] = data['Group'].map({'NonDemented': 0, 'Demented': 1})
        data = data.dropna(subset=['Group_num', 'Age'])
        return data
    except Exception as e:
        print(f"Erro ao carregar features_full.csv de '{set_name}': {e}")
        return None

def backend_run_training():
    """
    Executa a lógica de treinamento completa do classificador raso
    """
    try:
        print("Iniciando treinamento...")
        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(OUT_DIR, exist_ok=True)

        # 1. Carregar dados
        df_train = load_data_set_backend('treino')
        df_val = load_data_set_backend('validacao')
        df_test = load_data_set_backend('teste')

        if df_train is None or df_val is None or df_test is None:
            raise Exception("Falha no carregamento dos dados de treino/validação/teste.")
        
        print(f"Dados carregados: Treino({df_train.shape}), Validação({df_val.shape}), Teste({df_test.shape})")

        # Juntar treino e validação para os modelos finais
        X_train_val_unprocessed = pd.concat([df_train, df_val])
        y_train_val_unprocessed_dem = X_train_val_unprocessed['Group_num']
        y_train_val_unprocessed_age = X_train_val_unprocessed['Age']

        # --- Tarefa 1: Classificação (Demência) ---
        print("\n--- Treinando Modelos de Classificação (Demência) ---")
        
        numeric_transformer_t1 = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        preprocessor_t1 = ColumnTransformer(
            transformers=[('num', numeric_transformer_t1, COLS_DEMENCIA)],
            remainder='drop'
        )

        # Modelo 1: Regressão Logística
        pipeline_lr_dem = Pipeline(steps=[
            ('preprocessor', preprocessor_t1),
            ('classifier', LogisticRegression(C=0.01, random_state=42, max_iter=1000, solver='liblinear')) 
        ])
        pipeline_lr_dem.fit(X_train_val_unprocessed, y_train_val_unprocessed_dem)
        joblib.dump(pipeline_lr_dem, MODELS_DIR / 'modelo_lr_demencia.joblib')
        print("Modelo (LR Demência) salvo.")
        
        # Avaliar LR no Teste
        y_pred_lr_t1 = pipeline_lr_dem.predict(df_test)
        plot_confusion_matrix_backend(df_test['Group_num'], y_pred_lr_t1, "Matriz de Confusão - LR (Treino)", "cm_lr_demencia")

        # Modelo 2: XGBoost Classifier
        pipeline_xgb_dem = Pipeline(steps=[
            ('preprocessor', preprocessor_t1),
            ('classifier', XGBClassifier(random_state=42, eval_metric='logloss'))
        ])
        pipeline_xgb_dem.fit(X_train_val_unprocessed, y_train_val_unprocessed_dem)
        joblib.dump(pipeline_xgb_dem, MODELS_DIR / 'modelo_xgb_demencia.joblib')
        print("Modelo (XGB Demência) salvo.")
        
        # Avaliar XGB no Teste
        y_pred_xgb_t1 = pipeline_xgb_dem.predict(df_test)
        plot_confusion_matrix_backend(df_test['Group_num'], y_pred_xgb_t1, "Matriz de Confusão - XGBoost (Treino)", "cm_xgb_demencia")

        # --- Tarefa 2: Regressão (Idade) ---
        print("\n--- Treinando Modelos de Regressão (Idade) ---")

        numeric_transformer_t2 = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        preprocessor_t2 = ColumnTransformer(
            transformers=[('num', numeric_transformer_t2, COLS_IDADE)],
            remainder='drop'
        )
        
        # Modelo 1: Regressão Linear
        pipeline_lr_age = Pipeline(steps=[
            ('preprocessor', preprocessor_t2),
            ('regressor', LinearRegression())
        ])
        pipeline_lr_age.fit(X_train_val_unprocessed, y_train_val_unprocessed_age)
        joblib.dump(pipeline_lr_age, MODELS_DIR / 'modelo_lr_idade.joblib')
        print("Modelo (LR Idade) salvo.")
        
        # Avaliar LR no Teste
        y_pred_lr_t2 = pipeline_lr_age.predict(df_test)
        plot_age_scatterplot_backend(df_test['Age'], y_pred_lr_t2, "Predição de Idade - Regressão Linear (Treino)", "scatter_lr_age")

        # Modelo 2: XGBoost Regressor
        pipeline_xgb_age = Pipeline(steps=[
            ('preprocessor', preprocessor_t2),
            ('regressor', XGBRegressor(random_state=42, eval_metric='rmse'))
        ])
        pipeline_xgb_age.fit(X_train_val_unprocessed, y_train_val_unprocessed_age)
        joblib.dump(pipeline_xgb_age, MODELS_DIR / 'modelo_xgb_idade.joblib')
        print("Modelo (XGB Idade) salvo.")
        
        # Avaliar XGB no Teste
        y_pred_xgb_t2 = pipeline_xgb_age.predict(df_test)
        plot_age_scatterplot_backend(df_test['Age'], y_pred_xgb_t2, "Predição de Idade - XGBoost (Treino)", "scatter_xgb_age")
        
        print("Treinamento concluído.")
        return True

    except Exception as e:
        print(f"Erro during o treinamento: {e}")
        QMessageBox.critical(None, "Erro no Treinamento", f"Ocorreu uma falha: {e}\n\nVerifique se os arquivos de 'database/treino' e 'database/validacao' existem.")
        return False

# --- FIM: LÓGICA DE BACKEND ---

# Classe auxiliar para visualização de imagem com zoom
class PhotoViewer(QGraphicsView):
    """
    Classe QGraphicsView personalizada para permitir zoom com a roda do mouse
    e arrastar (pan) com o clique.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self.setScene(self._scene)
        
        # Correção para PySide6
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag) # Permite arrastar
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

    def setPixmap(self, pixmap):
        self._pixmap_item.setPixmap(pixmap)
        # NÃO chama fitInView ou scale aqui
        # O resizeEvent cuidará disso
    
    def wheelEvent(self, event):
        """ Zoom in/out com a roda do mouse """
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor
        
        # Salva a posição da cena sob o mouse
        old_pos = self.mapToScene(event.position().toPoint())
        
        # Zoom
        if event.angleDelta().y() > 0:
            scale_factor = zoom_in_factor
        else:
            scale_factor = zoom_out_factor
        
        self.scale(scale_factor, scale_factor)
        
        # Reposiciona a cena para que o ponto sob o mouse permaneça lá
        new_pos = self.mapToScene(event.position().toPoint())
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())

    def resizeEvent(self, event):
        """
        Chamado quando o widget é redimensionado (incluindo a primeira
        vez que é exibido após o .exec() do diálogo).
        """
        # Ajusta a imagem para caber na nova visualização
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        # Chama a implementação base
        super().resizeEvent(event)

# --- APLICAÇÃO GUI ---

class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trabalho PAI - Segmentação e Classificação de Ventrículos Cerebrais")
        self.setGeometry(100, 100, 1200, 900) 

        # --- Caminhos da Aplicação ---
        self.base_dir = pathlib.Path(__file__).parent.resolve()
        self.models_dir = self.base_dir / "models"
        self.out_dir = self.base_dir / "out"
        self.database_dir = self.base_dir.parent / "database"
        self.oasis_csv_path = self.database_dir / "oasis_longitudinal_demographic.csv"

        # --- Variáveis de estado ---
        self.caminho_imagem_original = None
        
        # Armazena os pixmaps em alta resolução para o diálogo de zoom
        self.full_pixmap_original = None
        self.full_pixmap_processada = None
        self.full_pixmap_segmentada = None
        
        # --- Layout Principal ---
        widget_central = QWidget()
        self.setCentralWidget(widget_central)
        layout_principal = QVBoxLayout(widget_central)
        
        # 1. Barra Superior (Botões e Predições)
        layout_barra_superior = self.criar_barra_superior()
        layout_principal.addLayout(layout_barra_superior)
        
        linha_h = QFrame()
        linha_h.setFrameShape(QFrame.Shape.HLine)
        linha_h.setFrameShadow(QFrame.Shadow.Sunken)
        layout_principal.addWidget(linha_h)

        # 2. Área de Conteúdo (Imagens)
        layout_conteudo = QHBoxLayout()
        layout_principal.addLayout(layout_conteudo, stretch=1) 

        layout_imagens = QGridLayout()
        layout_imagens.setSpacing(10)
        
        self.label_img_original = self.criar_placeholder_imagem("Imagem Original")
        self.label_img_processada = self.criar_placeholder_imagem("Pré-Processamento")
        self.label_img_segmentada = self.criar_placeholder_imagem("Original + Contorno")
        
        layout_imagens.addWidget(self.label_img_original, 0, 0)
        layout_imagens.addWidget(self.label_img_processada, 0, 1)
        layout_imagens.addWidget(self.label_img_segmentada, 0, 2)
        
        layout_conteudo.addLayout(layout_imagens, stretch=3)

        # 3. Painel Inferior Dividido (Features + Terminal)
        layout_inferior = QHBoxLayout()
        
        # 3.1. Lado Esquerdo: Features
        grupo_features = QGroupBox("Features Extraídas")
        layout_features = QVBoxLayout(grupo_features)
        self.text_edit_features = QTextEdit()
        self.text_edit_features.setReadOnly(True)
        self.text_edit_features.setFontFamily("Courier")
        self.text_edit_features.setText("Nenhuma imagem carregada.")
        layout_features.addWidget(self.text_edit_features)
        
        # 3.2. Lado Direito: Terminal
        grupo_terminal = QGroupBox("Terminal (Debug)")
        layout_terminal = QVBoxLayout(grupo_terminal)
        self.text_edit_terminal = QTextEdit()
        self.text_edit_terminal.setReadOnly(True)
        self.text_edit_terminal.setFontFamily("Courier")
        layout_terminal.addWidget(self.text_edit_terminal)

        # Adiciona os dois grupos ao layout inferior
        layout_inferior.addWidget(grupo_features, 1) # 1 parte de stretch
        layout_inferior.addWidget(grupo_terminal, 1) # 1 parte de stretch
        
        # Widget container para o layout inferior com altura fixa
        widget_inferior = QWidget()
        widget_inferior.setLayout(layout_inferior)
        widget_inferior.setFixedHeight(200) # Altura fixa para a área
        
        layout_principal.addWidget(widget_inferior)

        # --- Redirecionamento do stdout/stderr para o terminal ---
        sys.stdout = EmittingStream(textWritten=self.atualizar_terminal)
        sys.stderr = EmittingStream(textWritten=self.atualizar_terminal)
        
        print("Interface iniciada. Selecione uma imagem para começar.")

        # --- Adicionando funcionalidade de zoom nas imagens ---
        self.label_img_original.mousePressEvent = lambda event: self.abrir_imagem_em_dialogo(self.full_pixmap_original)
        self.label_img_processada.mousePressEvent = lambda event: self.abrir_imagem_em_dialogo(self.full_pixmap_processada)
        self.label_img_segmentada.mousePressEvent = lambda event: self.abrir_imagem_em_dialogo(self.full_pixmap_segmentada)

    @Slot(str)
    def atualizar_terminal(self, text):
        """ Slot para receber o texto do stdout/stderr e adicionar ao terminal """
        cursor = self.text_edit_terminal.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.text_edit_terminal.setTextCursor(cursor)
        self.text_edit_terminal.ensureCursorVisible()

    def criar_placeholder_imagem(self, texto):
        label = QLabel(texto)
        label.setFrameShape(QFrame.Shape.Box)
        label.setFrameShadow(QFrame.Shadow.Sunken)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setMinimumSize(300, 300) 
        label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        return label

    def criar_barra_superior(self):
        """
        Cria a barra superior com botões e predições (layout de linha única).
        """
        layout_superior = QHBoxLayout()
        layout_botoes = QHBoxLayout()

        btn_selecionar_img = QPushButton("Selecionar Imagem")
        btn_selecionar_img.clicked.connect(self.selecionar_imagem)
        layout_botoes.addWidget(btn_selecionar_img)

        btn_menu_graficos = QPushButton("Gráficos")
        menu_principal_graficos = QMenu(self)
        menu_modelo_raso = QMenu("Modelo Raso", self)
        
        acao_cm_lr = QAction("CM - Reg. Logística (Demência)", self)
        acao_cm_lr.triggered.connect(lambda: self.mostrar_grafico("cm_lr_demencia.png"))
        acao_cm_xgb = QAction("CM - XGBoost (Demência)", self)
        acao_cm_xgb.triggered.connect(lambda: self.mostrar_grafico("cm_xgb_demencia.png"))
        acao_scatter_lr = QAction("Scatter - Reg. Logística (Idade)", self)
        acao_scatter_lr.triggered.connect(lambda: self.mostrar_grafico("scatter_lr_age.png"))
        acao_scatter_xgb = QAction("Scatter - XGBoost (Idade)", self)
        acao_scatter_xgb.triggered.connect(lambda: self.mostrar_grafico("scatter_xgb_age.png"))

        menu_modelo_raso.addAction(acao_cm_lr)
        menu_modelo_raso.addAction(acao_cm_xgb)
        menu_modelo_raso.addAction(acao_scatter_lr)
        menu_modelo_raso.addAction(acao_scatter_xgb)
        
        menu_principal_graficos.addMenu(menu_modelo_raso)
        btn_menu_graficos.setMenu(menu_principal_graficos)
        layout_botoes.addWidget(btn_menu_graficos)

        btn_treinar_modelos = QPushButton("Treinar Modelos")
        btn_treinar_modelos.clicked.connect(self.confirmar_treinamento)
        layout_botoes.addWidget(btn_treinar_modelos)
        
        layout_superior.addLayout(layout_botoes)
        layout_superior.addStretch(1) 

        # --- Layout de Predição (com Título à Esquerda) ---
        
        # 1. Layout "wrapper" para o título e a caixa
        layout_pred_wrapper = QHBoxLayout()

        # 2. Adiciona o Título à esquerda
        font_pred = QFont()
        font_pred.setPointSize(9) # Fonte bem pequena
        label_titulo_pred = QLabel("<b>Predição Raso:</b>")
        label_titulo_pred.setFont(font_pred)
        layout_pred_wrapper.addWidget(label_titulo_pred)

        # 3. Cria o QGroupBox sem título (apenas para a borda)
        grupo_predicoes = QGroupBox("")
        layout_predicoes = QHBoxLayout(grupo_predicoes) 
        
        # Estilo para os labels de resultado
        style_label_pred = "color: #00008B; font-weight: bold;"

        # --- Labels de resultado ---
        self.label_pred_idade_xgb = QLabel("N/A")
        self.label_pred_demencia_xgb = QLabel("N/A")
        self.label_pred_idade_xgb.setFont(font_pred)
        self.label_pred_demencia_xgb.setFont(font_pred)
        self.label_pred_idade_xgb.setStyleSheet(style_label_pred)
        self.label_pred_demencia_xgb.setStyleSheet(style_label_pred)

        self.label_pred_idade_lr = QLabel("N/A")
        self.label_pred_demencia_lr = QLabel("N/A")
        self.label_pred_idade_lr.setFont(font_pred)
        self.label_pred_demencia_lr.setFont(font_pred)
        self.label_pred_idade_lr.setStyleSheet(style_label_pred)
        self.label_pred_demencia_lr.setStyleSheet(style_label_pred)

        # --- Labels de formatação (parênteses, etc) ---
        label_xgb_open = QLabel("<b>XGB</b> (")
        label_xgb_open.setFont(font_pred)
        label_xgb_sep = QLabel("|")
        label_xgb_sep.setFont(font_pred)
        label_xgb_close = QLabel(")")
        label_xgb_close.setFont(font_pred)

        label_lr_open = QLabel("<b>Linear</b> (")
        label_lr_open.setFont(font_pred)
        label_lr_sep = QLabel("|")
        label_lr_sep.setFont(font_pred)
        label_lr_close = QLabel(")")
        label_lr_close.setFont(font_pred)

        # 4. Adiciona os widgets ao layout *interno* (layout_predicoes)
        layout_predicoes.addWidget(label_xgb_open)
        layout_predicoes.addWidget(self.label_pred_idade_xgb)
        layout_predicoes.addWidget(label_xgb_sep)
        layout_predicoes.addWidget(self.label_pred_demencia_xgb)
        layout_predicoes.addWidget(label_xgb_close)
        
        layout_predicoes.addSpacing(10) # Pequeno espaço entre os grupos

        layout_predicoes.addWidget(label_lr_open)
        layout_predicoes.addWidget(self.label_pred_idade_lr)
        layout_predicoes.addWidget(label_lr_sep)
        layout_predicoes.addWidget(self.label_pred_demencia_lr)
        layout_predicoes.addWidget(label_lr_close)
        
        layout_predicoes.setSpacing(3) # Espaçamento bem justo
        
        # 5. Adiciona o GroupBox (com os resultados) ao wrapper
        layout_pred_wrapper.addWidget(grupo_predicoes)
        
        # 6. Adiciona o wrapper (Título + Caixa) ao layout superior
        layout_superior.addLayout(layout_pred_wrapper)
        
        return layout_superior

    def selecionar_imagem(self):
        filtro = "Imagens (*.png *.jpg *.jpeg *.bmp *.nii *.nii.gz)"
        caminho_arquivo, _ = QFileDialog.getOpenFileName(self, "Selecionar Imagem", str(self.database_dir), filtro)
        
        if caminho_arquivo:
            self.caminho_imagem_original = caminho_arquivo
            # Limpa o terminal antes de rodar
            self.text_edit_terminal.clear() 
            print(f"Imagem selecionada: {caminho_arquivo}")
            self.rodar_fluxo_completo()

    def rodar_fluxo_completo(self):
        if not self.caminho_imagem_original:
            return

        try:
            # --- 1. Carregar Imagem (.nii.gz ou .png) ---
            print("Carregando e processando slice...")
            slice_original, mri_id = backend_load_nii_slice(self.caminho_imagem_original)
            
            # --- 2. Segmentação (do aaa.py) ---
            print("Segmentando ventrículos...")
            seg_result = backend_segmentar_ventriculos(slice_original)
            img_preprocessada = seg_result['pre_processamento']
            mask_ventriculos = seg_result['ventriculos'] # Array booleano

            # --- 3. Gerar Imagem de Contorno (como em aaa.py) ---
            print("Gerando overlay de contorno...")
            overlay_bytes = backend_create_contour_overlay(slice_original, mask_ventriculos)
            if overlay_bytes is None:
                raise Exception("Falha ao gerar imagem de contorno.")

            # --- 4. Extração de Features (do aaa.py / feature_extraction.ipynb) ---
            print("Extraindo features...")
            features_dict = backend_extract_features(mask_ventriculos)
            
            # --- 5. Buscar Metadados (Age/Group) ---
            print(f"Buscando metadados para {mri_id}...")
            metadata_dict = backend_get_metadata(mri_id)

            # --- 6. Predição (do shallow_classifier.ipynb) ---
            print("Realizando predições...")
            preds = backend_run_prediction(features_dict, metadata_dict)
            
            if preds is None:
                raise Exception("Falha na predição (verifique console).")

            # --- 7. Atualizar GUI ---
            
            # 7.1. Atualizar Labels de Predição (apenas os valores)
            self.label_pred_idade_xgb.setText(f"{preds['xgb_age']}")
            self.label_pred_demencia_xgb.setText(f"{preds['xgb_dem']}")
            self.label_pred_idade_lr.setText(f"{preds['lr_age']}")
            self.label_pred_demencia_lr.setText(f"{preds['lr_dem']}")

            # 7.2. Atualizar Imagens (Agora 3 imagens corretas)
            self.atualizar_label_imagem(self.label_img_original, slice_original)
            self.atualizar_label_imagem(self.label_img_processada, img_preprocessada)
            self.atualizar_label_imagem(self.label_img_segmentada, overlay_bytes) # Exibe o contorno

            # 7.3. Atualizar Texto de Features
            features_texto = f"ID: {mri_id}\n"
            features_texto += f"Idade (Real): {metadata_dict['Age']} | Grupo (Real): {metadata_dict['Group']}\n"
            features_texto += "--- Features do Ventrículo ---\n"
            for key, val in features_dict.items():
                features_texto += f"{key}: {val:.4f}\n"
            self.text_edit_features.setText(features_texto)
            print("Fluxo concluído com sucesso.")

        except Exception as e:
            QMessageBox.critical(self, "Erro no Processamento", f"Ocorreu um erro: {e}")
            print(f"Erro em rodar_fluxo_completo: {e}")
            # Limpar placeholders em caso de erro
            self.label_img_original.setText("Falha ao carregar")
            self.label_img_processada.setText("N/A")
            self.label_img_segmentada.setText("N/A")
            self.text_edit_features.setText(f"Erro: {e}")

    def atualizar_label_imagem(self, label, imagem_data):
        """
        Atualiza um QLabel com uma nova imagem.
        Aceita array NumPy (qualquer tipo) ou bytes (do matplotlib).
        """
        pixmap = QPixmap()
        q_image = None
        
        try:
            if isinstance(imagem_data, bytes):
                # Carrega a partir de bytes (para o overlay PNG)
                pixmap.loadFromData(imagem_data)

            elif hasattr(imagem_data, 'shape'): # É NumPy array
                # Garante que dados 2D (grayscale) sejam normalizados para 0-255 uint8
                if len(imagem_data.shape) == 2:
                    if imagem_data.dtype != np.uint8:
                        # Normaliza qualquer tipo (float, bool, int16) para uint8
                        imagem_data_norm = cv2.normalize(imagem_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    else:
                        imagem_data_norm = imagem_data
                    
                    h, w = imagem_data_norm.shape
                    bytes_per_line = 1 * w
                    q_image = QImage(imagem_data_norm.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
                
                elif len(imagem_data.shape) == 3: # 3-channel (assume BGR)
                    h, w, _ = imagem_data.shape
                    bytes_per_line = 3 * w
                    q_image = QImage(imagem_data.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
                
                if q_image:
                    pixmap = QPixmap.fromImage(q_image)

            # Armazena o pixmap original (em alta resolução) antes de escalar
            # para ser usado no diálogo de zoom.
            # Fazemos uma cópia para evitar problemas de referência
            temp_pixmap = pixmap.copy()
            if label == self.label_img_original:
                self.full_pixmap_original = temp_pixmap
            elif label == self.label_img_processada:
                self.full_pixmap_processada = temp_pixmap
            elif label == self.label_img_segmentada:
                self.full_pixmap_segmentada = temp_pixmap

            if pixmap and not pixmap.isNull():
                label.setPixmap(pixmap.scaled(
                    label.size(), 
                    Qt.AspectRatioMode.KeepAspectRatio, 
                    Qt.TransformationMode.SmoothTransformation
                ))
            else:
                label.setText("Erro ao carregar imagem")
        
        except Exception as e:
            print(f"Erro em 'atualizar_label_imagem': {e}")
            label.setText("Erro")

    def confirmar_treinamento(self):
        titulo = "Confirmar Treinamento"
        texto = "Este processo pode demorar vários minutos e irá sobrescrever os modelos existentes. Você deseja treinar os modelos rasos novamente?"
        resposta = QMessageBox.question(self, titulo, texto, 
                                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                        QMessageBox.StandardButton.No)
        
        if resposta == QMessageBox.StandardButton.Yes:
            self.text_edit_terminal.clear()
            print("Iniciando treinamento...")
            self.setDisabled(True) 
            QApplication.processEvents() 
            
            try:
                success = backend_run_training()
                if success:
                    QMessageBox.information(self, "Concluído", "Treinamento concluído com sucesso. Modelos e gráficos salvos.")
                else:
                    QMessageBox.warning(self, "Falha", "O treinamento falhou. Verifique o console para mais detalhes.")
            except Exception as e:
                 QMessageBox.critical(self, "Erro Crítico no Treinamento", f"Falha: {e}")
            
            self.setDisabled(False)

    def mostrar_grafico(self, nome_arquivo_img):
        try:
            caminho_abs = self.out_dir / nome_arquivo_img
            
            if not caminho_abs.exists():
                QMessageBox.warning(self, "Gráfico não encontrado", f"Arquivo de gráfico não encontrado: {caminho_abs}\nTente treinar os modelos primeiro para gerar os gráficos.")
                return

            dialog = QDialog(self)
            dialog.setWindowTitle(f"Gráfico - {nome_arquivo_img}")
            
            layout = QVBoxLayout()
            label_grafico = QLabel()
            pixmap = QPixmap(str(caminho_abs))
            
            if pixmap.width() > 800 or pixmap.height() > 600:
                pixmap = pixmap.scaled(800, 600, Qt.AspectRatioMode.KeepAspectRatio)
                
            label_grafico.setPixmap(pixmap)
            layout.addWidget(label_grafico)
            
            btn_ok = QPushButton("OK")
            btn_ok.clicked.connect(dialog.accept)
            layout.addWidget(btn_ok)
            
            dialog.setLayout(layout)
            dialog.exec()
            
        except Exception as e:
            QMessageBox.warning(self, "Erro ao Abrir Gráfico", f"Não foi possível carregar o gráfico: {nome_arquivo_img}\nErro: {e}")
            print(f"Erro ao carregar gráfico: {e}")

    def abrir_imagem_em_dialogo(self, pixmap):
        """
        Abre a imagem em um diálogo com QGraphicsView para permitir zoom.
        Recebe o PIXMAP DE ALTA RESOLUÇÃO.
        """
        if pixmap is None or pixmap.isNull():
            # Ação de clique antes da primeira imagem ser carregada
            QMessageBox.warning(self, "Erro", "Nenhuma imagem para exibir. Carregue uma imagem primeiro.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Visualizar Imagem (Role para Zoom, Arraste para Mover)")
        layout = QVBoxLayout()

        # Substitui QLabel por PhotoViewer
        viewer = PhotoViewer(self)
        viewer.setPixmap(pixmap) # Passa o pixmap de alta resolução
        viewer.setMinimumSize(600, 600) # Define um tamanho inicial razoável
        layout.addWidget(viewer)

        btn_fechar = QPushButton("Fechar")
        btn_fechar.clicked.connect(dialog.accept)
        layout.addWidget(btn_fechar)

        dialog.setLayout(layout)
        dialog.exec()


# --- Bloco de Execução Principal ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    janela_principal = ImageProcessingApp()
    janela_principal.show()
    sys.exit(app.exec())