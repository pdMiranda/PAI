import sys
import platform
import io
import os
import matplotlib
matplotlib.use('Agg') #usamos para salvar figuras sem abrir janelas
import matplotlib.pyplot as plt
from PySide6.QtCore import Qt
from PySide6.QtGui import (QAction,QFont,QPixmap,QImage,QPainter,QColor,QTextCursor)
from PySide6.QtWidgets import (QApplication,QMainWindow, QWidget, QTabWidget,QVBoxLayout, QHBoxLayout, QStatusBar, QFileDialog,QGraphicsView, QGraphicsScene, 
    QGraphicsPixmapItem,QLabel,QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,QSplitter, QTextEdit, QMessageBox,QMenuBar,QMenu, QDialog)
import numpy as np
import nibabel as nib
from PIL import Image, ImageQt
import pandas as pd
import joblib
from skimage.morphology import remove_small_objects, binary_opening, disk
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from skimage.filters import threshold_otsu, gaussian
from skimage.exposure import rescale_intensity, equalize_adapthist
from sklearn.cluster import KMeans

#Imports do Deep Learning (EfficientNet)
import random
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error
#Variáveis globais e contendo caminhos (paths)
ROOT = Path('.')
OUTPUT_DIR = ROOT / 'out'
MODELS_DIR = ROOT / 'models'
DB_ROOT = ROOT / 'database'
AXL_DIR = DB_ROOT / 'axl'
OASIS_CSV_PATH = DB_ROOT / 'oasis_longitudinal_demographic.csv'
MODEL_LR_DEMENCIA_PATH = MODELS_DIR / 'modelo_lr_demencia.joblib'
MODEL_XGB_DEMENCIA_PATH = MODELS_DIR / 'modelo_xgb_demencia.joblib'
MODEL_LR_IDADE_PATH = MODELS_DIR / 'modelo_lr_idade.joblib'
MODEL_XGB_IDADE_PATH = MODELS_DIR / 'modelo_xgb_idade.joblib'
MODEL_DL_DEMENCIA_PATH = MODELS_DIR / 'efficientnet_classification.keras'
MODEL_DL_IDADE_PATH = MODELS_DIR / 'efficientnet_age_regression.keras'
MODEL_DL_AGE_STATS_PATH = MODELS_DIR / 'age_min_max.npy'

#gráficos gerados pelo treinamento
CM_DL_DEMENCIA_PATH = OUTPUT_DIR / 'dl_confusion_matrix.png'
CURVES_DL_DEMENCIA_PATH = OUTPUT_DIR / 'dl_classification_curves.png'
SCATTER_DL_IDADE_PATH = OUTPUT_DIR / 'dl_age_scatter_plot.png'
CURVES_DL_IDADE_PATH = OUTPUT_DIR / 'dl_regression_curves.png'

#semente para reprodutibilidade
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

#hiperparâmetros de treino (DL)
IMG_SIZE   = 224
BATCH_SIZE = 16
EPOCHS     = 20 # Primeira fase
FINE_TUNE_EPOCHS = 15 # Segunda fase
LEARNING_RATE = 0.0001
LEARNING_RATE_FT = LEARNING_RATE / 10

#mapeamento de classes (DL)
CLASS_MAP = {'NonDemented': 0, 'Demented': 1}
CLASS_NAMES = list(CLASS_MAP.keys())

#constantes da segmentação
N_CLUSTERS = 4
MIN_AREA_VENTRICLE = 100
MAX_AREA_VENTRICLE = 15000
MIN_AREA_BRAIN = 5000
CENTER_TOLERANCE_RATIO = 0.2

def adicionar_paths(df, target_col='y'):
    df = df.copy()
    df['path'] = df['MRI ID'].apply(lambda mid: str(AXL_DIR / f"{mid}_axl.nii.gz"))
    return df[['path', target_col]]

def montar_matriz_img(path_tensor):
        path_str = path_tensor.numpy().decode('utf-8')
        img = nib.load(path_str).get_fdata()
        if img.ndim == 3:
            slice_z = img.shape[2] // 2
            img_slice = img[:, :, slice_z]
        elif img.ndim == 4:
            slice_z = img.shape[2] // 2
            img_slice = img[:, :, slice_z, 0]
        elif img.ndim == 2:
            img_slice = img
        else:
            img_slice = np.zeros((IMG_SIZE, IMG_SIZE))
        
        img_slice = np.rot90(img_slice)
        return img_slice.astype(np.float32)

def realizar_preprocess_input(path, label):
    image_data = tf.py_function(montar_matriz_img, [path], tf.float32)
    image_data.set_shape([None, None])
    #resolve bug divisão por zero
    image_data = image_data / (tf.reduce_max(image_data) + 1e-6)
    
    #redimensionar e conversão para rgb (3 canais)
    image_data = tf.expand_dims(image_data, axis=-1)
    image_data = tf.image.resize(image_data, [IMG_SIZE, IMG_SIZE])
    image_data = tf.image.grayscale_to_rgb(image_data)

    #pre-process do EfficientNet (normaliza conforme esperado pela rede)
    image_data = tf.keras.applications.efficientnet.preprocess_input(image_data)
    image_data.set_shape([IMG_SIZE, IMG_SIZE, 3])
    return image_data, label

def criar_dataset(df, batch_size=BATCH_SIZE, is_training=True):
    dataset = tf.data.Dataset.from_tensor_slices((df['path'], df['y']))
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(df), seed=SEED)
    dataset = dataset.map(realizar_preprocess_input, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def buildar_modelo(img_size=IMG_SIZE):
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size, img_size, 3))
    base_model.trainable = False
    inputs = base_model.input
    x= base_model.output
    x= GlobalAveragePooling2D()(x)
    x= Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
   
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=[
            'accuracy', 
            tf.keras.metrics.Recall(name='sensitivity'),
            tf.keras.metrics.TrueNegatives(name='tn'),
            tf.keras.metrics.FalsePositives(name='fp')
        ])
    return model, base_model

def buildar_modelo_regressao(img_size=IMG_SIZE):
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size, img_size, 3))
    base_model.trainable = False
    inputs= base_model.input
    x= base_model.output
    x= GlobalAveragePooling2D()(x)
    x= Dropout(0.3)(x)
    outputs= Dense(1, activation='sigmoid')(x)
    model= Model(inputs, outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='mean_absolute_error', 
        metrics=['mae', 'mse'] )
    return model, base_model

def plot_learning_curves_dl(history, save_path):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    metric_key= 'accuracy' if 'accuracy' in history.history else 'mae'
    val_metric_key= 'val_' + metric_key
    ax[0].plot(history.history[metric_key], label=f'{metric_key.capitalize()} (treino)')
    ax[0].plot(history.history[val_metric_key], label=f'{metric_key.capitalize()} (validação)')
    ax[0].set_title(f'Curva de {metric_key.capitalize()}')
    ax[0].set_xlabel('Epoca')
    ax[0].set_ylabel(metric_key.capitalize())
    ax[0].legend()
    # perda
    ax[1].plot(history.history['loss'], label='Perda (treino)')
    ax[1].plot(history.history['val_loss'], label='Perda (validação)')
    ax[1].set_title('Curva de perda')
    ax[1].set_xlabel('Epoca')
    ax[1].set_ylabel('Perda')
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Curvas de aprendizado salvas em: {save_path}")

def carregar_df_csv():
    try:
        df_oasis = pd.read_csv(OASIS_CSV_PATH, sep=';', decimal=',')
    except FileNotFoundError:
        print(f"FATAL ERROR: Arquivo CSV não encontrado em {OASIS_CSV_PATH}")
        return None
    df_oasis['Group'] = df_oasis['Group'].astype(str).str.strip()
    df_oasis = df_oasis[df_oasis['Group'].isin(['Nondemented', 'Demented', 'Converted'])].copy()
    df_oasis['Group'] = df_oasis['Group'].replace({'Nondemented': 'NonDemented', 'Converted': 'Demented'})
    df_oasis['MRI ID'] = df_oasis['MRI ID'].astype(str).str.strip()
    df_oasis['y'] = df_oasis['Group'].map(CLASS_MAP)
    print(f"Verificando arquivos em:{AXL_DIR}")
    all_files = set(f.name for f in AXL_DIR.glob("*.nii.gz"))
    df_oasis['exists'] = df_oasis['MRI ID'].apply(lambda mid: f"{mid}_axl.nii.gz" in all_files)
    df_oasis = df_oasis[df_oasis['exists']].copy()
    if df_oasis.empty:
        print(f"ERRO:Nenhum arquivo .nii.gz encontrado em {AXL_DIR} que corresponda ao arquivo CSV")
        return None
    print(df_oasis['Group'].value_counts())

    trainval_df, test_df = train_test_split(
        df_oasis,
        test_size=0.2,
        stratify=df_oasis['y'],
        random_state=SEED)
    train_df, val_df = train_test_split(
        trainval_df,
        test_size=0.2,
        stratify=trainval_df['y'],
        random_state=SEED)
    return train_df, val_df, test_df


def iniciar_treino_classificacao():
    print("-- DEBUG -- iniciando treinamento: classificação (demência)")
    data_split = carregar_df_csv()
    if data_split is None:
        return False, "Falha ao carregar dados para classsificação"
    train_df, val_df, test_df = data_split
    train_paths = adicionar_paths(train_df, 'y')
    val_paths   = adicionar_paths(val_df, 'y')
    test_paths  = adicionar_paths(test_df, 'y')
    train_ds = criar_dataset(train_paths, is_training=True)
    val_ds = criar_dataset(val_paths, is_training=False)
    test_ds = criar_dataset(test_paths, is_training=False)
    model, base_model = buildar_modelo()
    
    print("-- DEBUG -- iniciando o treinamento..")
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])

    print("-- DEBUG -- realizando o fine-tuning...")
    base_model.trainable = True
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE_FT),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Recall(name='sensitivity'), tf.keras.metrics.TrueNegatives(name='tn'), tf.keras.metrics.FalsePositives(name='fp')])
    
    TOTAL_EPOCHS = EPOCHS + FINE_TUNE_EPOCHS
    history_fine_tune = model.fit(
        train_ds,
        epochs=TOTAL_EPOCHS,
        initial_epoch=history.epoch[-1] if history.epoch else 0,
        validation_data=val_ds)
    print("Treinamento de classificação finalizado com sucesso!")
    
    # Combinar históricos para plotagem
    history.history['accuracy'].extend(history_fine_tune.history['accuracy'])
    history.history['val_accuracy'].extend(history_fine_tune.history['val_accuracy'])
    history.history['loss'].extend(history_fine_tune.history['loss'])
    history.history['val_loss'].extend(history_fine_tune.history['val_loss'])
    plot_learning_curves_dl(history, CURVES_DL_DEMENCIA_PATH)
    print("Salvando modelo de classificação..")
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(MODEL_DL_DEMENCIA_PATH)
    print("Realizando avaliação no conjunto de teste...")
    results = model.evaluate(test_ds)
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    y_pred_probs = model.predict(test_ds)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    print("Relatório de classificação:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    # Salvar matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots()
    disp.plot(cmap='Blues', ax=ax)
    plt.title("Matriz de confusão (EfficientNet)")
    plt.savefig(CM_DL_DEMENCIA_PATH)
    plt.close(fig)
    print(f"Matriz de confusão salva em: {CM_DL_DEMENCIA_PATH}")
    
    return True, "Treinamento de classificação DL finalizado!"

def iniciar_treino_regressao():
    print("-- DEBUG -- Iniciando treinamento: regressão(idade)")
    data_split = carregar_df_csv()
    if data_split is None:
        return False, "Falha ao carregar dados para regressão"
    
    train_df, val_df, test_df = data_split
    # Normalizar idade
    age_min = train_df['Age'].min()
    age_max = train_df['Age'].max()
    os.makedirs(MODELS_DIR, exist_ok=True)
    np.save(MODEL_DL_AGE_STATS_PATH, np.array([age_min, age_max]))
    print(f"idade min/max (treino): {age_min:.1f} / {age_max:.1f} Salvo em {MODEL_DL_AGE_STATS_PATH}")
    train_df['y'] = (train_df['Age'] - age_min) / (age_max - age_min)
    val_df['y'] = (val_df['Age'] - age_min) / (age_max - age_min)
    test_df['y'] = (test_df['Age'] - age_min) / (age_max - age_min)
    train_paths = adicionar_paths(train_df, 'y')
    val_paths   = adicionar_paths(val_df, 'y')
    test_paths  = adicionar_paths(test_df, 'y')
    train_ds_reg = criar_dataset(train_paths, is_training=True)
    val_ds_reg = criar_dataset(val_paths, is_training=False)
    test_ds_reg = criar_dataset(test_paths, is_training=False)
    reg_model, reg_base_model = buildar_modelo_regressao()

    print("-- DEBUG -- iniciando treinamento do regressor")
    reg_history = reg_model.fit(
        train_ds_reg,
        epochs=EPOCHS,
        validation_data=val_ds_reg,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
    )

    print("-- DEBUG -- realizando o fine-tuning")
    reg_base_model.trainable = True
    reg_model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE_FT),
        loss='mean_absolute_error',
        metrics=['mae', 'mse']
    )
    
    TOTAL_EPOCHS = EPOCHS + FINE_TUNE_EPOCHS
    reg_history_ft = reg_model.fit(
        train_ds_reg,
        epochs=TOTAL_EPOCHS,
        initial_epoch=reg_history.epoch[-1] if reg_history.epoch else 0,
        validation_data=val_ds_reg
    )
    
    print("Treinamento de regressão finlizado com sucesso!")
    reg_history.history['mae'].extend(reg_history_ft.history['mae'])
    reg_history.history['val_mae'].extend(reg_history_ft.history['val_mae'])
    reg_history.history['loss'].extend(reg_history_ft.history['loss'])
    reg_history.history['val_loss'].extend(reg_history_ft.history['val_loss'])
    
    plot_learning_curves_dl(reg_history, CURVES_DL_IDADE_PATH)
    
    print("Salvando modelo de regressao..")
    reg_model.save(MODEL_DL_IDADE_PATH)
    
    print("Avaliando regressor(idade) no conjunto de teste...")
    y_pred_norm = reg_model.predict(test_ds_reg).flatten()
    
    #Obter idades reais (sem normalizacao)
    y_true_real = test_df['Age'].values
    
    #Desnormalizar as predições
    y_pred_real = (y_pred_norm * (age_max - age_min)) + age_min

    mae_real = mean_absolute_error(y_true_real, y_pred_real)
    print(f"Erro médio absoluto (MAE) do teste: {mae_real:.2f}  anos")

    #Salvar gráfico de dispersão
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true_real, y_pred_real, alpha=0.6)
    plt.title(f'Idade real vs. prevista (EfficientNet)\n = {mae_real:.2f} anos')
    plt.xlabel('Idade real (em anos)')
    plt.ylabel('Idade prevista (em anos')
    lims = [min(age_min, np.min(y_true_real)), max(age_max, np.max(y_true_real))]
    plt.plot(lims, lims, 'r--', label='Previsão perfeita (y=x)')
    plt.legend()
    plt.grid(True)
    plt.savefig(SCATTER_DL_IDADE_PATH)
    plt.close()
    print(f"Gráfico de dispersão da regressão salvo em: {SCATTER_DL_IDADE_PATH}")
    
    return True, "Treinamento de regressão finalizado!"



def predict_single_classification_dl(nii_path):
    try:
        model = load_model(MODEL_DL_DEMENCIA_PATH)
    except Exception as e:
        print(f"Erro ao carregar {MODEL_DL_DEMENCIA_PATH}: {e}")
        return 0.0, "Erro: Modelo não encontrado!!!"

    df_tmp = pd.DataFrame({"path": [nii_path], "y": [0]}) # 'y' é um placeholder
    ds_tmp = criar_dataset(df_tmp, batch_size=1, is_training=False)
    
    prob = model.predict(ds_tmp, verbose=0)[0, 0]
    pred_label_int = 1 if prob >= 0.5 else 0
    pred_label_str = "Demented" if pred_label_int == 1 else "NonDemented"

    return prob, pred_label_str

def predict_single_age_dl(nii_path):
    try:
        reg_model = load_model(MODEL_DL_IDADE_PATH)
        age_min, age_max = np.load(MODEL_DL_AGE_STATS_PATH)
    except Exception as e:
        print(f"Erro ao  carregar {MODEL_DL_IDADE_PATH} ou {MODEL_DL_AGE_STATS_PATH} : {e}")
        return 0.0, 0.0

    df_tmp = pd.DataFrame({"path": [nii_path], "y": [0.0]}) # 'y' é um placeholder
    ds_tmp = criar_dataset(df_tmp, batch_size=1, is_training=False)

    age_norm = reg_model.predict(ds_tmp, verbose=0)[0, 0]
    age_real = (age_norm * (age_max - age_min)) + age_min

    return age_real, age_norm


class InterfaceGrafica(QMainWindow):
    def __init__(self):
        super().__init__()
        self.imagem_carregada = False
        self.dataframe = self.load_dataframe()
        if self.dataframe is None:
            print("Erro: Dataframe não encontrado. Favor verificar  se o dataframe está no caminho esperado")
        self.setGeometry(105, 105, 1400, 900)
        self.load_menu_bar()
        self.load_barra_status()
        self.gerar_abas()
        self.update_abas()

    def load_menu_bar(self):
        menu_bar = QMenuBar(self)
        self.setMenuBar(menu_bar)
        menu_modelo = menu_bar.addMenu("Treinamento")
        acao_treinar = QAction("Realizar treinamento", self)
        acao_treinar.triggered.connect(self.iniciar_treinamento)
        menu_modelo.addAction(acao_treinar)
        menu_modelo.addSeparator()
        acao_visualizar = QAction("Visualizar graficos de treinamento", self)
        acao_visualizar.triggered.connect(self.visualizar_performance)
        menu_modelo.addAction(acao_visualizar)

    def load_barra_status(self):
        self.barra_status = QStatusBar()
        self.setStatusBar(self.barra_status)
        self.barra_status.showMessage("Carregue uma imagem para iniciar o processamento")

    def gerar_abas(self):
        self.abas_centrais = QTabWidget(self)
        self.abas_centrais.currentChanged.connect(self.ao_trocar_aba)
        self.aba_visualizador = QWidget()
        self.gerar_visualizador_imagens()
        self.abas_centrais.addTab(self.aba_visualizador, "Carregar imagem")
        self.aba_segmentacao = QWidget()
        self._montar_aba_segmentacao()
        self.abas_centrais.addTab(self.aba_segmentacao, "Resultados ")
        self.setCentralWidget(self.abas_centrais)

    def gerar_visualizador_imagens(self):
        layout = QVBoxLayout(self.aba_visualizador)
        controles_layout = QHBoxLayout()
        # Criar botão para carregar a imagem
        self.btn_abrir_imagem = QPushButton("Carregar imagem")
        self.btn_abrir_imagem.clicked.connect(self.abrir_processar_imagem)
        controles_layout.addWidget(self.btn_abrir_imagem)
        controles_layout.addStretch()
        # Criar botões de zoom
        self.btn_zoom_in = QPushButton("Zoom (+)")
        self.btn_zoom_in.clicked.connect(self.aplicar_zoom_in)
        controles_layout.addWidget(self.btn_zoom_in)
        self.btn_zoom_out = QPushButton("Zoom (-)")
        self.btn_zoom_out.clicked.connect(self.aplicar_zoom_out)
        controles_layout.addWidget(self.btn_zoom_out)
        self.btn_reset_zoom = QPushButton("Resetar Zoom")
        self.btn_reset_zoom.clicked.connect(self.resetar_zoom)
        controles_layout.addWidget(self.btn_reset_zoom)
        layout.addLayout(controles_layout)
        self.cena_visualizador = QGraphicsScene(self)
        self.view_visualizador = QGraphicsView(self.cena_visualizador)
        self.view_visualizador.setRenderHint(QPainter.RenderHint.Antialiasing) 
        self.view_visualizador.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.item_pixmap_visualizador = QGraphicsPixmapItem()
        self.cena_visualizador.addItem(self.item_pixmap_visualizador)
        layout.addWidget(self.view_visualizador)
        # Desabilitar botões de zoom antes de carregar imagem
        self.btn_zoom_in.setEnabled(False)
        self.btn_zoom_out.setEnabled(False)
        self.btn_reset_zoom.setEnabled(False)
        #Aumentar o tamanho dos botões
        self.btn_abrir_imagem.setFixedSize(150, 40)
        self.btn_zoom_in.setFixedSize(200, 65)
        self.btn_zoom_out.setFixedSize(200, 65)
        self.btn_reset_zoom.setFixedSize(200, 65)

    def _criar_view_processamento(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        cena = QGraphicsScene(self)
        view = QGraphicsView(cena)
        view.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        view.setRenderHint(QPainter.RenderHint.Antialiasing)
        view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        layout.addWidget(view)
        widget.setContentsMargins(0,0,0,0)
        layout.setContentsMargins(0,0,0,0)
        return widget, cena, view

    def _montar_aba_segmentacao(self):
        layout = QVBoxLayout(self.aba_segmentacao) 
        divisor = QSplitter(Qt.Orientation.Vertical) 
        painel_superior = QWidget()
        layout_superior = QVBoxLayout(painel_superior)
        controles_zoom_layout = QHBoxLayout()
        controles_zoom_layout.addStretch()
        btn_zoom_in_seg = QPushButton("Zoom +")
        btn_zoom_in_seg.clicked.connect(self.aplicar_zoom_in)
        controles_zoom_layout.addWidget(btn_zoom_in_seg)
        btn_zoom_out_seg = QPushButton("Zoom -")
        btn_zoom_out_seg.clicked.connect(self.aplicar_zoom_out)
        controles_zoom_layout.addWidget(btn_zoom_out_seg)
        btn_reset_zoom_seg = QPushButton("Resetar zoom")
        btn_reset_zoom_seg.clicked.connect(self.resetar_zoom)
        controles_zoom_layout.addWidget(btn_reset_zoom_seg)
        layout_superior.addLayout(controles_zoom_layout)
        btn_zoom_in_seg.setFixedSize(200, 65)
        btn_zoom_out_seg.setFixedSize(200, 65)
        btn_reset_zoom_seg.setFixedSize(200, 65)
        
        # Abas que ficarão disponíveis após o processamento
        self.abas_processamento = QTabWidget()
        widget_orig_proc, self.cena_orig_proc, self.view_orig_proc = self._criar_view_processamento()
        self.item_pixmap_orig_proc = QGraphicsPixmapItem()
        self.cena_orig_proc.addItem(self.item_pixmap_orig_proc)
        self.abas_processamento.addTab(widget_orig_proc, "Original")
        widget_preproc, self.cena_preproc, self.view_preproc = self._criar_view_processamento()
        self.item_pixmap_preproc = QGraphicsPixmapItem()
        self.cena_preproc.addItem(self.item_pixmap_preproc)
        self.abas_processamento.addTab(widget_preproc, "Pré-processamento")
        widget_seg_final, self.cena_seg_final, self.view_seg_final = self._criar_view_processamento()
        self.item_pixmap_seg_final = QGraphicsPixmapItem()
        self.cena_seg_final.addItem(self.item_pixmap_seg_final)
        self.abas_processamento.addTab(widget_seg_final, "Segmentada")
        layout_superior.addWidget(self.abas_processamento)
        divisor.addWidget(painel_superior)
        
        painel_inferior = QWidget()
        layout_inferior = QVBoxLayout(painel_inferior)
        
        layout_inferior.addWidget(QLabel("Características extraídas (usadas nos modelos Joblib)"))
        self.tabela_caracteristicas = QTableWidget()
        self.tabela_caracteristicas.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tabela_caracteristicas.setAlternatingRowColors(True)
        layout_inferior.addWidget(self.tabela_caracteristicas, 1) 
        
        # Layout para os resultados lado a lado
        layout_resultados = QHBoxLayout()
        widget_joblib = QWidget()
        layout_joblib = QVBoxLayout(widget_joblib)
        layout_joblib.addWidget(QLabel("Resultados (Baseado em Features - Joblib):"))
        self.log_classificacao = QTextEdit()
        self.log_classificacao.setReadOnly(True)
        self.log_classificacao.setMaximumHeight(100)
        layout_joblib.addWidget(self.log_classificacao)
        self.log_regressao = QTextEdit()
        self.log_regressao.setReadOnly(True)
        self.log_regressao.setMaximumHeight(100)
        layout_joblib.addWidget(self.log_regressao)
        layout_resultados.addWidget(widget_joblib)

        # Coluna de resultados Deeplearning
        widget_dl = QWidget()
        layout_dl = QVBoxLayout(widget_dl)
        layout_dl.addWidget(QLabel("Resultados (Deep Learning - EfficientNet):"))
        self.log_classificacao_dl = QTextEdit()
        self.log_classificacao_dl.setReadOnly(True)
        self.log_classificacao_dl.setMaximumHeight(100)
        layout_dl.addWidget(self.log_classificacao_dl)
        self.log_regressao_dl = QTextEdit()
        self.log_regressao_dl.setReadOnly(True)
        self.log_regressao_dl.setMaximumHeight(100)
        layout_dl.addWidget(self.log_regressao_dl)
        layout_resultados.addWidget(widget_dl)

        layout_inferior.addLayout(layout_resultados)
        
        divisor.addWidget(painel_inferior)
        divisor.setSizes([600, 400])
        layout.addWidget(divisor)

    def load_dataframe(self):
        try:
            df_oasis = pd.read_csv(OASIS_CSV_PATH, sep=';', decimal=',')
            df_oasis['Group'] = df_oasis['Group'].map({
                'Nondemented': 'NonDemented', 
                'Demented': 'Demented', 
                'Converted': 'Converted'
            })
            df_oasis = df_oasis.dropna(subset=['Group'])
            print("Metadados OK")
            return df_oasis
        except FileNotFoundError:
            print(f"Erro {OASIS_CSV_PATH} não encontrado.")
            return None
    def processar_img_input(self, caminho_arquivo):
        colunas_idade = ['Group_num', 'Ventricle_Area', 'Ventricle_Perimeter', 'Ventricle_Circularity', 'Ventricle_Eccentricity', 'Ventricle_Solidity', 'Ventricle_MajorAxisLength']
        colunas_demencia = ['Age','Ventricle_Area', 'Ventricle_Perimeter', 'Ventricle_Circularity','Ventricle_Eccentricity', 'Ventricle_Solidity', 'Ventricle_MajorAxisLength']
            
        resultados = { "error": None }
        try:
            if self.dataframe is None:
                return {"error": f"Falha ao carregar CSV ({OASIS_CSV_PATH}). Verifique o caminho."}
            if caminho_arquivo.endswith(('.nii', '.nii.gz')):
                nii_img = nib.load(caminho_arquivo)
                data = nii_img.get_fdata()
                if data.ndim == 3:
                    slice_z = data.shape[2] // 2
                    image_slice = data[:, :, slice_z]
                elif data.ndim == 2:
                    image_slice = data
                elif data.ndim == 4:
                    slice_z = data.shape[2] // 2
                    image_slice = data[:, :, slice_z, 0]
                else:
                    raise ValueError(f"Dimensionalidade inesperada: {data.ndim}")
            else:
                pil = Image.open(caminho_arquivo).convert('L')
                image_slice = np.array(pil)
            
            image_slice = np.rot90(image_slice)
            resultados["image_slice_np"] = image_slice

            print("Executando segmentação de ventrículos...")
            seg_results = self.exec_segmentacao(image_slice)
            ventricle_mask = seg_results['ventriculos']
            preprocessed_img = seg_results['pre_processamento']
            
            resultados["pixmap_original"] = self.conversao_np_qpixmap(image_slice)
            resultados["pixmap_preproc"] = self.conversao_np_qpixmap(preprocessed_img)

            print("Extraindo features morfológicas...")
            features = self.extract_features(ventricle_mask)

            mri_id = self.get_id_img(caminho_arquivo)
            
            # Buscar metadados reais
            try:
                metadata = self.dataframe[self.dataframe['MRI ID'] == mri_id].iloc[0]
                subject_id = metadata['Subject ID']
                age = metadata['Age']
                group = metadata['Group'] 
            except IndexError:
                 return {"error": f"MRI ID '{mri_id}' não encontrado no arquivo CSV: {OASIS_CSV_PATH}."}

            data_dict = {
                'Subject ID': subject_id, 'MRI ID': mri_id, 'Group': group, 'Age': age,
                **features
            }
            df_single = pd.DataFrame([data_dict])
            df_single['Group_num'] = df_single['Group'].map({'NonDemented': 0, 'Demented': 1, 'Converted': 1})
            
            resultados["features_tabela"] = df_single[['Subject ID','MRI ID','Group','Age','Ventricle_Area','Ventricle_Perimeter','Ventricle_Circularity','Ventricle_Eccentricity','Ventricle_Solidity','Ventricle_MajorAxisLength']].copy()

            print("Executando predição (Joblib)...")
            X_demencia = df_single[colunas_demencia]
            model_lr_dem = joblib.load(MODEL_LR_DEMENCIA_PATH)
            model_xgb_dem = joblib.load(MODEL_XGB_DEMENCIA_PATH)
            pred_lr_dem_val = model_lr_dem.predict(X_demencia)[0]
            pred_xgb_dem_val = model_xgb_dem.predict(X_demencia)[0]
            pred_lr_dem_label = 'Demented' if pred_lr_dem_val == 1 else 'NonDemented'
            pred_xgb_dem_label = 'Demented' if pred_xgb_dem_val == 1 else 'NonDemented'
            
            resultados["reporte_classificacao"] = (
                f"Grupo Real: {group}\n"
                f"Predição (Regressão Logística): {pred_lr_dem_label}\n"
                f"Predição (XGBoost): {pred_xgb_dem_label}")

            X_idade = df_single[colunas_idade]
            model_lr_age = joblib.load(MODEL_LR_IDADE_PATH)
            model_xgb_age = joblib.load(MODEL_XGB_IDADE_PATH)
            pred_lr_age = model_lr_age.predict(X_idade)[0]
            pred_xgb_age = model_xgb_age.predict(X_idade)[0]
            
            resultados["reporte_regressao"] = (
                f"Idade Real: {age}\n"
                f"Idade Predita (Regressão Linear): {pred_lr_age:.2f}\n"
                f"Idade Predita (XGBoost): {pred_xgb_age:.2f}")

            print("Executando predição (Deep Learning)...")
            prob_dl, label_dl = predict_single_classification_dl(caminho_arquivo)
            age_dl, _ = predict_single_age_dl(caminho_arquivo)

            resultados["reporte_classificacao_dl"] = (
                f"Grupo Real: {group}\n"
                f"Predição (EfficientNet): {label_dl} (Prob: {prob_dl:.4f})")
            
            resultados["reporte_regressao_dl"] = (
                f"Idade Real: {age}\n"
                f"Idade Predita (EfficientNet): {age_dl:.2f} anos")

            #Gerar imagem segmentada
            fig, ax = plt.subplots(figsize=(6,6), dpi=100)
            ax.imshow(image_slice, cmap='gray')
            ax.contour(ventricle_mask, levels=[0.5], colors='yellow', linewidths=1)
            ax.set_axis_off()
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            buf.seek(0)
            pil_seg = Image.open(buf).convert('RGB')
            q_img_seg = ImageQt.ImageQt(pil_seg)
            resultados["pixmap_segmentada"] = QPixmap.fromImage(q_img_seg)
            
            return resultados
            
        except FileNotFoundError as e:
            return {"error": f"Modelo não encontrado: {e.filename}. Execute o treinamento."}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": f"Erro inesperado no processamento: {e}"}

    def treinar_modelo(self):
        """
        Executa o pipeline de treinamento completo para os modelos de Deep Learning.
        """
        print("Iniciando pipeline de treinamento Deep Learning...")
        os.makedirs(OUTPUT_DIR, exist_ok=True) 
        os.makedirs(MODELS_DIR, exist_ok=True) 
        
        try:
            # Treino classificação
            sucesso_class, msg_class = iniciar_treino_classificacao()
            if not sucesso_class:
                return False, f"Falha na classificação: {msg_class}"
            
            # Treino regressão
            sucesso_reg, msg_reg = iniciar_treino_regressao()
            if not sucesso_reg:
                return False, f"Falha na regressão: {msg_reg}"

            msg_final = f"Treinamento DL finalizado!\nClassificação: {msg_class}\nRegressão: {msg_reg}"
            print(msg_final)
            return True, msg_final

        except Exception as e:
            import traceback
            traceback.print_exc()
            msg_erro = f"Erro crítico durante o treinamento: {e}"
            print(msg_erro)
            return False, msg_erro

    
    def exec_segmentacao(self, image_slice, n_clusters=N_CLUSTERS, min_area_brain=MIN_AREA_BRAIN, min_area_ventricle=MIN_AREA_VENTRICLE, max_area_ventricle=MAX_AREA_VENTRICLE, center_tolerance_ratio=CENTER_TOLERANCE_RATIO):
        original_shape = image_slice.shape
        img_norm = rescale_intensity(image_slice, out_range=(0, 1))
        img_clahe = equalize_adapthist(img_norm)
        img_smooth = gaussian(img_clahe, sigma=1)
        t = threshold_otsu(img_smooth)
        mask_cerebro = img_smooth > t
        mask_cerebro = binary_opening(mask_cerebro, disk(3))
        mask_cerebro = remove_small_objects(mask_cerebro, min_size=min_area_brain) 
        mask_cerebro = binary_fill_holes(mask_cerebro)
        labels_cerebro = label(mask_cerebro)
        if labels_cerebro.max() == 0:
            return {'ventriculos': np.zeros(original_shape, dtype=bool), 'pre_processamento': img_smooth}
        maior_comp_label = np.argmax([region.area for region in regionprops(labels_cerebro)]) + 1
        mask_cerebro = (labels_cerebro == maior_comp_label)
        img_sem_cranio = img_smooth * mask_cerebro
        pixels_cerebro = img_sem_cranio[mask_cerebro].reshape(-1, 1)
        if pixels_cerebro.shape[0] < n_clusters:
            return {'ventriculos': np.zeros(original_shape, dtype=bool), 'pre_processamento': img_sem_cranio}
        kmeans = KMeans(n_clusters=n_clusters, random_state=64, n_init=10).fit(pixels_cerebro)
        centers = kmeans.cluster_centers_.flatten()
        labels_flat = kmeans.labels_
        sorted_indices = np.argsort(centers)
        indice_lcr = sorted_indices[0] 
        labels_kmeans = np.zeros(original_shape, dtype=int)
        labels_kmeans[mask_cerebro] = labels_flat + 1
        mask_lcr_total = (labels_kmeans == (indice_lcr + 1))
        dist_transform = distance_transform_edt(mask_lcr_total)
        labels_lcr = label(dist_transform)
        regioes_lcr = regionprops(labels_lcr)
        mask_ventriculos_proc = np.zeros(original_shape, dtype=bool)
        if not regioes_lcr:
            return {'ventriculos': np.zeros(original_shape, dtype=bool), 'pre_processamento': img_sem_cranio}
        center_r, center_c = np.array(original_shape) / 2
        max_dist_r = original_shape[0] * center_tolerance_ratio
        max_dist_c = original_shape[1] * center_tolerance_ratio
        for r in regioes_lcr:
            is_correct_size = r.area > min_area_ventricle and r.area < max_area_ventricle
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

    def extract_features(self, ventricle_mask):
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
        features = {
            'Ventricle_Area': total_area,
            'Ventricle_Perimeter': total_perimeter,
            'Ventricle_Circularity': np.mean(metrics_list['Circularity']),
            'Ventricle_Eccentricity': np.mean(metrics_list['Eccentricity']),
            'Ventricle_Solidity': np.mean(metrics_list['Solidity']),
            'Ventricle_MajorAxisLength': np.mean(metrics_list['MajorAxisLength'])
        }
        return features

    def get_id_img(self, nii_path):
        filename = os.path.basename(nii_path)
        mri_id_base = filename.split('.')[0]
        if mri_id_base.endswith('_axl'):
            mri_id = mri_id_base.removesuffix('_axl')
        else:
            mri_id = mri_id_base
        return mri_id

    def conversao_np_qpixmap(self, arr):
        if arr is None:
            return QPixmap()
        a = np.array(arr, copy=True)
        if a.dtype in [np.float32, np.float64]:
            if a.max() > a.min():
                a = (a - a.min()) / (a.max() - a.min()) * 255.0
            else:
                a = a * 255.0
        elif a.dtype == np.uint16:
            a = (a / 256)
        a = a.astype(np.uint8)
        pil_img = Image.fromarray(a, mode='L')
        qim = ImageQt.ImageQt(pil_img)
        return QPixmap.fromImage(qim)

    def update_abas(self):
        self.btn_abrir_imagem.setEnabled(True) 
        self.abas_centrais.setTabEnabled(1, self.imagem_carregada)

    def janela_erro(self, titulo, mensagem):
        QApplication.restoreOverrideCursor()
        self.barra_status.showMessage(f"Erro: {mensagem}", 10000)
        dialogo_erro = QMessageBox(self)
        dialogo_erro.setWindowTitle(titulo)
        dialogo_erro.setText(mensagem)
        dialogo_erro.setIcon(QMessageBox.Icon.Warning)
        dialogo_erro.exec()

    def limpar_resultados_antigos(self):
        self.item_pixmap_visualizador.setPixmap(QPixmap())
        self.item_pixmap_orig_proc.setPixmap(QPixmap())
        self.item_pixmap_preproc.setPixmap(QPixmap())
        self.item_pixmap_seg_final.setPixmap(QPixmap())
        self.tabela_caracteristicas.clear()
        self.tabela_caracteristicas.setRowCount(0)
        self.tabela_caracteristicas.setColumnCount(0)
        self.log_classificacao.clear()
        self.log_regressao.clear()
        self.log_classificacao_dl.clear()
        self.log_regressao_dl.clear()

    def preencher_tabela(self, df_tabela):
        if df_tabela is None or df_tabela.empty:
            self.tabela_caracteristicas.clear()
            return
            
        self.tabela_caracteristicas.setColumnCount(len(df_tabela.columns))
        self.tabela_caracteristicas.setHorizontalHeaderLabels(list(df_tabela.columns))
        self.tabela_caracteristicas.setRowCount(len(df_tabela))
        
        for i, (idx, row) in enumerate(df_tabela.iterrows()):
            for j, col in enumerate(df_tabela.columns):
                val = row[col]
                if isinstance(val, (float, np.floating)):
                    item = f"{val:,.4f}"
                else:
                    item = str(val)
                self.tabela_caracteristicas.setItem(i, j, QTableWidgetItem(item))
                
        self.tabela_caracteristicas.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tabela_caracteristicas.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    
    def iniciar_treinamento(self):
        confirmacao = QMessageBox.question(self, 
            "Confirmação de Treinamento (Deep Learning)", 
            "Este processo irá treinar os modelos de Deep Learning (EfficientNet).\n"
            "Isso pode levar MUITOS minutos (ou horas) e irá sobrescrever os modelos .keras e gráficos atuais.\n\n"
            "**Requer uma GPU e TensorFlow configurado corretamente.**\n\n"
            "Deseja continuar?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if confirmacao == QMessageBox.StandardButton.No:
            self.barra_status.showMessage("Treinamento cancelado.", 5000)
            return
        self.barra_status.showMessage("Iniciando treinamento Deep Learning... (verifique o console)")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents() 
        sucesso, mensagem = self.treinar_modelo()
        QApplication.restoreOverrideCursor()
        
        if sucesso:
            self.barra_status.showMessage("Treinamento DL finalizado!", 10000)
            QMessageBox.information(self, "Treinamento Concluído", mensagem)
            self.visualizar_performance()
        else:
            QMessageBox.critical(self, "Erro no Treinamento", mensagem)
            self.barra_status.showMessage(f"Erro no treinamento: {mensagem}", 5000)

    def visualizar_performance(self):
        dialogo = QDialog(self)
        dialogo.setWindowTitle("Performance do Modelo (Deep Learning)")
        dialogo.setMinimumSize(1000, 700)
        main_layout = QVBoxLayout(dialogo)
        abas = QTabWidget()
        main_layout.addWidget(abas, 1)

        # Aba de classificação
        tab_classificacao = QWidget()
        layout_class = QHBoxLayout(tab_classificacao)
        lbl_cm_dl = QLabel("Matriz de Confusão (EfficientNet)")
        lbl_cm_dl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_curves_dl = QLabel("Curvas de Aprendizado (EfficientNet)")
        lbl_curves_dl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout_class_left = QVBoxLayout()
        layout_class_left.addWidget(QLabel("Matriz de Confusão (Teste)"), 0, Qt.AlignmentFlag.AlignCenter)
        layout_class_left.addWidget(lbl_cm_dl, 1)
        layout_class.addLayout(layout_class_left)
        layout_class_right = QVBoxLayout()
        layout_class_right.addWidget(QLabel("Curvas de Aprendizado (Treino/Val)"), 0, Qt.AlignmentFlag.AlignCenter)
        layout_class_right.addWidget(lbl_curves_dl, 1)
        layout_class.addLayout(layout_class_right)
        abas.addTab(tab_classificacao, "Resultados da Classificação (Demência)")
        
        # Aba de regressão
        tab_regressao = QWidget()
        layout_reg = QHBoxLayout(tab_regressao)
        lbl_scatter_dl = QLabel("Dispersão Idade (EfficientNet)")
        lbl_scatter_dl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_curves_reg_dl = QLabel("Curvas de Aprendizado (EfficientNet)")
        lbl_curves_reg_dl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout_reg_left = QVBoxLayout()
        layout_reg_left.addWidget(QLabel("Dispersão Idade Real vs. Prevista (Teste)"), 0, Qt.AlignmentFlag.AlignCenter)
        layout_reg_left.addWidget(lbl_scatter_dl, 1)
        layout_reg.addLayout(layout_reg_left)
        layout_reg_right = QVBoxLayout()
        layout_reg_right.addWidget(QLabel("Curvas de Aprendizado (Treino/Val)"), 0, Qt.AlignmentFlag.AlignCenter)
        layout_reg_right.addWidget(lbl_curves_reg_dl, 1)
        layout_reg.addLayout(layout_reg_right)
        abas.addTab(tab_regressao, "Resultados da Regressão (Idade)")
        btn_fechar = QPushButton("Fechar")
        btn_fechar.clicked.connect(dialogo.accept) 
        main_layout.addWidget(btn_fechar, 0, Qt.AlignmentFlag.AlignCenter)

        # Carregar as imagens salvas pelo novo script de treino
        self.load_image(lbl_cm_dl, CM_DL_DEMENCIA_PATH)
        self.load_image(lbl_curves_dl, CURVES_DL_DEMENCIA_PATH)
        self.load_image(lbl_scatter_dl, SCATTER_DL_IDADE_PATH)
        self.load_image(lbl_curves_reg_dl, CURVES_DL_IDADE_PATH)
        dialogo.exec()

    def load_image(self, label_widget, path_imgs):
        path_str = str(path_imgs)
        pixmap = QPixmap(path_str)
        if pixmap.isNull():
            pixmap = self.criar_box_img_nao_encontrada(path_str)
        label_widget.setPixmap(pixmap.scaled(600,500,Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation))

    def criar_box_img_nao_encontrada(self, path_str):
        pixmap = QPixmap(600, 500)
        pixmap.fill(Qt.GlobalColor.white)
        painter = QPainter(pixmap)
        painter.setPen(QColor(200, 0, 0))
        painter.setFont(QFont("Arial", 10))
        text = f"Imagem não encontrada.\n\nExecute o treinamento para gerar o gráfico:\n{os.path.basename(path_str)}"
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, text)
        painter.end()
        return pixmap

    def abrir_processar_imagem(self):
        filtro = "Arquivos NIfTI (*.nii *.nii.gz);;Imagens Padrão (*.png *.jpg *.jpeg)"
        caminho, _ = QFileDialog.getOpenFileName(self, "Abrir imagem", "", filtro)
        if not caminho:
            return
        self.barra_status.showMessage("Realizando o carregamento e processamento...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()  
        self.limpar_resultados_antigos()
        
        resultados = self.processar_img_input(caminho)
        QApplication.restoreOverrideCursor()

        if resultados.get("error"):
            self.janela_erro("Erro no processamento", resultados["error"])
            self.imagem_carregada = False
            self.update_abas()
            return

        # Popular interface gráfica com os resultados
        pix_orig = resultados["pixmap_original"]
        self.item_pixmap_visualizador.setPixmap(pix_orig)
        self.cena_visualizador.setSceneRect(pix_orig.rect())
        self.view_visualizador.fitInView(pix_orig.rect(), Qt.AspectRatioMode.KeepAspectRatio)    
        self.item_pixmap_orig_proc.setPixmap(pix_orig)
        self.cena_orig_proc.setSceneRect(pix_orig.rect())
        self.view_orig_proc.fitInView(pix_orig.rect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.item_pixmap_preproc.setPixmap(resultados["pixmap_preproc"])
        self.cena_preproc.setSceneRect(resultados["pixmap_preproc"].rect())
        self.view_preproc.fitInView(resultados["pixmap_preproc"].rect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.item_pixmap_seg_final.setPixmap(resultados["pixmap_segmentada"])
        self.cena_seg_final.setSceneRect(resultados["pixmap_segmentada"].rect())
        self.view_seg_final.fitInView(resultados["pixmap_segmentada"].rect(), Qt.AspectRatioMode.KeepAspectRatio)

        # Preencher Tabela (Joblib)
        self.preencher_tabela(resultados["features_tabela"])
        
        # Preencher Logs (Joblib)
        self.log_classificacao.setPlainText(resultados["reporte_classificacao"])
        self.log_regressao.setPlainText(resultados["reporte_regressao"])
        self.log_classificacao.moveCursor(QTextCursor.MoveOperation.Start)
        self.log_regressao.moveCursor(QTextCursor.MoveOperation.Start)
        
        # Preencher Logs (Deep Learning)
        self.log_classificacao_dl.setPlainText(resultados.get("reporte_classificacao_dl", "N/A"))
        self.log_regressao_dl.setPlainText(resultados.get("reporte_regressao_dl", "N/A"))
        self.log_classificacao_dl.moveCursor(QTextCursor.MoveOperation.Start)
        self.log_regressao_dl.moveCursor(QTextCursor.MoveOperation.Start)
        self.imagem_carregada = True
        self.update_abas()       
        self.abas_centrais.setCurrentIndex(1)
        self.abas_processamento.setCurrentIndex(2)
        self.barra_status.showMessage(f"Processamento finalizado: {caminho}", 10000)
        self.btn_zoom_in.setEnabled(True)
        self.btn_zoom_out.setEnabled(True)
        self.btn_reset_zoom.setEnabled(True)

    def aplicar_zoom_in(self):
        view = self.get_current_view()
        if view:
            view.scale(1.2, 1.2)
            self.barra_status.showMessage("Zoom aumentado")
        else:
            self.barra_status.showMessage("Imagem não carregada")

    def aplicar_zoom_out(self):
        view = self.get_current_view()
        if view:
            view.scale(0.8, 0.8)
            self.barra_status.showMessage("Zoom reduzido")
        else:
            self.barra_status.showMessage("Imagem não carregada")

    def resetar_zoom(self):
        view = self.get_current_view()
        if view:
            view.fitInView(view.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.barra_status.showMessage("Zoom resetado")
        else:
            self.barra_status.showMessage("Imagem não carregada")

    def ao_trocar_aba(self, index):
        nome = self.abas_centrais.tabText(index)
        self.barra_status.showMessage(f"Aba ativa: {nome}")

    def get_current_view(self):
        widget_aba_ativa = self.abas_centrais.currentWidget()
        
        if widget_aba_ativa == self.aba_visualizador:
            self.view_visualizador.setFocus()
            return self.view_visualizador
        
        if widget_aba_ativa == self.aba_segmentacao:
            indice_sub_aba = self.abas_processamento.currentIndex()
            if indice_sub_aba == 0:
                self.view_orig_proc.setFocus()
                return self.view_orig_proc
            elif indice_sub_aba == 1:
                self.view_preproc.setFocus()
                return self.view_preproc
            elif indice_sub_aba == 2:
                self.view_seg_final.setFocus()
                return self.view_seg_final
        
        return None

#Init app
if __name__ == "__main__":
    #Garantir que os diretórios de saída e modelos existam
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


    app = QApplication(sys.argv)
    janela = InterfaceGrafica()
    janela.show()
    sys.exit(app.exec())