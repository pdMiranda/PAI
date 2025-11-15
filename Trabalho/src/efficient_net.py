# ===== 0) Setup e Config =====
import random, os
from pathlib import Path
import numpy as np
import pandas as pd

import tensorflow as tf

import nibabel as nib
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error
import matplotlib.pyplot as plt
from pathlib import Path

# Reprodutibilidade
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

ROOT = Path('../')

DB_ROOT = os.path.join(ROOT, 'database')
AXL_DIR = os.path.join(DB_ROOT, 'axl')
TRAIN_DIR = os.path.join(DB_ROOT, 'treino')
TEST_DIR  = os.path.join(DB_ROOT, 'teste')

OASIS_CSV = os.path.join(DB_ROOT, 'oasis_longitudinal_demographic.csv')

# Hiperparâmetros
IMG_SIZE   = 224
BATCH_SIZE = 16
EPOCHS     = 20
LEARNING_RATE = 0.0001

# Mapeamento de classes
CLASS_MAP = {'NonDemented': 0, 'Demented': 1}


def add_paths(df):
    df = df.copy()
    df['path'] = df['MRI ID'].apply(lambda mid: str(AXL_DIR / f"{mid}_axl.nii.gz"))
    df['mri'] = df['MRI ID']
    return df[['path', 'y', 'mri']]

def _load_nifti(path_tensor):
        # Funcao auxiliar para carregar o Nifti
        path_str = path_tensor.numpy().decode('utf-8')
        img = nib.load(path_str).get_fdata()
        return img.astype(np.float32)

def load_and_split_data():
    # ===== 1) Carregar OASIS e fazer splits =====

    df_oasis = pd.read_csv(OASIS_CSV, sep=';', decimal=',')

    # Normaliza classes
    df_oasis['Group'] = df_oasis['Group'].astype(str).str.strip()

    df_oasis = df_oasis[df_oasis['Group'].isin(['Nondemented', 'Demented'])].copy()
    df_oasis['Group'] = df_oasis['Group'].replace({'Nondemented': 'NonDemented'})

    df_oasis['MRI ID'] = df_oasis['MRI ID'].astype(str).str.strip()
    df_oasis['y'] = df_oasis['Group'].map(CLASS_MAP)

    print('Distribuição original (apenas NonDemented/Demented):')
    print(df_oasis['Group'].value_counts())

    print('\nApós cruzar com volumes existentes:')
    print(df_oasis['Group'].value_counts())

    trainval_df, test_df = train_test_split(
        df_oasis,
        test_size=0.2,
        stratify=df_oasis['y'],
        random_state=SEED
    )

    train_df, val_df = train_test_split(
        trainval_df,
        test_size=0.2,
        stratify=trainval_df['y'],
        random_state=SEED
    )

    print('\nTreino:')
    print(train_df['Group'].value_counts())
    print('\nValidação:')
    print(val_df['Group'].value_counts())
    print('\nTeste:')
    print(test_df['Group'].value_counts())
    
    train_paths = add_paths(train_df)
    val_paths   = add_paths(val_df)
    test_paths  = add_paths(test_df)

    print(f"\nTrain volumes: {len(train_paths)}")
    print(f"Val   volumes: {len(val_paths)}")
    print(f"Test  volumes: {len(test_paths)}")
    return train_df, val_df, test_df, train_paths, val_paths, test_paths
    
def load_and_preprocess_nifti(path, label):
    # ===== 2) Carregar e pré-processar o arquivo .nii.gz =====
    
    image_data = tf.py_function(_load_nifti, [path], tf.float32)
    
    # Definir o shape manualmente.
    image_data.set_shape([None, None])
    
    # 2. Normalizar
    image_data = image_data / tf.reduce_max(image_data)
    
    # 3. Redimensionar para o tamanho do EfficientNet
    image_data = tf.expand_dims(image_data, axis=-1)
    image_data = tf.image.resize(image_data, [IMG_SIZE, IMG_SIZE])
    
    # 4. Converter de 1 canal (Grayscale) para 3 canais (RGB)
    image_data = tf.image.grayscale_to_rgb(image_data)
    
    # 5. Aplicar pré-processamento do EfficientNet
    image_data = tf.keras.applications.efficientnet.preprocess_input(image_data)
    
    # Definir o shape final
    image_data.set_shape([IMG_SIZE, IMG_SIZE, 3])
    
    return image_data, label

def create_dataset(df, batch_size=BATCH_SIZE, is_training=True):
    dataset = tf.data.Dataset.from_tensor_slices((df['path'], df['y']))
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(df), seed=SEED)
    
    # Mapear a função de carregamento
    dataset = dataset.map(load_and_preprocess_nifti, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Agrupar em lotes
    dataset = dataset.batch(batch_size)
    
    # Otimizar performance
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset

def build_model(img_size=IMG_SIZE):
    # Carregar a base pré-treinada ImageNet
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False, # Não incluir o classificador final da ImageNet
        input_shape=(img_size, img_size, 3)
    )
    
    # Congelar a base
    base_model.trainable = False
    
    # Adicionar cabeçalho de classificação
    inputs = base_model.input
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x) # Dropout para regularização
    
    # Saída: 1 neurônio (Demented/NonDemented) com ativação sigmoid
    outputs = Dense(1, activation='sigmoid')(x)
    
    # Criar modelo
    model = Model(inputs, outputs)
    
    model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=[
        'accuracy', 
        tf.keras.metrics.Recall(name='sensitivity'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalsePositives(name='fp')
        ]
    )
    
    return model, base_model

def plot_learning_curves(history):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    
    # Gráfico de Acurácia
    ax[0].plot(history.history['accuracy'], label='Acurácia (Treino)')
    ax[0].plot(history.history['val_accuracy'], label='Acurácia (Validação)')
    ax[0].set_title('Curva de Acurácia')
    ax[0].set_xlabel('Época')
    ax[0].set_ylabel('Acurácia')
    ax[0].legend()
    
    # Gráfico de Perda
    ax[1].plot(history.history['loss'], label='Perda (Treino)')
    ax[1].plot(history.history['val_loss'], label='Perda (Validação)')
    ax[1].set_title('Curva de Perda')
    ax[1].set_xlabel('Época')
    ax[1].set_ylabel('Perda')
    ax[1].legend()
    
    plt.tight_layout()
    plt.show()

def train_efficientnet_model():
    train_df, val_df, test_df, train_paths, val_paths, test_paths = load_and_split_data()
    # Criar os datasets
    train_ds = create_dataset(train_paths, is_training=True)
    val_ds = create_dataset(val_paths, is_training=False)
    test_ds = create_dataset(test_paths, is_training=False)
    
    print("Pipelines de dados criados.")
    
    model, base_model = build_model()
    
    print("Iniciando o treinamento...")

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        ]
    )

    print("Treinamento concluído.")
    
    plot_learning_curves(history) # Pra exibir na interface grafica, tem q mudar aqui
    
    # --- FINE-TUNING ---

    print("Iniciando Fine-Tuning...")

    # Descongelar o modelo base
    base_model.trainable = True

    # Taxa de aprendizado menor
    LEARNING_RATE_FT = LEARNING_RATE / 10

    # Re-compilar o modelo com a nova taxa

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE_FT),
        loss='binary_crossentropy',
        metrics=[
            'accuracy', 
            tf.keras.metrics.Recall(name='sensitivity'),
            tf.keras.metrics.TrueNegatives(name='tn'),
            tf.keras.metrics.FalsePositives(name='fp')
        ]
    )

    print(f"Modelo re-compilado para fine-tuning com LR = {LEARNING_RATE_FT}")

    FINE_TUNE_EPOCHS = 15 
    TOTAL_EPOCHS = EPOCHS + FINE_TUNE_EPOCHS

    print("Iniciando fine-tuning...")

    history_fine_tune = model.fit(
        train_ds,
        epochs=TOTAL_EPOCHS,
        initial_epoch=history.epoch[-1],
        validation_data=val_ds
    )

    print("Fine-tuning concluído.")
    
    print("Salvando modelo...")
    os.makedirs(os.path.join(ROOT, 'models'), exist_ok=True) # Cria o diretório se não existir
    model.save(os.path.join(ROOT, 'models', 'efficientnet.keras'))
    
    print("Avaliando no conjunto de teste...")

    # Avaliação do treinamnento
    results = model.evaluate(test_ds)
    loss = results[0]
    accuracy = results[1]
    sensitivity = results[2]
    tn = results[3] # True Negatives
    fp = results[4] # False Positives

    # Calcular Especificidade
    specificity = tn / (tn + fp ) if (tn + fp) > 0 else 0.0

    print("\nMétricas de Teste:")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}  (TN={int(tn)}, FP={int(fp)})")


    # Predições para a matriz de confusão
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    y_pred_probs = model.predict(test_ds)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()

    # Calcular e exibir Matriz de Confusão
    print("\nMatriz de Confusão (Teste):")
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_MAP.keys())
    disp.plot(cmap='Blues')
    plt.show()

    # Relatório de Classificação
    print("\nRelatório de Classificação (Teste):")
    print(classification_report(y_true, y_pred, target_names=CLASS_MAP.keys()))
    
def add_paths_regression_norm(df, v_min, v_max):
    df = df.copy()
    df['path'] = df['MRI ID'].apply(lambda mid: str(AXL_DIR / f"{mid}_axl.nii.gz"))
    
    # === MUDANÇA AQUI: Normalizar 'Age' para [0, 1] ===
    df['y'] = (df['Age'] - v_min) / (v_max - v_min)
    
    # Guardar a idade original para avaliação futura
    df['age_original'] = df['Age'] 
    
    return df[['path', 'y', 'age_original']]

def build_regression_model(img_size=IMG_SIZE):
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )
    base_model.trainable = False
    
    inputs = base_model.input
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(1, activation='sigmoid')(x) 
    
    model = Model(inputs, outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='mean_absolute_error', 
        metrics=['mae', 'mse'] 
    )
    
    return model, base_model

def train_efficientnet_regression_model():
    train_df, val_df, test_df, train_paths, val_paths, test_paths = load_and_split_data()
    
    # Calcular Min/Max no conjunto de treino
    age_min = train_df['Age'].min()
    age_max = train_df['Age'].max()
    
    np.save(os.path.join(ROOT, 'models', "age_min_max.npy"), np.array([age_min, age_max]))

    print(f"Idade Mín (Treino): {age_min}")
    print(f"Idade Máx (Treino): {age_max}")
    
    # Criar os dataframes
    train_paths_reg = add_paths_regression_norm(train_df, age_min, age_max)
    val_paths_reg   = add_paths_regression_norm(val_df, age_min, age_max)
    test_paths_reg  = add_paths_regression_norm(test_df, age_min, age_max)

    # Criar os datasets
    train_ds_reg = create_dataset(train_paths_reg, is_training=True)
    val_ds_reg = create_dataset(val_paths_reg, is_training=False)

    test_ds_reg = create_dataset(test_paths_reg, is_training=False)

    print("\nPipelines de dados de Regressão criados.")
    reg_model, reg_base_model = build_regression_model()

    print("Iniciando treinamento do Regressor...")

    reg_history = reg_model.fit(
        train_ds_reg,
        epochs=EPOCHS,
        validation_data=val_ds_reg
    )

    print("Treinamento concluído.")
    
    print("Iniciando Fine-Tuning do Regressor de Idade...")

    # =Descongelar
    reg_base_model.trainable = True

    # =Re-compilar com LR baixa
    LEARNING_RATE_FT = LEARNING_RATE / 10
    reg_model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE_FT),
        loss='mean_absolute_error',
        metrics=['mae', 'mse']
    )
    
    FINE_TUNE_EPOCHS = 15 
    TOTAL_EPOCHS = EPOCHS + FINE_TUNE_EPOCHS
    reg_history_ft = reg_model.fit(
        train_ds_reg,
        epochs=TOTAL_EPOCHS,
        initial_epoch=reg_history.epoch[-1],
        validation_data=val_ds_reg
    )

    print("Fine-Tuning do Regressor concluído.")
    
    print("Salvando modelo de Regressão...")
    os.makedirs(os.path.join(ROOT, 'models'), exist_ok=True) # Cria o diretório se não existir
    reg_model.save(os.path.join(ROOT, 'models', 'efficientnet_age.keras'))
    
    print("Avaliando Regressor de Idade no conjunto de teste...")

    # Avaliação (na escala [0, 1])
    results_reg_norm = reg_model.evaluate(test_ds_reg)
    print("\nMétricas de Teste (na escala normalizada [0, 1]):")
    print(f"Loss (MAE): {results_reg_norm[0]:.4f}")

    # Predições (em [0, 1])
    y_pred_norm = reg_model.predict(test_ds_reg).flatten()

    # Obter idades reais (não-normalizadas)
    y_true_real = test_paths_reg['age_original'].values

    # Desnormalizar as predições
    y_pred_real = (y_pred_norm * (age_max - age_min)) + age_min

    # Calcular MAE real (em anos)
    mae_real = mean_absolute_error(y_true_real, y_pred_real)

    print(f"Erro Médio Absoluto (MAE): {mae_real:.2f} anos.")

    # Gráfico de Dispersão
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true_real, y_pred_real, alpha=0.6)
    plt.title('Idade Real vs. Idade Prevista')
    plt.xlabel('Idade Real (Anos)')
    plt.ylabel('Idade Prevista (Anos)')

    lims = [age_min, age_max]
    plt.plot(lims, lims, 'r--', label='Previsão Perfeita (y=x)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def predict_single_classification(nii_path):
    """
    Faz a classificação (Demented / NonDemented) de UM exame .nii.gz
    usando o modelo 'model' já treinado.
    """
    df_tmp = pd.DataFrame({
        "path": [nii_path],
        "y": [0]
    })

    # Reaproveita o mesmo pipeline de treino (sem shuffle)
    ds_tmp = create_dataset(df_tmp, batch_size=1, is_training=False)

    # Predição
    model = load_model(os.path.join(ROOT, 'models', 'efficientnet.keras'))
    
    prob = model.predict(ds_tmp)[0, 0]  # saída sigmoid
    pred_label = 1 if prob >= 0.5 else 0

    return prob, CLASS_MAP[pred_label]


def predict_single_age(nii_path):
    """
    Faz a predição de idade de UM exame .nii.gz
    usando o modelo 'reg_model' já treinado.
    """
    df_tmp = pd.DataFrame({
        "path": [nii_path],
        "y": [0.0]   # modelo só usa X
    })

    ds_tmp = create_dataset(df_tmp, batch_size=1, is_training=False)

    # Predição normalizada [0, 1]
    reg_model = load_model(os.path.join(ROOT, 'models', 'efficientnet_age.keras'))
    age_norm = reg_model.predict(ds_tmp)[0, 0]

    # Desnormalizar para anos reais
    age_min, age_max = np.load(os.path.join(ROOT, 'models', "age_min_max.npy"))
    age_real = (age_norm * (age_max - age_min)) + age_min

    return age_real, age_norm