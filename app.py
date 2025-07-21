import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load cross-validation scores
try:
    mean_cv_accuracy = joblib.load('mean_cv_accuracy.joblib')
    std_cv_accuracy = joblib.load('std_cv_accuracy.joblib')
except FileNotFoundError:
    mean_cv_accuracy = None
    std_cv_accuracy = None
except Exception as e:
    mean_cv_accuracy = None
    std_cv_accuracy = None

# Load test set accuracy
try:
    test_accuracy = joblib.load('test_accuracy.joblib')
except FileNotFoundError:
    test_accuracy = None
except Exception as e:
    test_accuracy = None

# Load the original dataset
try:
    # Load only a small sample or just the column names if memory is a concern
    # For preprocessing and comparison plots, the full dataset is needed.
    # Keep loading the full dataset for now as required for the comparison page (which will be removed later).
    obesidade_df = pd.read_csv('Obesity.csv')
except FileNotFoundError:
    st.error("Dataset file 'Obesity.csv' not found. Please ensure the dataset is in the same directory as app.py")
    st.stop() # Stop the app if the dataset is not found
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop() # Stop the app if there's an error loading the dataset

# Define the features used for training the model globally
# This list MUST match the order and names of features the model was trained on
# Based on the notebook, the training features were:
# X = obesidade_df.drop(['obesidade', 'peso', 'altura'], axis=1)
# And after feature engineering (like one-hot encoding Gender), the columns became:
# 'vegetais', 'ref_principais', 'agua', 'atv_fisica', 'atv_eletronica',
# 'idade', 'historico', 'al_calorico', 'fumante', 'ctrl_caloria',
# 'entre_ref', 'alcool', 'transporte', 'feminino', 'masculino'
colunas_features_for_model = [
    'vegetais', 'ref_principais', 'agua', 'atv_fisica', 'atv_eletronica',
    'idade', 'historico', 'al_calorico', 'fumante', 'ctrl_caloria',
    'entre_ref', 'alcool', 'transporte', 'feminino', 'masculino'
]

# Function to preprocess new input data (replicates the notebook preprocessing)
def preprocess_input_data(input_df):
    """Applies the same preprocessing steps as used for the training data."""
    processed_df = input_df.copy()

    # Apply transformations for binary features
    processed_df['historico'] = processed_df['family_history'].replace({'yes': 1, 'no': 0})
    processed_df['al_calorico'] = processed_df['FAVC'].replace({'yes': 1, 'no': 0})
    processed_df['fumante'] = processed_df['SMOKE'].replace({'yes': 1, 'no': 0})
    processed_df['ctrl_caloria'] = processed_df['SCC'].replace({'yes': 1, 'no': 0})
    processed_df['entre_ref'] = processed_df['CAEC'].replace({'Sometimes': 1, 'Frequently' : 1, 'Always': 1, 'no': 0})
    processed_df['alcool'] = processed_df['CALC'].replace({'Sometimes': 1, 'Frequently' : 1, 'Always': 1, 'no': 0})
    processed_df['transporte'] = processed_df['MTRANS'].replace({'Public_Transportation': 1, 'Automobile' : 1, 'Motorbike': 1, 'Bike': 0, 'Walking': 0})

    # Apply transformations for numerical/categorical features
    # Need to handle potential NaN values from number_input if not required and left empty
    processed_df['vegetais'] = processed_df['FCVC'].fillna(0).astype(int)
    processed_df['ref_principais'] = processed_df['NCP'].fillna(0).astype(int)
    processed_df['agua'] = processed_df['CH2O'].fillna(0).astype(int)
    processed_df['atv_fisica'] = processed_df['FAF'].fillna(0).astype(int)
    processed_df['atv_eletronica'] = processed_df['TUE'].fillna(0).astype(int)
    processed_df['idade'] = processed_df['Age'].fillna(0).astype(int)
    processed_df['peso'] = processed_df['Weight'].fillna(0).astype(int)
    processed_df['altura'] = processed_df['Height'].fillna(0).round(2)


    # Create binary gender columns
    processed_df['feminino'] = processed_df['Gender'].map({'Female': 1, 'Male': 0})
    processed_df['masculino'] = processed_df['Gender'].map({'Female': 0, 'Male': 1})
    # Handle potential NaN if Gender input was somehow missed (unlikely with radio buttons)
    processed_df['feminino'] = processed_df['feminino'].fillna(0).astype(int)
    processed_df['masculino'] = processed_df['masculino'].fillna(0).astype(int)


    # Drop the original columns that have been transformed or replaced
    cols_to_drop = ['Age', 'Weight', 'Height', 'NCP', 'FCVC', 'CH2O', 'FAF', 'TUE',
                       'family_history', 'FAVC', 'SMOKE', 'SCC', 'CAEC', 'CALC', 'MTRANS',
                       'Gender']
    processed_df.drop(cols_to_drop, axis=1, inplace=True, errors='ignore')


    # Ensure integer types where appropriate after replacements
    for col in ['historico', 'al_calorico', 'fumante', 'ctrl_caloria', 'entre_ref', 'alcool', 'transporte']: # Gender already handled
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].fillna(0).astype(int) # Fill NaN with 0 for these binary features


    return processed_df


# Function for the main prediction page
def main_page(model, mean_cv_accuracy, std_cv_accuracy, test_accuracy):
    st.title('Previsão de Obesidade')

    # Initialize session state for prediction status
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False

    st.header("Insira seus dados para a previsão de Obesidade")

    # Define the questions and widgets using original column names
    perguntas_amigaveis_widgets_original = {
        'Gender': {"pergunta": "Qual o seu gênero?", "tipo": "radio", "opcoes": {'Female': 'Feminino', 'Male': 'Masculino'}},
        'Age': {"pergunta": "Qual a sua idade? (inteiro): ", "tipo": "number_input", "min_value": 18, "step": 1},
        'family_history': {"pergunta": "Você tem histórico familiar de obesidade? ", "tipo": "radio", "opcoes": {'yes': 'Sim', 'no': 'Não'}},
        'FAVC': {"pergunta": "Você consome frequentemente alimentos calóricos? ", "tipo": "radio", "opcoes": {'yes': 'Sim', 'no': 'Não'}},
        'FCVC': {"pergunta": "Com que frequência você come vegetais? (1 a 3): ", "tipo": "number_input", "min_value": 1.0, "max_value": 3.0, "step": 0.1},
        'NCP': {"pergunta": "Quantas refeições principais você faz por dia? (1 a 4): ", "tipo": "number_input", "min_value": 1.0, "max_value": 4.0, "step": 0.1},
        'CH2O': {"pergunta": "Quantos litros de água você bebe por dia? (1 a 3): ", "tipo": "number_input", "min_value": 1.0, "max_value": 3.0, "step": 0.1},
        'SMOKE': {"pergunta": "Você é fumante? ", "tipo": "radio", "opcoes": {'yes': 'Sim', 'no': 'Não'}},
        'SCC': {"pergunta": "Você monitora a ingestão de calorias? ", "tipo": "radio", "opcoes": {'yes': 'Sim', 'no': 'Não'}},
        'CALC': {"pergunta": "Você consome álcool? ", "tipo": "radio", "opcoes": {'Sometimes': 'Às vezes', 'Frequently' : 'Frequentemente', 'Always': 'Sempre', 'no': 'Não'}},
        'MTRANS': {"pergunta": "Qual seu meio de transporte principal?", "tipo": "radio", "opcoes": {'Public_Transportation': 'Transporte Público', 'Automobile' : 'Automóvel', 'Motorbike': 'Moto', 'Bike': 'Bicicleta', 'Walking': 'Caminhada'}},
        'CAEC': {"pergunta": "Você come entre as refeições principais? ", "tipo": "radio", "opcoes": {'Sometimes': 'Às vezes', 'Frequently' : 'Frequentemente', 'Always': 'Sempre', 'no': 'Não'}},
        'TUE': {"pergunta": "Com que frequência você usa dispositivos eletrônicos para lazer? (0 a 2): ", "tipo": "number_input", "min_value": 0.0, "max_value": 2.0, "step": 0.1},
        'FAF': {"pergunta": "Com que frequência você pratica atividade física? (0 a 3): ", "tipo": "number_input", "min_value": 0.0, "max_value": 3.0, "step": 0.1},
        'Weight': {"pergunta": "Qual o seu peso em kg? (inteiro): ", "tipo": "number_input", "min_value": 0.0, "step": 1.0},
        'Height': {"pergunta": "Qual a sua altura em metros? (ex: 1.75): ", "tipo": "number_input", "min_value": 0.0, "format": "%.2f", "step": 0.01},
    }

    dados_entrada_original = {}

    # Add input widgets for all original features
    for coluna_original, widget_info in perguntas_amigaveis_widgets_original.items():
        pergunta = widget_info["pergunta"]
        tipo_widget = widget_info["tipo"]

        if tipo_widget == "number_input":
            min_value = widget_info.get("min_value")
            max_value = widget_info.get("max_value")
            step = widget_info.get("step")
            format_str = widget_info.get("format")
            # Use a consistent key for each widget
            dados_entrada_original[coluna_original] = st.number_input(pergunta, min_value=min_value, max_value=max_value, step=step, format=format_str, key=f'{coluna_original}_input')
        elif tipo_widget == "radio":
            opcoes_internal = list(widget_info["opcoes"].keys()) # Use keys (original values) as internal values
            opcoes_labels = list(widget_info["opcoes"].values()) # Use values (friendly labels) as labels
            selected_label = st.radio(pergunta, opcoes_labels, key=f'{coluna_original}_input')
            # Find the internal value corresponding to the selected label
            dados_entrada_original[coluna_original] = opcoes_internal[opcoes_labels.index(selected_label)]


    # Display test set accuracy below the input fields
    if test_accuracy is not None:
        st.subheader('Performance do Modelo no Conjunto de Teste:')
        st.write(f"Acurácia no Teste: **{test_accuracy:.2f}**")
        st.info("Este valor indica a performance do modelo em dados que ele não viu durante o treinamento.")


    # Add a button to trigger the prediction
    if st.button('Prever Obesidade', key='predict_button'):
        if model is not None:
            # Create a DataFrame with the original input data
            novo_dado_original_df = pd.DataFrame([dados_entrada_original])

            # Apply the same preprocessing steps to the new input data
            # Need to handle potential missing columns if the user didn't fill all inputs (though Streamlit widgets handle this by providing default values)
            try:
                novo_dado_processed = preprocess_input_data(novo_dado_original_df)

                # Ensure the processed data has the same columns and order as the training data
                # Use colunas_features_for_model defined globally after initial preprocessing
                novo_dado_processed = novo_dado_processed.reindex(columns=colunas_features_for_model, fill_value=0)

                # Make the prediction
                previsao = model.predict(novo_dado_processed)
                previsao_proba = model.predict_proba(novo_dado_processed)[:, 1]

                # Display the prediction result
                st.subheader('Resultado da Previsão:')
                if previsao[0] == 1:
                    st.write(f"A previsão é: **Obeso**")
                else:
                    st.write(f"A previsão é: **Não Obeso**")

                st.write(f"Probabilidade de ser Obeso: **{previsao_proba[0]:.2f}**")

                # Display overall model performance metrics below the prediction
                st.subheader('Performance Geral do Modelo (Cross-Validation):')
                if mean_cv_accuracy is not None and std_cv_accuracy is not None:
                     st.write(f"Acurácia Média: {mean_cv_accuracy:.2f} (+/- {std_cv_accuracy*2:.2f})")
                     st.info("Estes valores indicam a performance geral do modelo em diferentes subconjuntos dos dados, não a confiança desta previsão específica.")

                # Set session state to indicate a prediction has been made
                st.session_state.prediction_made = True

            except Exception as e:
                st.error(f"Error during prediction preprocessing: {e}")


        else:
            st.error("Model not loaded. Cannot make prediction.")


# Load the trained model
try:
    model = joblib.load('obesity_model.joblib')
except FileNotFoundError:
    st.error("Model file 'obesity_model.joblib' not found. Please ensure the trained model is saved in the same directory as app.py")
    model = None
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None


# Create a sidebar for navigation
st.sidebar.title('Navegação')
# Removed the second page option for comparison graphs
page = st.sidebar.radio('Ir para', ['Previsão de Obesidade'])

# Display the selected page (only one page now)
if page == 'Previsão de Obesidade':
    main_page(model, mean_cv_accuracy, std_cv_accuracy, test_accuracy)

# Removed the comparison_graphs_page function definition as it is no longer needed.
