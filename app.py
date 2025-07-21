import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # Import numpy to handle potential numpy arrays from joblib

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
    obesidade_df = pd.read_csv('Obesity.csv')
    # Add a line to display column names for debugging
    st.write("Columns after loading CSV:", obesidade_df.columns.tolist())
except FileNotFoundError:
    st.error("Dataset file 'Obesity.csv' not found. Please ensure the dataset is in the same directory as app.py")
    st.stop() # Stop the app if the dataset is not found
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop() # Stop the app if there's an error loading the dataset

# Perform data preprocessing steps from the notebook
# These steps need to be executed after loading the raw data but before the pages are defined.
# Ensure original columns that are replaced or transformed are dropped.

try:
    # Create numerical features from original columns
    obesidade_df['vegetais'] = obesidade_df['FCVC'].astype(int)
    obesidade_df['ref_principais'] = obesidade_df['NCP'].astype(int)
    obesidade_df['agua'] = obesidade_df['CH2O'].astype(int)
    obesidade_df['atv_fisica'] = obesidade_df['FAF'].astype(int)
    obesidade_df['atv_eletronica'] = obesidade_df['TUE'].astype(int)
    obesidade_df['idade'] = obesidade_df['Age'].astype(int)
    # Keep peso and altura for potential future use, but drop the original columns
    obesidade_df['peso'] = obesidade_df['Weight'].astype(int)
    obesidade_df['altura'] = obesidade_df['Height'].round(2)

    # Handle categorical variables by replacing with numerical values
    obesidade_df['historico'] = obesidade_df['family_history'].replace({'yes': 1, 'no': 0})
    obesidade_df['al_calorico'] = obesidade_df['FAVC'].replace({'yes': 1, 'no': 0})
    obesidade_df['fumante'] = obesidade_df['SMOKE'].replace({'yes': 1, 'no': 0})
    obesidade_df['ctrl_caloria'] = obesidade_df['SCC'].replace({'yes': 1, 'no': 0})
    obesidade_df['entre_ref'] = obesidade_df['CAEC'].replace({'Sometimes': 1, 'Frequently' : 1, 'Always': 1, 'no': 0})
    obesidade_df['alcool'] = obesidade_df['CALC'].replace({'Sometimes': 1, 'Frequently' : 1, 'Always': 1, 'no': 0})
    obesidade_df['transporte'] = obesidade_df['MTRANS'].replace({'Public_Transportation': 1, 'Automobile' : 1, 'Motorbike': 1, 'Bike': 0, 'Walking': 0})

    # Create binary gender columns
    obesidade_df['feminino'] = obesidade_df['Gender'].map({'Female': 1, 'Male': 0})
    obesidade_df['masculino'] = obesidade_df['Gender'].map({'Female': 0, 'Male': 1})

    # Define the target variable 'obesidade' (binary) - Corrected column name to 'NObeyesdad'
    # Ensure 'NObeyesdad' exists before trying to access it
    if 'NObeyesdad' not in obesidade_df.columns:
         raise ValueError("Column 'NObeyesdad' not found in the dataset. Please check the CSV file.")
    obesidade_df['obesidade'] = obesidade_df['NObeyesdad'].replace({'Obesity_Type_III': 1, 'Obesity_Type_II' : 1, 'Obesity_Type_I': 1,  'Overweight_Level_II': 0, 'Overweight_Level_I': 0, 'Insufficient_Weight': 0, 'Normal_Weight': 0})

    # Ensure integer types where appropriate after replacements
    for col in ['historico', 'al_calorico', 'fumante', 'ctrl_caloria', 'entre_ref', 'alcool', 'transporte', 'feminino', 'masculino', 'obesidade']:
        if col in obesidade_df.columns:
            obesidade_df[col] = obesidade_df[col].astype(int)

    # Drop the original columns that have been transformed or replaced
    # Keep NObeyesdad temporarily to create 'obesidade', then drop
    cols_to_drop = ['Age', 'Weight', 'Height', 'NCP', 'FCVC', 'CH2O', 'FAF', 'TUE',
                       'family_history', 'FAVC', 'SMOKE', 'SCC', 'CAEC', 'CALC', 'MTRANS',
                       'Gender', 'NObeyesdad'] # Ensure 'NObeyesdad' is in the drop list AFTER it's used
    obesidade_df.drop(cols_to_drop, axis=1, inplace=True, errors='ignore') # Use errors='ignore' in case some columns were already dropped


except Exception as e:
    st.error(f"Error during data preprocessing: {e}")
    st.stop() # Stop the app if preprocessing fails

# Now define the functions using the preprocessed obesidade_df

# Function for the main prediction page
def main_page(model, mean_cv_accuracy, std_cv_accuracy, test_accuracy):
    st.title('Previsão de Obesidade')

    # Initialize session state for prediction status
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False

    st.header("Insira seus dados para a previsão de Obesidade")

    # Define the questions and widgets for input
    perguntas_amigaveis_widgets = {
        'vegetais': {"pergunta": "Com que frequência você come vegetais? (1 a 3): ", "tipo": "number_input", "min_value": 1, "max_value": 3, "step": 1},
        'ref_principais': {"pergunta": "Quantas refeições principais você faz por dia? (1 a 4): ", "tipo": "number_input", "min_value": 1, "max_value": 4, "step": 1},
        'agua': {"pergunta": "Quantos litros de água você bebe por dia? (1 a 3): ", "tipo": "number_input", "min_value": 1, "max_value": 3, "step": 1},
        'atv_fisica': {"pergunta": "Com que frequência você pratica atividade física? (0 a 3): ", "tipo": "number_input", "min_value": 0, "max_value": 3, "step": 1},
        'atv_eletronica': {"pergunta": "Com que frequência você usa dispositivos eletrônicos para lazer? (0 a 2): ", "tipo": "number_input", "min_value": 0, "max_value": 2, "step": 1},
        'idade': {"pergunta": "Qual a sua idade? (inteiro): ", "tipo": "number_input", "min_value": 18, "step": 1}, # Changed min_value to 18
        'historico': {"pergunta": "Você tem histórico familiar de obesidade? ", "tipo": "radio", "opcoes": {0: 'Não', 1: 'Sim'}},
        'al_calorico': {"pergunta": "Você consome frequentemente alimentos calóricos? ", "tipo": "radio", "opcoes": {0: 'Não', 1: 'Sim'}},
        'ctrl_caloria': {"pergunta": "Você monitora a ingestão de calorias? ", "tipo": "radio", "opcoes": {0: 'Não', 1: 'Sim'}},
        'entre_ref': {"pergunta": "Você come entre as refeições principais? ", "tipo": "radio", "opcoes": {0: 'Não', 1: 'Sim'}},
        'fumante': {"pergunta": "Você é fumante? ", "tipo": "radio", "opcoes": {0: 'Não', 1: 'Sim'}},
        'alcool': {"pergunta": "Você consome álcool? ", "tipo": "radio", "opcoes": {0: 'Não', 1: 'Sim'}},
        'transporte': {"pergunta": "Seu meio de transporte principal envolve caminhada ou bicicleta? ", "tipo": "radio", "opcoes": {0: 'Sim', 1: 'Não'}},
    }

    # Define the order of features expected by the model
    # This list MUST match the order and names of features the model was trained on
    # Based on the notebook, the training features were:
    # X = obesidade_df.drop(['obesidade', 'peso', 'altura'], axis=1)
    # And after feature engineering (like one-hot encoding Gender), the columns became:
    # 'vegetais', 'ref_principais', 'agua', 'atv_fisica', 'atv_eletronica',
    # 'idade', 'historico', 'al_calorico', 'fumante', 'ctrl_caloria',
    # 'entre_ref', 'alcool', 'transporte', 'feminino', 'masculino'
    colunas_features = [
        'vegetais', 'ref_principais', 'agua', 'atv_fisica', 'atv_eletronica',
        'idade', 'historico', 'al_calorico', 'fumante', 'ctrl_caloria',
        'entre_ref', 'alcool', 'transporte', 'feminino', 'masculino'
    ]


    dados_entrada = {}

    # Add a single input for Gender
    genero_selecionado = st.radio("Qual o seu gênero?", ['Feminino', 'Masculino'], key='genero_input_pred')

    # Map the single gender input back to the 'feminino' and 'masculino' columns
    if genero_selecionado == 'Feminino':
        dados_entrada['feminino'] = 1
        dados_entrada['masculino'] = 0
    else:
        dados_entrada['feminino'] = 0
        dados_entrada['masculino'] = 1

    # Add input widgets for all features except 'feminino' and 'masculino'
    for coluna in colunas_features:
        # Skip 'feminino' and 'masculino' as they are handled by the single gender input
        if coluna in ['feminino', 'masculino']:
            continue

        if coluna in perguntas_amigaveis_widgets:
            widget_info = perguntas_amigaveis_widgets[coluna]
            pergunta = widget_info["pergunta"]
            tipo_widget = widget_info["tipo"]

            if tipo_widget == "number_input":
                min_value = widget_info.get("min_value")
                max_value = widget_info.get("max_value")
                step = widget_info.get("step")
                format_str = widget_info.get("format")
                # Use a consistent key for each widget
                dados_entrada[coluna] = st.number_input(pergunta, min_value=min_value, max_value=max_value, step=step, format=format_str, key=f'{coluna}_input')
            elif tipo_widget == "radio":
                opcoes = list(widget_info["opcoes"].keys())
                opcoes_labels = list(widget_info["opcoes"].values())
                # Use a consistent key for each widget
                selected_label = st.radio(pergunta, opcoes_labels, key=f'{coluna}_input')
                dados_entrada[coluna] = opcoes[opcoes_labels.index(selected_label)]
        else:
            # Handle columns not in the friendly questions map, if any (shouldn't happen if colunas_features is correct)
            st.warning(f"Widget not defined for column: {coluna}. Using text input.")
            dados_entrada[coluna] = st.text_input(f"Enter value for '{coluna}': ", key=f'{coluna}_input_fallback')


    # Display test set accuracy below the input fields
    if test_accuracy is not None:
        st.subheader('Performance do Modelo no Conjunto de Teste:')
        st.write(f"Acurácia no Teste: **{test_accuracy:.2f}**")
        st.info("Este valor indica a performance do modelo em dados que ele não viu durante o treinamento.")


    # Add a button to trigger the prediction
    if st.button('Prever Obesidade', key='predict_button'):
        if model is not None:
            # Create a DataFrame with the input data
            # Ensure the order of columns matches the training data
            # Create the DataFrame directly from the dictionary
            novo_dado_df = pd.DataFrame([dados_entrada])
            # Reindex the columns to ensure they are in the correct order
            novo_dado_df = novo_dado_df.reindex(columns=colunas_features, fill_value=0)


            # Make the prediction
            previsao = model.predict(novo_dado_df)
            previsao_proba = model.predict_proba(novo_dado_df)[:, 1]

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

        else:
            st.error("Model not loaded. Cannot make prediction.")

# Function for the comparison graphs page
def comparison_graphs_page():
    st.title('Análise de Probabilidade de Obesidade por Fator')

    # Generate the necessary dataframes for graphs from the preprocessed obesidade_df
    try:
        # Use the preprocessed obesidade_df loaded and transformed globally

        obesidade_por_atv_fisica = obesidade_df.groupby('atv_fisica')['obesidade'].mean().reset_index()
        obesidade_por_al_calorico = obesidade_df.groupby('al_calorico')['obesidade'].mean().reset_index()
        obesidade_por_entre_ref = obesidade_df.groupby('entre_ref')['obesidade'].mean().reset_index()
        # For gender, use the binary columns already created
        obesidade_por_genero = obesidade_df.groupby(['feminino', 'masculino'])['obesidade'].mean().reset_index()
        obesidade_por_vegetais = obesidade_df.groupby('vegetais')['obesidade'].mean().reset_index()
        obesidade_por_ref_principais = obesidade_df.groupby('ref_principais')['obesidade'].mean().reset_index()
        obesidade_por_ctrl_caloria = obesidade_df.groupby('ctrl_caloria')['obesidade'].mean().reset_index()
        obesidade_por_agua = obesidade_df.groupby('agua')['obesidade'].mean().reset_index()
        obesidade_por_alcool = obesidade_df.groupby('alcool')['obesidade'].mean().reset_index()
        obesidade_por_transporte = obesidade_df.groupby('transporte')['obesidade'].mean().reset_index()


        # Display graphs
        st.write('**Probabilidade de Obesidade por Gênero:**')
        fig_gen, ax_gen = plt.subplots(figsize=(6, 4))
        # Ensure we handle cases where a gender might be missing in the data (though unlikely here)
        female_prob = obesidade_por_genero[obesidade_por_genero['feminino'] == 1]['obesidade'].iloc[0] if not obesidade_por_genero[obesidade_por_genero['feminino'] == 1].empty else 0
        male_prob = obesidade_por_genero[obesidade_por_genero['masculino'] == 1]['obesidade'].iloc[0] if not obesidade_por_genero[obesidade_por_genero['masculino'] == 1].empty else 0
        sns.barplot(x=['Feminino', 'Masculino'], y=[female_prob, male_prob], ax=ax_gen)
        ax_gen.set_title('Probabilidade de Obesidade por Gênero')
        ax_gen.set_ylabel('Probabilidade de Obesidade (Média)')
        ax_gen.set_ylim([0, 1])
        st.pyplot(fig_gen)
        plt.close(fig_gen)

        st.write('**Probabilidade de Obesidade por Nível de Atividade Física:**')
        fig_atv_comp, ax_atv_comp = plt.subplots(figsize=(8, 5))
        sns.barplot(x='atv_fisica', y='obesidade', data=obesidade_por_atv_fisica, ax=ax_atv_comp)
        ax_atv_comp.set_title('Probabilidade de Obesidade por Nível de Atividade Física')
        ax_atv_comp.set_xlabel('Nível de Atividade Física')
        ax_atv_comp.set_ylabel('Probabilidade de Obesidade (Média)')
        ax_atv_comp.set_ylim([0, 1])
        st.pyplot(fig_atv_comp)
        plt.close(fig_atv_comp)

        st.write('**Probabilidade de Obesidade por Consumo de Alimentos Calóricos:**')
        fig_cal_comp, ax_cal_comp = plt.subplots(figsize=(6, 4))
        # Ensure we handle cases where a category might be missing
        no_cal_prob = obesidade_por_al_calorico[obesidade_por_al_calorico['al_calorico'] == 0]['obesidade'].iloc[0] if not obesidade_por_al_calorico[obesidade_por_al_calico['al_calorico'] == 0].empty else 0
        yes_cal_prob = obesidade_por_al_calorico[obesidade_por_al_calorico['al_calorico'] == 1]['obesidade'].iloc[0] if not obesidade_por_al_calorico[obesidade_por_al_calorico['al_calorico'] == 1].empty else 0
        sns.barplot(x=['Não', 'Sim'], y=[no_cal_prob, yes_cal_prob], ax=ax_cal_comp)
        ax_cal_comp.set_title('Probabilidade de Obesidade por Consumo de Alimentos Calóricos')
        ax_cal_comp.set_xlabel('Consumo de Alimentos Calóricos')
        ax_cal_comp.set_ylabel('Probabilidade de Obesidade (Média)')
        ax_cal_comp.set_ylim([0, 1])
        st.pyplot(fig_cal_comp)
        plt.close(fig_cal_comp)

        st.write('**Probabilidade de Obesidade por Comer Entre as Refeições:**')
        fig_entre_comp, ax_entre_comp = plt.subplots(figsize=(6, 4))
        # Ensure we handle cases where a category might be missing
        no_entre_prob = obesidade_por_entre_ref[obesidade_por_entre_ref['entre_ref'] == 0]['obesidade'].iloc[0] if not obesidade_por_entre_ref[obesidade_por_entre_ref['entre_ref'] == 0].empty else 0
        yes_entre_prob = obesidade_por_entre_ref[obesidade_por_entre_ref['entre_ref'] == 1]['obesidade'].iloc[0] if not obesidade_por_entre_ref[obesidade_por_entre_ref['entre_ref'] == 1].empty else 0
        sns.barplot(x=['Não', 'Sim'], y=[no_entre_prob, yes_entre_prob], ax=ax_entre_comp)
        ax_entre_comp.set_title('Probabilidade de Obesidade por Comer Entre as Refeições')
        ax_entre_comp.set_xlabel('Comer Entre as Refeições')
        ax_entre_comp.set_ylabel('Probabilidade de Obesidade (Média)')
        ax_entre_comp.set_ylim([0, 1])
        st.pyplot(fig_entre_comp)
        plt.close(fig_entre_comp)

        st.write('**Probabilidade de Obesidade por Consumo de Vegetais:**')
        fig_veg, ax_veg = plt.subplots(figsize=(8, 5))
        sns.barplot(x='vegetais', y='obesidade', data=obesidade_por_vegetais, ax=ax_veg)
        ax_veg.set_title('Probabilidade de Obesidade por Consumo de Vegetais')
        ax_veg.set_xlabel('Consumo de Vegetais (vezes por dia)')
        ax_veg.set_ylabel('Probabilidade de Obesidade (Média)')
        ax_veg.set_ylim([0, 1])
        st.pyplot(fig_veg)
        plt.close(fig_veg)

        st.write('**Probabilidade de Obesidade por Quantidade de Refeições Principais:**')
        fig_ref, ax_ref = plt.subplots(figsize=(8, 5))
        sns.barplot(x='ref_principais', y='obesidade', data=obesidade_por_ref_principais, ax=ax_ref)
        ax_ref.set_title('Probabilidade de Obesidade por Quantidade de Refeições Principais')
        ax_ref.set_xlabel('Quantidade de Refeições Principais')
        ax_ref.set_ylabel('Probabilidade de Obesidade (Média)')
        ax_ref.set_ylim([0, 1])
        st.pyplot(fig_ref)
        plt.close(fig_ref)

        st.write('**Probabilidade de Obesidade por Controle de Calorias:**')
        fig_ctrl, ax_ctrl = plt.subplots(figsize=(6, 4))
        # Ensure we handle cases where a category might be missing
        no_ctrl_prob = obesidade_por_ctrl_caloria[obesidade_por_ctrl_caloria['ctrl_caloria'] == 0]['obesidade'].iloc[0] if not obesidade_por_ctrl_caloria[obesidade_por_ctrl_caloria['ctrl_caloria'] == 0].empty else 0
        yes_ctrl_prob = obesidade_por_ctrl_caloria[obesidade_por_ctrl_caloria['ctrl_caloria'] == 1]['obesidade'].iloc[0] if not obesidade_por_ctrl_caloria[obesidade_por_ctrl_caloria['ctrl_caloria'] == 1].empty else 0
        sns.barplot(x=['Não', 'Sim'], y=[no_ctrl_prob, yes_ctrl_prob], ax=ax_ctrl)
        ax_ctrl.set_title('Probabilidade de Obesidade por Controle de Calorias')
        ax_ctrl.set_xlabel('Controle de Calorias')
        ax_ctrl.set_ylabel('Probabilidade de Obesidade (Média)')
        ax_ctrl.set_ylim([0, 1])
        st.pyplot(fig_ctrl)
        plt.close(fig_ctrl)

        st.write('**Probabilidade de Obesidade por Consumo de Água:**')
        fig_agua, ax_agua = plt.subplots(figsize=(8, 5))
        sns.barplot(x='agua', y='obesidade', data=obesidade_por_agua, ax=ax_agua)
        ax_agua.set_title('Probabilidade de Obesidade por Consumo de Água')
        ax_agua.set_xlabel('Consumo de Água')
        ax_agua.set_ylabel('Probabilidade de Obesidade (Média)')
        ax_agua.set_ylim([0, 1])
        st.pyplot(fig_agua)
        plt.close(fig_agua)

        st.write('**Probabilidade de Obesidade por Consumo de Álcool:**')
        fig_alcool, ax_alcool = plt.subplots(figsize=(6, 4))
        # Ensure we handle cases where a category might be missing
        no_alcool_prob = obesidade_por_alcool[obesidade_por_alcool['alcool'] == 0]['obesidade'].iloc[0] if not obesidade_por_alcool[obesidade_por_alcool['alcool'] == 0].empty else 0
        yes_alcool_prob = obesidade_por_alcool[obesidade_por_alcool['alcool'] == 1]['obesidade'].iloc[0] if not obesidade_por_alcool[obesidade_por_alcool['alcool'] == 1].empty else 0
        sns.barplot(x=['Não', 'Sim'], y=[no_alcool_prob, yes_alcool_prob], ax=ax_alcool)
        ax_alcool.set_title('Probabilidade de Obesidade por Consumo de Álcool')
        ax_alcool.set_xlabel('Consumo de Álcool')
        ax_alcool.set_ylabel('Probabilidade de Obesidade (Média)')
        ax_alcool.set_ylim([0, 1])
        st.pyplot(fig_alcool)
        plt.close(fig_alcool)

        st.write('**Probabilidade de Obesidade por Meio de Transporte:**')
        fig_transporte, ax_transporte = plt.subplots(figsize=(6, 4))
        # Ensure we handle cases where a category might be missing
        walk_bike_prob = obesidade_por_transporte[obesidade_por_transporte['transporte'] == 0]['obesidade'].iloc[0] if not obesidade_por_transporte[obesidade_por_transporte['transporte'] == 0].empty else 0
        motorized_prob = obesidade_por_transporte[obesidade_por_transporte['transporte'] == 1]['obesidade'].iloc[0] if not obesidade_por_transporte[obesidade_por_transporte['transporte'] == 1].empty else 0
        sns.barplot(x=['Caminhar e Bicicleta', 'Transporte Público, Carro e Moto'], y=[walk_bike_prob, motorized_prob], ax=ax_transporte)
        ax_transporte.set_title('Probabilidade de Obesidade por Meio de Transporte')
        ax_transporte.set_xlabel('Meio de Transporte')
        ax_transporte.set_ylabel('Probabilidade de Obesidade (Média)')
        ax_transporte.set_ylim([0, 1])
        st.pyplot(fig_transporte)
        plt.close(fig_transporte)


    except FileNotFoundError:
        st.error("Could not load pre-calculated data for graphs. Please ensure the necessary CSV files are saved in the same directory as app.py")
        st.write("You can generate and save these files in your notebook using code like:")
        st.code("""
        obesidade_por_atv_fisica.to_csv('obesidade_por_atv_fisica.csv', index=False)
        obesidade_por_al_calorico.to_csv('obesidade_por_al_calorico.csv', index=False)
        obesidade_por_entre_ref.to_csv('obesidade_por_entre_ref.csv', index=False)
        # Add saving for other dataframes if needed for the comparison page
        obesidade_por_genero.to_csv('obesidade_por_genero.csv', index=False)
        obesidade_por_vegetais.to_csv('obesidade_por_vegetais.csv', index=False)
        obesidade_por_ref_principais.to_csv('obesidade_por_ref_principais.csv', index=False)
        obesidade_por_ctrl_caloria.to_csv('obesidade_por_ctrl_caloria.csv', index=False)
        obesidade_por_agua.to_csv('obesidade_por_agua.csv', index=False)
        obesidade_por_alcool.to_csv('obesidade_por_alcool.csv', index=False)
        obesidade_por_transporte.to_csv('obesidade_por_transporte.csv', index=False)
        """)
    except Exception as e:
        st.error(f"An error occurred while generating comparison graphs: {e}")


# Load the trained model
try:
    model = joblib.load('obesity_model.joblib')
except FileNotFoundError:
    st.error("Model file 'obesity_model.joblib' not found. Please ensure the trained model is saved in the same directory as app.py")
    model = None

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

# Create a sidebar for navigation
st.sidebar.title('Navegação')
page = st.sidebar.radio('Ir para', ['Previsão de Obesidade', 'Análise de Padrões Gerais'])

# Display the selected page
if page == 'Previsão de Obesidade':
    main_page(model, mean_cv_accuracy, std_cv_accuracy, test_accuracy)
elif page == 'Análise de Padrões Gerais':
    comparison_graphs_page()
