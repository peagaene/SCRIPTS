import os
import pandas as pd
import re

# === Funções auxiliares ===

# Extrair IP da rede secundária
def extrair_ip_rede_secundaria(rede):
    if isinstance(rede, str):
        ip_match = re.findall(r'(\d+\.\d+\.\d+\.\d+)', rede)
        return ip_match[0] if ip_match else None
    return None

# Remover softwares indesejados e organizar resumo
def remover_softwares_indesejados(softwares_resumo, recursos_clientes):
    # Palavras-chave de softwares a remover
    palavras_chave = [
        'Autodesk Material Library', 'Aplicativos da Autodesk em destaque', 'Autodesk App Manager',
        'AutoCAD Open in Desktop', 'Bentley DGN', 'Autodesk Access', 'CONNECTION Client',
        'HDR Preview', 'Verificação de integridade do PC Windows', 'Lightshot', 'Autodesk Genuine Service',
        'ArcGIS Pro Help', 'Autodesk Network License Manager', 'OpenStudio CLI', 'Personal Accelerator',
        'Git', 'Intel(R) Management Engine', 'AMD Catalyst Control Center', '7-Zip', 'Trellix Endpoint Security',
        'QGIS', 'Node.js', 'Brother MFL-Pro Suite', 'ActivityWatch', 'OP Auto Clicker', 'PyCharm', 'APP Center',
        'SITECHCS', 'UXP WebView Support', 'Revo Uninstaller', 'Adobe Reader', 'FileZilla', 'Zoom', 'Minitab',
        'ENVI', 'Bandicam', 'WhatsApp', 'VLC', 'PostgreSQL', 'ApowerREC'
    ]

    # Remover softwares pela palavra-chave
    for palavra in palavras_chave:
        softwares_resumo = softwares_resumo[~softwares_resumo.index.str.contains(palavra, case=False, regex=True)]

    # Remover softwares da Google
    recursos_google = recursos_clientes[recursos_clientes['Software - Fornecedor'].str.contains('Google', case=False, na=False)]
    softwares_google = recursos_google['Software - Nome'].value_counts()
    softwares_resumo = softwares_resumo[~softwares_resumo.index.isin(softwares_google.index)]

    # Microsoft: manter somente Office
    recursos_microsoft = recursos_clientes[recursos_clientes['Software - Fornecedor'].str.contains('Microsoft', case=False, na=False)]
    recursos_office = recursos_microsoft[recursos_microsoft['Software - Nome'].str.contains('Office', case=False)]
    softwares_office = recursos_office['Software - Nome'].value_counts()

    # Adicionar Office ao resumo
    softwares_resumo = pd.concat([softwares_resumo, softwares_office])

    # Remover os outros softwares Microsoft que não são Office
    outros_ms = recursos_microsoft[~recursos_microsoft['Software - Nome'].str.contains('Office', case=False)]
    softwares_resumo = softwares_resumo[~softwares_resumo.index.isin(outros_ms['Software - Nome'].value_counts().index)]

    # Remover drivers
    palavras_chave_drivers = ['Driver', 'Catalyst', 'RMS License', 'Management Engine']
    for palavra in palavras_chave_drivers:
        softwares_resumo = softwares_resumo[~softwares_resumo.index.str.contains(palavra, case=False, regex=True)]

    # Filtrar MicroStation
    recursos_bentley = recursos_clientes[recursos_clientes['Software - Fornecedor'].str.contains('Bentley Systems', case=False, na=False)]
    microstation = recursos_bentley[recursos_bentley['Software - Nome'].isin(['MicroStation CONNECT Edition', 'MicroStation 2024'])].copy()
    
    # Status para MicroStation
    if len(microstation) > 1:
        microstation['Status'] = 'MicroStation CONNECT Edition e MicroStation 2024'
    else:
        microstation['Status'] = microstation['Software - Nome']

    microstation_resumo = microstation['Software - Nome'].value_counts()
    softwares_resumo = pd.concat([softwares_resumo, microstation_resumo])

    # Mantém softwares em mais de uma máquina
    softwares_resumo = softwares_resumo[softwares_resumo > 1]

    # Remover softwares filtrados da aba recursos clientes
    recursos_clientes_filtrados = recursos_clientes[~recursos_clientes['Software - Nome'].isin(softwares_resumo.index)]

    return softwares_resumo, microstation, recursos_clientes_filtrados

# === Caminho da planilha ===
file_path = r'D:\00_Pedro\AUXILIARES\Recursos.xlsx'
xls = pd.ExcelFile(file_path)
diretorio_entrada = os.path.dirname(file_path)

# Carregar aba 'Recursos por clientes' (cabeçalho na linha 5)
recursos_clientes = pd.read_excel(xls, sheet_name='Recursos por clientes', header=4)

# Normalizar coluna 'Rede'
recursos_clientes['Rede'] = recursos_clientes['Rede'].apply(lambda x: extrair_ip_rede_secundaria(str(x)))

# Resumo de softwares
softwares_resumo = recursos_clientes['Software - Nome'].value_counts()

# Filtrar e remover softwares indesejados
softwares_resumo, microstation, recursos_clientes_filtrados = remover_softwares_indesejados(softwares_resumo, recursos_clientes)

# Criar aba de Hardware
hardware_cols = ['Rede', 'VGA', 'Discos', 'Memória', 'Processador']
hardware_maquinas = recursos_clientes[[col for col in hardware_cols if col in recursos_clientes.columns]].copy()
# --- Remover linhas totalmente vazias ---
hardware_maquinas = hardware_maquinas.dropna(how='all')

# === Salvar a nova planilha ===
output_file = os.path.join(diretorio_entrada, 'resumo_software_maquinas.xlsx')

with pd.ExcelWriter(output_file) as writer:
    recursos_clientes_filtrados.to_excel(writer, sheet_name='Recursos por clientes', index=False)

    softwares_resumo_df = pd.DataFrame(softwares_resumo).reset_index()
    softwares_resumo_df.columns = ['Software', 'Quantidade']
    softwares_resumo_df.to_excel(writer, sheet_name='Resumo Software', index=False)

    microstation_ips = microstation[['Rede', 'Software - Nome', 'Status']]
    microstation_ips.to_excel(writer, sheet_name='MicroStation Maquinas', index=False)

    hardware_maquinas.to_excel(writer, sheet_name='Hardware Maquinas', index=False)

print(f"Planilha de resumo gerada com sucesso no caminho: {output_file}")
