import json
import geopandas as gpd
import pandas as pd

# Caminhos dos arquivos
json_path = "id_json.json"
shp_path = "PONTO_PROPRIEDADE.shp"
out_path = "PONTO_PROPRIEDADE_JOIN.shp"

# 1) Carregar o JSON
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

items = data["results"]  # lista de propriedades

# 2) Montar tabelinha com ID, nome da propriedade, nome e CPF do proprietário
rows = []
for item in items:
    prop_id = item.get("_id")
    nome_prop = item.get("nomepropriedade") or item.get("nome")
    proprietarios = item.get("proprietarios") or []

    # escolhe o proprietário principal
    main_prop = None
    for p in proprietarios:
        if (p.get("tipo") or "").lower() == "proprietario":
            main_prop = p
            break
    if main_prop is None and proprietarios:
        main_prop = proprietarios[0]

    if main_prop:
        nome_pessoa = main_prop.get("nome")
        cpf = main_prop.get("cpf")
    else:
        nome_pessoa = None
        cpf = None

    rows.append({
        "_id": prop_id,
        "nm_propr": nome_prop,
        "prop_nome": nome_pessoa,
        "prop_cpf": cpf,
    })

df = pd.DataFrame(rows)

# 3) Carregar o SHP de pontos
gdf = gpd.read_file(shp_path)

# 4) Fazer o join: idpropried (SHP) ↔ _id (JSON)
merged = gdf.merge(df, how="left", left_on="idpropried", right_on="_id")

# Remove coluna auxiliar _id se não quiser
merged = merged.drop(columns=["_id"])

# 5) Salvar shapefile com os campos novos
merged.to_file(out_path)
