"""
Dashboard en temps reel pour le sondage sur l'immigration francophone en Ontario.
Lit les donnees depuis un CSV (Google Sheet publie en CSV, ou fichier local pour test).
"""

import os
import re
import unicodedata
from collections import Counter

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import spaces

@spaces.GPU(duration=1)
def _verification_gpu_hf():
    # Fonction technique requise par Hugging Face pour les Spaces ZeroGPU.
    # Cette app n'a pas besoin de GPU, cette fonction n'est jamais appelee.
    pass


from scipy import stats as spstats

# =============================================================
# CONFIG
# =============================================================

SHEET_CSV_URL = os.environ.get("SHEET_CSV_URL", "https://docs.google.com/spreadsheets/d/1vLkVKIM_hsRLuFFoN8oje6rqtS4xWa_vMzPtmg7K9LA/export?format=csv&gid=1945629438")

COL_TIMESTAMP = "Horodateur"
COL_ANNEES_CANADA = "Depuis combien d'années êtes-vous au Canada ?"
COL_VILLE_RESIDENCE = "Dans quelle ville d'Ontario résidez-vous ?"
COL_STATUT_TRAVAIL = "Quel est votre statut actuel sur le marche du travail ?"
COL_VILLE_TRAVAIL = "Dans quelle ville Travailler vous ?"
COL_DELAI_EMPLOI = "Combien de temps vous a-t-il fallu pour trouver votre premier emploi au Canada ?"
COL_DOMAINE = "Le domaine d'études/profession"
COL_FORMATION_CA = "Avez-vous suivi une formation ou mise à niveau canadienne avant de trouver un emploi ?"
COL_NIVEAU_ANGLAIS = "Quel est votre niveau d'anglais ?"
COL_SERVICES = "Avez-vous eu recours a des services d'etablissement ou d'aide a l'emploi a votre arrivee ?"
COL_AGE = "Quelle est votre tranche d'âge ?"
COL_REGION_ORIGINE = "Quelle est votre région d'origine ?"
COL_GENRE = "Quel est votre genre ?"
COL_CONSEIL = "Un conseil que vous donneriez aux futurs arrivants ? (réponse longue, question facultative"

COL_OBSTACLES = {
    "Pas un obstacle": "Veuillez evaluer l'importance des obstacles suivants dans votre parcours [Pas un obstacle]",
    "Obstacle mineur": "Veuillez evaluer l'importance des obstacles suivants dans votre parcours [Obstacle mineur]",
    "Obstacle majeur": "Veuillez evaluer l'importance des obstacles suivants dans votre parcours [Obstacle majeur]",
}

# Colonnes texte a nettoyer (virgules parasites laissees par Google Forms sur les cases a cocher uniques)
COLONNES_A_NETTOYER = [
    COL_ANNEES_CANADA, COL_STATUT_TRAVAIL, COL_DELAI_EMPLOI, COL_VILLE_RESIDENCE,
    COL_VILLE_TRAVAIL, COL_DOMAINE, COL_FORMATION_CA, COL_NIVEAU_ANGLAIS,
    COL_SERVICES, COL_AGE, COL_REGION_ORIGINE, COL_GENRE,
]

# Ordre logique des delais (du plus rapide au plus long)
DELAI_ORDER = ["1 à 3 mois", "3 à 6 mois", "6 mois à 1 an", "Plus d'un an", "Toujours en recherche"]
DELAI_RAPIDE = {"1 à 3 mois", "3 à 6 mois"}

STOPWORDS_FR = {
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "a", "au", "aux",
    "en", "pour", "dans", "sur", "avec", "sans", "ce", "cet", "cette", "ces",
    "il", "elle", "ils", "elles", "je", "tu", "on", "nous", "vous", "que",
    "qui", "est", "sont", "etre", "avoir", "ont", "pas", "plus", "tres",
    "son", "sa", "ses", "leur", "leurs", "mon", "ma", "mes", "votre", "vos",
    "ne", "se", "si", "ou", "mais", "car", "donc", "par", "d", "l", "s",
    "n", "c", "j", "y", "vraiment", "meme", "aussi", "peu", "bien",
    "faut", "faire", "faites", "fait", "chose", "tout", "toute", "toutes",
    "tous", "cela", "ainsi", "dont", "etes", "etais", "sera", "seront",
}

# Mots propres au contexte du sondage (immigration/emploi) qui reviennent dans presque
# toutes les reponses sans etre informatifs : ils reprennent le vocabulaire de la
# question elle-meme plutot que d'apporter un signal distinctif.
STOPWORDS_DOMAINE = {
    "canada", "canadien", "canadienne", "canadiens", "canadiennes",
    "travail", "travailler", "travaillez",
    "emploi", "emplois",
    "temps", "avant", "lieu",
    "ontario",
    "arrivee", "arrivant", "arrivants",
    "conseil", "conseille", "conseillerais",
    "personne", "personnes",
}


def enlever_accents(mot):
    """Normalise un mot en retirant ses accents, pour comparer 'tres' et 'très' correctement."""
    return "".join(c for c in unicodedata.normalize("NFD", mot) if unicodedata.category(c) != "Mn")


STOPWORDS_FR_SANS_ACCENTS = {enlever_accents(m) for m in STOPWORDS_FR | STOPWORDS_DOMAINE}

# Palette de couleurs coherente
COULEUR_LIGNE = "#0891b2"
COULEUR_BARRE = "#2563eb"
COULEUR_OK = "#22c55e"
COULEUR_ATTENTION = "#f59e0b"
COULEUR_ALERTE = "#ef4444"
# Couleur neutre pour les graphiques descriptifs (region, statut) : la valeur est deja
# encodee par la hauteur/position, une echelle de couleur en plus serait redondante.
COULEUR_NEUTRE = "#94a3b8"
# Vert profond inspire du drapeau franco-ontarien (accent secondaire, identite communautaire).
COULEUR_VERT_FRANCO = "#1f4f3d"
PALETTE_DELAI = {
    "1 à 3 mois": "#ffedd5",
    "3 à 6 mois": "#fdba74",
    "6 mois à 1 an": "#fb923c",
    "Plus d'un an": "#ea580c",
    "Toujours en recherche": "#9a3412",
}
GRADIENT_HEATMAP = ["#fff1f2", "#e11d48", "#7f1d1d"]  # blanc/rose pale -> rouge vif -> bordeaux
TEMPLATE = "plotly_white"

# En dessous de ce nombre de reponses, un groupe est trop petit pour etre compare de
# maniere fiable : on le fusionne dans "Autres" (graphiques) ou on ne le compare pas
# (insights/tests). Seuil unique reutilise partout dans le dashboard pour la coherence.
SEUIL_MIN_GROUPE = 5
# Sous ce nombre de reponses, un groupe reste affiche mais avec une opacite reduite,
# pour signaler visuellement une fiabilite plus faible sans le masquer completement.
SEUIL_OPACITE_REDUITE = SEUIL_MIN_GROUPE * 2

# =============================================================
# Chargement et nettoyage des donnees
# =============================================================

def nettoyer_dataframe(df):
    for col in COLONNES_A_NETTOYER:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.rstrip(",").str.strip()
            df.loc[df[col].isin(["nan", "None", ""]), col] = pd.NA
    return df


def charger_donnees():
    df = pd.read_csv(SHEET_CSV_URL)
    df.columns = [c.strip() for c in df.columns]
    df = nettoyer_dataframe(df)
    return df


# =============================================================
# Filtres
# =============================================================

ORDRE_AGE = ["18-24 ans", "25-34 ans", "35-44 ans", "45-54 ans", "55 ans et plus"]
ORDRE_ANNEES = ["6 mois à 1 an", "1 à 2 ans", "2 à 5 ans", "plus que 5 ans"]


def trier_selon(liste, ordre_ref):
    dans_ordre = [x for x in ordre_ref if x in liste]
    hors_ordre = sorted(x for x in liste if x not in ordre_ref)
    return dans_ordre + hors_ordre


def obtenir_choix(df, col, ordre_ref=None):
    if col not in df.columns:
        return []
    valeurs = df[col].dropna().unique().tolist()
    if ordre_ref:
        return trier_selon(valeurs, ordre_ref)
    return sorted(valeurs)


try:
    _df_init = charger_donnees()
except Exception:
    _df_init = pd.DataFrame()

CHOIX_REGION = obtenir_choix(_df_init, COL_REGION_ORIGINE)
CHOIX_AGE = obtenir_choix(_df_init, COL_AGE, ORDRE_AGE)
CHOIX_ANNEES = obtenir_choix(_df_init, COL_ANNEES_CANADA, ORDRE_ANNEES)


def appliquer_filtres(df, regions, ages, genre, annees, domaine_travail=None):
    if regions:
        df = df[df[COL_REGION_ORIGINE].isin(regions)]
    if ages:
        df = df[df[COL_AGE].isin(ages)]
    if genre and genre != "Tous":
        df = df[df[COL_GENRE] == genre]
    if annees:
        df = df[df[COL_ANNEES_CANADA].isin(annees)]
    if domaine_travail and domaine_travail != "Tous":
        d = creer_domaine_binaire(df)
        df = d[d["_domaine_bin"] == domaine_travail].drop(columns=["_domaine_bin"])
    return df


# =============================================================
# Graphiques
# =============================================================

def fusionner_petits_groupes(serie, seuil=SEUIL_MIN_GROUPE):
    """Fusionne les categories comptant moins de `seuil` occurrences dans 'Autres',
    pour eviter des comparaisons peu fiables entre groupes de tailles tres differentes.
    Le label reste simple ; l'effectif total du groupe fusionne apparait ensuite via
    l'annotation "(n=X)" ajoutee par les graphiques appelants."""
    effectifs = serie.value_counts()
    petits = effectifs[effectifs < seuil].index.tolist()
    if not petits:
        return serie
    return serie.apply(lambda v: "Autres" if v in petits else v)


def fig_region(df):
    if COL_REGION_ORIGINE not in df.columns:
        return px.bar(title="Colonne region d'origine introuvable")
    c = df[COL_REGION_ORIGINE].dropna().value_counts().sort_values(ascending=False).reset_index()
    c.columns = ["region", "nombre"]
    c["pct"] = c["nombre"] / c["nombre"].sum() * 100
    fig = px.bar(
        c, x="region", y="nombre", template=TEMPLATE,
        color="nombre", color_continuous_scale=["#bfdbfe", "#1d4ed8"],
        custom_data=["pct"],
    )
    fig.update_traces(
        texttemplate="%{customdata[0]:.0f}%", textposition="outside",
        hovertemplate="%{x}<br>n=%{y} (%{customdata[0]:.0f}%)<extra></extra>",
    )
    fig.update_layout(coloraxis_showscale=False)
    return styliser_titre(fig, "Region d'origine")


def fig_delai_bar(df, col_groupe, titre):
    """Barres empilees en % (100% stacked) pour les croisements simples (peu de categories).
    Chaque groupe est normalise a 100% pour rester comparable meme si les groupes ont des
    tailles tres differentes ; le nombre de reponses (n) apparait dans l'etiquette et dans
    l'infobulle, et les groupes proches du seuil minimal sont affiches avec une opacite reduite."""
    if col_groupe not in df.columns or COL_DELAI_EMPLOI not in df.columns:
        return styliser_titre(px.bar(template=TEMPLATE), f"Donnees indisponibles: {titre}")
    sub = df[[col_groupe, COL_DELAI_EMPLOI]].dropna().copy()
    if sub.empty:
        return styliser_titre(px.bar(template=TEMPLATE), f"Pas encore de donnees: {titre}")

    sub[col_groupe] = fusionner_petits_groupes(sub[col_groupe])
    effectifs = sub[col_groupe].value_counts()
    ordre_groupe = effectifs.index.tolist()
    etiquettes = [f"{g} (n={effectifs[g]})" for g in ordre_groupe]

    ct = pd.crosstab(sub[col_groupe], sub[COL_DELAI_EMPLOI])
    colonnes = [c for c in DELAI_ORDER if c in ct.columns]
    ct = ct.reindex(index=ordre_groupe, columns=colonnes, fill_value=0)
    ct_pct = ct.div(effectifs, axis=0) * 100

    opacites = [0.55 if effectifs[g] < SEUIL_OPACITE_REDUITE else 1.0 for g in ordre_groupe]

    fig = go.Figure()
    for delai in colonnes:
        fig.add_trace(go.Bar(
            x=etiquettes, y=ct_pct[delai], name=delai,
            marker=dict(color=PALETTE_DELAI.get(delai, COULEUR_NEUTRE), opacity=opacites),
            customdata=ct[delai].values,
            hovertemplate=f"%{{x}}<br>{delai} : %{{y:.0f}}% (n=%{{customdata}})<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack", template=TEMPLATE,
        legend_title_text="Delai", legend_font_size=11,
        xaxis=dict(title=None),
        yaxis=dict(title="% des repondants", range=[0, 100], ticksuffix="%"),
    )
    return styliser_titre(fig, titre)


DELAI_GROUPE_MAP = {
    "1 à 3 mois": "Rapide (≤6 mois)",
    "3 à 6 mois": "Rapide (≤6 mois)",
    "6 mois à 1 an": "Moyen (6 mois-1 an)",
    "Plus d'un an": "Lent (+1 an)",
    "Toujours en recherche": "Lent (+1 an)",
}
DELAI_GROUPE_ORDER = ["Rapide (≤6 mois)", "Moyen (6 mois-1 an)", "Lent (+1 an)"]


def fig_delai_heatmap(df, col_groupe, titre, seuil_min_ligne=SEUIL_MIN_GROUPE):
    """Carte de chaleur en % par ligne pour les croisements avec beaucoup de categories
    (ex: secteur d'activite). Le delai est regroupe en 3 niveaux, les categories peu
    representees sont fusionnees dans 'Autres', et chaque ligne est normalisee en %
    pour rester comparable malgre des tailles de groupe tres differentes."""
    if col_groupe not in df.columns or COL_DELAI_EMPLOI not in df.columns:
        return styliser_titre(px.imshow([[0]], template=TEMPLATE), f"Donnees indisponibles: {titre}")
    sub = df[[col_groupe, COL_DELAI_EMPLOI]].dropna().copy()
    if sub.empty:
        return styliser_titre(px.imshow([[0]], template=TEMPLATE), f"Pas encore de donnees: {titre}")

    sub["delai_groupe"] = sub[COL_DELAI_EMPLOI].map(DELAI_GROUPE_MAP)
    sub[col_groupe] = fusionner_petits_groupes(sub[col_groupe], seuil_min_ligne)

    ct = pd.crosstab(sub[col_groupe], sub["delai_groupe"])
    colonnes_ordonnees = [c for c in DELAI_GROUPE_ORDER if c in ct.columns]
    ct = ct[colonnes_ordonnees]
    effectifs = ct.sum(axis=1).sort_values(ascending=False)
    ct = ct.loc[effectifs.index]
    ct_pct = ct.div(effectifs, axis=0) * 100
    etiquettes_lignes = [f"{idx} (n={effectifs[idx]})" for idx in ct.index]

    fig = px.imshow(
        ct_pct.values, x=ct_pct.columns, y=etiquettes_lignes,
        color_continuous_scale=GRADIENT_HEATMAP, zmin=0, zmax=100,
        aspect="auto", template=TEMPLATE,
    )
    fig.update_traces(
        texttemplate="%{z:.0f}%",
        customdata=ct.values,
        hovertemplate="%{y} / %{x}<br>%{z:.0f}% (n=%{customdata})<extra></extra>",
    )
    fig.update_layout(coloraxis_colorbar=dict(ticksuffix="%"))
    fig.update_xaxes(title=None, tickfont=dict(size=11))
    fig.update_yaxes(title=None, tickfont=dict(size=11))
    return styliser_titre(fig, titre)


def styliser_titre(fig, titre, taille=13):
    fig.update_layout(
        title=dict(text=titre, font=dict(size=taille), x=0.02, xanchor="left"),
        margin=dict(t=42, l=10, r=10, b=10),
    )
    # Desactive le zoom a la molette sur les graphiques, pour que la molette
    # continue de faire defiler la page normalement au lieu d'etre captee par Plotly.
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def diviser_obstacles(cellule):
    """Decoupe une cellule contenant plusieurs obstacles.
    Google Forms a parfois utilise ';' et parfois ',' comme separateur
    selon les reponses -> on gere les deux pour ne rater aucun cas."""
    return [item.strip() for item in re.split(r"[;,]", str(cellule)) if item.strip()]


def fig_obstacles_complet(df):
    """Barres empilees en % : chaque obstacle est normalise a 100% des repondants qui
    l'ont evalue, pour montrer la composition (part majeur/mineur/pas un obstacle)
    plutot que des comptes bruts."""
    lignes = []
    for niveau, col in COL_OBSTACLES.items():
        if col not in df.columns:
            continue
        for cell in df[col].dropna():
            for item in diviser_obstacles(cell):
                lignes.append({"obstacle": item, "niveau": niveau})

    if not lignes:
        return px.bar(title="Aucune donnee obstacle")

    df_obs = pd.DataFrame(lignes)
    ordre_niveaux = [n for n in ["Pas un obstacle", "Obstacle mineur", "Obstacle majeur"] if n in df_obs["niveau"].unique()]
    pivot = df_obs.groupby(["obstacle", "niveau"]).size().unstack(fill_value=0).reindex(columns=ordre_niveaux, fill_value=0)

    effectifs = pivot.sum(axis=1)
    majeurs = pivot["Obstacle majeur"] if "Obstacle majeur" in pivot.columns else pd.Series(0, index=pivot.index)
    ordre_obstacles = majeurs.sort_values(ascending=True).index.tolist()
    pivot = pivot.loc[ordre_obstacles]
    effectifs = effectifs.loc[ordre_obstacles]
    pivot_pct = pivot.div(effectifs, axis=0) * 100
    etiquettes = [f"{o} (n={effectifs[o]})" for o in ordre_obstacles]

    couleurs_niveau = {
        "Pas un obstacle": COULEUR_OK,
        "Obstacle mineur": COULEUR_ATTENTION,
        "Obstacle majeur": COULEUR_ALERTE,
    }

    fig = go.Figure()
    for niveau in ordre_niveaux:
        fig.add_trace(go.Bar(
            y=etiquettes, x=pivot_pct[niveau], name=niveau, orientation="h",
            marker=dict(color=couleurs_niveau.get(niveau)),
            customdata=pivot[niveau].values,
            hovertemplate=f"%{{y}}<br>{niveau} : %{{x:.0f}}% (n=%{{customdata}})<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack", template=TEMPLATE,
        legend_title_text="Niveau",
        xaxis=dict(title="% des repondants", range=[0, 100], ticksuffix="%"),
    )
    return styliser_titre(fig, "Obstacles rencontres")


STATUTS_EN_EMPLOI = {
    "Travail qualifié dans mon domaine",
    "Travail qualifié autre domaine",
    "Travail non qualifié",
    "Auto-entrepreneur",
}


def fig_statut_travail(df):
    """Distribution du statut sur le marche du travail (emploi dans le domaine ou non)."""
    if COL_STATUT_TRAVAIL not in df.columns:
        return styliser_titre(px.bar(template=TEMPLATE), "Statut sur le marche du travail")
    c = df[COL_STATUT_TRAVAIL].dropna().value_counts().sort_values(ascending=False).reset_index()
    c.columns = ["statut", "nombre"]
    fig = px.bar(
        c, x="statut", y="nombre", template=TEMPLATE,
        color="nombre", color_continuous_scale=["#bfdbfe", "#1d4ed8"],
    )
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(title=None, tickfont=dict(size=10))
    return styliser_titre(fig, "Statut sur le marche du travail")

# =============================================================
# KPI (cartes en haut du dashboard)
# =============================================================

def intervalle_confiance_wilson(succes, n, z=1.96):
    """Intervalle de confiance de Wilson pour une proportion (95% par defaut).
    Plus fiable qu'une approximation normale simple quand n est petit,
    ce qui est notre cas avec un echantillon de sondage de quelques dizaines de reponses."""
    if n == 0:
        return (0.0, 0.0)
    p = succes / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    marge = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5)
    borne_inf = max(0.0, (centre - marge) / denom)
    borne_sup = min(1.0, (centre + marge) / denom)
    return borne_inf, borne_sup


def formater_ic(succes, n):
    """Formate un pourcentage avec son intervalle de confiance a 95%, ex: '78% (IC95: 62-88%)'."""
    if n == 0:
        return "n/a"
    pct = succes / n * 100
    bas, haut = intervalle_confiance_wilson(succes, n)
    return f"{pct:.0f}% (IC95: {bas*100:.0f}-{haut*100:.0f}%)"


# Fond pastel + texte contraste associes a chaque couleur semantique des KPI. Le fond
# seul (pastel de la couleur) ne suffit pas a garantir un bon contraste pour le vert et
# l'orange une fois eclaircis : le chiffre utilise donc une teinte plus foncee de la
# meme famille plutot que la couleur d'origine, pour rester lisible (WCAG AA).
TEINTES_KPI = {
    COULEUR_BARRE:     ("#eff6ff", "#1d4ed8"),
    COULEUR_LIGNE:     ("#ecfeff", "#0e7490"),
    COULEUR_OK:        ("#f0fdf4", "#15803d"),
    COULEUR_ATTENTION: ("#fffbeb", "#b45309"),
    COULEUR_ALERTE:    ("#fef2f2", "#b91c1c"),
}


def carte_kpi(valeur, label, couleur, ic=None):
    fond, texte = TEINTES_KPI.get(couleur, ("#f8fafc", "#334155"))
    ic_html = f'<div style="font-size:11px; color:#9ca3af; margin-top:2px;">{ic}</div>' if ic else ""
    return f"""
    <div style="flex:1; min-width:150px; background:{fond}; border-radius:10px;
                padding:16px 18px; box-shadow:0 1px 3px rgba(0,0,0,0.08);">
        <div style="font-size:28px; font-weight:700; color:{texte}; line-height:1.1;">{valeur}</div>
        <div style="font-size:13px; color:#6b7280; margin-top:4px;">{label}</div>
        {ic_html}
    </div>
    """


def generer_kpis(df):
    n = len(df)
    cartes = [carte_kpi(n, "Reponses totales", COULEUR_BARRE)]

    if COL_DELAI_EMPLOI in df.columns:
        delais_connus = df[COL_DELAI_EMPLOI].dropna()
        delais_connus = delais_connus[delais_connus != "Toujours en recherche"]
        if not delais_connus.empty:
            mode_delai = delais_connus.value_counts().idxmax()
            n_mode = int((delais_connus == mode_delai).sum())
            n_delais = len(delais_connus)
            ic = formater_ic(n_mode, n_delais)
            cartes.append(carte_kpi(mode_delai, "Delai le plus frequent", COULEUR_LIGNE, ic=ic))

    if COL_STATUT_TRAVAIL in df.columns:
        statuts = df[COL_STATUT_TRAVAIL].dropna()
        if not statuts.empty:
            n_emploi = int(statuts.isin(STATUTS_EN_EMPLOI).sum())
            ic = formater_ic(n_emploi, len(statuts))
            pct_emploi = n_emploi / len(statuts) * 100
            cartes.append(carte_kpi(f"{pct_emploi:.0f}%", "Actuellement en emploi", COULEUR_OK, ic=ic))

            n_domaine = int((statuts == "Travail qualifié dans mon domaine").sum())
            ic_domaine = formater_ic(n_domaine, len(statuts))
            pct_domaine = n_domaine / len(statuts) * 100
            cartes.append(carte_kpi(f"{pct_domaine:.0f}%", "Emploi dans leur domaine de competence", COULEUR_LIGNE, ic=ic_domaine))

    if COL_FORMATION_CA in df.columns:
        f = df[COL_FORMATION_CA].dropna()
        if not f.empty:
            n_oui = int((f == "Oui").sum())
            ic = formater_ic(n_oui, len(f))
            pct_formation = n_oui / len(f) * 100
            cartes.append(carte_kpi(f"{pct_formation:.0f}%", "Ont suivi une formation canadienne", COULEUR_ATTENTION, ic=ic))

    if COL_SERVICES in df.columns:
        s = df[COL_SERVICES].dropna()
        if not s.empty:
            n_oui = int((s == "Oui").sum())
            ic = formater_ic(n_oui, len(s))
            pct_services = n_oui / len(s) * 100
            cartes.append(carte_kpi(f"{pct_services:.0f}%", "Ont utilise des services d'etablissement", COULEUR_ATTENTION, ic=ic))

    col_majeur = COL_OBSTACLES.get("Obstacle majeur")
    if col_majeur in df.columns:
        compteur = Counter()
        for cell in df[col_majeur].dropna():
            for item in diviser_obstacles(cell):
                compteur[item] += 1
        if compteur:
            top_obstacle, top_n = compteur.most_common(1)[0]
            ic = formater_ic(top_n, n) if n else None
            cartes.append(carte_kpi(top_obstacle, "1er obstacle majeur cite", COULEUR_ALERTE, ic=ic))

    return f'<div style="display:flex; gap:14px; flex-wrap:wrap;">{"".join(cartes)}</div>'


# =============================================================
# Insights automatiques (croisements)
# =============================================================

def taux_rapide(df, col_groupe):
    """Retourne, pour chaque valeur de col_groupe, le (%, n) de repondants ayant trouve un emploi en 6 mois ou moins."""
    if col_groupe not in df.columns or COL_DELAI_EMPLOI not in df.columns:
        return {}
    sub = df[[col_groupe, COL_DELAI_EMPLOI]].dropna()
    sub = sub[sub[COL_DELAI_EMPLOI] != "Toujours en recherche"]
    if sub.empty:
        return {}
    sub = sub.copy()
    sub["rapide"] = sub[COL_DELAI_EMPLOI].isin(DELAI_RAPIDE)
    grp = sub.groupby(col_groupe)["rapide"].agg(["mean", "count"])
    return {idx: (row["mean"], int(row["count"])) for idx, row in grp.iterrows()}


def generer_insights(df):
    n = len(df)
    lignes = [
        f"_Echantillon actuel : {n} reponses au total. Les pourcentages ci-dessous sont accompagnes "
        f"de leur intervalle de confiance a 95% (IC95) : plus il est large, moins l'estimation est fiable._",
        "",
    ]
    au_moins_un_insight = False

    # Ville de residence : Ottawa vs reste
    if COL_VILLE_RESIDENCE in df.columns:
        d = df.copy()
        d["groupe_ville"] = d[COL_VILLE_RESIDENCE].apply(lambda v: "Ottawa" if v == "Ottawa" else "Ailleurs en Ontario")
        taux = taux_rapide(d, "groupe_ville")
        pct_o, n_o = taux.get("Ottawa", (None, 0))
        pct_a, n_a = taux.get("Ailleurs en Ontario", (None, 0))
        if n_o >= SEUIL_MIN_GROUPE and n_a >= SEUIL_MIN_GROUPE:
            ic_o = formater_ic(round(pct_o * n_o), n_o)
            ic_a = formater_ic(round(pct_a * n_a), n_a)
            lignes.append(f"- **Ottawa** : {ic_o} ont trouve un emploi en 6 mois ou moins, contre **{ic_a}** ailleurs en Ontario.")
            au_moins_un_insight = True
        elif n_o >= SEUIL_MIN_GROUPE:
            ic_o = formater_ic(round(pct_o * n_o), n_o)
            lignes.append(f"- **Ottawa** : {ic_o} ont trouve un emploi en 6 mois ou moins (pas assez de reponses hors Ottawa pour comparer, n={n_a}).")
            au_moins_un_insight = True

    # Formation canadienne
    taux = taux_rapide(df, COL_FORMATION_CA)
    pct_oui, n_oui = taux.get("Oui", (None, 0))
    pct_non, n_non = taux.get("Non", (None, 0))
    if n_oui >= SEUIL_MIN_GROUPE and n_non >= SEUIL_MIN_GROUPE:
        ic_oui = formater_ic(round(pct_oui * n_oui), n_oui)
        ic_non = formater_ic(round(pct_non * n_non), n_non)
        lignes.append(f"- **Formation canadienne suivie** : {ic_oui} trouvent rapidement un emploi, contre {ic_non} sans formation.")
        au_moins_un_insight = True

    # Services d'etablissement
    taux = taux_rapide(df, COL_SERVICES)
    pct_oui, n_oui = taux.get("Oui", (None, 0))
    pct_non, n_non = taux.get("Non", (None, 0))
    if n_oui >= SEUIL_MIN_GROUPE and n_non >= SEUIL_MIN_GROUPE:
        ic_oui = formater_ic(round(pct_oui * n_oui), n_oui)
        ic_non = formater_ic(round(pct_non * n_non), n_non)
        lignes.append(f"- **Recours a des services d'etablissement** : {ic_oui} trouvent rapidement un emploi, contre {ic_non} sans ce recours.")
        au_moins_un_insight = True

    # Obstacle le plus cite comme majeur
    col_majeur = COL_OBSTACLES.get("Obstacle majeur")
    if col_majeur in df.columns:
        compteur = Counter()
        for cell in df[col_majeur].dropna():
            for item in diviser_obstacles(cell):
                compteur[item] += 1
        if compteur:
            top_obstacle, top_n = compteur.most_common(1)[0]
            ic_obstacle = formater_ic(top_n, n) if n else "n/a"
            lignes.append(f"- **Obstacle le plus cite comme majeur** : {top_obstacle}, cite par {ic_obstacle} des repondants.")
            au_moins_un_insight = True

    # --- Emploi dans le domaine : quels facteurs sont associes ? ---
    taux = taux_domaine(df, COL_FORMATION_CA)
    pct_oui, n_oui = taux.get("Oui", (None, 0))
    pct_non, n_non = taux.get("Non", (None, 0))
    if n_oui >= SEUIL_MIN_GROUPE and n_non >= SEUIL_MIN_GROUPE:
        ic_oui = formater_ic(round(pct_oui * n_oui), n_oui)
        ic_non = formater_ic(round(pct_non * n_non), n_non)
        lignes.append(f"- **Formation canadienne et emploi dans le domaine** : {ic_oui} ont un emploi dans leur domaine chez ceux qui ont suivi une formation, contre {ic_non} chez ceux qui n'en ont pas suivi.")
        au_moins_un_insight = True

    taux = taux_domaine(df, COL_SERVICES)
    pct_oui, n_oui = taux.get("Oui", (None, 0))
    pct_non, n_non = taux.get("Non", (None, 0))
    if n_oui >= SEUIL_MIN_GROUPE and n_non >= SEUIL_MIN_GROUPE:
        ic_oui = formater_ic(round(pct_oui * n_oui), n_oui)
        ic_non = formater_ic(round(pct_non * n_non), n_non)
        lignes.append(f"- **Services d'etablissement et emploi dans le domaine** : {ic_oui} ont un emploi dans leur domaine chez ceux qui ont eu recours a des services, contre {ic_non} chez ceux qui n'y ont pas eu recours.")
        au_moins_un_insight = True

    taux = taux_domaine(df, COL_NIVEAU_ANGLAIS)
    niveaux_suffisants = {k: v for k, v in taux.items() if v[1] >= SEUIL_MIN_GROUPE}
    if len(niveaux_suffisants) >= 2:
        details = ", ".join(f"{niveau} : {formater_ic(round(p*n), n)}" for niveau, (p, n) in niveaux_suffisants.items())
        lignes.append(f"- **Niveau d'anglais et emploi dans le domaine** : {details}.")
        au_moins_un_insight = True

    if not au_moins_un_insight:
        lignes.append(f"_Pas encore assez de donnees pour comparer des groupes de maniere fiable (minimum {SEUIL_MIN_GROUPE} reponses par groupe requis)._")

    return "\n".join(lignes)


def creer_domaine_binaire(df):
    """Classe chaque reponse en 'Dans le domaine' vs 'Hors domaine', selon le statut de travail.
    Les personnes en etudes/formation sont exclues (question non applicable)."""
    d = df.copy()
    if COL_STATUT_TRAVAIL not in d.columns:
        return d
    def classer(v):
        if pd.isna(v) or v == "En etudes ou en formation":
            return None
        return "Dans le domaine" if v == "Travail qualifié dans mon domaine" else "Hors domaine"
    d["_domaine_bin"] = d[COL_STATUT_TRAVAIL].apply(classer)
    return d


def taux_domaine(df, col_groupe):
    """Retourne, pour chaque valeur de col_groupe, le (%, n) ayant un emploi dans leur domaine."""
    if col_groupe not in df.columns or COL_STATUT_TRAVAIL not in df.columns:
        return {}
    d = creer_domaine_binaire(df)
    sub = d[[col_groupe, "_domaine_bin"]].dropna()
    if sub.empty:
        return {}
    sub = sub.copy()
    sub["dans_domaine"] = sub["_domaine_bin"] == "Dans le domaine"
    grp = sub.groupby(col_groupe)["dans_domaine"].agg(["mean", "count"])
    return {idx: (row["mean"], int(row["count"])) for idx, row in grp.iterrows()}


# =============================================================
# Tests statistiques (page separee)
# =============================================================

DELAI_RANG = {val: i for i, val in enumerate(DELAI_ORDER)}
ANGLAIS_RANG = {"Intermédiaire": 1, "Bilingue / courant": 2, "Avancé": 3}
SEUIL_TEST_GROUPE = 5
SEUIL_TEST_CORRELATION = 15


def preparer_delai_ordinal(df):
    d = df.copy()
    d["_delai_rang"] = d[COL_DELAI_EMPLOI].map(DELAI_RANG)
    return d


def preparer_delai_binaire(df):
    d = df.copy()
    def classer(v):
        if v in DELAI_RAPIDE:
            return "Rapide (<=6 mois)"
        if v in {"6 mois à 1 an", "Plus d'un an", "Toujours en recherche"}:
            return "Lent (>6 mois)"
        return None
    d["_delai_bin"] = d[COL_DELAI_EMPLOI].apply(classer)
    return d


def bloc_spearman_anglais_delai(df):
    """Correlation entre deux variables ordonnees : niveau d'anglais et delai d'emploi."""
    if COL_NIVEAU_ANGLAIS not in df.columns or COL_DELAI_EMPLOI not in df.columns:
        return None
    d = preparer_delai_ordinal(df)
    d["_anglais_rang"] = d[COL_NIVEAU_ANGLAIS].map(ANGLAIS_RANG)
    sub = d[["_anglais_rang", "_delai_rang"]].dropna()
    n = len(sub)
    if n < SEUIL_TEST_CORRELATION:
        return None

    rho, p = spstats.spearmanr(sub["_anglais_rang"], sub["_delai_rang"])
    signif = "statistiquement significatif" if p < 0.05 else "non significatif"
    if abs(rho) < 0.1:
        sens = "aucune tendance claire ne se degage"
    elif rho < 0:
        sens = "plus le niveau d'anglais est eleve, plus le delai tend a etre court"
    else:
        sens = "plus le niveau d'anglais est eleve, plus le delai tend a etre long (contre-intuitif, a verifier avec plus de donnees)"

    return f"""### Correlation de Spearman — niveau d'anglais et delai d'emploi

*Ce test mesure si deux variables ordonnees (ici le niveau d'anglais et le delai, du plus rapide au plus lent) evoluent ensemble, sans supposer une relation lineaire. Le coefficient rho va de -1 (relation inverse parfaite) a +1 (relation directe parfaite). Contrairement a une correlation de Pearson, il ne suppose pas que les donnees sont continues.*

- Echantillon : n={n}
- Coefficient de Spearman : rho = {rho:.2f}
- p-value : {p:.3f} → resultat **{signif}** au seuil de 5%
- Interpretation : {sens}
"""


def bloc_fisher(df, col_groupe, nom_variable):
    """Test exact de Fisher pour un tableau 2x2 (variable binaire vs delai regroupe rapide/lent)."""
    if col_groupe not in df.columns or COL_DELAI_EMPLOI not in df.columns:
        return None
    d = preparer_delai_binaire(df)
    sub = d[[col_groupe, "_delai_bin"]].dropna()
    if sub.empty:
        return None

    table = pd.crosstab(sub[col_groupe], sub["_delai_bin"])
    if table.shape != (2, 2):
        return None  # le test de Fisher classique s'applique uniquement a un tableau 2x2

    effectifs = sub[col_groupe].value_counts()
    if (effectifs < SEUIL_TEST_GROUPE).any():
        return None

    oddsratio, p = spstats.fisher_exact(table.values)
    chi2, _, _, _ = spstats.chi2_contingency(table.values, correction=False)
    n_total = int(table.values.sum())
    v_cramer = (chi2 / n_total) ** 0.5

    signif = "statistiquement significatif" if p < 0.05 else "non significatif"
    if v_cramer < 0.1:
        force = "tres faible"
    elif v_cramer < 0.3:
        force = "faible a modere"
    elif v_cramer < 0.5:
        force = "modere a fort"
    else:
        force = "fort"

    tableau_txt = table.to_string()

    return f"""### Test exact de Fisher — {nom_variable}

*Ce test verifie si deux variables categorielles sont liees, en calculant la probabilite exacte d'observer ce tableau (ou plus extreme) si elles etaient independantes l'une de l'autre. Il est plus fiable que le test du Chi2 classique quand l'echantillon est petit.*

```
{tableau_txt}
```

- p-value (Fisher) : {p:.3f} → resultat **{signif}** au seuil de 5%
- V de Cramer (force du lien, 0 = aucun lien, 1 = lien parfait) : {v_cramer:.2f} → lien **{force}**
"""


def bloc_mannwhitney(df, col_groupe, nom_variable):
    """Compare la distribution complete des delais (5 niveaux) entre 2 groupes, sans supposer une loi normale."""
    if col_groupe not in df.columns or COL_DELAI_EMPLOI not in df.columns:
        return None
    d = preparer_delai_ordinal(df)
    sub = d[[col_groupe, "_delai_rang"]].dropna()
    if sub.empty:
        return None

    groupes = sub[col_groupe].unique().tolist()
    if len(groupes) != 2:
        return None

    effectifs = sub[col_groupe].value_counts()
    if (effectifs < SEUIL_TEST_GROUPE).any():
        return None

    g1, g2 = groupes
    x = sub.loc[sub[col_groupe] == g1, "_delai_rang"]
    y = sub.loc[sub[col_groupe] == g2, "_delai_rang"]
    stat, p = spstats.mannwhitneyu(x, y, alternative="two-sided")

    def rang_vers_delai(valeur_mediane):
        idx = int(round(valeur_mediane))
        idx = min(max(idx, 0), len(DELAI_ORDER) - 1)
        return DELAI_ORDER[idx]

    signif = "statistiquement significatif" if p < 0.05 else "non significatif"
    plus_rapide = g1 if x.median() < y.median() else (g2 if y.median() < x.median() else "aucun (egalite)")

    return f"""### Test de Mann-Whitney — {nom_variable}

*Ce test non parametrique compare la distribution des delais entre deux groupes sans supposer que les donnees suivent une loi normale — adapte a une variable ordonnee en 5 niveaux comme notre delai d'emploi.*

- {g1} : n={len(x)}, delai median ≈ {rang_vers_delai(x.median())}
- {g2} : n={len(y)}, delai median ≈ {rang_vers_delai(y.median())}
- p-value : {p:.3f} → resultat **{signif}** au seuil de 5%
- Groupe avec le delai median le plus court : **{plus_rapide}**
"""


def bloc_test_association(df, col_groupe, col_resultat, nom_variable, nom_resultat):
    """Test d'association generique entre deux variables categorielles.
    Utilise le test exact de Fisher pour un tableau 2x2, et le test du Chi2
    (avec verification des effectifs attendus) pour un tableau plus grand."""
    if col_groupe not in df.columns or col_resultat not in df.columns:
        return None
    sub = df[[col_groupe, col_resultat]].dropna()
    if sub.empty:
        return None

    table = pd.crosstab(sub[col_groupe], sub[col_resultat])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return None

    effectifs = sub[col_groupe].value_counts()
    if (effectifs < SEUIL_TEST_GROUPE).any():
        return None

    n_total = int(table.values.sum())

    if table.shape == (2, 2):
        nom_test = "Test exact de Fisher"
        _, p = spstats.fisher_exact(table.values)
        chi2, _, _, _ = spstats.chi2_contingency(table.values, correction=False)
        explication = (
            "Ce test verifie si deux variables categorielles sont liees, en calculant la "
            "probabilite exacte d'observer ce tableau (ou plus extreme) si elles etaient "
            "independantes. Bien adapte aux petits echantillons."
        )
    else:
        chi2, p, dof, expected = spstats.chi2_contingency(table.values)
        cellules_fiables = (expected >= 5).sum() / expected.size >= 0.8 and (expected >= 1).all()
        if not cellules_fiables:
            return None
        nom_test = "Test du Chi2"
        explication = (
            "Ce test verifie si deux variables categorielles a plusieurs niveaux sont liees, "
            "en comparant les effectifs observes aux effectifs attendus si elles etaient "
            "independantes l'une de l'autre."
        )

    v_cramer = (chi2 / (n_total * (min(table.shape) - 1))) ** 0.5
    signif = "statistiquement significatif" if p < 0.05 else "non significatif"
    if v_cramer < 0.1:
        force = "tres faible"
    elif v_cramer < 0.3:
        force = "faible a modere"
    elif v_cramer < 0.5:
        force = "modere a fort"
    else:
        force = "fort"

    tableau_txt = table.to_string()

    return f"""### {nom_test} — {nom_variable} vs {nom_resultat}

*{explication}*

```
{tableau_txt}
```

- p-value : {p:.3f} → resultat **{signif}** au seuil de 5%
- V de Cramer (force du lien) : {v_cramer:.2f} → lien **{force}**
"""


def construire_page_stats():
    df = charger_donnees()
    n = len(df)

    blocs = []

    blocs.append(bloc_spearman_anglais_delai(df))

    blocs.append(bloc_fisher(df, COL_FORMATION_CA, "Formation canadienne suivie"))
    blocs.append(bloc_mannwhitney(df, COL_FORMATION_CA, "Formation canadienne suivie"))

    blocs.append(bloc_fisher(df, COL_SERVICES, "Recours a des services d'etablissement"))
    blocs.append(bloc_mannwhitney(df, COL_SERVICES, "Recours a des services d'etablissement"))

    if COL_VILLE_RESIDENCE in df.columns:
        d_ville = df.copy()
        d_ville["_groupe_ville"] = d_ville[COL_VILLE_RESIDENCE].apply(
            lambda v: "Ottawa" if v == "Ottawa" else "Ailleurs en Ontario"
        )
        blocs.append(bloc_fisher(d_ville, "_groupe_ville", "Ville de residence (Ottawa vs ailleurs)"))
        blocs.append(bloc_mannwhitney(d_ville, "_groupe_ville", "Ville de residence (Ottawa vs ailleurs)"))

    # --- Qu'est-ce qui est associe a un emploi DANS le domaine (pas juste un emploi) ? ---
    d_domaine = creer_domaine_binaire(df)
    blocs.append(bloc_test_association(d_domaine, COL_FORMATION_CA, "_domaine_bin", "Formation canadienne", "Emploi dans le domaine"))
    blocs.append(bloc_test_association(d_domaine, COL_NIVEAU_ANGLAIS, "_domaine_bin", "Niveau d'anglais", "Emploi dans le domaine"))
    blocs.append(bloc_test_association(d_domaine, COL_SERVICES, "_domaine_bin", "Recours aux services d'etablissement", "Emploi dans le domaine"))
    blocs.append(bloc_mannwhitney(d_domaine, "_domaine_bin", "Emploi dans le domaine (delai d'obtention)"))

    blocs = [b for b in blocs if b]

    intro = f"""# Tests statistiques

Echantillon actuel : **{n} reponses**.

Cette page applique des tests formels aux croisements deja presentes dans le dashboard principal, pour vérifier lesquels sont statistiquement fiables plutot que de simples variations dues au hasard d'un petit echantillon. **Seuls les tests dont les conditions sont remplies (taille de groupe suffisante, structure de tableau adaptee) sont affiches ci-dessous** — les autres sont ecartes automatiquement plutot que d'afficher un resultat peu fiable.

A retenir avec un echantillon de cette taille : l'absence de resultat "significatif" ne veut pas dire qu'il n'y a pas de difference reelle dans la population generale — seulement que cet echantillon est peut-etre trop petit pour la detecter avec certitude (ce qu'on appelle un manque de "puissance statistique").

---

"""
    if not blocs:
        corps = "_Aucun test ne remplit actuellement les conditions minimales de faisabilite (groupes trop petits, moins de {SEUIL_TEST_GROUPE} reponses). Reessayez avec un echantillon plus grand._"
    else:
        corps = "\n\n---\n\n".join(blocs)

    return intro + corps


# =============================================================
# Carte de mots-cles
# =============================================================

FREQ_MIN_NUAGE = 2  # frequence minimale pour apparaitre dans le nuage visuel (reduit le bruit des mentions isolees)
TOP_MOTS_LISTE = 15  # nombre de mots affiches dans la liste de repli


def liste_mots_frequents(frequences, top_n=TOP_MOTS_LISTE):
    """Liste HTML des mots les plus frequents, utilisee en repli quand l'echantillon est
    trop petit pour un nuage de mots representatif (tous les mots sont gardes, y compris
    les mentions uniques, pour rester transparent sur un tres petit echantillon)."""
    top = frequences.most_common(top_n)
    items = "".join(f"<li>{mot} <span style='color:#9ca3af;'>({n})</span></li>" for mot, n in top)
    return f"<ul style='columns:2; font-size:14px; line-height:1.8; padding-left:20px;'>{items}</ul>"


def generer_carte_mots_cles(df):
    """Retourne (image_nuage, html_liste_repli) : un seul des deux est renseigne.
    Sous SEUIL_MIN_GROUPE reponses textuelles, le nuage visuel n'est pas assez
    representatif -> on bascule sur une liste brute des mots les plus frequents."""
    if COL_CONSEIL not in df.columns:
        return None, None
    textes = df[COL_CONSEIL].dropna().astype(str).tolist()
    if not textes:
        return None, None
    texte_complet = " ".join(textes).lower()
    mots = re.findall(r"[a-zàâäéèêëïîôöùûüç]+", texte_complet)
    mots_filtres = [m for m in mots if enlever_accents(m) not in STOPWORDS_FR_SANS_ACCENTS and len(m) > 2]
    if not mots_filtres:
        return None, None
    frequences = Counter(mots_filtres)

    if len(textes) < SEUIL_MIN_GROUPE:
        return None, liste_mots_frequents(frequences)

    frequences_nuage = Counter({m: n for m, n in frequences.items() if n >= FREQ_MIN_NUAGE})
    if not frequences_nuage:
        return None, liste_mots_frequents(frequences)

    wc = WordCloud(
        width=900, height=450, background_color="white",
        colormap="viridis", prefer_horizontal=0.9,
    ).generate_from_frequencies(frequences_nuage)
    return wc.to_array(), None


# =============================================================
# Assemblage du dashboard
# =============================================================

def fig_heatmap_generique(df, col_lignes, col_colonnes, titre, ordre_colonnes=None):
    """Carte de chaleur generique en % par ligne entre deux variables categorielles
    quelconques. Les categories de ligne peu representees sont fusionnees dans 'Autres'
    pour rester comparables malgre des tailles de groupe tres differentes."""
    if col_lignes not in df.columns or col_colonnes not in df.columns:
        return styliser_titre(px.imshow([[0]], template=TEMPLATE), f"Donnees indisponibles: {titre}")
    sub = df[[col_lignes, col_colonnes]].dropna().copy()
    if sub.empty:
        return styliser_titre(px.imshow([[0]], template=TEMPLATE), f"Pas encore de donnees: {titre}")

    sub[col_lignes] = fusionner_petits_groupes(sub[col_lignes])

    ct = pd.crosstab(sub[col_lignes], sub[col_colonnes])
    if ordre_colonnes:
        colonnes_presentes = [c for c in ordre_colonnes if c in ct.columns]
        ct = ct[colonnes_presentes]
    effectifs = ct.sum(axis=1).sort_values(ascending=False)
    ct = ct.loc[effectifs.index]
    ct_pct = ct.div(effectifs, axis=0) * 100
    etiquettes_lignes = [f"{idx} (n={effectifs[idx]})" for idx in ct.index]

    fig = px.imshow(
        ct_pct.values, x=ct_pct.columns, y=etiquettes_lignes,
        color_continuous_scale=GRADIENT_HEATMAP, zmin=0, zmax=100,
        aspect="auto", template=TEMPLATE,
    )
    fig.update_traces(
        texttemplate="%{z:.0f}%",
        customdata=ct.values,
        hovertemplate="%{y} / %{x}<br>%{z:.0f}% (n=%{customdata})<extra></extra>",
    )
    fig.update_layout(coloraxis_colorbar=dict(ticksuffix="%"))
    fig.update_xaxes(title=None, tickfont=dict(size=11))
    fig.update_yaxes(title=None, tickfont=dict(size=10))
    return styliser_titre(fig, titre)


def construire_dashboard(regions=None, ages=None, genre="Tous", annees=None, domaine_travail="Tous"):
    df = charger_donnees()
    n_avant_filtre = len(df)
    df = appliquer_filtres(df, regions, ages, genre, annees, domaine_travail)
    n_total = len(df)

    kpis = generer_kpis(df)
    if n_total < n_avant_filtre:
        kpis = (
            f'<div style="margin-bottom:8px; font-size:13px; color:#6b7280;">'
            f'Filtre actif : {n_total} sur {n_avant_filtre} reponses affichees</div>'
        ) + kpis
    insights = generer_insights(df)

    g1 = fig_region(df)
    g2 = fig_obstacles_complet(df)
    g3 = fig_delai_bar(df, COL_VILLE_RESIDENCE, "Delai par ville")
    g4 = fig_delai_heatmap(df, COL_DOMAINE, "Delai par secteur")
    g5 = fig_delai_bar(df, COL_NIVEAU_ANGLAIS, "Delai par niveau d'anglais")
    g6 = fig_delai_bar(df, COL_SERVICES, "Delai par recours aux services")
    g7 = fig_statut_travail(df)
    g8 = fig_heatmap_generique(
        df, COL_STATUT_TRAVAIL, COL_FORMATION_CA,
        "Statut d'emploi par formation canadienne",
    )
    img_wordcloud, html_repli_mots = generer_carte_mots_cles(df)
    img_wordcloud = gr.update(value=img_wordcloud, visible=img_wordcloud is not None)
    html_repli_mots = gr.update(value=html_repli_mots, visible=html_repli_mots is not None)

    return kpis, insights, g1, g2, g3, g4, g5, g6, g7, g8, img_wordcloud, html_repli_mots


# =============================================================
# Interface Gradio
# =============================================================

# Rampe de vert franco-ontarien (profond/foret, pas vif) utilisee comme teinte secondaire
# du theme -- colore entre autres les boutons "secondary" (ex: Rafraichir les donnees).
VERT_FRANCO_HUE = gr.themes.Color(
    c50="#f1f8f4", c100="#dcefe3", c200="#b9dfc8", c300="#8cc7a3",
    c400="#5aa87c", c500="#357f5c", c600="#26634a", c700=COULEUR_VERT_FRANCO,
    c800="#1a4033", c900="#16342a", c950="#0b1d17",
)
THEME = gr.themes.Soft(primary_hue="blue", secondary_hue=VERT_FRANCO_HUE, neutral_hue="slate")

# CSS personnalise : coins/ombres plus marques, espacement plus genereux entre sections
# (via elem_classes plutot que des classes internes Gradio, plus stable), et une police
# a empattements pour les titres (H1-H3) qui tranche avec Montserrat en corps de texte.
CSS_PERSONNALISE = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&display=swap');

.gradio-container {
    --block-radius: 14px;
    --block-shadow: 0 4px 18px rgba(15, 23, 42, 0.07);
    --block-border-width: 1px;
    --block-border-color: rgba(15, 23, 42, 0.06);
    max-width: 1400px !important;
}

.gradio-container h1,
.gradio-container h2,
.gradio-container h3 {
    font-family: 'Fraunces', Georgia, serif;
    letter-spacing: -0.01em;
}

.section-row {
    margin-bottom: 28px !important;
}

.section-heading {
    margin-top: 22px !important;
    margin-bottom: 12px !important;
}

/* Etiquettes des filtres : texte gras neutre, sans fond/bordure de badge, pour que le
   menu deroulant (l'element actionnable) reste visuellement plus proeminent que son
   etiquette (norme de design Gestalt du projet, voir CLAUDE.md).
   NB : le libelle d'un Dropdown est rendu via un <span> qui lit --block-title-*
   (pas --block-label-*, qui ne s'applique qu'aux badges type "Plot"/"Graphique"
   portes par un <label>) -- verifie par inspection du CSS compile de Gradio. */
.filtre-label {
    --block-title-background-fill: transparent;
    --block-title-border-width: 0px;
    --block-title-text-color: #334155;
    --block-title-text-weight: 700;
    --block-title-padding: 0 0 6px 0;
    --block-title-radius: 0px;
}
"""

# Petit trille stylise (fleur emblematique de l'Ontario) en trait geometrique tres discret,
# utilise uniquement comme touche d'identite a cote du titre principal.
TRILLE_SVG = f"""<svg width="24" height="24" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <g fill="{COULEUR_VERT_FRANCO}" fill-opacity="0.85">
    <ellipse cx="50" cy="28" rx="13" ry="23" transform="rotate(0 50 50)"/>
    <ellipse cx="50" cy="28" rx="13" ry="23" transform="rotate(120 50 50)"/>
    <ellipse cx="50" cy="28" rx="13" ry="23" transform="rotate(240 50 50)"/>
  </g>
  <circle cx="50" cy="50" r="6" fill="#fafaf9"/>
</svg>"""

with gr.Blocks(title="Dashboard - Sondage immigration francophone Ontario") as demo:
    gr.HTML(f"""
    <div style="display:flex; align-items:center; gap:10px;">
        {TRILLE_SVG}
        <h1 style="font-size:26px; font-weight:700; color:#111827; margin:0;">Immigration francophone en Ontario</h1>
    </div>
    <div style="height:3px; width:64px; background:{COULEUR_VERT_FRANCO}; border-radius:2px; margin:8px 0 4px;"></div>
    """)

    with gr.Row():
        with gr.Column(scale=1, min_width=220):
            gr.Markdown("### Filtres")
            filtre_region = gr.Dropdown(
                choices=CHOIX_REGION, multiselect=True, label="Region d'origine", value=[],
                elem_classes=["filtre-label"],
            )
            filtre_age = gr.Dropdown(
                choices=CHOIX_AGE, multiselect=True, label="Tranche d'age", value=[],
                elem_classes=["filtre-label"],
            )
            filtre_genre = gr.Dropdown(
                choices=["Tous", "Femme", "Homme"], value="Tous", label="Genre",
                elem_classes=["filtre-label"],
            )
            filtre_annees = gr.Dropdown(
                choices=CHOIX_ANNEES, multiselect=True, label="Annees au Canada", value=[],
                elem_classes=["filtre-label"],
            )
            filtre_domaine = gr.Dropdown(
                choices=["Tous", "Dans le domaine", "Hors domaine"], value="Tous",
                label="Emploi dans le domaine ?",
                elem_classes=["filtre-label"],
            )
            bouton_refresh = gr.Button("Rafraichir les donnees")

        with gr.Column(scale=4):
            with gr.Tabs():
                with gr.Tab("Vue d'ensemble"):
                    kpi_html = gr.HTML(elem_classes=["section-row"])

                    gr.Markdown("## Insights cles", elem_classes=["section-heading"])
                    insights_txt = gr.Markdown()

                    with gr.Row(elem_classes=["section-row"]):
                        g1 = gr.Plot()
                        g7 = gr.Plot()
                    with gr.Row(elem_classes=["section-row"]):
                        g2 = gr.Plot()

                    gr.Markdown("## Carte des mots-cles - conseils aux futurs arrivants", elem_classes=["section-heading"])
                    img_nuage = gr.Image(label="Mots les plus frequents", show_label=False)
                    liste_mots_html = gr.HTML()

                with gr.Tab("Croisements"):
                    gr.Markdown("## Croisements avec le delai d'obtention d'un emploi, et avec la correspondance au domaine", elem_classes=["section-heading"])
                    with gr.Row(elem_classes=["section-row"]):
                        g3 = gr.Plot()
                        g4 = gr.Plot()
                    with gr.Row(elem_classes=["section-row"]):
                        g5 = gr.Plot()
                        g6 = gr.Plot()
                    with gr.Row(elem_classes=["section-row"]):
                        g8 = gr.Plot()

                with gr.Tab("Tests statistiques"):
                    bouton_stats = gr.Button("Lancer / rafraichir les tests")
                    stats_md = gr.Markdown()

                    demo.load(construire_page_stats, outputs=stats_md)
                    bouton_stats.click(construire_page_stats, outputs=stats_md)

    entrees = [filtre_region, filtre_age, filtre_genre, filtre_annees, filtre_domaine]
    sorties = [kpi_html, insights_txt, g1, g2, g3, g4, g5, g6, g7, g8, img_nuage, liste_mots_html]

    demo.load(construire_dashboard, inputs=entrees, outputs=sorties)
    bouton_refresh.click(construire_dashboard, inputs=entrees, outputs=sorties)
    filtre_region.change(construire_dashboard, inputs=entrees, outputs=sorties)
    filtre_age.change(construire_dashboard, inputs=entrees, outputs=sorties)
    filtre_genre.change(construire_dashboard, inputs=entrees, outputs=sorties)
    filtre_annees.change(construire_dashboard, inputs=entrees, outputs=sorties)
    filtre_domaine.change(construire_dashboard, inputs=entrees, outputs=sorties)

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS_PERSONNALISE)
