"""
Dashboard en temps reel pour le sondage sur l'immigration francophone en Ontario.
Lit les donnees depuis un CSV (Google Sheet publie en CSV, ou fichier local pour test).
"""

import re
import unicodedata
from collections import Counter

import gradio as gr
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
from scipy import stats as spstats

# =============================================================
# CONFIG
# =============================================================

SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/1vLkVKIM_hsRLuFFoN8oje6rqtS4xWa_vMzPtmg7K9LA/export?format=csv&gid=1945629438"

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


def enlever_accents(mot):
    """Normalise un mot en retirant ses accents, pour comparer 'tres' et 'très' correctement."""
    return "".join(c for c in unicodedata.normalize("NFD", mot) if unicodedata.category(c) != "Mn")


STOPWORDS_FR_SANS_ACCENTS = {enlever_accents(m) for m in STOPWORDS_FR}

# Palette de couleurs coherente
COULEUR_LIGNE = "#0891b2"
COULEUR_BARRE = "#2563eb"
COULEUR_OK = "#22c55e"
COULEUR_ATTENTION = "#f59e0b"
COULEUR_ALERTE = "#ef4444"
PALETTE_DELAI = {
    "1 à 3 mois": "#ffedd5",
    "3 à 6 mois": "#fdba74",
    "6 mois à 1 an": "#fb923c",
    "Plus d'un an": "#ea580c",
    "Toujours en recherche": "#9a3412",
}
GRADIENT_HEATMAP = ["#fff7ed", "#9a3412"]
TEMPLATE = "plotly_white"

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


def appliquer_filtres(df, regions, ages, genre, annees):
    if regions:
        df = df[df[COL_REGION_ORIGINE].isin(regions)]
    if ages:
        df = df[df[COL_AGE].isin(ages)]
    if genre and genre != "Tous":
        df = df[df[COL_GENRE] == genre]
    if annees:
        df = df[df[COL_ANNEES_CANADA].isin(annees)]
    return df


# =============================================================
# Graphiques
# =============================================================

def fig_region(df):
    if COL_REGION_ORIGINE not in df.columns:
        return px.bar(title="Colonne region d'origine introuvable")
    c = df[COL_REGION_ORIGINE].dropna().value_counts().sort_values(ascending=False).reset_index()
    c.columns = ["region", "nombre"]
    fig = px.bar(
        c, x="region", y="nombre", template=TEMPLATE,
        color="nombre", color_continuous_scale=["#bfdbfe", "#1d4ed8"],
    )
    fig.update_layout(coloraxis_showscale=False)
    return styliser_titre(fig, "Region d'origine")


def fig_delai_bar(df, col_groupe, titre):
    """Barres empilees pour les croisements simples (peu de categories)."""
    if col_groupe not in df.columns or COL_DELAI_EMPLOI not in df.columns:
        return styliser_titre(px.bar(template=TEMPLATE), f"Donnees indisponibles: {titre}")
    sub = df[[col_groupe, COL_DELAI_EMPLOI]].dropna()
    if sub.empty:
        return styliser_titre(px.bar(template=TEMPLATE), f"Pas encore de donnees: {titre}")
    ct = sub.groupby([col_groupe, COL_DELAI_EMPLOI]).size().reset_index(name="nombre")
    ordre_groupe = sub[col_groupe].value_counts().index.tolist()
    fig = px.bar(
        ct, x=col_groupe, y="nombre", color=COL_DELAI_EMPLOI,
        category_orders={COL_DELAI_EMPLOI: DELAI_ORDER, col_groupe: ordre_groupe},
        color_discrete_map=PALETTE_DELAI, barmode="stack", template=TEMPLATE,
    )
    fig.update_layout(legend_title_text="Delai", legend_font_size=11)
    fig.update_xaxes(title=None)
    return styliser_titre(fig, titre)


DELAI_GROUPE_MAP = {
    "1 à 3 mois": "Rapide (≤6 mois)",
    "3 à 6 mois": "Rapide (≤6 mois)",
    "6 mois à 1 an": "Moyen (6 mois-1 an)",
    "Plus d'un an": "Lent (+1 an)",
    "Toujours en recherche": "Lent (+1 an)",
}
DELAI_GROUPE_ORDER = ["Rapide (≤6 mois)", "Moyen (6 mois-1 an)", "Lent (+1 an)"]


def fig_delai_heatmap(df, col_groupe, titre, seuil_min_ligne=3):
    """Carte de chaleur pour les croisements avec beaucoup de categories (ex: secteur d'activite).
    Le delai est regroupe en 3 niveaux et les categories peu representees sont fusionnees
    dans 'Autres', pour eviter une grille clairsemee pleine de 0 et de 1."""
    if col_groupe not in df.columns or COL_DELAI_EMPLOI not in df.columns:
        return styliser_titre(px.imshow([[0]], template=TEMPLATE), f"Donnees indisponibles: {titre}")
    sub = df[[col_groupe, COL_DELAI_EMPLOI]].dropna().copy()
    if sub.empty:
        return styliser_titre(px.imshow([[0]], template=TEMPLATE), f"Pas encore de donnees: {titre}")

    sub["delai_groupe"] = sub[COL_DELAI_EMPLOI].map(DELAI_GROUPE_MAP)

    effectifs = sub[col_groupe].value_counts()
    petits = effectifs[effectifs < seuil_min_ligne].index.tolist()
    if petits:
        libelle_autres = f"Autres (n<{seuil_min_ligne} chacun)"
        sub[col_groupe] = sub[col_groupe].apply(lambda v: libelle_autres if v in petits else v)

    ct = pd.crosstab(sub[col_groupe], sub["delai_groupe"])
    colonnes_ordonnees = [c for c in DELAI_GROUPE_ORDER if c in ct.columns]
    ct = ct[colonnes_ordonnees]
    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]

    fig = px.imshow(
        ct.values, x=ct.columns, y=ct.index,
        color_continuous_scale=GRADIENT_HEATMAP,
        text_auto=True, aspect="auto", template=TEMPLATE,
    )
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(title=None, tickfont=dict(size=11))
    fig.update_yaxes(title=None, tickfont=dict(size=11))
    return styliser_titre(fig, titre)


def styliser_titre(fig, titre, taille=13):
    fig.update_layout(
        title=dict(text=titre, font=dict(size=taille), x=0.02, xanchor="left"),
        margin=dict(t=42, l=10, r=10, b=10),
    )
    return fig


def diviser_obstacles(cellule):
    """Decoupe une cellule contenant plusieurs obstacles.
    Google Forms a parfois utilise ';' et parfois ',' comme separateur
    selon les reponses -> on gere les deux pour ne rater aucun cas."""
    return [item.strip() for item in re.split(r"[;,]", str(cellule)) if item.strip()]


def fig_obstacles_complet(df):
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
    pivot = df_obs.groupby(["obstacle", "niveau"]).size().reset_index(name="nombre")

    majeurs = pivot[pivot["niveau"] == "Obstacle majeur"].set_index("obstacle")["nombre"]
    ordre_obstacles = (
        majeurs.reindex(pivot["obstacle"].unique()).fillna(0).sort_values(ascending=True).index.tolist()
    )

    fig = px.bar(
        pivot, x="nombre", y="obstacle", color="niveau", orientation="h",
        category_orders={
            "obstacle": ordre_obstacles,
            "niveau": ["Pas un obstacle", "Obstacle mineur", "Obstacle majeur"],
        },
        color_discrete_map={
            "Pas un obstacle": COULEUR_OK,
            "Obstacle mineur": COULEUR_ATTENTION,
            "Obstacle majeur": COULEUR_ALERTE,
        },
        template=TEMPLATE,
    )
    fig.update_layout(legend_title_text="Niveau")
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
        color="nombre", color_continuous_scale=["#d1fae5", "#047857"],
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


def carte_kpi(valeur, label, couleur, ic=None):
    ic_html = f'<div style="font-size:11px; color:#9ca3af; margin-top:2px;">{ic}</div>' if ic else ""
    return f"""
    <div style="flex:1; min-width:150px; background:white; border-radius:10px;
                border-left:5px solid {couleur}; padding:16px 18px; box-shadow:0 1px 3px rgba(0,0,0,0.08);">
        <div style="font-size:28px; font-weight:700; color:#111827; line-height:1.1;">{valeur}</div>
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

SEUIL_MIN_GROUPE = 5  # en dessous de ce nombre de reponses, on ne compare pas (trop peu fiable)


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

    if not au_moins_un_insight:
        lignes.append(f"_Pas encore assez de donnees pour comparer des groupes de maniere fiable (minimum {SEUIL_MIN_GROUPE} reponses par groupe requis)._")

    return "\n".join(lignes)


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

def generer_carte_mots_cles(df):
    if COL_CONSEIL not in df.columns:
        return None
    textes = df[COL_CONSEIL].dropna().astype(str).tolist()
    if not textes:
        return None
    texte_complet = " ".join(textes).lower()
    mots = re.findall(r"[a-zàâäéèêëïîôöùûüç]+", texte_complet)
    mots_filtres = [m for m in mots if enlever_accents(m) not in STOPWORDS_FR_SANS_ACCENTS and len(m) > 2]
    if not mots_filtres:
        return None
    frequences = Counter(mots_filtres)
    wc = WordCloud(
        width=900, height=450, background_color="white",
        colormap="viridis", prefer_horizontal=0.9,
    ).generate_from_frequencies(frequences)
    return wc.to_array()


# =============================================================
# Assemblage du dashboard
# =============================================================

def fig_heatmap_generique(df, col_lignes, col_colonnes, titre, ordre_colonnes=None):
    """Carte de chaleur generique entre deux variables categorielles quelconques."""
    if col_lignes not in df.columns or col_colonnes not in df.columns:
        return styliser_titre(px.imshow([[0]], template=TEMPLATE), f"Donnees indisponibles: {titre}")
    sub = df[[col_lignes, col_colonnes]].dropna()
    if sub.empty:
        return styliser_titre(px.imshow([[0]], template=TEMPLATE), f"Pas encore de donnees: {titre}")

    ct = pd.crosstab(sub[col_lignes], sub[col_colonnes])
    if ordre_colonnes:
        colonnes_presentes = [c for c in ordre_colonnes if c in ct.columns]
        ct = ct[colonnes_presentes]
    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]

    fig = px.imshow(
        ct.values, x=ct.columns, y=ct.index,
        color_continuous_scale=GRADIENT_HEATMAP,
        text_auto=True, aspect="auto", template=TEMPLATE,
    )
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(title=None, tickfont=dict(size=11))
    fig.update_yaxes(title=None, tickfont=dict(size=10))
    return styliser_titre(fig, titre)


def construire_dashboard(regions=None, ages=None, genre="Tous", annees=None):
    df = charger_donnees()
    n_avant_filtre = len(df)
    df = appliquer_filtres(df, regions, ages, genre, annees)
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
    img_wordcloud = generer_carte_mots_cles(df)

    return kpis, insights, g1, g2, g3, g4, g5, g6, g7, g8, img_wordcloud


# =============================================================
# Interface Gradio
# =============================================================

with gr.Blocks(title="Dashboard - Sondage immigration francophone Ontario") as demo:
    gr.Markdown("# Immigration francophone en Ontario")

    with gr.Tabs():
        with gr.Tab("Dashboard"):
            with gr.Row():
                filtre_region = gr.Dropdown(choices=CHOIX_REGION, multiselect=True, label="Region d'origine", value=[])
                filtre_age = gr.Dropdown(choices=CHOIX_AGE, multiselect=True, label="Tranche d'age", value=[])
                filtre_genre = gr.Dropdown(choices=["Tous", "Femme", "Homme"], value="Tous", label="Genre")
                filtre_annees = gr.Dropdown(choices=CHOIX_ANNEES, multiselect=True, label="Annees au Canada", value=[])

            bouton_refresh = gr.Button("Rafraichir les donnees")

            kpi_html = gr.HTML()

            gr.Markdown("## Insights cles")
            insights_txt = gr.Markdown()

            with gr.Row():
                g1 = gr.Plot()
                g2 = gr.Plot()
            with gr.Row():
                g3 = gr.Plot()
                g4 = gr.Plot()
            with gr.Row():
                g5 = gr.Plot()
                g6 = gr.Plot()
            with gr.Row():
                g7 = gr.Plot()
                g8 = gr.Plot()

            gr.Markdown("## Carte des mots-cles - conseils aux futurs arrivants")
            img_nuage = gr.Image(label="Mots les plus frequents", show_label=False)

            entrees = [filtre_region, filtre_age, filtre_genre, filtre_annees]
            sorties = [kpi_html, insights_txt, g1, g2, g3, g4, g5, g6, g7, g8, img_nuage]

            demo.load(construire_dashboard, inputs=entrees, outputs=sorties)
            bouton_refresh.click(construire_dashboard, inputs=entrees, outputs=sorties)
            filtre_region.change(construire_dashboard, inputs=entrees, outputs=sorties)
            filtre_age.change(construire_dashboard, inputs=entrees, outputs=sorties)
            filtre_genre.change(construire_dashboard, inputs=entrees, outputs=sorties)
            filtre_annees.change(construire_dashboard, inputs=entrees, outputs=sorties)

        with gr.Tab("Tests statistiques"):
            bouton_stats = gr.Button("Lancer / rafraichir les tests")
            stats_md = gr.Markdown()

            demo.load(construire_page_stats, outputs=stats_md)
            bouton_stats.click(construire_page_stats, outputs=stats_md)

if __name__ == "__main__":
    demo.launch()
