# -*- coding: utf-8 -*-
"""
COPAG — Dashboard d'Analyse Logistique
========================================
Application Streamlit avec authentification, nettoyage de données,
KPIs, filtres dynamiques et visualisations (Matplotlib / Seaborn).

Auteur : généré avec Claude
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ----------------------------------------------------------------------------
# CONFIGURATION GÉNÉRALE DE LA PAGE
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="COPAG | Dashboard Logistique",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(APP_DIR, "dataset.csv")
LOGO_PATH = os.path.join(APP_DIR, "logo.png")

# Identifiants de connexion (démo)
VALID_USERNAME = "zhrdgmi"
VALID_PASSWORD = "1234"

sns.set_theme(style="whitegrid")
COPAG_GREEN = "#007A3D"
COPAG_DARK = "#00542D"
COPAG_ACCENT = "#F2A900"
PALETTE = [COPAG_GREEN, COPAG_ACCENT, COPAG_DARK, "#4CAF50", "#8BC34A", "#C8E6C9", "#1B5E20"]
sns.set_palette(sns.color_palette(PALETTE))

# ----------------------------------------------------------------------------
# STYLE CSS PERSONNALISÉ
# ----------------------------------------------------------------------------
CUSTOM_CSS = f"""
<style>
    .main {{
        background-color: #F7FAF8;
    }}
    section[data-testid="stSidebar"] {{
        background-color: {COPAG_DARK};
    }}
    section[data-testid="stSidebar"] * {{
        color: #FFFFFF !important;
    }}
    div[data-testid="stMetric"] {{
        background-color: #FFFFFF;
        border: 1px solid #E3ECE6;
        border-left: 6px solid {COPAG_GREEN};
        border-radius: 10px;
        padding: 14px 16px 10px 16px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }}
    div[data-testid="stMetricLabel"] {{
        color: {COPAG_DARK};
        font-weight: 600;
    }}
    h1, h2, h3 {{
        color: {COPAG_DARK};
    }}
    .copag-header {{
        display: flex;
        align-items: center;
        gap: 18px;
        padding: 6px 0 18px 0;
    }}
    .copag-badge {{
        background-color: {COPAG_GREEN};
        color: white;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }}
    .login-card {{
        max-width: 420px;
        margin: 60px auto 0 auto;
        padding: 36px 34px 28px 34px;
        background: white;
        border-radius: 16px;
        border: 1px solid #E3ECE6;
        box-shadow: 0 4px 18px rgba(0,0,0,0.08);
    }}
    .stButton>button {{
        background-color: {COPAG_GREEN};
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        padding: 0.5em 1.2em;
    }}
    .stButton>button:hover {{
        background-color: {COPAG_DARK};
        color: white;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# AUTHENTIFICATION
# ----------------------------------------------------------------------------
def login_page():
    col_l, col_c, col_r = st.columns([1, 1.2, 1])
    with col_c:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        if os.path.exists(LOGO_PATH):
            lc1, lc2, lc3 = st.columns([1, 1, 1])
            with lc2:
                st.image(LOGO_PATH, width=110)
        st.markdown(
            f"<h2 style='text-align:center;margin-top:0;'>COPAG</h2>"
            f"<p style='text-align:center;color:#555;margin-top:-10px;'>"
            f"Dashboard Logistique &amp; Transport</p>",
            unsafe_allow_html=True,
        )
        st.write("")
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Nom d'utilisateur")
            password = st.text_input("Mot de passe", type="password")
            submitted = st.form_submit_button("Se connecter", use_container_width=True)

        if submitted:
            if username == VALID_USERNAME and password == VALID_PASSWORD:
                st.session_state["authenticated"] = True
                st.session_state["user"] = username
                st.rerun()
            else:
                st.error("Identifiants incorrects. Veuillez réessayer.")

        st.markdown(
            "<p style='text-align:center;color:#999;font-size:0.8rem;margin-top:18px;'>"
            "Accès réservé au personnel COPAG</p>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


def logout_button():
    with st.sidebar:
        st.write("---")
        if st.button("🚪 Se déconnecter", use_container_width=True):
            st.session_state["authenticated"] = False
            st.rerun()


# ----------------------------------------------------------------------------
# CHARGEMENT & NETTOYAGE DES DONNÉES
# ----------------------------------------------------------------------------
@st.cache_data
def load_and_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # 1. Suppression des doublons
    df = df.drop_duplicates(subset=["shipment_id"]).reset_index(drop=True)

    # 2. Conversion des types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    numeric_cols = [
        "distance_km", "avg_speed_kmh", "planned_duration_min", "actual_duration_min",
        "delay_min", "quantity_tons", "capacity_utilization", "fuel_liters",
        "co2_kg", "cost_mad", "on_time", "temperature_breach",
        "o_lat", "o_lon", "d_lat", "d_lon", "lat", "lon",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 3. Traitement des valeurs manquantes : imputation par la médiane
    for c in ["delay_min", "fuel_liters"]:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    # recalcul cohérent après imputation
    df["co2_kg"] = df["co2_kg"].fillna(df["fuel_liters"] * 2.68)

    # 4. Suppression des lignes sans date ou sans coordonnées essentielles
    df = df.dropna(subset=["date", "origin_hub", "destination_city"]).reset_index(drop=True)

    # 5. Colonnes dérivées utiles au dashboard
    df["lane"] = df["origin_hub"] + " → " + df["destination_city"]
    df["cost_per_km"] = df["cost_mad"] / df["distance_km"].replace(0, np.nan)
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["weekday"] = df["date"].dt.day_name()
    df["speed_gap_kmh"] = df["avg_speed_kmh"] - (
        df["distance_km"] / (df["planned_duration_min"] / 60)
    )

    return df


# ----------------------------------------------------------------------------
# TABLEAU DE BORD PRINCIPAL
# ----------------------------------------------------------------------------
def render_header():
    c1, c2 = st.columns([0.12, 0.88])
    with c1:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=80)
    with c2:
        st.markdown(
            "<div class='copag-header'>"
            "<div><h1 style='margin-bottom:0;'>COPAG — Dashboard Logistique &amp; Transport</h1>"
            "<span class='copag-badge'>LIVRAISON À TRAVERS TOUT LE MAROC</span></div>"
            "</div>",
            unsafe_allow_html=True,
        )


def sidebar_filters(df: pd.DataFrame):
    st.sidebar.markdown("## 🔎 Filtres")

    min_date, max_date = df["date"].min().date(), df["date"].max().date()
    date_range = st.sidebar.date_input(
        "Période",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    regions = sorted(df["destination_region"].dropna().unique().tolist())
    sel_regions = st.sidebar.multiselect("Région de destination", regions, default=regions)

    hubs = sorted(df["origin_hub"].dropna().unique().tolist())
    sel_hubs = st.sidebar.multiselect("Hub d'origine", hubs, default=hubs)

    products = sorted(df["product_category"].dropna().unique().tolist())
    sel_products = st.sidebar.multiselect("Catégorie de produit", products, default=products)

    modes = sorted(df["mode"].dropna().unique().tolist())
    sel_modes = st.sidebar.multiselect("Mode de transport", modes, default=modes)

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Connecté en tant que **{st.session_state.get('user','')}**")

    mask = (
        (df["date"].dt.date >= start_date)
        & (df["date"].dt.date <= end_date)
        & (df["destination_region"].isin(sel_regions))
        & (df["origin_hub"].isin(sel_hubs))
        & (df["product_category"].isin(sel_products))
        & (df["mode"].isin(sel_modes))
    )
    return df.loc[mask].copy()


def render_kpis(df: pd.DataFrame):
    st.subheader("📊 Indicateurs clés (KPIs)")
    if df.empty:
        st.warning("Aucune donnée pour les filtres sélectionnés.")
        return

    shipments = len(df)
    on_time_rate = df["on_time"].mean() * 100
    avg_cost = df["cost_mad"].mean()
    avg_distance = df["distance_km"].mean()
    avg_speed = df["avg_speed_kmh"].mean()
    avg_delay = df["delay_min"].mean()
    avg_fuel = df["fuel_liters"].mean()
    avg_co2 = df["co2_kg"].mean()
    breach_rate = df["temperature_breach"].mean() * 100

    row1 = st.columns(4)
    row1[0].metric("Expéditions", f"{shipments:,}".replace(",", " "))
    row1[1].metric("Taux à l'heure", f"{on_time_rate:.1f} %")
    row1[2].metric("Coût moyen", f"{avg_cost:,.0f} MAD".replace(",", " "))
    row1[3].metric("Retard moyen", f"{avg_delay:.1f} min")

    row2 = st.columns(4)
    row2[0].metric("Distance moyenne", f"{avg_distance:,.0f} km".replace(",", " "))
    row2[1].metric("Vitesse moyenne", f"{avg_speed:.1f} km/h")
    row2[2].metric("Carburant moyen", f"{avg_fuel:.1f} L")
    row2[3].metric("Rupture froid", f"{breach_rate:.1f} %", delta_color="inverse")


def fig_daily_on_time(df):
    daily = df.groupby(df["date"].dt.date)["on_time"].mean().mul(100).reset_index()
    daily.columns = ["date", "on_time_rate"]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(daily["date"], daily["on_time_rate"], color=COPAG_GREEN, linewidth=2)
    ax.fill_between(daily["date"], daily["on_time_rate"], color=COPAG_GREEN, alpha=0.12)
    ax.set_title("Performance quotidienne des livraisons à l'heure")
    ax.set_xlabel("Date")
    ax.set_ylabel("Taux à l'heure (%)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def fig_top_routes(df):
    top_routes = df["lane"].value_counts().head(10).reset_index()
    top_routes.columns = ["lane", "count"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.barplot(data=top_routes, x="count", y="lane", ax=ax, color=COPAG_GREEN)
    ax.set_title("Top 10 des routes par nombre d'expéditions")
    ax.set_xlabel("Nombre d'expéditions")
    ax.set_ylabel("Route")
    fig.tight_layout()
    return fig


def fig_delay_by_product(df):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.boxplot(data=df, x="delay_min", y="product_category", ax=ax, color=COPAG_GREEN)
    ax.set_title("Distribution des retards par catégorie de produit")
    ax.set_xlabel("Retard (minutes)")
    ax.set_ylabel("Catégorie de produit")
    fig.tight_layout()
    return fig


def fig_speed_vs_distance(df):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.scatter(df["distance_km"], df["avg_speed_kmh"], alpha=0.45, color=COPAG_GREEN, s=22)
    ax.set_title("Vitesse moyenne en fonction de la distance")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Vitesse moyenne (km/h)")
    fig.tight_layout()
    return fig


def fig_speed_distribution(df):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.hist(df["avg_speed_kmh"].dropna(), bins=30, color=COPAG_GREEN, edgecolor="white")
    ax.set_title("Distribution des vitesses moyennes des trajets")
    ax.set_xlabel("Vitesse (km/h)")
    ax.set_ylabel("Nombre d'expéditions")
    fig.tight_layout()
    return fig


def fig_cost_per_km_region(df):
    cost_region = (
        df.groupby("destination_region")["cost_per_km"].mean().sort_values(ascending=False).reset_index()
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(cost_region["destination_region"], cost_region["cost_per_km"], color=COPAG_GREEN)
    ax.set_title("Coût moyen par km selon la région")
    ax.set_ylabel("MAD / km")
    ax.tick_params(axis="x", rotation=45)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    fig.tight_layout()
    return fig


def fig_corr_heatmap(df):
    numeric_cols = [
        "distance_km", "avg_speed_kmh", "planned_duration_min", "actual_duration_min",
        "delay_min", "on_time", "capacity_utilization", "fuel_liters", "co2_kg",
        "cost_mad", "quantity_tons",
    ]
    corr = df[numeric_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9.5, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Greens", ax=ax)
    ax.set_title("Matrice de corrélation")
    fig.tight_layout()
    return fig


def fig_product_share(df):
    counts = df["product_category"].value_counts()
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90, colors=PALETTE)
    ax.set_title("Répartition des expéditions par produit")
    ax.axis("equal")
    fig.tight_layout()
    return fig

def fig_capacity_by_mode(df):
    cap = df.groupby("mode")["capacity_utilization"].mean().mul(100).reset_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(cap["mode"], cap["capacity_utilization"], color=COPAG_GREEN)
    ax.set_title("Utilisation moyenne de la capacité par mode de transport")
    ax.set_ylabel("Utilisation (%)")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    return fig


def fig_breach_by_product(df):
    breach = df.groupby("product_category")["temperature_breach"].mean().mul(100).reset_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(breach["product_category"], breach["temperature_breach"], color=COPAG_ACCENT)
    ax.set_title("Taux de rupture de chaîne du froid par produit")
    ax.set_ylabel("Taux de rupture (%)")
    ax.tick_params(axis="x", rotation=30)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    fig.tight_layout()
    return fig


def fig_geo_positions(df):
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.scatter(df["lon"], df["lat"], alpha=0.5, color=COPAG_GREEN, s=22)
    hubs = df.groupby("origin_hub")[["o_lat", "o_lon"]].first().reset_index()
    ax.scatter(hubs["o_lon"], hubs["o_lat"], color=COPAG_ACCENT, s=140, marker="*", edgecolor="black", zorder=5)
    for _, r in hubs.iterrows():
        ax.annotate(r["origin_hub"], (r["o_lon"], r["o_lat"]), fontsize=9, weight="bold")
    ax.set_title("Positions géographiques des livraisons (Maroc)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig_flows(df):
    flows = (
        df.groupby(["origin_hub", "destination_city", "o_lat", "o_lon", "d_lat", "d_lon"])
        .size()
        .reset_index(name="count")
    )
    fig, ax = plt.subplots(figsize=(8.5, 8))
    for _, row in flows.iterrows():
        ax.plot(
            [row["o_lon"], row["d_lon"]], [row["o_lat"], row["d_lat"]],
            linewidth=max(0.4, row["count"] / 25), alpha=0.3, color=COPAG_GREEN,
        )
    hubs = df.groupby("origin_hub")[["o_lat", "o_lon"]].first().reset_index()
    for _, r in hubs.iterrows():
        ax.scatter(r["o_lon"], r["o_lat"], color=COPAG_DARK, s=130, zorder=5)
        ax.text(r["o_lon"], r["o_lat"], r["origin_hub"], fontsize=9, weight="bold")
    ax.set_title("Flux de transport entre hubs et villes")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig_delay_by_region(df):
    delay_region = df.groupby("destination_region")["delay_min"].mean().sort_values(ascending=False).reset_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(delay_region["destination_region"], delay_region["delay_min"], color=COPAG_GREEN)
    ax.set_title("Retard moyen par région")
    ax.set_ylabel("Retard moyen (min)")
    ax.tick_params(axis="x", rotation=45)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    fig.tight_layout()
    return fig


def render_dashboard(df_full: pd.DataFrame):
    render_header()
    df = sidebar_filters(df_full)
    logout_button()

    render_kpis(df)
    st.write("")

    tabs = st.tabs([
        "🏁 Performance", "🗺️ Routes & Régions", "🌡️ Qualité & Environnement",
        "📍 Carte", "🧾 Données"
    ])

    with tabs[0]:
        if not df.empty:
            st.pyplot(fig_daily_on_time(df))
            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(fig_speed_vs_distance(df))
            with c2:
                st.pyplot(fig_speed_distribution(df))
            st.pyplot(fig_corr_heatmap(df))

    with tabs[1]:
        if not df.empty:
            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(fig_top_routes(df))
            with c2:
                st.pyplot(fig_delay_by_region(df))
            st.pyplot(fig_cost_per_km_region(df))
            st.pyplot(fig_delay_by_product(df))

    with tabs[2]:
        if not df.empty:
            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(fig_product_share(df))
            with c2:
                st.pyplot(fig_capacity_by_mode(df))
            st.pyplot(fig_breach_by_product(df))

    with tabs[3]:
        if not df.empty:
            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(fig_geo_positions(df))
            with c2:
                st.pyplot(fig_flows(df))

    with tabs[4]:
        st.markdown("#### Aperçu des données filtrées")
        st.dataframe(df.head(200), use_container_width=True)
        st.download_button(
            "⬇️ Télécharger les données filtrées (CSV)",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name="copag_donnees_filtrees.csv",
            mime="text/csv",
        )


# ----------------------------------------------------------------------------
# POINT D'ENTRÉE
# ----------------------------------------------------------------------------
def main():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        login_page()
        return

    if not os.path.exists(DATA_PATH):
        st.error(f"Fichier de données introuvable : {DATA_PATH}")
        return

    df_full = load_and_clean_data(DATA_PATH)
    render_dashboard(df_full)


if __name__ == "__main__":
    main()
