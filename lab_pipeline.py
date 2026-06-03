from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. PODEŠAVANJA

DATA_PATH = Path("ai_student_impact_dataset.csv")
OUTPUT_DIR = Path("output")
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"

AI_GROUP_ORDER = ["0-2h", "2-6h", "6-12h", "12h+"]
DEPENDENCY_ORDER = ["Niska zavisnost", "Srednja zavisnost", "Visoka zavisnost"]

# Purple / gray / gold paleta za glavne grafikone
PURPLE = "#6A0DAD"
LIGHT_PURPLE = "#B57EDC"
DARK_PURPLE = "#3B0764"

GOLD = "#D4AF37"
LIGHT_GOLD = "#F5D76E"

GRAY = "#6B7280"
LIGHT_GRAY = "#D1D5DB"
DARK_GRAY = "#374151"

AI_USAGE_COLORS = [LIGHT_GRAY, LIGHT_PURPLE, GOLD, DARK_GRAY]

# Boje za KMeans segmente
BLUE = "#2563EB"
GREEN = "#16A34A"
RED = "#DC2626"
YELLOW = "#EAB308"

SEGMENT_COLORS = {
    "Tradicionalni učenici": BLUE,
    "Produktivni sweet spot": GREEN,
    "Rizična AI zavisnost": RED,
    "Umjereni korisnici": YELLOW
}

RISK_CMAP = LinearSegmentedColormap.from_list(
    "risk_map",
    [LIGHT_GRAY, LIGHT_GOLD, PURPLE]
)

# 2. OSNOVNE FUNKCIJE

def setup_plot_style():
    """
    Profesionalniji izgled grafikona.
    """
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def create_folders():
    OUTPUT_DIR.mkdir(exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset():
    df = pd.read_csv(DATA_PATH)

    print("INFO O DATASETU")
    print("Broj redova:", df.shape[0])
    print("Broj kolona:", df.shape[1])
    print("Duplikati:", df.duplicated().sum())
    print("Nedostajuće vrijednosti:", df.isna().sum().sum())

    print("\nKolone:")
    for col in df.columns:
        print("-", col)

    return df


def scale_0_100(series):
    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series([50] * len(series), index=series.index)

    return (series - minimum) / (maximum - minimum) * 100


def add_features(df):
    df = df.copy()

    # Promjena GPA tokom semestra.
    df["GPA_Change"] = df["Post_Semester_GPA"] - df["Pre_Semester_GPA"]

    # Podjela studenata prema sedmičnoj upotrebi GenAI alata.
    df["AI_Usage_Group"] = pd.cut(
        df["Weekly_GenAI_Hours"],
        bins=[-0.001, 2, 6, 12, np.inf],
        labels=AI_GROUP_ORDER,
        include_lowest=True
    )

    # Podjela prema percipiranoj AI zavisnosti.
    df["Dependency_Group"] = pd.cut(
        df["Perceived_AI_Dependency"],
        bins=[0, 3, 6, 10],
        labels=DEPENDENCY_ORDER,
        include_lowest=True
    )

    # Binary kolona: 1 ako je burnout visok, 0 ako nije.
    df["High_Burnout"] = (df["Burnout_Risk_Level"] == "High").astype(int)

    # AI Learning Risk Index:
    # veći AI sati + veća zavisnost + veća anksioznost + manji retention = veći rizik.
    ai_hours_scaled = scale_0_100(df["Weekly_GenAI_Hours"]) / 100
    dependency_scaled = scale_0_100(df["Perceived_AI_Dependency"]) / 100
    anxiety_scaled = scale_0_100(df["Anxiety_Level_During_Exams"]) / 100
    retention_scaled = scale_0_100(df["Skill_Retention_Score"]) / 100

    df["AI_Learning_Risk_Index"] = (
        ai_hours_scaled
        + dependency_scaled
        + anxiety_scaled
        + (1 - retention_scaled)
    ) / 4

    return df


def save_figure(filename):
    path = FIGURES_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Sačuvana slika:", path)


def get_ai_summary(df):
    summary = (
        df.groupby("AI_Usage_Group", observed=True)
        .agg(
            broj_studenata=("Student_ID", "count"),
            prosjecna_promjena_gpa=("GPA_Change", "mean"),
            prosjecni_retention=("Skill_Retention_Score", "mean"),
            prosjecna_ai_zavisnost=("Perceived_AI_Dependency", "mean"),
            prosjecna_anksioznost=("Anxiety_Level_During_Exams", "mean"),
            visoki_burnout_proc=("High_Burnout", lambda x: x.mean() * 100),
            prosjecni_risk_index=("AI_Learning_Risk_Index", "mean")
        )
        .reindex(AI_GROUP_ORDER)
        .reset_index()
    )

    summary["AI_Usage_Group"] = summary["AI_Usage_Group"].astype(str)
    return summary


def annotate_bars(ax, decimals=2, suffix=""):
    for patch in ax.patches:
        height = patch.get_height()
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height,
            f"{height:.{decimals}f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold"
        )


def add_total_note(ax, total_students):
    ax.text(
        0.98,
        0.95,
        f"Ukupno: {total_students:,} studenata".replace(",", "."),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=LIGHT_GRAY)
    )

# 3. SUMMARY TABELE

def create_summary_tables(df):
    ai_summary = get_ai_summary(df).round(3)
    ai_summary.to_csv(TABLES_DIR / "ai_grupe_summary.csv", index=False)

    use_case_summary = (
        df.groupby("Primary_Use_Case", observed=True)
        .agg(
            broj_studenata=("Student_ID", "count"),
            prosjecni_retention=("Skill_Retention_Score", "mean"),
            prosjecna_promjena_gpa=("GPA_Change", "mean"),
            visoki_burnout_proc=("High_Burnout", lambda x: x.mean() * 100)
        )
        .round(3)
    )
    use_case_summary.to_csv(TABLES_DIR / "nacin_koristenja_ai_summary.csv")

    basic_info = pd.DataFrame({
        "metrika": [
            "Broj redova",
            "Broj kolona",
            "Broj duplikata",
            "Nedostajuće vrijednosti"
        ],
        "vrijednost": [
            df.shape[0],
            df.shape[1],
            df.duplicated().sum(),
            df.isna().sum().sum()
        ]
    })
    basic_info.to_csv(TABLES_DIR / "osnovne_informacije_o_datasetu.csv", index=False)

    print("\nSUMMARY PO AI GRUPAMA")
    print(ai_summary)

    print("\nSUMMARY PO NAČINU KORIŠTENJA AI")
    print(use_case_summary)

    print("\nTabele su sačuvane u:", TABLES_DIR)

# 4. OSNOVNI GRAFIKONI

def chart_00_basic_dataset_overview(df):
    """
    Osnovni 3-u-1 pregled:
    - distribucija AI sati
    - udio studenata po burnout nivou
    - prosječni GPA prije i poslije semestra
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    total_students = len(df)

    # A) Histogram AI sati
    axes[0].hist(
        df["Weekly_GenAI_Hours"],
        bins=25,
        color=PURPLE,
        edgecolor=DARK_GRAY,
        alpha=0.85
    )
    axes[0].set_title("A) Distribucija sedmičnih AI sati")
    axes[0].set_xlabel("Sedmični GenAI sati")
    axes[0].set_ylabel("Broj studenata po intervalu")
    axes[0].grid(axis="y", alpha=0.25)
    add_total_note(axes[0], total_students)

    # B) Burnout nivoi kao procenti
    burnout_order = ["Low", "Medium", "High"]
    burnout_percent = (
        df["Burnout_Risk_Level"]
        .value_counts(normalize=True)
        .reindex(burnout_order)
        .dropna()
        * 100
    )

    burnout_labels = {
        "Low": "Nizak",
        "Medium": "Srednji",
        "High": "Visok"
    }

    axes[1].bar(
        [burnout_labels.get(x, x) for x in burnout_percent.index],
        burnout_percent.values,
        color=[LIGHT_GRAY, GOLD, PURPLE][:len(burnout_percent)],
        edgecolor=DARK_GRAY
    )
    axes[1].set_title("B) Udio studenata po burnout nivou")
    axes[1].set_xlabel("Burnout nivo")
    axes[1].set_ylabel("Udio studenata (%)")
    axes[1].grid(axis="y", alpha=0.25)

    for i, value in enumerate(burnout_percent.values):
        axes[1].text(
            i,
            value,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )

    # C) GPA prije i poslije
    gpa_values = [
        df["Pre_Semester_GPA"].mean(),
        df["Post_Semester_GPA"].mean()
    ]

    axes[2].bar(
        ["GPA prije\nsemestra", "GPA poslije\nsemestra"],
        gpa_values,
        color=[LIGHT_PURPLE, GOLD],
        edgecolor=DARK_GRAY
    )
    axes[2].set_title("C) Prosječni GPA prije i poslije semestra")
    axes[2].set_ylabel("Prosječni GPA")
    axes[2].set_ylim(0, 4)
    axes[2].grid(axis="y", alpha=0.25)

    for i, value in enumerate(gpa_values):
        axes[2].text(
            i,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )

    fig.suptitle(
        "Osnovni pregled dataseta",
        fontsize=15,
        fontweight="bold"
    )

    save_figure("00_osnovni_pregled_dataseta_3in1.png")


def chart_01_ai_group_basic_comparison(df):
    """
    Osnovni grafikon:
    prikazuje procenat i broj studenata u svakoj AI usage grupi.
    """
    summary = get_ai_summary(df)
    total_students = summary["broj_studenata"].sum()
    summary["procenat"] = summary["broj_studenata"] / total_students * 100

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        summary["AI_Usage_Group"],
        summary["procenat"],
        color=AI_USAGE_COLORS,
        edgecolor=DARK_GRAY,
        linewidth=0.8
    )

    ax.set_title("Struktura studenata prema sedmičnom korištenju GenAI alata", fontweight="bold")
    ax.set_xlabel("Sedmična upotreba GenAI alata")
    ax.set_ylabel("Udio studenata (%)")
    ax.grid(axis="y", alpha=0.25)

    for i, row in summary.iterrows():
        ax.text(
            i,
            row["procenat"],
            f"{row['procenat']:.1f}%\n(n={int(row['broj_studenata'])})",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold"
        )

    save_figure("01_struktura_studenata_po_ai_grupama.png")

# 5. WOW GRAFIKONI

def chart_02_ai_paradox_dashboard(df):
    """
    Glavni 3-u-1 grafikon:
    - GPA rast
    - zadržavanje znanja
    - burnout rizik
    """
    summary = get_ai_summary(df)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].bar(
        summary["AI_Usage_Group"],
        summary["prosjecna_promjena_gpa"],
        color=AI_USAGE_COLORS,
        edgecolor=DARK_GRAY,
        linewidth=0.8
    )
    axes[0].set_title("A) Promjena GPA")
    axes[0].set_xlabel("Sedmična upotreba GenAI alata")
    axes[0].set_ylabel("Prosječna promjena GPA")
    axes[0].set_ylim(0, 0.26)
    annotate_bars(axes[0], decimals=3)

    axes[1].bar(
        summary["AI_Usage_Group"],
        summary["prosjecni_retention"],
        color=AI_USAGE_COLORS,
        edgecolor=DARK_GRAY,
        linewidth=0.8
    )
    axes[1].set_title("B) Zadržavanje znanja")
    axes[1].set_xlabel("Sedmična upotreba GenAI alata")
    axes[1].set_ylabel("Prosječni retention score")
    axes[1].set_ylim(70, 79)
    annotate_bars(axes[1], decimals=1)

    axes[2].bar(
        summary["AI_Usage_Group"],
        summary["visoki_burnout_proc"],
        color=AI_USAGE_COLORS,
        edgecolor=DARK_GRAY,
        linewidth=0.8
    )
    axes[2].set_title("C) Visok burnout rizik")
    axes[2].set_xlabel("Sedmična upotreba GenAI alata")
    axes[2].set_ylabel("Udio visokog burnout rizika")
    axes[2].set_ylim(0, 65)
    annotate_bars(axes[2], decimals=1, suffix="%")

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "AI paradoks učenja: ocjene, zadržavanje znanja i burnout",
        fontsize=15,
        fontweight="bold"
    )

    save_figure("02_ai_paradoks_dashboard_3in1.png")


def chart_03_risk_heatmap(df):
    """
    Heatmapa:
    kombinuje AI sate i AI zavisnost.
    """
    pivot = (
        df.groupby(["Dependency_Group", "AI_Usage_Group"], observed=True)
        .agg(visoki_burnout_proc=("High_Burnout", lambda x: x.mean() * 100))
        .reset_index()
        .pivot(
            index="Dependency_Group",
            columns="AI_Usage_Group",
            values="visoki_burnout_proc"
        )
        .reindex(index=DEPENDENCY_ORDER, columns=AI_GROUP_ORDER)
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))

    im = ax.imshow(pivot.values, cmap=RISK_CMAP)

    ax.set_xticks(np.arange(len(AI_GROUP_ORDER)))
    ax.set_yticks(np.arange(len(DEPENDENCY_ORDER)))

    ax.set_xticklabels(AI_GROUP_ORDER)
    ax.set_yticklabels(["Niska\n1-3", "Srednja\n4-6", "Visoka\n7-10"])

    ax.set_xlabel("Sedmična upotreba GenAI alata")
    ax.set_ylabel("Percipirana AI zavisnost")
    ax.set_title("AI mapa rizika: udio visokog burnout rizika", fontweight="bold")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if pd.notna(value):
                ax.text(
                    j,
                    i,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold"
                )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Udio visokog burnout rizika (%)")

    save_figure("03_ai_mapa_rizika_heatmap.png")


def chart_04_ai_learning_radar(df):
    """
    Radar chart:
    sve ose su okrenute tako da veća vrijednost znači zdraviji profil učenja.
    """
    summary = get_ai_summary(df)

    radar = summary.copy()

    radar["GPA napredak"] = scale_0_100(radar["prosjecna_promjena_gpa"])
    radar["Zadržavanje znanja"] = scale_0_100(radar["prosjecni_retention"])

    # Ove metrike okrećemo jer je niža vrijednost bolja.
    radar["Sigurnost od burnouta"] = 100 - radar["visoki_burnout_proc"]
    radar["Niska AI zavisnost"] = (10 - radar["prosjecna_ai_zavisnost"]) / 10 * 100
    radar["Niska anksioznost"] = (10 - radar["prosjecna_anksioznost"]) / 10 * 100
    radar["Nizak AI rizik"] = (1 - radar["prosjecni_risk_index"]) * 100

    metrics = [
        "GPA napredak",
        "Zadržavanje znanja",
        "Sigurnost od burnouta",
        "Niska AI zavisnost",
        "Niska anksioznost",
        "Nizak AI rizik"
    ]

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(9, 8))
    ax = plt.subplot(111, polar=True)

    for i, row in radar.iterrows():
        values = [row[m] for m in metrics]
        values += values[:1]

        ax.plot(
            angles,
            values,
            linewidth=2.2,
            label=row["AI_Usage_Group"],
            color=AI_USAGE_COLORS[i]
        )

        ax.fill(
            angles,
            values,
            alpha=0.10,
            color=AI_USAGE_COLORS[i]
        )

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)

    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8)

    ax.set_title(
        "Radar profil AI učenja\nVeća vrijednost = zdraviji profil učenja",
        pad=25,
        fontsize=14,
        fontweight="bold"
    )

    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10), frameon=False)
    ax.grid(alpha=0.35)

    save_figure("04_radar_profil_ai_ucenja.png")


def chart_05_burnout_compass_radial(df):
    """
    Radial chart:
    upečatljivo prikazuje skok burnout rizika.
    """
    summary = get_ai_summary(df)

    labels = summary["AI_Usage_Group"].tolist()
    values = summary["visoki_burnout_proc"].values

    theta = np.linspace(0.0, 2 * np.pi, len(labels), endpoint=False)
    width = 2 * np.pi / len(labels) * 0.62

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    ax.bar(
        theta,
        values,
        width=width,
        color=AI_USAGE_COLORS,
        edgecolor=DARK_GRAY,
        linewidth=1.2,
        alpha=0.95
    )

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(theta)
    ax.set_xticklabels(labels, fontsize=13, fontweight="bold")

    ax.set_ylim(0, 65)
    ax.set_yticks([10, 25, 40, 55])
    ax.set_yticklabels(["10%", "25%", "40%", "55%"], fontsize=9)

    for angle, value in zip(theta, values):
        ax.text(
            angle,
            value + 5,
            f"{value:.1f}%",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold"
        )

    ax.set_title(
        "Kompas burnout rizika\nUdio visokog burnout rizika po AI grupama",
        pad=28,
        fontsize=14,
        fontweight="bold"
    )

    ax.grid(alpha=0.35)

    save_figure("05_kompas_burnout_rizika_radial.png")


def chart_06_sweet_spot_quadrant(df):
    """
    Bubble / quadrant chart:
    X = promjena GPA
    Y = zadržavanje znanja
    veličina kruga = burnout rizik
    """
    summary = get_ai_summary(df)

    fig, ax = plt.subplots(figsize=(9, 6.5))

    sizes = summary["visoki_burnout_proc"] * 25

    ax.scatter(
        summary["prosjecna_promjena_gpa"],
        summary["prosjecni_retention"],
        s=sizes,
        c=AI_USAGE_COLORS,
        edgecolors=DARK_GRAY,
        linewidths=1.2,
        alpha=0.88
    )

    x_mid = summary["prosjecna_promjena_gpa"].mean()
    y_mid = summary["prosjecni_retention"].mean()

    ax.axvline(x_mid, color=GRAY, linestyle="--", linewidth=1.2)
    ax.axhline(y_mid, color=GRAY, linestyle="--", linewidth=1.2)

    for _, row in summary.iterrows():
        ax.annotate(
            f"{row['AI_Usage_Group']}\nBurnout {row['visoki_burnout_proc']:.1f}%",
            (row["prosjecna_promjena_gpa"], row["prosjecni_retention"]),
            textcoords="offset points",
            xytext=(10, 8),
            fontsize=10,
            fontweight="bold"
        )

    ax.text(
        summary["prosjecna_promjena_gpa"].max(),
        summary["prosjecni_retention"].max() + 0.05,
        "Sweet spot",
        ha="right",
        fontsize=12,
        color=DARK_PURPLE,
        fontweight="bold"
    )

    ax.text(
        summary["prosjecna_promjena_gpa"].min(),
        summary["prosjecni_retention"].min() - 0.30,
        "Rizična zona",
        ha="left",
        fontsize=12,
        color=DARK_GRAY,
        fontweight="bold"
    )

    ax.set_title(
        "AI sweet spot kvadrant\nGPA napredak vs. zadržavanje znanja, veličina kruga = burnout rizik",
        fontsize=14,
        fontweight="bold"
    )
    ax.set_xlabel("Prosječna promjena GPA")
    ax.set_ylabel("Prosječni retention score")
    ax.grid(alpha=0.25)

    save_figure("06_ai_sweet_spot_kvadrant.png")


def chart_07_use_case_dashboard(df):
    """
    3-u-1 dashboard:
    pokazuje da nije bitno samo koliko se AI koristi, nego i za šta se koristi.
    """
    summary = (
        df.groupby("Primary_Use_Case", observed=True)
        .agg(
            prosjecni_retention=("Skill_Retention_Score", "mean"),
            prosjecna_promjena_gpa=("GPA_Change", "mean"),
            visoki_burnout_proc=("High_Burnout", lambda x: x.mean() * 100)
        )
        .reset_index()
    )

    label_map = {
        "Copywriting/Drafting": "Pisanje / draft",
        "Ideation": "Ideje",
        "Summarizing_Reading": "Sažimanje",
        "Debugging/Troubleshooting": "Debugging",
        "Direct_Answer_Generation": "Direktni odgovori"
    }

    summary["Nacin_koristenja"] = summary["Primary_Use_Case"].map(label_map).fillna(summary["Primary_Use_Case"])

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    retention = summary.sort_values("prosjecni_retention", ascending=True)
    axes[0].barh(
        retention["Nacin_koristenja"],
        retention["prosjecni_retention"],
        color=PURPLE,
        edgecolor=DARK_GRAY
    )
    axes[0].set_title("A) Zadržavanje znanja")
    axes[0].set_xlabel("Prosječni retention score")

    gpa = summary.sort_values("prosjecna_promjena_gpa", ascending=True)
    axes[1].barh(
        gpa["Nacin_koristenja"],
        gpa["prosjecna_promjena_gpa"],
        color=GOLD,
        edgecolor=DARK_GRAY
    )
    axes[1].set_title("B) Promjena GPA")
    axes[1].set_xlabel("Prosječna promjena GPA")

    burnout = summary.sort_values("visoki_burnout_proc", ascending=True)
    axes[2].barh(
        burnout["Nacin_koristenja"],
        burnout["visoki_burnout_proc"],
        color=GRAY,
        edgecolor=DARK_GRAY
    )
    axes[2].set_title("C) Burnout rizik")
    axes[2].set_xlabel("Udio visokog burnout rizika (%)")

    for ax in axes:
        ax.grid(axis="x", alpha=0.25)

    fig.suptitle(
        "Nije svaka AI upotreba ista: poređenje načina korištenja",
        fontsize=15,
        fontweight="bold"
    )

    save_figure("07_nacin_koristenja_ai_dashboard_3in1.png")


def chart_08_random_forest_importance(df):
    """
    Random Forest:
    pokazuje koje numeričke varijable su najvažnije za predviđanje visokog burnout rizika.
    """
    df_model = df.copy()

    year_map = {
        "Freshman": 1,
        "Sophomore": 2,
        "Junior": 3,
        "Senior": 4,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4
    }

    df_model["Year_of_Study_Num"] = df_model["Year_of_Study"].astype(str).map(year_map)
    df_model["Year_of_Study_Num"] = df_model["Year_of_Study_Num"].fillna(
        df_model["Year_of_Study_Num"].median()
    )

    features = [
        "Weekly_GenAI_Hours",
        "Traditional_Study_Hours",
        "Perceived_AI_Dependency",
        "Anxiety_Level_During_Exams",
        "Skill_Retention_Score",
        "GPA_Change",
        "AI_Learning_Risk_Index",
        "Pre_Semester_GPA",
        "Post_Semester_GPA",
        "Tool_Diversity",
        "Year_of_Study_Num"
    ]

    X = df_model[features]
    y = df_model["High_Burnout"]

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        max_depth=8,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("\nRANDOM FOREST")
    print("Tačnost modela:", round(accuracy, 3))

    feature_names_bosnian = {
        "Weekly_GenAI_Hours": "Sedmični AI sati",
        "Traditional_Study_Hours": "Tradicionalno učenje",
        "Perceived_AI_Dependency": "AI zavisnost",
        "Anxiety_Level_During_Exams": "Anksioznost tokom ispita",
        "Skill_Retention_Score": "Zadržavanje znanja",
        "GPA_Change": "Promjena GPA",
        "AI_Learning_Risk_Index": "AI risk indeks",
        "Pre_Semester_GPA": "GPA prije semestra",
        "Post_Semester_GPA": "GPA poslije semestra",
        "Tool_Diversity": "Raznolikost AI alata",
        "Year_of_Study_Num": "Godina studija"
    }

    importance = pd.DataFrame({
        "varijabla": [feature_names_bosnian[f] for f in features],
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=True)

    importance.to_csv(TABLES_DIR / "random_forest_vaznost_varijabli.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.barh(
        importance["varijabla"],
        importance["importance"],
        color=PURPLE,
        edgecolor=DARK_GRAY,
        linewidth=0.8
    )

    ax.set_title(
        f"Random Forest: važnost varijabli za burnout rizik\nPredviđanje visokog burnout rizika | Tačnost: {accuracy:.3f}",
        fontsize=14,
        fontweight="bold"
    )
    ax.set_xlabel("Važnost varijable")
    ax.set_ylabel("Varijabla")
    ax.grid(axis="x", alpha=0.25)

    save_figure("08_random_forest_vaznost_varijabli.png")


def assign_segment_names(cluster_summary):
    """
    KMeans daje samo brojeve klastera: 0, 1, 2, 3.
    Ovdje tim klasterima dajemo smislenija imena na osnovu karakteristika.
    """
    summary = cluster_summary.copy()
    segment_names = {}

    # 1. Najrizičniji segment
    risk_cluster = summary["prosjecni_risk_index"].idxmax()
    segment_names[risk_cluster] = "Rizična AI zavisnost"

    remaining = summary.drop(index=risk_cluster)

    # 2. Produktivni segment: dobar GPA rast + dobar retention
    productivity_score = (
        scale_0_100(remaining["prosjecna_promjena_gpa"])
        + scale_0_100(remaining["prosjecni_retention"])
    ) / 2

    productive_cluster = productivity_score.idxmax()
    segment_names[productive_cluster] = "Produktivni sweet spot"

    remaining = remaining.drop(index=productive_cluster)

    # 3. Tradicionalni segment: najmanje AI sati
    traditional_cluster = remaining["prosjecni_ai_sati"].idxmin()
    segment_names[traditional_cluster] = "Tradicionalni učenici"

    remaining = remaining.drop(index=traditional_cluster)

    # 4. Preostali segment
    for cluster_id in remaining.index:
        segment_names[cluster_id] = "Umjereni korisnici"

    return segment_names


def chart_09_kmeans_student_segments(df):
    """
    KMeans segmentacija:
    automatski dijeli studente u 4 segmenta prema AI ponašanju i ishodima učenja.
    """
    features = [
        "Weekly_GenAI_Hours",
        "Traditional_Study_Hours",
        "Perceived_AI_Dependency",
        "Anxiety_Level_During_Exams",
        "Skill_Retention_Score",
        "GPA_Change",
        "AI_Learning_Risk_Index"
    ]

    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)

    df_clustered = df.copy()
    df_clustered["Student_Cluster"] = kmeans.fit_predict(X_scaled)

    cluster_summary = (
        df_clustered.groupby("Student_Cluster")
        .agg(
            broj_studenata=("Student_ID", "count"),
            prosjecni_ai_sati=("Weekly_GenAI_Hours", "mean"),
            prosjecna_promjena_gpa=("GPA_Change", "mean"),
            prosjecni_retention=("Skill_Retention_Score", "mean"),
            prosjecna_ai_zavisnost=("Perceived_AI_Dependency", "mean"),
            prosjecna_anksioznost=("Anxiety_Level_During_Exams", "mean"),
            visoki_burnout_proc=("High_Burnout", lambda x: x.mean() * 100),
            prosjecni_risk_index=("AI_Learning_Risk_Index", "mean")
        )
        .round(3)
    )

    segment_names = assign_segment_names(cluster_summary)

    df_clustered["Segment"] = df_clustered["Student_Cluster"].map(segment_names)
    cluster_summary["Segment"] = cluster_summary.index.map(segment_names)

    cols = ["Segment"] + [col for col in cluster_summary.columns if col != "Segment"]
    cluster_summary = cluster_summary[cols]

    cluster_summary.to_csv(TABLES_DIR / "kmeans_segmenti_studenata.csv")

    print("\nKMEANS SEGMENTI STUDENATA")
    print(cluster_summary)

    sample = df_clustered.sample(n=min(6000, len(df_clustered)), random_state=42)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    segment_order = [
        "Tradicionalni učenici",
        "Produktivni sweet spot",
        "Umjereni korisnici",
        "Rizična AI zavisnost"
    ]

    # A) Scatter mapa segmenata
    for segment in segment_order:
        subset = sample[sample["Segment"] == segment]

        if len(subset) == 0:
            continue

        axes[0].scatter(
            subset["GPA_Change"],
            subset["Skill_Retention_Score"],
            s=20,
            alpha=0.35,
            color=SEGMENT_COLORS[segment],
            label=segment
        )

    axes[0].set_title("A) Mapa segmenata studenata")
    axes[0].set_xlabel("Promjena GPA")
    axes[0].set_ylabel("Retention score")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    # B) Burnout rizik po segmentima
    plot_summary = cluster_summary.copy()
    plot_summary = plot_summary.sort_values("visoki_burnout_proc", ascending=True)

    bar_colors = [SEGMENT_COLORS[segment] for segment in plot_summary["Segment"]]

    axes[1].barh(
        plot_summary["Segment"],
        plot_summary["visoki_burnout_proc"],
        color=bar_colors,
        edgecolor=DARK_GRAY,
        linewidth=0.8
    )

    axes[1].set_title("B) Visok burnout rizik po segmentima")
    axes[1].set_xlabel("Udio visokog burnout rizika (%)")
    axes[1].set_ylabel("Segment")
    axes[1].grid(axis="x", alpha=0.25)

    for i, value in enumerate(plot_summary["visoki_burnout_proc"]):
        axes[1].text(
            value,
            i,
            f"{value:.1f}%",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold"
        )

    fig.suptitle(
        "KMeans segmentacija studenata: od tradicionalnih učenika do rizične AI zavisnosti",
        fontsize=15,
        fontweight="bold"
    )

    save_figure("09_kmeans_segmentacija_studenata.png")


def create_all_charts(df):
    """
    Pokreće sve grafikone redom.
    """
    print("\nGENERISANJE GRAFIKONA...")

    # Osnovni grafikoni
    chart_00_basic_dataset_overview(df)
    chart_01_ai_group_basic_comparison(df)

    # Glavni wow grafikoni
    chart_02_ai_paradox_dashboard(df)
    chart_03_risk_heatmap(df)
    chart_04_ai_learning_radar(df)
    chart_05_burnout_compass_radial(df)
    chart_06_sweet_spot_quadrant(df)
    chart_07_use_case_dashboard(df)

    # Machine Learning dio
    chart_08_random_forest_importance(df)
    chart_09_kmeans_student_segments(df)

    print("\nSvi grafikoni su sačuvani u folderu output/figures.")

# 6. GLAVNI PROGRAM

def main():
    setup_plot_style()
    create_folders()

    df = load_dataset()
    df = add_features(df)

    df.to_csv(OUTPUT_DIR / "ocisceni_ai_student_dataset.csv", index=False)

    create_summary_tables(df)
    create_all_charts(df)

    print("\nGOTOVO.")
    print("Očišćeni dataset:", OUTPUT_DIR / "ocisceni_ai_student_dataset.csv")
    print("Tabele:", TABLES_DIR)
    print("Grafikoni:", FIGURES_DIR)


if __name__ == "__main__":
    main()