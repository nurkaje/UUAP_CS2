# Case Study 2 – The AI Learning Paradox: When Help Becomes Dependency

This project was developed for the course **Introduction to Data Analysis**. The topic of the project is the analysis of how generative artificial intelligence affects students’ academic performance, knowledge retention, and burnout risk.

The main idea of the study is to examine whether there is a boundary between useful and risky use of GenAI tools. The project focuses on whether moderate AI usage can support learning, while excessive AI usage may increase AI dependency, exam anxiety, and burnout risk.

---

## Research Question

Does moderate use of GenAI tools improve students’ academic performance and knowledge retention, while excessive use increases AI dependency, anxiety, and burnout risk?

---

## Dataset

The dataset used in this project is:

```text
ai_student_impact_dataset.csv
```

The dataset contains:

* 50,000 rows
* 16 columns
* students’ GPA before and after the semester
* weekly GenAI usage hours
* traditional study hours
* perceived AI dependency
* exam anxiety level
* skill retention score
* burnout risk level

No duplicate rows or missing values were found during the initial data check.

---

## Key Variables

The most important variables used in the analysis are:

* `Pre_Semester_GPA` – GPA before the semester
* `Post_Semester_GPA` – GPA after the semester
* `Weekly_GenAI_Hours` – weekly hours of GenAI tool usage
* `Traditional_Study_Hours` – weekly hours of traditional studying
* `Primary_Use_Case` – main purpose of AI tool usage
* `Prompt_Engineering_Skill` – prompt engineering skill level
* `Perceived_AI_Dependency` – perceived dependency on AI tools
* `Anxiety_Level_During_Exams` – anxiety level during exams
* `Skill_Retention_Score` – knowledge retention score
* `Burnout_Risk_Level` – burnout risk level

---

## Created Variables

Several additional variables were created during data processing.

### `GPA_Change`

```text
GPA_Change = Post_Semester_GPA - Pre_Semester_GPA
```

This variable shows the change in GPA during the semester.

### `AI_Usage_Group`

Students were divided into four groups based on their weekly GenAI usage:

* `0-2h`
* `2-6h`
* `6-12h`
* `12h+`

### `Dependency_Group`

Perceived AI dependency was divided into three groups:

* low dependency
* medium dependency
* high dependency

### `High_Burnout`

A binary variable that indicates whether a student belongs to the high burnout risk group.

### `AI_Learning_Risk_Index`

An index that combines:

* weekly AI usage hours
* perceived AI dependency
* exam anxiety
* inversely scaled skill retention score

A higher index value indicates a higher risk pattern in AI-supported learning.

---

## Methodology

The analysis was conducted through several phases:

1. Loading and checking the dataset
2. Checking duplicate rows and missing values
3. Creating new analytical variables
4. Grouping students based on GenAI usage intensity
5. Exploratory Data Analysis
6. Visualizing the relationship between GPA change, knowledge retention, and burnout risk
7. Random Forest feature importance analysis
8. KMeans student segmentation

---

## Technologies Used

The project was developed in **Python**.

The following libraries were used:

* `pandas` – data loading, processing, and analysis
* `numpy` – numerical operations
* `matplotlib` – data visualization
* `scikit-learn` – Random Forest and KMeans analysis

---

## Project Structure

```text
UUAP_CS2/
│
├── ai_student_impact_dataset.csv
├── main.py
├── lab_pipeline.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── output/
│   ├── ocisceni_ai_student_dataset.csv
│   │
│   ├── figures/
│   │   ├── 00_osnovni_pregled_dataseta_3in1.png
│   │   ├── 01_struktura_studenata_po_ai_grupama.png
│   │   ├── 02_ai_paradoks_dashboard_3in1.png
│   │   ├── 03_ai_mapa_rizika_heatmap.png
│   │   ├── 04_radar_profil_ai_ucenja.png
│   │   ├── 05_kompas_burnout_rizika_radial.png
│   │   ├── 06_ai_sweet_spot_kvadrant.png
│   │   ├── 07_nacin_koristenja_ai_dashboard_3in1.png
│   │   ├── 08_random_forest_vaznost_varijabli.png
│   │   └── 09_kmeans_segmentacija_studenata.png
│   │
│   └── tables/
│       ├── ai_grupe_summary.csv
│       ├── nacin_koristenja_ai_summary.csv
│       ├── osnovne_informacije_o_datasetu.csv
│       ├── random_forest_vaznost_varijabli.csv
│       └── kmeans_segmenti_studenata.csv
```

---

## How to Run the Project

First, install the required libraries:

```bash
pip install -r requirements.txt
```

Then run the project with:

```bash
python main.py
```

The program automatically:

* loads the dataset
* checks basic dataset information
* creates new variables
* generates a cleaned dataset
* creates summary tables
* generates 300 dpi charts
* runs Random Forest analysis
* runs KMeans segmentation

---

## Output Files

After running the program, the following folder is generated:

```text
output/
```

It contains the generated dataset, charts, and summary tables.

### `output/figures/`

This folder contains all charts used in the documentation and poster.

The most important charts are:

* `02_ai_paradoks_dashboard_3in1.png`
* `03_ai_mapa_rizika_heatmap.png`
* `04_radar_profil_ai_ucenja.png`
* `08_random_forest_vaznost_varijabli.png`
* `09_kmeans_segmentacija_studenata.png`

### `output/tables/`

This folder contains summary tables used to verify results and support the written documentation.

### `output/ocisceni_ai_student_dataset.csv`

This is the cleaned version of the dataset with additional created variables.

---

## Main Findings

The analysis indicates the existence of an **AI sweet spot**.

Students who use GenAI tools for approximately **6 to 12 hours per week** show the best balance between GPA improvement and knowledge retention.

On the other hand, students who use AI tools for **12 or more hours per week** show a higher burnout risk and lower knowledge retention.

The AI risk heatmap shows that the riskiest combination is high weekly AI usage together with high perceived AI dependency.

Random Forest analysis shows which variables are most important for predicting high burnout risk, while KMeans segmentation divides students into different behavioral patterns based on their AI usage and learning outcomes.

---

## Conclusion

GenAI tools can be useful for learning, but their value depends on how and how much they are used. Moderate and conscious AI usage can be associated with better academic outcomes, while excessive usage can increase the risk of AI dependency, anxiety, and burnout.

The main conclusion of the project is that AI is not automatically good or bad for learning. The key factor is whether students use it as a tool for understanding or as a replacement for independent thinking.

---

