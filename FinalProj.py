#loading data into the data frame
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
import openpyxl
from scipy.signal import find_peaks
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
import streamlit as st

# Loading data sheets according to sample type
treated_waves= pd.read_excel("Sample no Filter.xlsx", sheet_name="Samples", header=None)
clean_waves= pd.read_excel("Sample no Filter.xlsx", sheet_name="Clean Water", header=None)
bw_waves= pd.read_excel("Sample no Filter.xlsx", sheet_name="Blackwater", header=None)

#grouping triplicate measurements and taking the average, then replacing the triplicate measurements with the average in the
#data frame

##blackwater
# Assume first column = sample names, rest = numeric data
bw_waves.columns = ["Sample"] + [f"Col{i}" for i in range(1, bw_waves.shape[1])]
# Extract group prefix (BW1, BW2, BW3...)
bw_waves["Group"] = bw_waves["Sample"].str.extract(r'^(BW\d+)')
# Compute averages across all numeric columns per group
bw_waves = bw_waves.groupby("Group").mean(numeric_only=True).reset_index()


##treated_water
treated_waves = treated_waves.iloc[1:].reset_index(drop=True)

# Rename columns: first col = Sample, rest = numeric data
treated_waves.columns = ["Sample"] + [f"Col{i}" for i in range(1, treated_waves.shape[1])]

# Extract group prefix: everything before the first dash
treated_waves["Group"] = treated_waves["Sample"].astype(str).str.extract(r'^([A-Za-z0-9]+)')

# Compute averages across all numeric columns per group
treated_waves = treated_waves.groupby("Group").mean(numeric_only=True).reset_index()

##clean_water
clean_waves = clean_waves.iloc[1:].reset_index(drop=True)

# Rename columns: first col = Sample, rest = numeric data
clean_waves.columns = ["Sample"] + [f"Col{i}" for i in range(1, clean_waves.shape[1])]

# Extract group prefix: everything before the first dash
clean_waves["Group"] = clean_waves["Sample"].astype(str).str.extract(r'^([A-Za-z0-9]+)')

# Compute averages across all numeric columns per group
clean_waves = clean_waves.groupby("Group").mean(numeric_only=True).reset_index()

#merging related data
tw_bw_clean_waves = pd.concat([treated_waves, bw_waves, clean_waves])


#importing turbidity measurements 
turb = pd.read_excel('physiochemical data.xlsx', sheet_name='Master Physiochem', usecols=['Turbidity (NTU)'])
#checking that data was correctly imported

#importing total suspended solids measurements 
tss = pd.read_excel('physiochemical data.xlsx', sheet_name='TSS', usecols=['TSS (% w/v)'])

#importing total solids measurements 
ts = pd.read_excel('physiochemical data.xlsx', sheet_name='TS', usecols=['%TS (w/w)'])

#missing and not missing
tss_missing = tss[tss.isnull().any(axis=1)]
tss_not_missing = tss.dropna()

#preparing scalar for KNN
scaler = StandardScaler()
tss_scaled = pd.DataFrame(scaler.fit_transform(tss_not_missing), columns = tss_not_missing.columns)

#intialize and fit KNN imputer
imputer = KNNImputer(n_neighbors=5, weights ='distance')
#here we have to make the scatter plot of the data without missing values so they aren't skewed by missing values, then we overlay the missing values 
#on the scatter plot 
#what does .fit do? Training on non-missing data ---it's machine learning and has never seen this data set before, so it has to train
imputer.fit(tss_scaled)

# function to impute and inverse transform the data
def impute_and_inverse_transform(data):
    # Ensure 'data' is always a DataFrame with proper column names
    scaled_data = pd.DataFrame(scaler.transform(data), columns=data.columns, index=data.index)
    imputed_scaled = imputer.transform(scaled_data)
    return pd.DataFrame(scaler.inverse_transform(imputed_scaled), columns=data.columns, index=data.index)

# impute missing values
tss_imputed = impute_and_inverse_transform(tss)

#missing and not missing
turb_missing = turb[turb.isnull().any(axis=1)]
turb_not_missing = turb.dropna()

#preparing scalar for KNN
scaler = StandardScaler()
turb_scaled = pd.DataFrame(scaler.fit_transform(turb_not_missing), columns = turb_not_missing.columns)

#intialize and fit KNN imputer
#imputer = KNNImputer(n_neighbors=5)
#modded above function to include 10 neighbors and weight the imputed value depending on how close the points were-the closer the point the more it 
#affects the imputing value
imputer = KNNImputer(n_neighbors=3, weights ='distance')
#here we have to make the scatter plot of the data without missing values so they aren't skewed by missing values, then we overlay the missing values 
#on the scatter plot 
#what does .fit do? Training on non-missing data ---it's machine learning and has never seen this data set before, so it has to train
imputer.fit(turb_scaled)
# impute missing values
turb_imputed = impute_and_inverse_transform(turb)

#tss and turb data was imputed, no data missing for ts

tw_tss = tss_imputed[0:96]
tw_ts = ts[0:96]
tw_turb = turb_imputed[0:96]

bw_tss = tss_imputed[96:106]
bw_ts = ts[96:106]
bw_turb = turb_imputed[96:106]

#reading sample names
sample_IDs = pd.read_excel('physiochemical data.xlsx', sheet_name='Master Physiochem', usecols=['Sample'])

#merging related data
tw_physio = pd.concat([tw_tss, tw_ts,tw_turb], axis=1)
bw_physio = pd.concat([bw_tss, bw_ts,bw_turb], axis=1)
tw_bw_physio = pd.concat([ tw_physio,bw_physio])

# Add the Sample IDs as a new column
tw_bw_physio.insert(0, 'Sample', sample_IDs['Sample'].values)

# Define column names (excluding 'Sample')
value_cols = [col for col in tw_bw_physio.columns if col != 'Sample']

# Generate random values with different ranges
col1 = np.random.uniform(0, 0.00002, size=(7,))   # TSS (% w/v)
col2 = np.random.uniform(0, 0.00002, size=(7,))   # %TS (w/w)
col3 = np.random.uniform(0, 0.02, size=(7,))     # Turbidity (NTU)

# Stack into DataFrame
new_df = pd.DataFrame(
    np.column_stack([col1, col2, col3]),
    columns=value_cols
)

# Add the Sample IDs as the first column
new_df = pd.concat(
    [clean_waves[['Group']].rename(columns={'Group':'Sample'}).head(7), new_df],
    axis=1
)

# Append to the original DataFrame
tw_bw_clean_physio = pd.concat([tw_bw_physio, new_df], ignore_index=True)

# Assume your DataFrame is called df with columns:
# ['Sample', 'TSS (% w/v)', '%TS (w/w)', 'Turbidity (NTU)']

# Segregating samples according to sample name [treated water (A1, B2, etc), clean (Blank1, Blank2), blackwater feed (BW1, BW2) etc)
def classify_sample(sample):
    if sample.startswith("Blank"):
        return "Clean Water"
    elif sample.startswith("BW"):
        return "Contaminated Water"
    else:
        return "Treated Water"

# Use classify sample
tw_bw_clean_physio['Type'] = tw_bw_clean_physio['Sample'].apply(classify_sample)

# Describe statistics in a separate value
stats = tw_bw_clean_physio.groupby('Type').describe()


grouped_stats = tw_bw_clean_physio.groupby('Type')[['TSS (% w/v)', '%TS (w/w)', 'Turbidity (NTU)']].agg(['mean','std'])

# Extract mean and std separately for plotting
means = grouped_stats.xs('mean', axis=1, level=1)
stds  = grouped_stats.xs('std', axis=1, level=1)

#tw_bw_clean_waves['Type'] = tw_bw_clean_waves['Sample'].apply(classify_sample)
wavelengths = pd.read_excel("Sample no Filter.xlsx", sheet_name="Samples", nrows=1, header = None)
wavelengths = wavelengths.drop(columns=[0])

tw_bw_clean_waves['Type'] = tw_bw_clean_waves['Group'].apply(classify_sample)

# --- Step 2: Extract wavelength values ---
wavelengths_values = wavelengths.values.flatten()  # drop first col if needed

# --- Step 1: Compute mean spectra and find maxima/minima indices ---
mean_spectra = {}
for sample_type, group in tw_bw_clean_waves.groupby('Type'):
    data = group.drop(columns=['Group','Type']).values
    mean_spectra[sample_type] = data.mean(axis=0)

# Find maxima/minima for each type
maxima_sets = []
minima_sets = []
for mean in mean_spectra.values():
    maxima_idx, _ = find_peaks(mean)
    minima_idx, _ = find_peaks(-mean)
    maxima_sets.append(maxima_idx)
    minima_sets.append(minima_idx)

# Function to find common peaks within tolerance
def find_common_peaks(sets, tolerance=5):
    common = []
    ref = sets[0]
    for idx in ref:
        if all(any(abs(idx - other) <= tolerance for other in s) for s in sets[1:]):
            common.append(idx)
    return np.array(common)
common_maxima = find_common_peaks(maxima_sets, tolerance=5)
common_minima = find_common_peaks(minima_sets, tolerance=5)

# --- Step 2: Extract wavelengths at these indices ---
selected_indices = np.sort(np.concatenate([common_maxima, common_minima]))
selected_wavelengths = wavelengths.iloc[0, selected_indices].values

# --- Step 3: Build new DataFrame ---
rows = []
# First row: wavelengths
header_row = ["Sample_ID"] + selected_wavelengths.tolist()
rows.append(header_row)

# Subsequent rows: sample ID + values at selected indices
for _, row in tw_bw_clean_waves.iterrows():
    sample_id = row['Group']
    values = row.drop(labels=['Group','Type']).values[selected_indices]
    rows.append([sample_id] + values.tolist())

# Convert to DataFrame
compeak_sum = pd.DataFrame(rows)

# --- Step 4: Optional formatting ---
compeak_sum.columns = ["Sample_ID"] + [f"Wavelength_{w}" for w in selected_wavelengths]
# Compute summary statistics for each sample type
summary_stats = []

for sample_type, group in tw_bw_clean_waves.groupby('Type'):
    data = group.drop(columns=['Group','Type']).values[:, selected_indices]
    
    stats = {
        "Sample_Type": sample_type,
        "N_samples": data.shape[0]
    }
    
    # For each wavelength, compute stats
    for i, wl in enumerate(selected_wavelengths):
        stats[f"{wl}_mean"]   = data[:, i].mean()
        stats[f"{wl}_std"]    = data[:, i].std()
        stats[f"{wl}_min"]    = data[:, i].min()
        stats[f"{wl}_max"]    = data[:, i].max()
        stats[f"{wl}_median"] = np.median(data[:, i])
    
    summary_stats.append(stats)

# --- Step 4: Convert to DataFrame ---
compeak_summary = pd.DataFrame(summary_stats)
# Ensure both DataFrames have Sample_ID as the first column
if compeak_sum.columns[0] != "Sample_ID":
    compeak_sum = compeak_sum.rename(columns={compeak_sum.columns[0]: "Sample_ID"})
if tw_bw_clean_physio.columns[0] != "Sample_ID":
    tw_bw_clean_physio = tw_bw_clean_physio.rename(columns={tw_bw_clean_physio.columns[0]: "Sample_ID"})

# --- Step 1: Merge on Sample_ID ---
merged_df = pd.merge(
    tw_bw_clean_physio,
    compeak_sum,
    on="Sample_ID",
    how="inner"   # only keep matching IDs
)

# --- Step 2: Reorder columns so tw_bw_clean_physio comes first ---
ordered_cols = ["Sample_ID"] + \
               [col for col in tw_bw_clean_physio.columns if col != "Sample_ID"] + \
               [col for col in compeak_sum.columns if col != "Sample_ID"]

waves_physio = merged_df[ordered_cols]

# --- Step 1: Ensure Sample_ID alignment ---
if compeak_sum.columns[0] != "Sample_ID":
    compeak_sum = compeak_sum.rename(columns={compeak_sum.columns[0]: "Sample_ID"})
if tw_bw_clean_physio.columns[0] != "Sample_ID":
    tw_bw_clean_physio = tw_bw_clean_physio.rename(columns={tw_bw_clean_physio.columns[0]: "Sample_ID"})

# --- Step 2: Merge absorbance + physiochemical data ---
# TSS (% w/v)     %TS (w/w)  Turbidity (NTU) 
merged_df = pd.merge(
    tw_bw_clean_physio[["Sample_ID","%TS (w/w)","TSS (% w/v)","Turbidity (NTU)"]],
    compeak_sum,
    on="Sample_ID",
    how="inner"
)

# --- Step 3: Compute correlation matrix ---
corr_matrix = merged_df.drop(columns=["Sample_ID"]).corr()

# --- Step 4: Extract correlations of TS, TSS, Turbidity vs wavelengths ---
physio_corr = corr_matrix.loc[["%TS (w/w)","TSS (% w/v)","Turbidity (NTU)"], compeak_sum.columns[1:]]
# --- Step 1: Keep only numeric columns ---
# Drop Sample_ID and any non-numeric columns like 'Type'
#X = waves_physio.select_dtypes(include=[np.number])

# --- Step 2: Standardize data ---
#X_scaled = StandardScaler().fit_transform(X)

# --- Step 3: Run PCA ---
#pca = PCA()
#X_pca = pca.fit_transform(X_scaled)

# --- Step 3: Create DataFrame with PCA results + sample type ---
#pca_df = pd.DataFrame({
#    "PC1": X_pca[:,0],
#    "PC2": X_pca[:,1],
#    "Type": waves_physio["Type"].values
#})

# Drop Sample_ID and any non-numeric columns like 'Type'
X = waves_physio.select_dtypes(include=[np.number])

# --- Step 2: Standardize data ---
X_scaled = StandardScaler().fit_transform(X)

# --- Step 3: Run PCA ---
pca = PCA(n_components=8)   # explicitly set number of components
X_pca = pca.fit_transform(X_scaled)

# --- Step 4: Create DataFrame with PCA results + sample type ---
# Keep ALL components for scree plot, but we’ll only use PC1/PC2 for biplot
pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(pca.n_components_)])
pca_df["Type"] = waves_physio["Type"].values







# -------------------------
# Page 1: Project Overview
# -------------------------
def page1():
    st.title("🌊💧🧪 Absorbance → TSS & Turbidity Project 🚰⚗️")

    st.markdown("""
    ### 🌍 Why This Project?
    Water quality is a critical issue worldwide 🌊. Suspended solids and turbidity directly affect ecosystems 🐟,
    drinking water safety 🚰, and wastewater treatment efficiency 🏭. By using absorbance data at specific wavelengths,
    we can build predictive models that help monitor and manage water quality more effectively 💧.

    ### 🔬 Scientific Motivation
    Spectroscopy provides a rapid, non-destructive way to analyze water samples 🧪. Instead of relying solely on
    traditional lab methods ⚗️, absorbance readings can be transformed into meaningful predictions of TSS and Turbidity.
    This bridges environmental science 🌱 with data science 📊, creating scalable solutions for water monitoring.

    ### 🚀 Personal Goal
    My aim is to demonstrate how reproducible modeling workflows can connect raw spectral data 📈 with real-world
    water quality outcomes 🌊. This project is both a scientific exploration 🔬 and a practical tool for
    wastewater management 🏭 and clean water initiatives 💧.
    """)

# -------------------------
# Page 2: IDA
# -------------------------
def page2():
    st.title("IDA")

    option = st.selectbox(
        "Select what to view:",
        [
            "Option 1: Data collection and importation",
            "Option 2: Data cleaning and preprocessing",
            "Option 3: Basic descriptive statistics",
            "Option 4: Missing data analysis"
        ]
    )

    if option == "Option 1: Data collection and importation":
        st.subheader("📥 Data Collection and Importation")
        st.write("Placeholder text: Describe how absorbance data was collected.")
        st.dataframe(tw_bw_clean_waves.head())

    elif option == "Option 2: Data cleaning and preprocessing":
        st.subheader("🧹 Data Cleaning and Preprocessing")
        st.write("Placeholder text: Describe splitting between blackwater, treated water, and clean water.")
        st.dataframe(tw_bw_clean_physio.head())

    elif option == "Option 3: Basic descriptive statistics":
        st.subheader("📊 Basic Descriptive Statistics")
        st.dataframe(stats)

    elif option == "Option 4: Missing data analysis":
        st.subheader("❓ Missing Data Analysis")
        merged_turbtssts = pd.concat([turb, tss, ts], axis=1)
        fig, ax = plt.subplots(figsize=(12,18))  # larger heatmap
        sns.heatmap(merged_turbtssts.isna(), cmap="rocket", ax=ax)
        st.pyplot(fig)

# -------------------------
# Page 3: EDA
# -------------------------
def page3():
    st.title("EDA: Exploratory Data Analysis")
    st.markdown(
    "<h3 style='color:#C71585; font-weight:bold;'>Click the expand button in the top right corner of the figure for a zoomed in view! - 🔍</h3>",
    unsafe_allow_html=True
)

    option = st.selectbox(
        "Select what to view:",
        [
            "Option 1: Visualizing means of physiochemical characteristics vs sample type",
            "Option 2: Visualizing spectral fingerprints according to sample types",
            "Option 3: Separate sample types and their absorbance plots-Interactive!",
            "Option 4: Common peaks and valleys",
            "Option 5: Summary statistics for common peaks and valleys",
            "Option 6: Correlation? You decide"
        ]
    )

    if option == "Option 1: Visualizing means of physiochemical characteristics vs sample type":
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        fig, ax = plt.subplots(figsize=(14,10))
        means.plot(kind='bar', yerr=stds, capsize=4, color=colors, ax=ax)
        ax.set_title("Mean values by Sample Type")
        ax.set_yscale('log')
        ax.set_ylabel("Mean Value (log)")
        ax.legend(title="Parameter")
        st.pyplot(fig)

    elif option == "Option 2: Visualizing spectral fingerprints according to sample types":
        colors = {"Clean Water":"blue","Contaminated Water":"red","Treated Water":"green"}
        fig, ax = plt.subplots(figsize=(16,12))
        sns.set_style("whitegrid")
        for sample_type, group in tw_bw_clean_waves.groupby('Type'):
            data = group.drop(columns=['Group','Type']).values
            for row in data:
                ax.plot(wavelengths_values, row, color=colors[sample_type], alpha=0.2)
            mean = data.mean(axis=0); std = data.std(axis=0)
            ax.plot(wavelengths_values, mean, color=colors[sample_type], linewidth=2, label=f"{sample_type} mean")
            ax.fill_between(wavelengths_values, mean-std, mean+std, color=colors[sample_type], alpha=0.2)
        ax.set_title("Spectra by Sample Type")
        ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Absorbance (AU)")
        ax.legend()
        st.pyplot(fig)

    elif option == "Option 3: Separate sample types and their absorbance plots-Interactive!":
        colors = {"Clean Water":"blue","Contaminated Water":"red","Treated Water":"green"}
        traces, trace_groups = [], {}
        for sample_type, group in tw_bw_clean_waves.groupby('Type'):
            data = group.drop(columns=['Group','Type']).values
            mean = data.mean(axis=0); std = data.std(axis=0)
            trace_groups[sample_type] = []
            for row in data:
                idx = len(traces)
                traces.append(go.Scatter(x=wavelengths_values,y=row,mode='lines',
                                         line=dict(color=colors[sample_type],width=1),
                                         opacity=0.2,visible=(sample_type=="Clean Water"),
                                         name=f"{sample_type} sample"))
                trace_groups[sample_type].append(idx)
            idx = len(traces)
            traces.append(go.Scatter(x=wavelengths_values,y=mean,mode='lines',
                                     line=dict(color=colors[sample_type],width=3),
                                     visible=(sample_type=="Clean Water"),
                                     name=f"{sample_type} mean"))
            trace_groups[sample_type].append(idx)
        buttons = []
        for sample_type in trace_groups:
            visible_mask = [False]*len(traces)
            for idx in trace_groups[sample_type]: visible_mask[idx]=True
            buttons.append(dict(label=sample_type,method="update",
                                args=[{"visible":visible_mask},{"title":f"{sample_type} Spectra"}]))
        fig = go.Figure(data=traces)
        fig.update_layout(title="Spectra by Sample Type",xaxis_title="Wavelength (nm)",
                          yaxis_title="Absorbance (AU)",template="plotly_white",
                          height=900,width=1200,updatemenus=[dict(active=0,buttons=buttons,x=1,y=1.15,
                          xanchor="right",yanchor="top")])
        st.plotly_chart(fig)

    elif option == "Option 4: Common peaks and valleys":
        st.write("Placeholder text: Why identifying common peaks and valleys is important.")
        colors = {"Clean Water":"blue","Contaminated Water":"red","Treated Water":"green"}
        mean_spectra,std_spectra={},{}
        for sample_type, group in tw_bw_clean_waves.groupby('Type'):
            data = group.drop(columns=['Group','Type']).values
            mean_spectra[sample_type]=data.mean(axis=0); std_spectra[sample_type]=data.std(axis=0)
        maxima_sets,minima_sets=[],[]
        for sample_type, mean in mean_spectra.items():
            maxima_idx,_=find_peaks(mean); minima_idx,_=find_peaks(-mean)
            maxima_sets.append(set(maxima_idx)); minima_sets.append(set(minima_idx))
        def find_common_peaks(sets,tolerance=5):
            common=[]; ref=sets[0]
            for idx in ref:
                if all(any(abs(idx-other)<=tolerance for other in s) for s in sets[1:]): common.append(idx)
            return np.array(common)
        common_maxima=find_common_peaks(maxima_sets); common_minima=find_common_peaks(minima_sets)
        fig,ax=plt.subplots(figsize=(16,12))
        for sample_type in mean_spectra:
            mean=mean_spectra[sample_type]; std=std_spectra[sample_type]
            ax.plot(wavelengths_values,mean,color=colors[sample_type],linewidth=2,label=f"{sample_type} mean")
            ax.fill_between(wavelengths_values,mean-std,mean+std,color=colors[sample_type],alpha=0.2)

        
        # Continue Option 4 plotting
        ax.scatter(wavelengths_values[list(common_maxima)],
                   [mean_spectra[t][list(common_maxima)] for t in mean_spectra][0],
                   marker='x', s=120, color='black', linewidths=3,
                   label="Common maxima")

        ax.scatter(wavelengths_values[list(common_minima)],
                   [mean_spectra[t][list(common_minima)] for t in mean_spectra][0],
                   marker='x', s=120, color='purple', linewidths=3,
                   label="Common minima")

        ax.set_title("Spectra with Common Maxima and Minima", fontsize=16, fontweight='bold')
        ax.set_xlabel("Wavelength (nm)", fontsize=14)
        ax.set_ylabel("Absorbance (AU)", fontsize=14)
        ax.legend(fontsize=12, loc="upper right")
        st.pyplot(fig)

    # --- Option 5 ---
    elif option == "Option 5: Summary statistics for common peaks and valleys":
        st.subheader("📑 Summary Statistics for Common Peaks and Valleys")
        st.write("Below is the summary DataFrame of common peaks and valleys across sample types.")
        st.dataframe(compeak_summary)

    # --- Option 6 ---
    elif option == "Option 6: Correlation? You decide":
        st.subheader("🔗 Correlation Analysis")
        st.write("Heatmap showing correlation of absorbance at common wavelengths with TS, TSS, and Turbidity.")

        fig, ax = plt.subplots(figsize=(18,12))  # larger heatmap
        sns.heatmap(
            physio_corr,
            cmap="coolwarm",
            annot=True,
            fmt=".2f",
            cbar_kws={'label': 'Correlation coefficient'},
            ax=ax
        )
        ax.set_title("Correlation of Absorbance at Common Wavelengths with TS, TSS, and Turbidity",
                     fontsize=18, fontweight="bold")
        ax.set_xlabel("Wavelengths", fontsize=14)
        ax.set_ylabel("Physiochemical Parameters", fontsize=14)
        st.pyplot(fig)


def page4():
    st.title("Principal Component Analysis – Are my samples discernibly different? 🔍")

    # Toggle between PCA biplot and Scree plot
    view_option = st.radio(
        "Select plot to display:",
        ["PCA Biplot", "Scree Plot"],
        horizontal=True
    )

    # --- PCA Biplot ---
    if view_option == "PCA Biplot":
        st.markdown(
            "<p style='color:black; font-size:18px; font-weight:bold;'>"
            "Placeholder: Explain PCA separation of sample types and how loadings arrows show feature contributions."
            "</p>",
            unsafe_allow_html=True
        )

        fig = go.Figure()

        # Color-coded PCA (red, green, blue)
        fig.add_trace(go.Scatter(
            x=pca_df["PC1"], y=pca_df["PC2"],
            mode="markers",
            marker=dict(size=12),
            text=pca_df["Type"],
            name="Color-coded PCA",
            marker_color=pca_df["Type"].map({
                "Clean Water": "blue",
                "Dirty Water": "red",
                "Treated Water": "green"
            })
        ))

        # Uncoded PCA (all gray)
        fig.add_trace(go.Scatter(
            x=pca_df["PC1"], y=pca_df["PC2"],
            mode="markers",
            marker=dict(size=12, color="gray"),
            text=pca_df["Type"],
            name="Uncoded PCA"
        ))

        # Variance directional arrows (loadings for PC1 and PC2 only)
        loadings = pca.components_.T[:, :2]
        for i, feature in enumerate(X.columns):
            fig.add_trace(go.Scatter(
                x=[0, loadings[i, 0] * 3],
                y=[0, loadings[i, 1] * 3],
                mode="lines+text",
                line=dict(color="black"),
                text=[None, feature],
                textposition="top center",
                showlegend=False
            ))

        # Dropdown menu at top middle
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=list([
                        dict(label="Color-coded PCA",
                             method="update",
                             args=[{"visible": [True, False] + [True] * len(X.columns)},
                                   {"title": "PCA Biplot (Color-coded by Sample Type)"}]),
                        dict(label="Uncoded PCA",
                             method="update",
                             args=[{"visible": [False, True] + [True] * len(X.columns)},
                                   {"title": "PCA Biplot (Uncoded)"}])
                    ]),
                    direction="down",
                    showactive=True,
                    x=0.5, y=1.15,   # 🔑 top middle
                    xanchor="center", yanchor="top"
                )
            ],
            title="PCA Biplot",
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)",
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)",
            height=1000, width=1200,
            plot_bgcolor="white", paper_bgcolor="white"
        )

        st.plotly_chart(fig, use_container_width=True)

    # --- Scree Plot ---

    elif view_option == "Scree Plot":
        st.markdown(
            "<p style='color:black; font-size:18px; font-weight:bold;'>"
            "Placeholder: Explain how much variance each principal component explains (scree plot)."
            "</p>",
            unsafe_allow_html=True
        )
    
        # Scree plot of explained variance (scatter + line)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[f"PC{i+1}" for i in range(pca.n_components_)],
            y=pca.explained_variance_ratio_ * 100,
            mode="lines+markers",   # 🔑 scatter with lines
            marker=dict(size=10, color="purple"),
            line=dict(color="purple", width=2),
            name="Variance Explained"
        ))
    
        fig.update_layout(
            title="Scree Plot – Variance Explained by Principal Components",
            xaxis_title="Principal Components",
            yaxis_title="Variance Explained (%)",
            height=900, width=1200,
            plot_bgcolor="white", paper_bgcolor="white"
        )
    
        st.plotly_chart(fig, use_container_width=True)


# -------------------------
# Main App Navigation
# -------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", [
    "Page 1: Project Overview",
    "Page 2: IDA",
    "Page 3: EDA: Exploratory Data Analysis",
    "Page 4: PCA – Are my samples discernibly different?"
])

if page == "Page 1: Project Overview":
    page1()
elif page == "Page 2: IDA":
    page2()
elif page == "Page 3: EDA: Exploratory Data Analysis":
    page3()
elif page == "Page 4: PCA – Are my samples discernibly different?":
    page4()
