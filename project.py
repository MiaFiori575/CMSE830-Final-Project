

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.signal import find_peaks
#import plotly.express as px

#importing turbidity measurements 
turb = pd.read_excel('physiochemical data.xlsx', sheet_name='Master Physiochem', usecols=['Turbidity (NTU)'])
#checking that data was correctly imported
turb.head()

#importing total suspended solids measurements 
tss = pd.read_excel('physiochemical data.xlsx', sheet_name='TSS', usecols=['TSS (% w/v)'])

#importing total solids measurements 
ts = pd.read_excel('physiochemical data.xlsx', sheet_name='TS', usecols=['%TS (w/w)'])

#1 cont. importing spectral data
#importing wavelengths 
wavelengths = pd.read_excel('SAMPLE SET 1.xlsx', sheet_name='SAMPLE SET 1', nrows=1, header = None)
#column one does not have a value, dropping this column
wavelengths = wavelengths.drop(columns=wavelengths.columns[0])
#dropping the last two columns which are strings, not floats
wavelengths = wavelengths.drop(columns=wavelengths.columns[125:])
#extracting values so I can plot
wavelengths = wavelengths.iloc[0].values
#checking that data was correctly imported
#print(wavelengths)

#I think this could be a for loop for all three data types
#we could put a for loop here for all sets of samples
#importing absorbances and wavelengths of treated water (tw)
tw = pd.read_excel('SAMPLE SET 2.xlsx', sheet_name='SAMPLE SET 2', skiprows = range(1,109), header = None)
#dropping first column with sample names -might undo this later
tw = tw.drop(columns=tw.columns[0])
#dropping last two columns which are non numeric
tw = tw.drop(columns=tw.columns[125:])
#extracting individual samples
#A1 triplicate value average, averaging three rows element-wise
A1 = tw.iloc[[1, 3]].mean(axis=0)
B1 = tw.iloc[[4, 6]].mean(axis=0)
C1 = tw.iloc[[7, 9]].mean(axis=0)
D1 = tw.iloc[[10, 12]].mean(axis=0)
E1= tw.iloc[[13, 15]].mean(axis=0)
F1 = tw.iloc[[16, 18]].mean(axis=0)
G1 = tw.iloc[[19, 21]].mean(axis=0)
H1 = tw.iloc[[22, 24]].mean(axis=0)
I1 = tw.iloc[[25, 27]].mean(axis=0)
J1 = tw.iloc[[28, 30]].mean(axis=0)
K1 = tw.iloc[[31, 33]].mean(axis=0)
L1 = tw.iloc[[34, 36]].mean(axis=0)

#extracting blank readings (no contamination, clean water)
blank = pd.read_excel('SAMPLE SET 2.xlsx', sheet_name='BLANK', skiprows = 1, header = None)
#dropping first column with sample nams -might undo this later
blank = blank.drop(columns=blank.columns[0])
#dropping last two columns which are non numeric
blank = blank.drop(columns=blank.columns[125:])

blnk1 = blank.iloc[0].values
blnk2 = blank.iloc[1].values
blnk3 = blank.iloc[2].values

#extracting blackwater readings (contaminated water without any treatment)
bw = pd.read_excel('SAMPLE SET 2.xlsx', sheet_name='BW SAMPLES', skiprows = 1, header = None)
#dropping first column with sample names -might undo this later
bw = bw.drop(columns=bw.columns[0])
#dropping last two columns which are non numeric
bw = bw.drop(columns=bw.columns[125:])


bw1 = bw.iloc[[1, 3]].mean(axis=0)
bw2 = bw.iloc[[4, 6]].mean(axis=0)
bw3 = bw.iloc[[7, 9]].mean(axis=0)
bw4 = bw.iloc[[10, 12]].mean(axis=0)
bw5 = bw.iloc[[13, 15]].mean(axis=0)
bw6 = bw.iloc[[16, 18]].mean(axis=0)
bw7 = bw.iloc[[19, 21]].mean(axis=0)
bw8= bw.iloc[[22, 24]].mean(axis=0)


plt.rcParams.update({                     # This is the big difference to the code before, here we customize the setting for all the following subplots
    # Figure
    'figure.figsize': (5, 7),
    'figure.dpi': 20,                    # dpi stand for dots per inch
    # Font
    'font.family': 'fantasy',
    'font.size': 8,
    # Axes
    'axes.titlesize': 10,                 # Setting for axis like title and label
    'axes.labelsize': 14,                   
    'axes.spines.top': False,             # Spines note boundaries of the data area and are disabled here
    'axes.spines.right': False,
    # Ticks
    'xtick.direction': 'out',             # ticks are pointing outwards now for x-axis
    'ytick.direction': 'out',
    'xtick.labelsize': 2,
    'ytick.labelsize': 2,
    # Legend
    'legend.fontsize': 2,
    'legend.frameon': False,
    # Lines
    'lines.linewidth': 1,
    'lines.markersize': 5,
    # Grid
    'grid.linewidth': 0.5,
    'grid.alpha': 0.8,
    # Color cycle
    'axes.prop_cycle': plt.cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']),
                                            # Setting a cycle for colors to be used
})
st.set_page_config(layout="wide")
st.title("💩🦠🧫🧪💧 Wastewater: Using Spectral Signatures to Predict Total Solid Contamination")
st.markdown(
    "<p style='text-align: center; font-size: 1.1em;'>Because sometimes you just need to know...</p>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; font-size: 1.1em;'>AI Windows Copilot was used to polish app look and adjust tabs from horizontal to vertical</p>",
    unsafe_allow_html=True
)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from scipy.signal import find_peaks
# Precompute imputed data so it's available in all tabs
tss_missing = tss[tss.isnull().any(axis=1)]
tss_not_missing = tss.dropna()
turb_missing = turb[turb.isnull().any(axis=1)]
turb_not_missing = turb.dropna()

# TSS pipeline
scaler_tss = StandardScaler()
tss_scaled = pd.DataFrame(scaler_tss.fit_transform(tss_not_missing), columns=tss_not_missing.columns)
imputer_tss = KNNImputer(n_neighbors=5, weights='distance')
imputer_tss.fit(tss_scaled)

def impute_and_inverse_transform_tss(data):
    scaled_data = pd.DataFrame(scaler_tss.transform(data), columns=data.columns, index=data.index)
    imputed_scaled = imputer_tss.transform(scaled_data)
    return pd.DataFrame(scaler_tss.inverse_transform(imputed_scaled), columns=data.columns, index=data.index)

tss_imputed = impute_and_inverse_transform_tss(tss)

# Turbidity pipeline
scaler_turb = StandardScaler()
turb_scaled = pd.DataFrame(scaler_turb.fit_transform(turb_not_missing), columns=turb_not_missing.columns)
imputer_turb = KNNImputer(n_neighbors=3, weights='distance')
imputer_turb.fit(turb_scaled)

def impute_and_inverse_transform_turb(data):
    scaled_data = pd.DataFrame(scaler_turb.transform(data), columns=data.columns, index=data.index)
    imputed_scaled = imputer_turb.transform(scaled_data)
    return pd.DataFrame(scaler_turb.inverse_transform(imputed_scaled), columns=data.columns, index=data.index)

turb_imputed = impute_and_inverse_transform_turb(turb)


# --- Vertical tab selector ---
selected_tab = st.radio("📂 Navigation", [
    "🧠 Introduction",
    "🧼 Data Cleaning",
    "🔍 IDA #6 Missing Data Analysis",
    "🏠 KNN Imputation",
    "📊 Spectral Signatures and Peak Detection",
    "📈 Correlations and Statistics"
])

# --- Tab 1: Introduction ---
if selected_tab == "🧠 Introduction":
    st.header("Introduction")
    st.write("This tab contains an overview of the project, its importance and goals")
    st.markdown("""
    - 🔬 Step 1: Scientific Question  
    - ❓ Step 2: Why is it important?  
    - 🌈 Step 3: What is NIR?  
    - 🛠️📐 Step 4: What physiochemical parameters are we trying to correlate?  
    """)
    st.write("🔬 Step 1: Scientific Question")
    st.markdown("Can the total amount of solid contamination in a water sample be determined using its near infrared (NIR) spectral signature?")
    st.write("❓ Step 2: Why is it important?")
    st.markdown("Scientists can take water quality measurements in real time! No waiting 24-hours for total solids measurements!")
    st.write("🌈 Step 3: What is NIR?")
    st.markdown("NIR stands for Near-Infrared Spectroscopy, an analytical technique that utilizes the near-infrared region of the electromagnetic spectrum (from 780 nm to 2500 nm) to analyze samples for compositional or characteristic traits.")
    st.markdown("Light in this region interacts with OH, NH and CH bonds and certain wavelengths (frequencies) are associated with each bond type.")
    st.markdown("When NIR light is presented to samples high in chemical compounds containing these bonds, some of energy is absorbed by the sample in these specific wavelengths, and thus the reflected light has less intensity in these regions.")
    st.write("🛠️📐 Step 4: What physiochemical parameters are we trying to correlate?")
    st.markdown("I am trying to correlate total solids (total solids in a sample), total suspended solids (total solids minus dissolved solids), and turbidity of a sample (measured in NTU)")

elif selected_tab == "🧼 Data Cleaning":
    st.markdown("## 🧼 Data Cleaning Steps")

    # --- Physiochemical Data Section ---
    st.markdown("### ⚗️ Physiochemical Data")
    st.write("First I split the physiochemical parameter data into TS, TSS, and Turbidity.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Turbidity Data**")
        st.dataframe(turb, height=200)

    with col2:
        st.markdown("**TSS Data**")
        st.dataframe(tss, height=200)

    with col3:
        st.markdown("**TS Data**")
        st.dataframe(ts, height=200)

    # --- Spectral Data Section ---
    st.markdown("### 🌈 Spectral Data")
    st.markdown("**Clean Water Absorbances**")
    st.write("""
        From a different Excel file with multiple absorbances across a multitude of wavelengths, I separated the different sample types
        with the goal of identifying similar peaks and valleys. These will later be combined with the physiochemical parameters at select wavelengths
        to perform a correlation analysis.
    """)
    col1, col2, col3 = st.columns([3, 3, 3])
    with col1:
        st.markdown("**Clean Water**")
        st.dataframe(blank, height=200,  use_container_width=True)

    with col2:
        st.markdown("**Treated Water**")
        st.dataframe(tw, height=200,  use_container_width=True)

    with col3:
        st.markdown("**Blackwater**")
        st.dataframe(bw, height=200,  use_container_width=True)

# --- Tab 2: Missing Data Analysis ---
elif selected_tab == "🔍 IDA #6 Missing Data Analysis":
    st.header("IDA #6 Missing Data Analysis")
    merged_turbtssts = pd.concat([turb, tss, ts], axis=1)
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(merged_turbtssts.isna(), cmap="rocket", cbar=True, ax=ax)
    ax.set_title("Missing Data Heatmap", fontsize=10)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=2)
    st.pyplot(fig)

# --- Tab 3: KNN Imputation ---
elif selected_tab == "🏠 KNN Imputation":
    st.header("KNN Imputation")
    st.write("Assuming MCAR—because any missing values are because I dropped the sample 😬, but I will actually prove it the way we did in class")

    tss_missing = tss[tss.isnull().any(axis=1)]
    tss_not_missing = tss.dropna()
    turb_missing = turb[turb.isnull().any(axis=1)]
    turb_not_missing = turb.dropna()

    scaler_tss = StandardScaler()
    tss_scaled = pd.DataFrame(scaler_tss.fit_transform(tss_not_missing), columns=tss_not_missing.columns)
    imputer_tss = KNNImputer(n_neighbors=5, weights='distance')
    imputer_tss.fit(tss_scaled)

    def impute_and_inverse_transform_tss(data):
        scaled_data = pd.DataFrame(scaler_tss.transform(data), columns=data.columns, index=data.index)
        imputed_scaled = imputer_tss.transform(scaled_data)
        return pd.DataFrame(scaler_tss.inverse_transform(imputed_scaled), columns=data.columns, index=data.index)

    tss_imputed = impute_and_inverse_transform_tss(tss)

    scaler_turb = StandardScaler()
    turb_scaled = pd.DataFrame(scaler_turb.fit_transform(turb_not_missing), columns=turb_not_missing.columns)
    imputer_turb = KNNImputer(n_neighbors=3, weights='distance')
    imputer_turb.fit(turb_scaled)

    def impute_and_inverse_transform_turb(data):
        scaled_data = pd.DataFrame(scaler_turb.transform(data), columns=data.columns, index=data.index)
        imputed_scaled = imputer_turb.transform(scaled_data)
        return pd.DataFrame(scaler_turb.inverse_transform(imputed_scaled), columns=data.columns, index=data.index)

    turb_imputed = impute_and_inverse_transform_turb(turb)

    fig1 = plt.figure(figsize=(8, 4))
    sns.histplot(tss_not_missing.dropna(), kde=True, color='blue', alpha=0.5, label='Original (non-missing)')
    sns.histplot(tss_imputed['TSS (% w/v)'], kde=True, color='red', alpha=0.5, label='Imputed')
    plt.title('Distribution of Original vs Imputed TSS')
    plt.legend(fontsize=10)
    plt.tick_params(axis='x', labelsize=10)
    plt.tick_params(axis='y', labelsize=10)
    plt.tight_layout()
    st.pyplot(fig1)

    fig2 = plt.figure(figsize=(8, 4))
    sns.histplot(turb_not_missing.dropna(), kde=True, color='blue', alpha=0.5, label='Original (non-missing)')
    sns.histplot(turb_imputed['Turbidity (NTU)'], kde=True, color='red', alpha=0.5, label='Imputed')
    plt.title('Distribution of Original vs Imputed Turbidity')
    plt.legend(fontsize=10)
    plt.tick_params(axis='x', labelsize=10)
    plt.tick_params(axis='y', labelsize=10)
    plt.tight_layout()
    st.pyplot(fig2)

# --- Tab 4: Spectral Signatures and Peak Detection ---
elif selected_tab == "📊 Spectral Signatures and Peak Detection":
    st.header("📊 Spectral Signature Analysis")

    # Update plot style
    plt.rcParams.update({
        'figure.figsize': (4, 3),  # Smaller height
        'figure.dpi': 100,
        'font.family': 'fantasy',
        'font.size': 10,
        'axes.titlesize': 10,
        'axes.labelsize': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'legend.frameon': False,
        'lines.linewidth': 2,
        'lines.markersize': 10,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.8,
        'axes.prop_cycle': plt.cycler(color=[
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
            '#bcbd22', '#17becf'
        ]),
    })

    # --- Column 1: Absorbance Signatures ---
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Absorbance Signatures")
        st.markdown("Visual comparison of absorbance across treated, blackwater, and clean water samples.")

        fig1, ax1 = plt.subplots()
        ax1.plot(wavelengths, A1, label='Treated water')
        ax1.plot(wavelengths, bw1, label='Blackwater')
        ax1.plot(wavelengths, blnk1, label='Clean water')
        ax1.set_title('Absorbance Signatures')
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_ylabel('Absorbance (AU)')
        ax1.legend()
        st.pyplot(fig1)

    # --- Column 2: Detected Peaks ---
    with col2:
        st.subheader("Detected Peaks")
        st.markdown("Peaks identified in each sample using `scipy.signal.find_peaks`.")

        blnk1_peaks, _ = find_peaks(blnk1)
        A1_peaks, _ = find_peaks(A1)
        bw1_peaks, _ = find_peaks(bw1)

        fig2, ax2 = plt.subplots()
        ax2.plot(wavelengths, A1, label='Treated water')
        ax2.plot(wavelengths, bw1, label='Blackwater')
        ax2.plot(wavelengths, blnk1, label='Clean water')
        ax2.plot(wavelengths[blnk1_peaks], blnk1[blnk1_peaks], "x", label='Clean peaks')
        ax2.plot(wavelengths[bw1_peaks], bw1[bw1_peaks], "x", label='Blackwater peaks')
        ax2.plot(wavelengths[A1_peaks], A1[A1_peaks], "x", label='Treated peaks')
        ax2.set_title("Detected Peaks")
        ax2.set_xlabel("Wavelength (nm)")
        ax2.set_ylabel("Absorbance (AU)")
        ax2.legend()
        st.pyplot(fig2)

    # --- Common Peaks Section ---
    st.subheader("Common Peaks Across All Samples")
    st.markdown("Peaks shared across all three sample types within a ±2 nm tolerance.")

    tolerance = 2
    com_peaks_tol = [
        val for val in A1_peaks
        if any(abs(val - p) <= tolerance for p in bw1_peaks)
        and any(abs(val - p) <= tolerance for p in blnk1_peaks)
    ]

    fig3, ax3 = plt.subplots(figsize=(6, 3))
    ax3.plot(wavelengths, A1, label='Treated water')
    ax3.plot(wavelengths, bw1, label='Blackwater')
    ax3.plot(wavelengths, blnk1, label='Clean water')
    ax3.plot(wavelengths[com_peaks_tol], A1[com_peaks_tol], "x", label='Common peaks (Treated)')
    ax3.plot(wavelengths[com_peaks_tol], bw1[com_peaks_tol], "x", label='Common peaks (Blackwater)')
    ax3.plot(wavelengths[com_peaks_tol], blnk1[com_peaks_tol], "x", label='Common peaks (Clean)')
    ax3.set_title("Common Peaks")
    ax3.set_xlabel("Wavelength (nm)")
    ax3.set_ylabel("Absorbance (AU)")
    ax3.legend()
    st.pyplot(fig3)

    # st.subheader("**Different visualization**")
    # df = pd.DataFrame({
    # 'Wavelength': wavelengths,
    # 'Clean': blnk1,
    # 'Treated': A1,
    # 'Blackwater': bw1})

    # df_long = pd.melt(
    # df,
    # id_vars='Wavelength',
    # var_name='Sample',
    # value_name='Absorbance')

    # fig = px.line(
    # df_long,
    # x='Wavelength',
    # y='Absorbance',
    # color='Sample',
    # title='Interactive Absorbance Spectra',
    # labels={'Absorbance': 'Absorbance (AU)', 'Wavelength': 'Wavelength (nm)'})
    # st.plotly_chart(fig, use_container_width=True)


    # df_long = pd.melt(df.reset_index(), id_vars='Wavelength', var_name='Sample', value_name='Absorbance')
    # fig = px.line(df_long, x='Wavelength', y='Absorbance', color='Sample', title='Interactive Absorbance')
    # st.plotly_chart(fig)

    
  
elif selected_tab == "📈 Correlations and Statistics":
    st.header("Correlations and Statistics")
    #tss and turb data was imputed, no data missing for ts

    tw_tss = tss_imputed[0:96]
    tw_ts = ts[0:96]
    tw_turb = turb_imputed[0:96]

    bw_tss = tss_imputed[96:106]
    bw_ts = ts[96:106]
    bw_turb = turb_imputed[96:106]

    #merging related data
    tw_physio = pd.concat([tw_tss, tw_ts,tw_turb], axis=1)
    bw_physio = pd.concat([bw_tss, bw_ts,bw_turb], axis=1)
    tw_bw_physio = pd.concat([tw_physio,bw_physio])
    st.write("Descriptive Statistics for Treated Water's Physiochemical Parameters")
    st.dataframe(tw_physio.describe())

    st.write("Descriptive Statistics for Blackwater's Physiochemical Parameters")
    st.dataframe(bw_physio.describe())
    
   # Set global context for consistent styling
    sns.set_context("notebook", font_scale=1.2)  # Slightly scaled for annotations

    # --- General correlation matrix ---
    merged_data = pd.concat([tss, turb, ts], axis=1)
    corr = merged_data.corr()
    st.write("Here is a correlation matrix analyzing the different physiochemical parameters. As expected, there is a strong positive correlation between turbidity, TS, and TSS-they are all measurements on particles in a solution ")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax1, annot_kws={"size": 12})
    ax1.set_title("Correlation Matrix: TSS, Turbidity, TS", fontsize=16)
    ax1.tick_params(axis='x', labelsize=10)
    ax1.tick_params(axis='y', labelsize=10)
    st.pyplot(fig1)


    # --- Valley absorbance correlation matrix ---
    st.write("Here is a correlation matrix between physiochemical parameters and a common valley (local minimum) between treated water (samples A1-L1) and blackwater (samples BW1-BW8) spectral signatures. This tries to preliminarily answer the question: are these physiochemical variables correlated to spectral signature?")
    st.write("There is a moderate correlation (0.5 < r < 0.75) between the Absorbance value at wavelength 945.266 nm and TSS, Turbidity, and TS for these samples!")
    cond_twbw = pd.concat([tw_physio[0:12], bw_physio])
    tw_valley7 = pd.concat([
        pd.Series([A1[7]]), pd.Series([B1[7]]), pd.Series([C1[7]]),
        pd.Series([D1[7]]), pd.Series([E1[7]]), pd.Series([F1[7]]),
        pd.Series([G1[7]]), pd.Series([H1[7]]), pd.Series([I1[7]]),
        pd.Series([J1[7]]), pd.Series([K1[1]]), pd.Series([L1[1]])
    ], ignore_index=True)

    bw_valley7 = pd.concat([
        pd.Series([bw1[7]]), pd.Series([bw2[7]]), pd.Series([bw3[7]]),
        pd.Series([bw4[7]]), pd.Series([bw5[7]]), pd.Series([bw6[7]]),
        pd.Series([bw7[7]]), pd.Series([bw8[7]])
    ], ignore_index=True)

    valley7 = pd.concat([tw_valley7, bw_valley7]).reset_index(drop=True)
    cond_twbw = cond_twbw.reset_index(drop=True)
    cond_data = cond_twbw.copy()
    cond_data['Valley ABS at Index 7'] = valley7

    fig3, ax3 = plt.subplots(figsize=(10, 8))
    sns.heatmap(cond_data.corr(), annot=True, cmap='Spectral', ax=ax3, annot_kws={"size": 12})
    ax3.set_title("Correlation: Physiochemical Data + Valley Absorbance @ Index 7", fontsize=16)
    ax3.tick_params(axis='x', labelsize=10)
    ax3.tick_params(axis='y', labelsize=10)
    st.pyplot(fig3)
    


#import streamlit as st

# Dropdown options
#options = ['Red Cedar River', 'Grand River', 'Sycamore Creek', 'Rose Lake']

# Create dropdown
#selected = st.selectbox("Choose a watershed or trail:", options)

# Display selection
#st.write(f"You selected: {selected}")
