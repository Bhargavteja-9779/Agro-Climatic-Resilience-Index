import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Exploratory Data Analysis (EDA)\n",
    "## Climate-Aware Crop Yield and Risk Prediction in South Indian Agriculture\n",
    "\n",
    "This notebook performs EDA on the compiled South Indian agricultural dataset. It includes:\n",
    "1. Dataset summary statistics (for Table 1 of the paper).\n",
    "2. Yield distribution analysis.\n",
    "3. Correlation matrix between climate/soil features and crop yield.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import os\n",
    "\n",
    "# Set plot style\n",
    "plt.style.use('seaborn-v0_8-whitegrid')\n",
    "sns.set_palette('viridis')\n",
    "\n",
    "data_path = '../data/datasets/south_india_crop_data.csv'\n",
    "df = pd.read_csv(data_path)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1. Dataset Description\n",
    "Let's identify the missing values and numeric distributions."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"Dataset Shape:\", df.shape)\n",
    "print(\"\\nMissing Values:\")\n",
    "print(df.isnull().sum())\n",
    "print(\"\\nSummary Statistics:\")\n",
    "display(df.describe())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2. Crop Yield Distribution"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(10, 6))\n",
    "sns.histplot(df['Yield'], bins=30, kde=True)\n",
    "plt.title('Distribution of Crop Yield (Tons/Hectare)')\n",
    "plt.xlabel('Yield')\n",
    "plt.ylabel('Frequency')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3. Correlation Heatmap\n",
    "Analyzing the relationship between climate indicators (Rainfall, Temperature, Humidity) and soil nutrients (N, P, K, pH) against Yield."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "numeric_cols = df.select_dtypes(include=[np.number]).columns\n",
    "plt.figure(figsize=(12, 8))\n",
    "correlation_matrix = df[numeric_cols].corr()\n",
    "sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=\".2f\")\n",
    "plt.title('Correlation Heatmap of Agricultural Features')\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open('notebooks/data_analysis.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
print("Notebook created successfully.")
