import json

# Create a Jupyter notebook for proxy-based spatial evaluation
notebook = {
    'cells': [
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '# Proxy-Based Spatial Evaluation: Surrogate Reference Values\n',
                '## Air Quality Spatial Interpolation Assessment\n',
                '\n',
                '**Date**: January 2, 2026  \n',
                '**Context**: Sri Lanka Air Quality Monitoring (Battaramulla & Kandy stations)\n',
                '\n',
                '---\n',
                '\n',
                '## Overview\n',
                '\n',
                'This notebook demonstrates **proxy-based spatial evaluation** using LSTM predictions as **surrogate reference values** (NOT true ground truth) to assess spatial interpolation performance.\n',
                '\n',
                '### Key Concept\n',
                '> **LSTM predictions serve as proxy benchmarks** to evaluate relative spatial consistency between monitoring stations, not absolute prediction accuracy.\n',
                '\n',
                '### Evaluation Scenarios\n',
                '- **Scenario A**: Use only Battaramulla LSTM prediction → predict at Kandy location\n',
                '- **Scenario B**: Use only Kandy LSTM prediction → predict at Battaramulla location\n',
                '\n',
                '### Metrics\n',
                '- **MAE**: Mean Absolute Error\n',
                '- **RMSE**: Root Mean Square Error  \n',
                '- **MAPE**: Mean Absolute Percentage Error'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 1. Setup and Imports'
            ]
        },
        {
            'cell_type': 'code',
            'metadata': {},
            'source': [
                '# Import required libraries\n',
                'import sys\n',
                'import os\n',
                'from pathlib import Path\n',
                'import pandas as pd\n',
                'import numpy as np\n',
                'import matplotlib.pyplot as plt\n',
                'import seaborn as sns\n',
                'from datetime import datetime, timedelta\n',
                'import warnings\n',
                '\n',
                '# Set plotting style\n',
                'plt.style.use(\"seaborn-v0_8\")\n',
                'sns.set_palette(\"husl\")\n',
                'warnings.filterwarnings(\"ignore\")\n',
                '\n',
                '# Add project root to path\n',
                'project_root = Path.cwd().parent\n',
                'sys.path.append(str(project_root))\n',
                'sys.path.append(str(project_root / \"backend\"))\n',
                '\n',
                'print(f\"Project root: {project_root}\")\n',
                'print(f\"Current working directory: {Path.cwd()}\")\n',
                'print(\"Libraries imported successfully!\")'
            ],
            'outputs': []
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 2. Load Models and Configurations'
            ]
        },
        {
            'cell_type': 'code',
            'metadata': {},
            'source': [
                '# Import air quality logic\n',
                'from backend.air_quality_logic import (\n',
                '    proxy_based_spatial_evaluation,\n',
                '    load_artifacts,\n',
                '    STATION_META\n',
                ')\n',
                '\n',
                'print(\"Air quality logic imported successfully!\")\n',
                '\n',
                '# Display station information\n',
                'print(\"\\nMonitoring Stations:\")\n',
                'for station, info in STATION_META.items():\n',
                '    print(f\"  {station}: {info[\'lat\']:.6f}, {info[\'lon\']:.6f}\")\n',
                '\n',
                '# Load LSTM models and preprocessors\n',
                'print(\"\\nLoading LSTM models...\")\n',
                '\n',
                'try:\n',
                '    # Load Battaramulla model\n',
                '    model_b, preprocess_b, cfg_b = load_artifacts(\"Battaramulla\")\n',
                '    print(\"✓ Battaramulla model loaded\")\n',
                '    \n',
                '    # Load Kandy model  \n',
                '    model_k, preprocess_k, cfg_k = load_artifacts(\"Kandy\")\n',
                '    print(\"✓ Kandy model loaded\")\n',
                '    \n',
                '    models_loaded = True\n',
                '    \n',
                'except Exception as e:\n',
                '    print(f\"✗ Error loading models: {e}\")\n',
                '    print(\"Please ensure model files exist in the expected locations\")\n',
                '    models_loaded = False'
            ],
            'outputs': []
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 3. Generate Evaluation Timestamps\n',
                '\n',
                'We need to select timestamps for evaluation. Let\'s use a sample period from the available data.'
            ]
        },
        {
            'cell_type': 'code',
            'metadata': {},
            'source': [
                '# Load observation data to determine available timestamps\n',
                'from backend.air_quality_logic import obs\n',
                '\n',
                'print(f\"Total observations: {len(obs):,}\")\n',
                'print(f\"Date range: {obs[\'datetime\'].min()} to {obs[\'datetime\'].max()}\")\n',
                '\n',
                '# Get unique timestamps (hourly)\n',
                'all_timestamps = obs[\'datetime\'].sort_values().unique()\n',
                'print(f\"Unique timestamps: {len(all_timestamps):,}\")\n',
                '\n',
                '# Select evaluation period (e.g., one week in 2023)\n',
                'start_date = pd.Timestamp(\"2023-06-01\")\n',
                'end_date = pd.Timestamp(\"2023-06-07\")  # One week\n',
                '\n',
                'eval_timestamps = [ts for ts in all_timestamps \n',
                '                   if start_date <= ts <= end_date]\n',
                '\n',
                'print(f\"\\nEvaluation period: {start_date.date()} to {end_date.date()}\")\n',
                'print(f\"Evaluation timestamps: {len(eval_timestamps)}\")\n',
                '\n',
                '# Display sample timestamps\n',
                'print(\"\\nSample timestamps:\")\n',
                'for i, ts in enumerate(eval_timestamps[:5]):\n',
                '    print(f\"  {i+1}. {ts}\")\n',
                'if len(eval_timestamps) > 5:\n',
                '    print(f\"  ... and {len(eval_timestamps)-5} more\")'
            ],
            'outputs': []
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 4. Run Proxy-Based Spatial Evaluation\n',
                '\n',
                'Now we execute the proxy-based evaluation for PM₂.₅ (you can change this to PM₁₀ or NO₂).'
            ]
        },
        {
            'cell_type': 'code',
            'metadata': {},
            'source': [
                '# Run evaluation only if models are loaded\n',
                'if models_loaded:\n',
                '    # Choose pollutant\n',
                '    pollutant = \"PM25\"  # Options: \"PM25\", \"PM10\", \"NO2\"\n',
                '    \n',
                '    print(f\"Starting proxy-based spatial evaluation for {pollutant}\")\n',
                '    print(\"=\" * 60)\n',
                '    \n',
                '    # Execute evaluation\n',
                '    evaluation_results = proxy_based_spatial_evaluation(\n',
                '        model_battaramulla=model_b,\n',
                '        model_kandy=model_k,\n',
                '        preprocess_battaramulla=preprocess_b,\n',
                '        preprocess_kandy=preprocess_k,\n',
                '        cfg_battaramulla=cfg_b,\n',
                '        cfg_kandy=cfg_k,\n',
                '        timestamps=eval_timestamps,\n',
                '        pollutant=pollutant\n',
                '    )\n',
                '    \n',
                '    print(\"\\n\" + \"=\" * 60)\n',
                '    print(\"EVALUATION COMPLETE\")\n',
                '    print(\"=\" * 60)\n',
                '    \n',
                'else:\n',
                '    print(\"Models not loaded. Cannot run evaluation.\")\n',
                '    evaluation_results = pd.DataFrame()'
            ],
            'outputs': []
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 5. Results Analysis and Visualization'
            ]
        },
        {
            'cell_type': 'code',
            'metadata': {},
            'source': [
                '# Display and analyze results\n',
                'if not evaluation_results.empty:\n',
                '    print(\"EVALUATION RESULTS SUMMARY\")\n',
                '    print(\"=\" * 50)\n',
                '    \n',
                '    # Format the results table\n',
                '    formatted_results = evaluation_results.copy()\n',
                '    formatted_results = formatted_results.round({\n',
                '        \'mae\': 2,\n',
                '        \'rmse\': 2,\n',
                '        \'mape\': 1\n',
                '    })\n',
                '    \n',
                '    print(formatted_results.to_string(index=False))\n',
                '    \n',
                '    # Create visualization\n',
                '    fig, axes = plt.subplots(1, 3, figsize=(18, 6))\n',
                '    fig.suptitle(f\'Proxy-Based Spatial Evaluation Results: {pollutant}\\n\' +\n',
                '                  \'LSTM Predictions as Surrogate Reference Values\',\n',
                '                  fontsize=14, fontweight=\'bold\')\n',
                '    \n',
                '    metrics = [\'mae\', \'rmse\', \'mape\']\n',
                '    metric_names = [\'Mean Absolute Error\', \'Root Mean Square Error\', \'Mean Absolute % Error\']\n',
                '    \n',
                '    for i, (metric, name) in enumerate(zip(metrics, metric_names)):\n',
                '        ax = axes[i]\n',
                '        \n',
                '        # Filter out NaN values for MAPE\n',
                '        plot_data = evaluation_results.dropna(subset=[metric])\n',
                '        \n',
                '        if not plot_data.empty:\n',
                '            bars = ax.bar(plot_data[\'scenario\'], plot_data[metric], \n',
                '                         color=[\'#2E86AB\', \'#A23B72\'], alpha=0.8)\n',
                '            \n',
                '            ax.set_title(f\'{name}\', fontsize=12, fontweight=\'bold\')\n',
                '            ax.set_ylabel(f\'{name} ({pollutant} units)\' if i < 2 else f\'{name} (%)\', fontsize=10)\n',
                '            ax.set_xlabel(\'Evaluation Scenario\', fontsize=10)\n',
                '            ax.grid(True, alpha=0.3)\n',
                '            \n',
                '            # Add value labels\n',
                '            for bar, value in zip(bars, plot_data[metric]):\n',
                '                height = bar.get_height()\n',
                '                ax.text(bar.get_x() + bar.get_width()/2., height + max(plot_data[metric])*0.02,\n',
                '                       f\'{value:.1f}\' if metric == \'mape\' else f\'{value:.2f}\',\n',
                '                       ha=\'center\', va=\'bottom\', fontsize=10)\n',
                '        \n',
                '    plt.tight_layout()\n',
                '    plt.show()\n',
                '    \n',
                '    # Interpretation\n',
                '    print(\"\\n\" + \"=\" * 50)\n',
                '    print(\"INTERPRETATION\")\n',
                '    print(\"=\" * 50)\n',
                '    print(\"\\nKey Points:\")\n',
                '    print(\"• These metrics measure spatial interpolation consistency\")\n',
                '    print(\"• Lower values indicate better spatial agreement between stations\")\n',
                '    print(\"• LSTM predictions serve as surrogate reference values\")\n',
                '    print(\"• Results assess relative spatial patterns, not absolute accuracy\")\n',
                '    \n',
                '    # Scenario comparison\n',
                '    if len(evaluation_results) >= 2:\n',
                '        scenario_a = evaluation_results.iloc[0]\n',
                '        scenario_b = evaluation_results.iloc[1]\n',
                '        \n',
                '        print(f\"\\nScenario Comparison:\")\n',
                '        print(f\"• Scenario A (B→K): MAE={scenario_a[\'mae\']:.2f}, RMSE={scenario_a[\'rmse\']:.2f}, MAPE={scenario_a[\'mape\']:.1f}%\")\n',
                '        print(f\"• Scenario B (K→B): MAE={scenario_b[\'mae\']:.2f}, RMSE={scenario_b[\'rmse\']:.2f}, MAPE={scenario_b[\'mape\']:.1f}%\")\n',
                '        \n',
                '        if scenario_a[\'mae\'] < scenario_b[\'mae\']:\n',
                '            print(\"• Scenario A shows better spatial consistency\")\n',
                '        else:\n',
                '            print(\"• Scenario B shows better spatial consistency\")\n',
                '    \n',
                'else:\n',
                '    print(\"No evaluation results to display\")'
            ],
            'outputs': []
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 6. Methodological Notes\n',
                '\n',
                '### Important Distinctions\n',
                '- **Surrogate Reference Values**: LSTM predictions are used as proxy benchmarks, NOT true measurements\n',
                '- **Spatial Consistency**: This evaluates how well spatial interpolation preserves relative patterns\n',
                '- **NOT Validation**: This is NOT true spatial cross-validation with ground truth data\n',
                '\n',
                '### Use Cases\n',
                '- Assess spatial interpolation method performance\n',
                '- Compare different spatial algorithms\n',
                '- Evaluate consistency between monitoring stations\n',
                '- Support method selection for air quality mapping\n',
                '\n',
                '### Limitations\n',
                '- Depends on LSTM prediction quality as reference\n',
                '- Measures relative consistency, not absolute accuracy\n',
                '- Limited to available monitoring station pairs\n',
                '- Temporal variability may affect spatial relationships'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 7. Integration with FastAPI Pipeline\n',
                '\n',
                'This evaluation can be integrated into your existing FastAPI pipeline:\n',
                '\n',
                '```python\n',
                '# Example FastAPI endpoint\n',
                '@app.post(\"/evaluate-spatial\")\n',
                'async def evaluate_spatial_performance(pollutant: str = \"PM25\"):\n',
                '    # Get recent timestamps\n',
                '    timestamps = get_recent_timestamps(hours=168)  # Last week\n',
                '    \n',
                '    # Run evaluation\n',
                '    results = proxy_based_spatial_evaluation(\n',
                '        model_b, model_k, preprocess_b, preprocess_k,\n',
                '        cfg_b, cfg_k, timestamps, pollutant\n',
                '    )\n',
                '    \n',
                '    return {\"evaluation_results\": results.to_dict(\'records\')}\n',
                '```\n',
                '\n',
                'This provides ongoing monitoring of spatial interpolation performance.'
            ]
        }
    ],
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'codemirror_mode': {'name': 'ipython', 'version': 3},
            'file_extension': '.py',
            'mimetype': 'text/x-python',
            'name': 'python',
            'nbconvert_exporter': 'python',
            'pygments_lexer': 'ipython3',
            'version': '3.8.5'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 4
}

# Write the notebook
with open('proxy_spatial_evaluation_demo.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print('Proxy-based spatial evaluation notebook created!')
print('File: proxy_spatial_evaluation_demo.ipynb')
print('Ready for demonstration and results analysis!')