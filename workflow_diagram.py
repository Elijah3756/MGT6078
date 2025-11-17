#!/usr/bin/env python3
"""
Workflow Diagram Generator
Creates a visual representation of the project's workflow showing inputs, processes, and outputs
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch, Shadow
import numpy as np
from matplotlib import font_manager

# Set professional font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'Verdana']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0

def create_workflow_diagram(output_path='output/workflow_diagram.png'):
    """Create a workflow diagram showing the project's data flow"""
    
    fig, ax = plt.subplots(1, 1, figsize=(18, 13))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12.5)
    ax.axis('off')
    
    # Professional color scheme
    input_color = '#E8F4F8'  # Soft blue
    input_border = '#2C5F8D'  # Deep blue
    process_color = '#FFF8E1'  # Soft cream
    process_border = '#E65100'  # Deep orange
    output_color = '#E8F5E9'  # Soft green
    output_border = '#2E7D32'  # Deep green
    decision_color = '#F3E5F5'  # Soft purple
    decision_border = '#6A1B9A'  # Deep purple
    bg_color = '#FAFAFA'  # Off-white background
    
    # Set background
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    # Title with subtitle
    title_box = FancyBboxPatch((0.5, 11.2), 9.0, 1.0,
                              boxstyle="round,pad=0.1", 
                              facecolor='white',
                              edgecolor='#424242',
                              linewidth=2.5,
                              linestyle='-')
    ax.add_patch(title_box)
    ax.text(5, 11.85, 'LLM-Powered Black-Litterman Portfolio Optimization', 
            ha='center', va='center', fontsize=22, fontweight='bold', color='#212121')
    ax.text(5, 11.45, 'System Workflow Diagram', 
            ha='center', va='center', fontsize=14, style='italic', color='#616161')
    
    # ========== INPUTS SECTION ==========
    section_header_box = FancyBboxPatch((0.3, 10.2), 2.4, 0.5,
                                       boxstyle="round,pad=0.05", 
                                       facecolor=input_border,
                                       edgecolor=input_border,
                                       linewidth=2)
    ax.add_patch(section_header_box)
    ax.text(1.5, 10.45, 'INPUTS', ha='center', va='center', 
            fontsize=16, fontweight='bold', color='white')
    
    # Input boxes with shadows
    inputs = [
        ('config.yaml\n(Configuration)', 1.5, 9.5),
        ('Stock Data\n(Price History)', 1.5, 8.5),
        ('Political News\n(GDELT API)', 1.5, 7.5),
        ('Bloomberg\nHeadlines', 1.5, 6.5),
        ('FOMC PDFs\n(Federal Reserve)', 1.5, 5.5),
        ('Investor Research\n(Quarterly Reports)', 1.5, 4.5),
    ]
    
    input_boxes = []
    for text, x, y in inputs:
        # Shadow effect
        shadow = FancyBboxPatch((x-0.72, y-0.32), 1.44, 0.64,
                               boxstyle="round,pad=0.05", 
                               facecolor='#CCCCCC',
                               edgecolor='none',
                               alpha=0.3,
                               zorder=0)
        ax.add_patch(shadow)
        
        box = FancyBboxPatch((x-0.7, y-0.3), 1.4, 0.6,
                            boxstyle="round,pad=0.08", 
                            facecolor=input_color,
                            edgecolor=input_border,
                            linewidth=2.5,
                            zorder=1)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=10, weight='bold', color='#212121', zorder=2)
        input_boxes.append((x, y))
    
    # ========== PROCESSING SECTION ==========
    section_header_box2 = FancyBboxPatch((3.8, 10.2), 2.4, 0.5,
                                        boxstyle="round,pad=0.05", 
                                        facecolor=process_border,
                                        edgecolor=process_border,
                                        linewidth=2)
    ax.add_patch(section_header_box2)
    ax.text(5, 10.45, 'PROCESSING PIPELINE', ha='center', va='center', 
            fontsize=16, fontweight='bold', color='white')
    
    # Process boxes (main flow) with shadows
    processes = [
        ('DataLoader\nLoad & Format\nQuarterly Data', 5, 9.2),
        ('LLMAnalyzer\nGenerate Views\nfrom Data', 5, 8.0),
        ('ViewsConverter\nConvert to\nBL Format', 5, 6.8),
        ('PortfolioOptimizer\nBlack-Litterman\nOptimization', 5, 5.6),
        ('Backtester\nSimulate Performance\nQuarter by Quarter', 5, 4.4),
        ('PerformanceMetrics\nCalculate Risk/Return\nMetrics', 5, 3.2),
    ]
    
    process_boxes = []
    for text, x, y in processes:
        # Shadow effect
        shadow = FancyBboxPatch((x-1.02, y-0.42), 2.04, 0.84,
                               boxstyle="round,pad=0.05", 
                               facecolor='#CCCCCC',
                               edgecolor='none',
                               alpha=0.3,
                               zorder=0)
        ax.add_patch(shadow)
        
        box = FancyBboxPatch((x-1.0, y-0.4), 2.0, 0.8,
                            boxstyle="round,pad=0.1", 
                            facecolor=process_color,
                            edgecolor=process_border,
                            linewidth=3,
                            zorder=1)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=10, weight='bold', color='#212121', zorder=2)
        process_boxes.append((x, y))
    
    # ========== OUTPUTS SECTION ==========
    section_header_box3 = FancyBboxPatch((7.3, 10.2), 2.4, 0.5,
                                        boxstyle="round,pad=0.05", 
                                        facecolor=output_border,
                                        edgecolor=output_border,
                                        linewidth=2)
    ax.add_patch(section_header_box3)
    ax.text(8.5, 10.45, 'OUTPUTS', ha='center', va='center', 
            fontsize=16, fontweight='bold', color='white')
    
    # Output boxes with shadows
    outputs = [
        ('Portfolio\nAllocations\n(JSON)', 8.5, 9.5),
        ('LLM Views\n(JSON)', 8.5, 8.5),
        ('Performance\nCharts\n(PNG)', 8.5, 7.5),
        ('Backtest\nResults\n(JSON)', 8.5, 6.5),
        ('Full Report\n(TXT, JSON)', 8.5, 5.5),
        ('Metrics\nSummary', 8.5, 4.5),
    ]
    
    output_boxes = []
    for text, x, y in outputs:
        # Shadow effect
        shadow = FancyBboxPatch((x-0.72, y-0.32), 1.44, 0.64,
                               boxstyle="round,pad=0.05", 
                               facecolor='#CCCCCC',
                               edgecolor='none',
                               alpha=0.3,
                               zorder=0)
        ax.add_patch(shadow)
        
        box = FancyBboxPatch((x-0.7, y-0.3), 1.4, 0.6,
                            boxstyle="round,pad=0.08", 
                            facecolor=output_color,
                            edgecolor=output_border,
                            linewidth=2.5,
                            zorder=1)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=10, weight='bold', color='#212121', zorder=2)
        output_boxes.append((x, y))
    
    # ========== ARROWS: Inputs to Processes ==========
    # From inputs to DataLoader - professional arrows
    for i, (x_in, y_in) in enumerate(input_boxes):
        arrow = FancyArrowPatch((x_in + 0.7, y_in), (4.0, 9.2),
                               arrowstyle='->', lw=2.5, 
                               color=input_border, alpha=0.8,
                               connectionstyle="arc3,rad=0.1",
                               zorder=1,
                               mutation_scale=20)
        ax.add_patch(arrow)
    
    # ========== ARROWS: Process to Process ==========
    # Between processes (vertical flow) - thicker, more prominent
    for i in range(len(process_boxes) - 1):
        x1, y1 = process_boxes[i]
        x2, y2 = process_boxes[i + 1]
        arrow = FancyArrowPatch((x1, y1 - 0.4), (x2, y2 + 0.4),
                               arrowstyle='->', lw=3.5, 
                               color=process_border, alpha=0.9,
                               zorder=1,
                               mutation_scale=25)
        ax.add_patch(arrow)
    
    # ========== ARROWS: Processes to Outputs ==========
    # DataLoader -> Portfolio Allocations
    arrow1 = FancyArrowPatch((6.0, 9.2), (7.8, 9.5),
                             arrowstyle='->', lw=2.5, 
                             color=output_border, alpha=0.8,
                             connectionstyle="arc3,rad=-0.2",
                             zorder=1,
                             mutation_scale=20)
    ax.add_patch(arrow1)
    
    # LLMAnalyzer -> LLM Views
    arrow2 = FancyArrowPatch((6.0, 8.0), (7.8, 8.5),
                             arrowstyle='->', lw=2.5, 
                             color=output_border, alpha=0.8,
                             connectionstyle="arc3,rad=-0.2",
                             zorder=1,
                             mutation_scale=20)
    ax.add_patch(arrow2)
    
    # PortfolioOptimizer -> Portfolio Allocations
    arrow3 = FancyArrowPatch((6.0, 5.6), (7.8, 9.5),
                             arrowstyle='->', lw=2.5, 
                             color=output_border, alpha=0.8,
                             connectionstyle="arc3,rad=-0.3",
                             zorder=1,
                             mutation_scale=20)
    ax.add_patch(arrow3)
    
    # Backtester -> Charts, Results, Report
    arrow4 = FancyArrowPatch((6.0, 4.4), (7.8, 7.5),
                             arrowstyle='->', lw=2.5, 
                             color=output_border, alpha=0.8,
                             connectionstyle="arc3,rad=-0.2",
                             zorder=1,
                             mutation_scale=20)
    ax.add_patch(arrow4)
    
    arrow5 = FancyArrowPatch((6.0, 4.4), (7.8, 6.5),
                             arrowstyle='->', lw=2.5, 
                             color=output_border, alpha=0.8,
                             connectionstyle="arc3,rad=-0.2",
                             zorder=1,
                             mutation_scale=20)
    ax.add_patch(arrow5)
    
    arrow6 = FancyArrowPatch((6.0, 4.4), (7.8, 5.5),
                             arrowstyle='->', lw=2.5, 
                             color=output_border, alpha=0.8,
                             connectionstyle="arc3,rad=-0.2",
                             zorder=1,
                             mutation_scale=20)
    ax.add_patch(arrow6)
    
    # PerformanceMetrics -> Metrics Summary
    arrow7 = FancyArrowPatch((6.0, 3.2), (7.8, 4.5),
                             arrowstyle='->', lw=2.5, 
                             color=output_border, alpha=0.8,
                             connectionstyle="arc3,rad=-0.2",
                             zorder=1,
                             mutation_scale=20)
    ax.add_patch(arrow7)
    
    # ========== HYPERPARAMETER OPTIMIZATION SECTION ==========
    opt_header_box = FancyBboxPatch((2.8, 1.9), 4.4, 0.4,
                                   boxstyle="round,pad=0.05", 
                                   facecolor=decision_border,
                                   edgecolor=decision_border,
                                   linewidth=2)
    ax.add_patch(opt_header_box)
    ax.text(5, 2.1, 'HYPERPARAMETER OPTIMIZATION (Optional)', ha='center', va='center', 
            fontsize=13, fontweight='bold', color='white')
    
    # Shadow for optimization box
    opt_shadow = FancyBboxPatch((3.52, 1.22), 3.04, 0.64,
                               boxstyle="round,pad=0.05", 
                               facecolor='#CCCCCC',
                               edgecolor='none',
                               alpha=0.3,
                               zorder=0)
    ax.add_patch(opt_shadow)
    
    opt_box = FancyBboxPatch((3.5, 1.2), 3.0, 0.6,
                            boxstyle="round,pad=0.1", 
                            facecolor=decision_color,
                            edgecolor=decision_border,
                            linewidth=2.5,
                            zorder=1)
    ax.add_patch(opt_box)
    ax.text(5, 1.5, 'HyperparameterOptimizer\n(Grid Search / Random Search / ML-based)', 
            ha='center', va='center', fontsize=10, weight='bold', color='#212121', zorder=2)
    
    # Arrow from optimization to config (dashed for optional)
    opt_arrow = FancyArrowPatch((3.5, 1.5), (1.5, 4.8),
                               arrowstyle='->', lw=2, 
                               color=decision_border, alpha=0.7,
                               connectionstyle="arc3,rad=0.3",
                               linestyle='--',
                               zorder=1,
                               mutation_scale=18)
    ax.add_patch(opt_arrow)
    
    # Arrow from optimization to main pipeline (dashed)
    opt_arrow2 = FancyArrowPatch((5, 1.2), (5, 3.6),
                                arrowstyle='->', lw=2, 
                                color=decision_border, alpha=0.7,
                                linestyle='--',
                                zorder=1,
                                mutation_scale=18)
    ax.add_patch(opt_arrow2)
    
    # ========== LEGEND ==========
    legend_box = FancyBboxPatch((0.2, 0.1), 2.2, 0.9,
                               boxstyle="round,pad=0.1", 
                               facecolor='white',
                               edgecolor='#757575',
                               linewidth=2,
                               alpha=0.95)
    ax.add_patch(legend_box)
    
    legend_elements = [
        mpatches.Patch(facecolor=input_color, edgecolor=input_border, label='Inputs', linewidth=2.5),
        mpatches.Patch(facecolor=process_color, edgecolor=process_border, label='Processing Steps', linewidth=3),
        mpatches.Patch(facecolor=output_color, edgecolor=output_border, label='Outputs', linewidth=2.5),
        mpatches.Patch(facecolor=decision_color, edgecolor=decision_border, label='Optional Optimization', linewidth=2.5),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=11, 
             framealpha=1.0, edgecolor='#757575', fancybox=True, 
             shadow=True, frameon=True)
    
    # ========== QUARTERLY LOOP INDICATOR ==========
    loop_box = FancyBboxPatch((3.5, 0.1), 3.0, 0.5,
                              boxstyle="round,pad=0.1", 
                              facecolor='white',
                              edgecolor='#757575',
                              linewidth=2,
                              alpha=0.95)
    ax.add_patch(loop_box)
    loop_text = 'For each quarter:\nQ1_2024 → Q2_2024 → Q3_2024 → Q4_2024 → Q1_2025 → ...'
    ax.text(5, 0.35, loop_text, ha='center', va='center', 
            fontsize=10, style='italic', color='#424242', weight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=bg_color, edgecolor='none')
    print(f"Workflow diagram saved to: {output_path}")
    return fig


def create_detailed_workflow_diagram(output_path='output/workflow_diagram_detailed.png'):
    """Create a more detailed workflow diagram with sub-processes"""
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 15.5)
    ax.axis('off')
    
    # Professional color scheme
    input_color = '#E8F4F8'
    input_border = '#2C5F8D'
    process_color = '#FFF8E1'
    process_border = '#E65100'
    output_color = '#E8F5E9'
    output_border = '#2E7D32'
    subprocess_color = '#FFFDE7'
    subprocess_border = '#F57F17'
    bg_color = '#FAFAFA'
    
    # Set background
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    # Title with subtitle
    title_box = FancyBboxPatch((0.5, 14.0), 11.0, 1.0,
                              boxstyle="round,pad=0.1", 
                              facecolor='white',
                              edgecolor='#424242',
                              linewidth=2.5)
    ax.add_patch(title_box)
    ax.text(6, 14.65, 'Detailed System Workflow', 
            ha='center', va='center', fontsize=24, fontweight='bold', color='#212121')
    ax.text(6, 14.25, 'Component-Level Process Flow', 
            ha='center', va='center', fontsize=15, style='italic', color='#616161')
    
    # ========== CONFIGURATION ==========
    config_shadow = FancyBboxPatch((0.52, 12.82), 2.04, 1.04,
                                 boxstyle="round,pad=0.05", 
                                 facecolor='#CCCCCC',
                                 edgecolor='none',
                                 alpha=0.3,
                                 zorder=0)
    ax.add_patch(config_shadow)
    
    config_box = FancyBboxPatch((0.5, 12.8), 2.0, 1.0,
                               boxstyle="round,pad=0.1", 
                               facecolor=input_color,
                               edgecolor=input_border,
                               linewidth=2.5,
                               zorder=1)
    ax.add_patch(config_box)
    ax.text(1.5, 13.3, 'Configuration\n(config.yaml)', 
            ha='center', va='center', fontsize=11, weight='bold', color='#212121', zorder=2)
    
    # ========== DATA COLLECTION ==========
    section_header = FancyBboxPatch((3.5, 13.0), 5.0, 0.5,
                                   boxstyle="round,pad=0.05", 
                                   facecolor=input_border,
                                   edgecolor=input_border,
                                   linewidth=2)
    ax.add_patch(section_header)
    ax.text(6, 13.25, 'DATA COLLECTION', ha='center', va='center', 
            fontsize=14, fontweight='bold', color='white')
    
    data_sources = [
        ('Stock\nPrices', 4.5, 12.3),
        ('Political\nNews', 6.0, 12.3),
        ('Bloomberg\nHeadlines', 7.5, 12.3),
        ('FOMC\nPDFs', 9.0, 12.3),
        ('Research\nReports', 10.5, 12.3),
    ]
    
    for text, x, y in data_sources:
        shadow = FancyBboxPatch((x-0.62, y-0.32), 1.24, 0.64,
                               boxstyle="round,pad=0.05", 
                               facecolor='#CCCCCC',
                               edgecolor='none',
                               alpha=0.3,
                               zorder=0)
        ax.add_patch(shadow)
        
        box = FancyBboxPatch((x-0.6, y-0.3), 1.2, 0.6,
                            boxstyle="round,pad=0.08", 
                            facecolor=input_color,
                            edgecolor=input_border,
                            linewidth=2.5,
                            zorder=1)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, weight='bold', color='#212121', zorder=2)
    
    # ========== DATA LOADING & PROCESSING ==========
    section_header2 = FancyBboxPatch((3.0, 11.0), 6.0, 0.5,
                                    boxstyle="round,pad=0.05", 
                                    facecolor=process_border,
                                    edgecolor=process_border,
                                    linewidth=2)
    ax.add_patch(section_header2)
    ax.text(6, 11.25, 'DATA LOADING & PROCESSING', ha='center', va='center', 
            fontsize=14, fontweight='bold', color='white')
    
    loader_shadow = FancyBboxPatch((4.02, 10.32), 4.04, 0.84,
                                  boxstyle="round,pad=0.05", 
                                  facecolor='#CCCCCC',
                                  edgecolor='none',
                                  alpha=0.3,
                                  zorder=0)
    ax.add_patch(loader_shadow)
    
    loader_box = FancyBboxPatch((4.0, 10.3), 4.0, 0.8,
                               boxstyle="round,pad=0.1", 
                               facecolor=process_color,
                               edgecolor=process_border,
                               linewidth=3,
                               zorder=1)
    ax.add_patch(loader_box)
    ax.text(6, 10.7, 'DataLoader.load_quarterly_data()', 
            ha='center', va='center', fontsize=11, weight='bold', color='#212121', zorder=2)
    
    # Sub-processes
    subprocesses = [
        ('DataFormatter\nFormat & Clean', 4.5, 9.3),
        ('StockProcessor\nCalculate Returns', 6.0, 9.3),
        ('Quarter Dates\nMapping', 7.5, 9.3),
    ]
    
    for text, x, y in subprocesses:
        shadow = FancyBboxPatch((x-0.72, y-0.32), 1.44, 0.64,
                               boxstyle="round,pad=0.05", 
                               facecolor='#CCCCCC',
                               edgecolor='none',
                               alpha=0.3,
                               zorder=0)
        ax.add_patch(shadow)
        
        box = FancyBboxPatch((x-0.7, y-0.3), 1.4, 0.6,
                            boxstyle="round,pad=0.08", 
                            facecolor=subprocess_color,
                            edgecolor=subprocess_border,
                            linewidth=2.5,
                            zorder=1)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, weight='bold', color='#212121', zorder=2)
        # Arrow from loader to subprocess
        arrow = FancyArrowPatch((x, 10.3), (x, 9.6),
                               arrowstyle='->', lw=2, 
                               color=process_border, alpha=0.7,
                               zorder=1,
                               mutation_scale=15)
        ax.add_patch(arrow)
    
    # ========== LLM ANALYSIS ==========
    section_header3 = FancyBboxPatch((4.5, 8.0), 3.0, 0.5,
                                     boxstyle="round,pad=0.05", 
                                     facecolor=process_border,
                                     edgecolor=process_border,
                                     linewidth=2)
    ax.add_patch(section_header3)
    ax.text(6, 8.25, 'LLM ANALYSIS', ha='center', va='center', 
            fontsize=14, fontweight='bold', color='white')
    
    llm_shadow = FancyBboxPatch((4.02, 7.32), 4.04, 0.84,
                               boxstyle="round,pad=0.05", 
                               facecolor='#CCCCCC',
                               edgecolor='none',
                               alpha=0.3,
                               zorder=0)
    ax.add_patch(llm_shadow)
    
    llm_box = FancyBboxPatch((4.0, 7.3), 4.0, 0.8,
                             boxstyle="round,pad=0.1", 
                             facecolor=process_color,
                             edgecolor=process_border,
                             linewidth=3,
                             zorder=1)
    ax.add_patch(llm_box)
    ax.text(6, 7.7, 'LLMAnalyzer.generate_views()', 
            ha='center', va='center', fontsize=11, weight='bold', color='#212121', zorder=2)
    
    llm_subprocesses = [
        ('Create Prompt\nfrom Data', 4.5, 6.3),
        ('Finance-LLM\nModel Inference', 6.0, 6.3),
        ('ViewsParser\nExtract Views', 7.5, 6.3),
    ]
    
    for text, x, y in llm_subprocesses:
        shadow = FancyBboxPatch((x-0.72, y-0.32), 1.44, 0.64,
                               boxstyle="round,pad=0.05", 
                               facecolor='#CCCCCC',
                               edgecolor='none',
                               alpha=0.3,
                               zorder=0)
        ax.add_patch(shadow)
        
        box = FancyBboxPatch((x-0.7, y-0.3), 1.4, 0.6,
                            boxstyle="round,pad=0.08", 
                            facecolor=subprocess_color,
                            edgecolor=subprocess_border,
                            linewidth=2.5,
                            zorder=1)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, weight='bold', color='#212121', zorder=2)
        arrow = FancyArrowPatch((x, 7.3), (x, 6.6),
                               arrowstyle='->', lw=2, 
                               color=process_border, alpha=0.7,
                               zorder=1,
                               mutation_scale=15)
        ax.add_patch(arrow)
    
    # ========== PORTFOLIO OPTIMIZATION ==========
    section_header4 = FancyBboxPatch((3.5, 5.0), 5.0, 0.5,
                                     boxstyle="round,pad=0.05", 
                                     facecolor=process_border,
                                     edgecolor=process_border,
                                     linewidth=2)
    ax.add_patch(section_header4)
    ax.text(6, 5.25, 'PORTFOLIO OPTIMIZATION', ha='center', va='center', 
            fontsize=14, fontweight='bold', color='white')
    
    opt_shadow = FancyBboxPatch((4.02, 4.32), 4.04, 0.84,
                               boxstyle="round,pad=0.05", 
                               facecolor='#CCCCCC',
                               edgecolor='none',
                               alpha=0.3,
                               zorder=0)
    ax.add_patch(opt_shadow)
    
    opt_box = FancyBboxPatch((4.0, 4.3), 4.0, 0.8,
                            boxstyle="round,pad=0.1", 
                            facecolor=process_color,
                            edgecolor=process_border,
                            linewidth=3,
                            zorder=1)
    ax.add_patch(opt_box)
    ax.text(6, 4.7, 'PortfolioOptimizer.optimize_quarter()', 
            ha='center', va='center', fontsize=11, weight='bold', color='#212121', zorder=2)
    
    opt_subprocesses = [
        ('ViewsConverter\nP, Q, Ω Matrices', 4.5, 3.3),
        ('Black-Litterman\nModel', 6.0, 3.3),
        ('Optimization\nSolver', 7.5, 3.3),
    ]
    
    for text, x, y in opt_subprocesses:
        shadow = FancyBboxPatch((x-0.72, y-0.32), 1.44, 0.64,
                               boxstyle="round,pad=0.05", 
                               facecolor='#CCCCCC',
                               edgecolor='none',
                               alpha=0.3,
                               zorder=0)
        ax.add_patch(shadow)
        
        box = FancyBboxPatch((x-0.7, y-0.3), 1.4, 0.6,
                            boxstyle="round,pad=0.08", 
                            facecolor=subprocess_color,
                            edgecolor=subprocess_border,
                            linewidth=2.5,
                            zorder=1)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, weight='bold', color='#212121', zorder=2)
        arrow = FancyArrowPatch((x, 4.3), (x, 3.6),
                               arrowstyle='->', lw=2, 
                               color=process_border, alpha=0.7,
                               zorder=1,
                               mutation_scale=15)
        ax.add_patch(arrow)
    
    # ========== BACKTESTING ==========
    section_header5 = FancyBboxPatch((2.5, 1.8), 7.0, 0.5,
                                     boxstyle="round,pad=0.05", 
                                     facecolor=process_border,
                                     edgecolor=process_border,
                                     linewidth=2)
    ax.add_patch(section_header5)
    ax.text(6, 2.05, 'BACKTESTING & REPORTING', ha='center', va='center', 
            fontsize=14, fontweight='bold', color='white')
    
    backtest_shadow = FancyBboxPatch((2.02, 1.12), 8.04, 0.84,
                                    boxstyle="round,pad=0.05", 
                                    facecolor='#CCCCCC',
                                    edgecolor='none',
                                    alpha=0.3,
                                    zorder=0)
    ax.add_patch(backtest_shadow)
    
    backtest_box = FancyBboxPatch((2.0, 1.1), 8.0, 0.8,
                                 boxstyle="round,pad=0.1", 
                                 facecolor=process_color,
                                 edgecolor=process_border,
                                 linewidth=3,
                                 zorder=1)
    ax.add_patch(backtest_box)
    ax.text(6, 1.5, 'Backtester.run_backtest() → PerformanceMetrics → Visualizer → ReportGenerator', 
            ha='center', va='center', fontsize=10, weight='bold', color='#212121', zorder=2)
    
    # ========== OUTPUTS ==========
    outputs = [
        ('Portfolio\nJSON', 1.0, 0.3),
        ('Views\nJSON', 2.5, 0.3),
        ('Charts\nPNG', 4.0, 0.3),
        ('Results\nJSON', 5.5, 0.3),
        ('Report\nTXT', 7.0, 0.3),
        ('Metrics', 8.5, 0.3),
    ]
    
    for text, x, y in outputs:
        shadow = FancyBboxPatch((x-0.52, y-0.32), 1.04, 0.64,
                               boxstyle="round,pad=0.05", 
                               facecolor='#CCCCCC',
                               edgecolor='none',
                               alpha=0.3,
                               zorder=0)
        ax.add_patch(shadow)
        
        box = FancyBboxPatch((x-0.5, y-0.3), 1.0, 0.6,
                            boxstyle="round,pad=0.08", 
                            facecolor=output_color,
                            edgecolor=output_border,
                            linewidth=2.5,
                            zorder=1)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, weight='bold', color='#212121', zorder=2)
        # Arrow from backtest to outputs
        arrow = FancyArrowPatch((x, 1.1), (x, 0.6),
                               arrowstyle='->', lw=2.5, 
                               color=output_border, alpha=0.8,
                               zorder=1,
                               mutation_scale=18)
        ax.add_patch(arrow)
    
    # ========== MAIN FLOW ARROWS ==========
    # Config to data collection
    arrow1 = FancyArrowPatch((2.5, 13.3), (4.0, 12.6),
                            arrowstyle='->', lw=2.5, 
                            color=input_border, alpha=0.8,
                            zorder=1,
                            mutation_scale=20)
    ax.add_patch(arrow1)
    
    # Data sources to loader
    for x in [4.5, 6.0, 7.5, 9.0, 10.5]:
        arrow = FancyArrowPatch((x, 12.0), (x, 11.1),
                               arrowstyle='->', lw=2, 
                               color=input_border, alpha=0.7,
                               zorder=1,
                               mutation_scale=18)
        ax.add_patch(arrow)
    
    # Loader to LLM
    arrow2 = FancyArrowPatch((6, 10.3), (6, 8.1),
                            arrowstyle='->', lw=3.5, 
                            color=process_border, alpha=0.9,
                            zorder=1,
                            mutation_scale=25)
    ax.add_patch(arrow2)
    
    # LLM to Optimization
    arrow3 = FancyArrowPatch((6, 7.3), (6, 5.1),
                            arrowstyle='->', lw=3.5, 
                            color=process_border, alpha=0.9,
                            zorder=1,
                            mutation_scale=25)
    ax.add_patch(arrow3)
    
    # Optimization to Backtesting
    arrow4 = FancyArrowPatch((6, 4.3), (6, 1.9),
                            arrowstyle='->', lw=3.5, 
                            color=process_border, alpha=0.9,
                            zorder=1,
                            mutation_scale=25)
    ax.add_patch(arrow4)
    
    # Config arrow to main flow (dashed)
    config_arrow = FancyArrowPatch((1.5, 12.8), (5.5, 10.8),
                                  arrowstyle='->', lw=2, 
                                  color=input_border, alpha=0.6,
                                  linestyle='--',
                                  zorder=1,
                                  mutation_scale=18)
    ax.add_patch(config_arrow)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=bg_color, edgecolor='none')
    print(f"Detailed workflow diagram saved to: {output_path}")
    return fig


if __name__ == "__main__":
    import os
    os.makedirs('output', exist_ok=True)
    
    print("Generating workflow diagrams...")
    create_workflow_diagram()
    create_detailed_workflow_diagram()
    print("\nDone! Check the output/ directory for the diagrams.")

