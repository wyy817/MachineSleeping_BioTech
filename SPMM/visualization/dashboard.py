"""
Dashboard Module: Create integrated visualization dashboards
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from .spatial import SpatialVisualizer
from .attention import AttentionVisualizer
from .multimodal import MultimodalVisualizer
from .tumor_microenvironment import TMEVisualizer
from .interactive import InteractiveVisualizer


class DashboardCreator:
    """Comprehensive visualization dashboard creator"""
    
    def __init__(self, output_dir='./visualizations/dashboard'):
        """
        Initialize the dashboard creator
        
        Args:
            output_dir: Output directory path
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize various visualizers
        self.spatial_visualizer = SpatialVisualizer(os.path.join(output_dir, 'spatial'))
        self.attention_visualizer = AttentionVisualizer(os.path.join(output_dir, 'attention'))
        self.multimodal_visualizer = MultimodalVisualizer(os.path.join(output_dir, 'multimodal'))
        self.tme_visualizer = TMEVisualizer(os.path.join(output_dir, 'tme'))
        self.interactive_visualizer = InteractiveVisualizer(os.path.join(output_dir, 'interactive'))
    
    def create_dashboard(self, model_outputs, spatial_data, text_data, vision_data, 
                       gene_names, cell_types, spatial_coords, class_names):
        """
        Create comprehensive visualization dashboard
        
        Args:
            model_outputs: Model output dictionary
            spatial_data: Spatial feature data
            text_data: Text feature data
            vision_data: Vision feature data
            gene_names: List of gene names
            cell_types: Cell type labels
            spatial_coords: Spatial coordinates
            class_names: List of class names
            
        Returns:
            dashboard_paths: Dictionary containing paths to all visualization files
        """
        # 1. Visualize attention weights
        attention_path = os.path.join(self.output_dir, 'attention_weights.png')
        self.attention_visualizer.visualize_attention_weights(
            model_outputs['attention_weights'], 
            gene_names, 
            attention_path
        )
        
        # 2. Visualize prediction uncertainty
        uncertainty_path = os.path.join(self.output_dir, 'uncertainty.png')
        self.attention_visualizer.visualize_uncertainty(
            model_outputs['uncertainty'], 
            class_names, 
            uncertainty_path
        )
        
        # 3. Visualize spatial features
        spatial_features_path = os.path.join(self.output_dir, 'spatial_features.png')
        self.spatial_visualizer.visualize_spatial_features(
            spatial_data, 
            model_outputs['classification'].argmax(dim=1) if torch.is_tensor(model_outputs['classification']) else np.argmax(model_outputs['classification'], axis=1), 
            spatial_coords, 
            spatial_features_path
        )
        
        # 4. Visualize multimodal feature integration
        multimodal_path = os.path.join(self.output_dir, 'multimodal_integration.png')
        self.multimodal_visualizer.visualize_multimodal_integration(
            spatial_data, 
            text_data, 
            vision_data, 
            multimodal_path
        )
        
        # 5. Visualize gene expression
        gene_expression_path = os.path.join(self.output_dir, 'gene_expression.png')
        self.spatial_visualizer.visualize_gene_expression_spatial(
            spatial_data, 
            spatial_coords, 
            gene_names, 
            None, 
            gene_expression_path
        )
        
        # 6. Visualize cell type distribution
        cell_types_path = os.path.join(self.output_dir, 'cell_types.png')
        self.spatial_visualizer.visualize_cell_type_distribution(
            cell_types, 
            spatial_coords, 
            cell_types_path
        )
        
        # 7. Visualize tumor microenvironment
        # Assume some marker genes
        tumor_marker_genes = ['MKI67', 'EGFR', 'MYC', 'CCNE1']
        immune_marker_genes = ['CD3D', 'CD8A', 'CD4', 'FOXP3', 'CD68']
        tme_path = os.path.join(self.output_dir, 'tumor_microenvironment.png')
        self.tme_visualizer.visualize_tumor_microenvironment(
            model_outputs['features'], 
            cell_types, 
            spatial_data, 
            spatial_coords, 
            gene_names, 
            tumor_marker_genes, 
            immune_marker_genes, 
            tme_path
        )
        
        # 8. Create integrated analysis visualization
        integrated_path = os.path.join(self.output_dir, 'integrated_analysis.png')
        self.multimodal_visualizer.visualize_integrated_analysis(
            model_outputs['classification'][0] if len(model_outputs['classification'].shape) > 1 else model_outputs['classification'], 
            model_outputs['attention_weights'], 
            model_outputs['uncertainty'][0] if len(model_outputs['uncertainty'].shape) > 1 else model_outputs['uncertainty'],
            gene_names if len(gene_names) == model_outputs['attention_weights'].shape[0] else 
            [f"Feature_{i}" for i in range(model_outputs['attention_weights'].shape[0])], 
            class_names, 
            integrated_path
        )
        
        # 9. Create interactive visualization
        interactive_path = os.path.join(self.output_dir, 'interactive_spatial.html')
        self.interactive_visualizer.create_interactive_spatial_plot(
            spatial_coords, 
            cell_types, 
            spatial_data, 
            gene_names, 
            interactive_path
        )
        
        # 10. Create interactive feature explorer
        feature_explorer_path = os.path.join(self.output_dir, 'feature_explorer.html')
        self.interactive_visualizer.create_interactive_feature_explorer(
            model_outputs['features'],
            model_outputs['classification'].argmax(dim=1) if torch.is_tensor(model_outputs['classification']) else np.argmax(model_outputs['classification'], axis=1),
            gene_names[:model_outputs['features'].shape[1]] if len(gene_names) >= model_outputs['features'].shape[1] else 
            [f"Feature_{i}" for i in range(model_outputs['features'].shape[1])],
            feature_explorer_path
        )
        
        # 11. Create interactive gene heatmap
        heatmap_path = os.path.join(self.output_dir, 'gene_heatmap.html')
        self.interactive_visualizer.create_interactive_gene_heatmap(
            spatial_data,
            gene_names,
            cell_types,
            None,
            heatmap_path
        )
        
        # Collect all visualization paths
        dashboard_paths = {
            'attention_weights_path': attention_path,
            'uncertainty_path': uncertainty_path,
            'spatial_features_path': spatial_features_path,
            'multimodal_path': multimodal_path,
            'gene_expression_path': gene_expression_path,
            'cell_types_path': cell_types_path,
            'tme_path': tme_path,
            'integrated_path': integrated_path,
            'interactive_path': interactive_path,
            'feature_explorer_path': feature_explorer_path,
            'heatmap_path': heatmap_path
        }
        
        # Create HTML index page
        self._create_index_page(dashboard_paths)
        
        print(f"All visualizations have been saved to the {self.output_dir} directory")
        
        return dashboard_paths
    
    def _create_index_page(self, dashboard_paths):
        """
        Create dashboard index page
        
        Args:
            dashboard_paths: Dictionary of visualization file paths
        """
        index_path = os.path.join(self.output_dir, 'index.html')
        
        # Create HTML content
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Spatial Transcriptomics Multimodal Analysis Dashboard</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                }
                h1 {
                    color: #2c3e50;
                    text-align: center;
                    margin-bottom: 30px;
                }
                h2 {
                    color: #3498db;
                    margin-top: 30px;
                }
                .dashboard-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                    gap: 20px;
                    margin-top: 20px;
                }
                .dashboard-item {
                    border: 1px solid #ddd;
                    border-radius: 8px;
                    overflow: hidden;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    transition: transform 0.3s ease;
                }
                .dashboard-item:hover {
                    transform: translateY(-5px);
                }
                .dashboard-item img {
                    width: 100%;
                    height: auto;
                    display: block;
                }
                .dashboard-item-caption {
                    padding: 15px;
                    background: #f9f9f9;
                }
                .dashboard-item-caption h3 {
                    margin: 0 0 10px 0;
                    color: #2c3e50;
                }
                .dashboard-item-caption p {
                    margin: 0;
                    color: #7f8c8d;
                }
                .interactive-section {
                    margin-top: 40px;
                }
                .interactive-links {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 15px;
                    margin-top: 20px;
                }
                .interactive-link {
                    background: #3498db;
                    color: white;
                    padding: 10px 20px;
                    border-radius: 5px;
                    text-decoration: none;
                    font-weight: bold;
                    transition: background 0.3s ease;
                }
                .interactive-link:hover {
                    background: #2980b9;
                }
            </style>
        </head>
        <body>
            <h1>Spatial Transcriptomics Multimodal Analysis Dashboard</h1>
            
            <h2>Static Visualizations</h2>
            <div class="dashboard-grid">
        """
        
        # Add static visualization items
        static_items = {
            'attention_weights_path': {'title': 'Attention Weights Visualization', 'description': 'Shows importance weights of model features.'},
            'uncertainty_path': {'title': 'Prediction Uncertainty', 'description': 'Shows uncertainty in model predictions.'},
            'spatial_features_path': {'title': 'Spatial Features Visualization', 'description': 'Shows feature distribution in space.'},
            'multimodal_path': {'title': 'Multimodal Integration', 'description': 'Shows integration of features from different modalities.'},
            'gene_expression_path': {'title': 'Gene Expression Visualization', 'description': 'Shows spatial expression patterns of key genes.'},
            'cell_types_path': {'title': 'Cell Type Distribution', 'description': 'Shows spatial distribution of different cell types.'},
            'tme_path': {'title': 'Tumor Microenvironment Analysis', 'description': 'Shows cell interactions in tumor microenvironment.'},
            'integrated_path': {'title': 'Integrated Analysis', 'description': 'Comprehensive display of prediction results and key features.'}
        }
        
        for path_key, item_info in static_items.items():
            if path_key in dashboard_paths:
                # Get file name
                file_path = dashboard_paths[path_key]
                file_name = os.path.basename(file_path)
                
                # Get relative path
                rel_path = os.path.relpath(file_path, self.output_dir)
                
                html_content += f"""
                <div class="dashboard-item">
                    <img src="{rel_path}" alt="{item_info['title']}">
                    <div class="dashboard-item-caption">
                        <h3>{item_info['title']}</h3>
                        <p>{item_info['description']}</p>
                    </div>
                </div>
                """
        
        # Add interactive visualization links
        html_content += """
            </div>
            
            <h2 class="interactive-section">Interactive Visualizations</h2>
            <div class="interactive-links">
        """
        
        # Add interactive visualization items
        interactive_items = {
            'interactive_path': {'title': 'Interactive Spatial Browser', 'description': 'Interactively explore spatial data and gene expression.'},
            'feature_explorer_path': {'title': 'Feature Explorer', 'description': 'Interactively explore feature space and model representations.'},
            'heatmap_path': {'title': 'Gene Expression Heatmap', 'description': 'Interactively explore gene expression patterns.'}
        }
        
        for path_key, item_info in interactive_items.items():
            if path_key in dashboard_paths:
                # Get relative path
                file_path = dashboard_paths[path_key]
                rel_path = os.path.relpath(file_path, self.output_dir)
                
                html_content += f"""
                <a href="{rel_path}" class="interactive-link" target="_blank">{item_info['title']}</a>
                """
        
        # Complete HTML
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Write HTML file
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Dashboard index page created: {index_path}")
    
    def visualize_domain_generalization(self, source_features, target_features, source_labels, target_labels, 
                                      source_domain_name="Source", target_domain_name="Target"):
        """
        Visualize domain generalization performance
        
        Args:
            source_features: Source domain features
            target_features: Target domain features
            source_labels: Source domain labels
            target_labels: Target domain labels
            source_domain_name: Source domain name
            target_domain_name: Target domain name
            
        Returns:
            dg_path: Domain generalization visualization path
        """
        dg_path = os.path.join(self.output_dir, 'domain_generalization.png')
        
        self.multimodal_visualizer.visualize_domain_generalization(
            source_features, 
            target_features, 
            source_labels, 
            target_labels, 
            source_domain_name, 
            target_domain_name, 
            dg_path
        )
        
        return dg_path
    
    def visualize_prediction_results(self, predictions, true_labels, class_names):
        """
        Visualize prediction results
        
        Args:
            predictions: Prediction results
            true_labels: True labels
            class_names: Class names
            
        Returns:
            results_path: Prediction results visualization path
        """
        results_path = os.path.join(self.output_dir, 'prediction_results.png')
        
        # Ensure data is on CPU
        preds = predictions.cpu().detach().numpy() if torch.is_tensor(predictions) else predictions
        labels = true_labels.cpu().detach().numpy() if torch.is_tensor(true_labels) else np.array(true_labels)
        
        # If probability distribution, convert to class indices
        if len(preds.shape) > 1:
            pred_classes = np.argmax(preds, axis=1)
        else:
            pred_classes = preds
        
        # Create confusion matrix
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        cm = confusion_matrix(labels, pred_classes)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                  xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(results_path)
        plt.close()
        
        # Calculate classification metrics
        from sklearn.metrics import classification_report, accuracy_score
        
        accuracy = accuracy_score(labels, pred_classes)
        report = classification_report(labels, pred_classes, target_names=class_names)
        
        # Save classification report
        report_path = os.path.join(self.output_dir, 'classification_report.txt')
        with open(report_path, 'w') as f:
            f.write(f"Accuracy: {accuracy}\n\n")
            f.write(report)
        
        print(f"Prediction results visualization saved: {results_path}")
        print(f"Classification report saved: {report_path}")
        
        return {'confusion_matrix': results_path, 'report': report_path}
    
    def create_model_overview_page(self, model_architecture, training_history=None):
        """
        Create model overview page
        
        Args:
            model_architecture: Model architecture string
            training_history: Training history data, including loss and accuracy
            
        Returns:
            overview_path: Model overview page path
        """
        overview_path = os.path.join(self.output_dir, 'model_overview.html')
        
        # Create HTML content
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Model Architecture and Training Overview</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                }
                h1, h2 {
                    color: #2c3e50;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                }
                .architecture {
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 5px;
                    margin-bottom: 30px;
                    overflow-x: auto;
                }
                pre {
                    white-space: pre-wrap;
                    font-family: 'Courier New', Courier, monospace;
                }
                .training-history {
                    margin-top: 30px;
                }
                .chart-container {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    margin-top: 20px;
                }
                .chart {
                    flex: 1;
                    min-width: 400px;
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                }
            </style>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        </head>
        <body>
            <div class="container">
                <h1>Model Architecture and Training Overview</h1>
                
                <h2>Model Architecture</h2>
                <div class="architecture">
                    <pre><code>""" + model_architecture + """</code></pre>
                </div>
        """
        
        # If there is training history, add related charts
        if training_history is not None:
            html_content += """
                <h2 class="training-history">Training History</h2>
                <div class="chart-container">
                    <div class="chart">
                        <canvas id="lossChart"></canvas>
                    </div>
                    <div class="chart">
                        <canvas id="accuracyChart"></canvas>
                    </div>
                </div>
                
                <script>
                    // Loss chart
                    const lossCtx = document.getElementById('lossChart').getContext('2d');
                    const lossChart = new Chart(lossCtx, {
                        type: 'line',
                        data: {
                            labels: """ + str(list(range(1, len(training_history['loss']) + 1))) + """,
                            datasets: [
                                {
                                    label: 'Training Loss',
                                    data: """ + str(training_history['loss']) + """,
                                    borderColor: 'rgb(54, 162, 235)',
                                    tension: 0.1
                                }"""
            
            if 'val_loss' in training_history:
                html_content += """,
                                {
                                    label: 'Validation Loss',
                                    data: """ + str(training_history['val_loss']) + """,
                                    borderColor: 'rgb(255, 99, 132)',
                                    tension: 0.1
                                }"""
                
            html_content += """
                            ]
                        },
                        options: {
                            responsive: true,
                            plugins: {
                                title: {
                                    display: true,
                                    text: 'Loss During Training'
                                }
                            }
                        }
                    });
                    
                    // Accuracy chart
                    const accCtx = document.getElementById('accuracyChart').getContext('2d');
                    const accChart = new Chart(accCtx, {
                        type: 'line',
                        data: {
                            labels: """ + str(list(range(1, len(training_history['accuracy']) + 1))) + """,
                            datasets: [
                                {
                                    label: 'Training Accuracy',
                                    data: """ + str(training_history['accuracy']) + """,
                                    borderColor: 'rgb(75, 192, 192)',
                                    tension: 0.1
                                }"""
                
            if 'val_accuracy' in training_history:
                html_content += """,
                                {
                                    label: 'Validation Accuracy',
                                    data: """ + str(training_history['val_accuracy']) + """,
                                    borderColor: 'rgb(153, 102, 255)',
                                    tension: 0.1
                                }"""
                
            html_content += """
                            ]
                        },
                        options: {
                            responsive: true,
                            plugins: {
                                title: {
                                    display: true,
                                    text: 'Accuracy During Training'
                                }
                            }
                        }
                    });
                </script>
            """
        
        # Complete HTML
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Write HTML file
        with open(overview_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Model overview page created: {overview_path}")
        
        return overview_path