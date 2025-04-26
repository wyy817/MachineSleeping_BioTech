"""
Interactive Visualization Module: Create interactive visualizations
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import torch
import pandas as pd
import os


class InteractiveVisualizer:
    """Interactive visualization"""
    
    def __init__(self, output_dir='./visualizations/interactive'):
        """
        Initialize the visualizer
        
        Args:
            output_dir: Output directory path
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def create_interactive_spatial_plot(self, spatial_coords, cell_types, gene_expression, gene_names, output_path=None):
        """
        Create interactive spatial visualization, can switch between different gene expressions
        
        Args:
            spatial_coords: Spatial coordinates
            cell_types: Cell types
            gene_expression: Gene expression data
            gene_names: Gene names
            output_path: Output file path
            
        Returns:
            fig: Plotly figure object
        """
        # Ensure data is on CPU
        coords = spatial_coords.cpu().detach().numpy() if torch.is_tensor(spatial_coords) else spatial_coords
        types = cell_types.cpu().detach().numpy() if torch.is_tensor(cell_types) else np.array(cell_types)
        expression = gene_expression.cpu().detach().numpy() if torch.is_tensor(gene_expression) else gene_expression
        
        # Create interactive figure
        fig = make_subplots(rows=1, cols=2, 
                          subplot_titles=("Cell Type Distribution", "Gene Expression"),
                          specs=[[{"type": "scatter"}, {"type": "scatter"}]])
        
        # Add cell type distribution
        unique_types = np.unique(types)
        colors = px.colors.qualitative.Plotly[:len(unique_types)]
        
        for i, cell_type in enumerate(unique_types):
            mask = types == cell_type
            fig.add_trace(
                go.Scatter(
                    x=coords[mask, 0], 
                    y=coords[mask, 1],
                    mode='markers',
                    marker=dict(color=colors[i], size=8),
                    name=f'Cell Type {cell_type}',
                    legendgroup='cell_types',
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # Add gene expression (default displays the first gene)
        selected_gene_idx = 0
        fig.add_trace(
            go.Scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                mode='markers',
                marker=dict(
                    size=8,
                    color=expression[:, selected_gene_idx],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title=f"{gene_names[selected_gene_idx]} Expression")
                ),
                name=f"{gene_names[selected_gene_idx]} Expression",
                legendgroup='gene_expression',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Add dropdown menu for gene selection
        buttons = []
        for i, gene_name in enumerate(gene_names[:min(50, len(gene_names))]):  # Limit to the first 50 genes to avoid too many
            buttons.append(
                dict(
                    method="update",
                    label=gene_name,
                    args=[
                        {"marker.color": [None, expression[:, i]]},
                        {"marker.colorbar.title": f"{gene_name} Expression"}
                    ],
                    args2=[{"title": f"{gene_name} Expression"}]
                )
            )
        
        fig.update_layout(
            updatemenus=[
                dict(
                    active=0,
                    buttons=buttons,
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.5,
                    xanchor="center",
                    y=1.15,
                    yanchor="top"
                )
            ],
            height=600,
            width=1200,
            title_text="Interactive Spatial Visualization",
            legend_title="Cell Types"
        )
        
        fig.update_xaxes(title_text="X Coordinate", row=1, col=1)
        fig.update_yaxes(title_text="Y Coordinate", row=1, col=1)
        fig.update_xaxes(title_text="X Coordinate", row=1, col=2)
        fig.update_yaxes(title_text="Y Coordinate", row=1, col=2)
        
        if output_path:
            fig.write_html(output_path)
        
        return fig
    
    def create_interactive_feature_explorer(self, features, labels, feature_names, output_path=None):
        """
        Create interactive feature explorer
        
        Args:
            features: Feature matrix
            labels: Labels
            feature_names: Feature names
            output_path: Output file path
            
        Returns:
            fig: Plotly figure object
        """
        # Ensure data is on CPU
        feature_data = features.cpu().detach().numpy() if torch.is_tensor(features) else features
        label_data = labels.cpu().detach().numpy() if torch.is_tensor(labels) else np.array(labels)
        
        # Create dataframe
        df = pd.DataFrame(feature_data, columns=feature_names if feature_names else [f"Feature_{i}" for i in range(feature_data.shape[1])])
        df['Label'] = label_data
        
        # Calculate principal components (for initial visualization)
        from sklearn.decomposition import PCA
        
        # If there are more than 2 features, apply PCA
        if feature_data.shape[1] > 2:
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(feature_data)
            df['PCA1'] = pca_result[:, 0]
            df['PCA2'] = pca_result[:, 1]
            
            # Calculate principal component contribution for each feature
            loadings = pd.DataFrame(
                pca.components_.T,
                columns=['PC1', 'PC2'],
                index=df.columns[:-3]  # Exclude Label and PCA columns
            )
        else:
            # If there are only two features, use them directly
            df['PCA1'] = feature_data[:, 0]
            df['PCA2'] = feature_data[:, 1]
            loadings = None
        
        # Create scatter plot
        fig = go.Figure()
        
        # Create a scatter plot for each label
        unique_labels = np.unique(label_data)
        for label in unique_labels:
            mask = label_data == label
            fig.add_trace(go.Scatter(
                x=df.loc[mask, 'PCA1'],
                y=df.loc[mask, 'PCA2'],
                mode='markers',
                name=f'Class {label}',
                marker=dict(size=8)
            ))
        
        # Add feature vectors (if PCA is available)
        if loadings is not None:
            max_val = max(abs(df['PCA1'].max()), abs(df['PCA1'].min()),
                         abs(df['PCA2'].max()), abs(df['PCA2'].min()))
            scale_factor = max_val * 0.2
            
            for i, feature in enumerate(loadings.index):
                fig.add_trace(go.Scatter(
                    x=[0, loadings.loc[feature, 'PC1'] * scale_factor],
                    y=[0, loadings.loc[feature, 'PC2'] * scale_factor],
                    mode='lines+text',
                    line=dict(color='gray', width=1),
                    text=['', feature],
                    textposition='top center',
                    showlegend=False
                ))
        
        # Add interactive functionality: feature selection
        buttons = []
        
        # Add PCA option
        buttons.append(dict(
            method='update',
            label='PCA',
            args=[{'x': [df.loc[df['Label'] == label, 'PCA1'] for label in unique_labels],
                  'y': [df.loc[df['Label'] == label, 'PCA2'] for label in unique_labels]}],
            args2=[{'title': 'PCA Visualization'}]
        ))
        
        # Add options for each pair of features
        n_features = min(10, feature_data.shape[1])  # Limit to first 10 features
        for i in range(n_features):
            for j in range(i+1, n_features):
                feature_i = df.columns[i]
                feature_j = df.columns[j]
                
                buttons.append(dict(
                    method='update',
                    label=f'{feature_i} vs {feature_j}',
                    args=[{'x': [df.loc[df['Label'] == label, feature_i] for label in unique_labels],
                          'y': [df.loc[df['Label'] == label, feature_j] for label in unique_labels]}],
                    args2=[{'title': f'{feature_i} vs {feature_j}'}]
                ))
        
        # Update layout
        fig.update_layout(
            title='Interactive Feature Explorer',
            xaxis_title='Feature 1',
            yaxis_title='Feature 2',
            updatemenus=[dict(
                active=0,
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.15,
                yanchor="top"
            )],
            legend_title='Classes',
            height=600,
            width=800
        )
        
        if output_path:
            fig.write_html(output_path)
        
        return fig
    
    def create_interactive_gene_heatmap(self, gene_expression, gene_names, cell_types=None, cell_metadata=None, output_path=None):
        """
        Create interactive gene expression heatmap
        
        Args:
            gene_expression: Gene expression data
            gene_names: Gene names
            cell_types: Cell types
            cell_metadata: Cell metadata
            output_path: Output file path
            
        Returns:
            fig: Plotly figure object
        """
        # Ensure data is on CPU
        expr_data = gene_expression.cpu().detach().numpy() if torch.is_tensor(gene_expression) else gene_expression
        
        # If there are too many genes, only select the top 100 with highest expression variance
        if len(gene_names) > 100:
            gene_var = np.var(expr_data, axis=0)
            top_var_indices = np.argsort(gene_var)[-100:]
            expr_data = expr_data[:, top_var_indices]
            selected_genes = [gene_names[i] for i in top_var_indices]
        else:
            selected_genes = gene_names
        
        # Create heatmap
        fig = go.Figure()
        
        # Add heatmap
        heatmap = go.Heatmap(
            z=expr_data,
            x=selected_genes,
            y=None,  # Use index as row labels
            colorscale='Viridis',
            colorbar=dict(title='Expression Level')
        )
        fig.add_trace(heatmap)
        
        # If cell types are available, add a sidebar
        if cell_types is not None:
            # Ensure cell_types is a NumPy array
            type_data = cell_types.cpu().detach().numpy() if torch.is_tensor(cell_types) else np.array(cell_types)
            
            # Get unique cell types
            unique_types = np.unique(type_data)
            
            # Convert cell types to numerical values
            type_values = np.zeros(len(type_data))
            for i, t in enumerate(unique_types):
                type_values[type_data == t] = i
            
            # Create sidebar heatmap
            type_heatmap = go.Heatmap(
                z=type_values.reshape(-1, 1),
                colorscale='Rainbow',
                showscale=False,
                x=['Cell Type'],
                y=None
            )
            
            # Create subplot layout
            fig = make_subplots(rows=1, cols=2, column_widths=[0.05, 0.95], 
                             shared_yaxes=True, horizontal_spacing=0.01)
            
            # Add sidebar and main heatmap
            fig.add_trace(type_heatmap, row=1, col=1)
            fig.add_trace(heatmap, row=1, col=2)
            
            # Add legend
            for i, t in enumerate(unique_types):
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(size=10, color=px.colors.sequential.Rainbow[int(i * (len(px.colors.sequential.Rainbow)-1) / (len(unique_types)-1))]),
                    name=f'Cell Type {t}',
                    showlegend=True
                ), row=1, col=1)
        
        # Add clustering functionality
        buttons = []
        
        # Original order
        buttons.append(dict(
            method='update',
            label='Original Order',
            args=[{'y': [None, None] if cell_types is not None else [None]}],  # Keep original order
        ))
        
        # Cluster by cell type
        if cell_types is not None:
            # Sort by cell type
            sorted_indices = np.argsort(type_data)
            
            buttons.append(dict(
                method='update',
                label='Group by Cell Type',
                args=[{'y': [None, None] if cell_types is not None else [None],
                      'z': [type_values[sorted_indices].reshape(-1, 1), expr_data[sorted_indices]] if cell_types is not None else [expr_data[sorted_indices]]}],
            ))
        
        # Cluster by gene expression
        try:
            from scipy.cluster.hierarchy import linkage, dendrogram
            
            # Hierarchical clustering on samples
            Z = linkage(expr_data, 'ward')
            # Get clustering order
            cluster_indices = dendrogram(Z, no_plot=True)['leaves']
            
            buttons.append(dict(
                method='update',
                label='Hierarchical Clustering',
                args=[{'y': [None, None] if cell_types is not None else [None],
                      'z': [type_values[cluster_indices].reshape(-1, 1), expr_data[cluster_indices]] if cell_types is not None else [expr_data[cluster_indices]]}],
            ))
        except Exception as e:
            print(f"Hierarchical clustering failed: {e}")
        
        # Add dropdown menu
        fig.update_layout(
            updatemenus=[dict(
                active=0,
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.15,
                yanchor="top"
            )],
            height=800,
            width=1200,
            title='Interactive Gene Expression Heatmap'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Genes", row=1, col=2)
        fig.update_yaxes(title_text="Cells", row=1, col=1)
        
        if output_path:
            fig.write_html(output_path)
        
        return fig
    
    def create_interactive_cell_trajectory(self, spatial_coords, pseudotime, cell_types=None, gene_expression=None, 
                                        gene_names=None, output_path=None):
        """
        Create interactive cell trajectory visualization
        
        Args:
            spatial_coords: Spatial coordinates
            pseudotime: Pseudotime values
            cell_types: Cell types
            gene_expression: Gene expression data
            gene_names: Gene names
            output_path: Output file path
            
        Returns:
            fig: Plotly figure object
        """
        # Ensure data is on CPU
        coords = spatial_coords.cpu().detach().numpy() if torch.is_tensor(spatial_coords) else spatial_coords
        time_values = pseudotime.cpu().detach().numpy() if torch.is_tensor(pseudotime) else np.array(pseudotime)
        
        # Create dataframe
        df = pd.DataFrame({
            'x': coords[:, 0],
            'y': coords[:, 1],
            'Pseudotime': time_values
        })
        
        # If cell types are available, add to dataframe
        if cell_types is not None:
            type_data = cell_types.cpu().detach().numpy() if torch.is_tensor(cell_types) else np.array(cell_types)
            df['Cell Type'] = type_data
        
        # Create basic scatter plot
        fig = px.scatter(df, x='x', y='y', color='Pseudotime', 
                       color_continuous_scale='Viridis',
                       title='Cell Trajectory Visualization')
        
        # Add trajectory line
        # Sort points by pseudotime
        sorted_indices = np.argsort(time_values)
        sorted_coords = coords[sorted_indices]
        
        # Add trajectory line
        fig.add_trace(go.Scatter(
            x=sorted_coords[:, 0],
            y=sorted_coords[:, 1],
            mode='lines',
            line=dict(color='rgba(0,0,0,0.3)', width=1),
            showlegend=False
        ))
        
        # If gene expression data is available, add expression pattern animation functionality
        if gene_expression is not None and gene_names is not None:
            expr_data = gene_expression.cpu().detach().numpy() if torch.is_tensor(gene_expression) else gene_expression
            
            # Create frames
            frames = []
            buttons = []
            
            # First color by pseudotime (default view)
            buttons.append(dict(
                method='animate',
                label='Pseudotime',
                args=[None, {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}]
            ))
            
            # For each gene, create a frame
            for i, gene in enumerate(gene_names[:min(20, len(gene_names))]):  # Limit to first 20 genes
                gene_expr = expr_data[:, i]
                
                frame = go.Frame(
                    data=[go.Scatter(
                        x=df['x'],
                        y=df['y'],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=gene_expr,
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title=f"{gene} Expression")
                        ),
                        text=df['Pseudotime'],
                        name=gene
                    ),
                    go.Scatter(
                        x=sorted_coords[:, 0],
                        y=sorted_coords[:, 1],
                        mode='lines',
                        line=dict(color='rgba(0,0,0,0.3)', width=1),
                        showlegend=False
                    )],
                    name=gene
                )
                frames.append(frame)
                
                # Add button
                buttons.append(dict(
                    method='animate',
                    label=gene,
                    args=[[gene], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}]
                ))
            
            # Add cell type view button (if available)
            if cell_types is not None:
                buttons.append(dict(
                    method='update',
                    label='Cell Types',
                    args=[{'marker.color': df['Cell Type'], 'marker.colorscale': 'Rainbow'}],
                    args2=[{'title': 'Cell Types Along Trajectory'}]
                ))
            
            # Update layout
            fig.update_layout(
                updatemenus=[dict(
                    active=0,
                    buttons=buttons,
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.1,
                    xanchor="left",
                    y=1.15,
                    yanchor="top"
                )],
                sliders=[{
                    'active': 0,
                    'yanchor': 'top',
                    'xanchor': 'left',
                    'currentvalue': {
                        'visible': True,
                        'prefix': 'Gene: ',
                        'xanchor': 'right'
                    },
                    'steps': [{'args': [[f.name], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
                             'label': f.name,
                             'method': 'animate'} for f in frames]
                }],
                height=700,
                width=900
            )
            
            # Add frames
            fig.frames = frames
        
        if output_path:
            fig.write_html(output_path)
        
        return fig
    
    def create_interactive_domain_explorer(self, spatial_coords, domain_labels, gene_expression=None, 
                                        gene_names=None, cell_metadata=None, output_path=None):
        """
        Create interactive spatial domain explorer
        
        Args:
            spatial_coords: Spatial coordinates
            domain_labels: Domain labels
            gene_expression: Gene expression data
            gene_names: Gene names
            cell_metadata: Cell metadata
            output_path: Output file path
            
        Returns:
            fig: Plotly figure object
        """
        # Ensure data is on CPU
        coords = spatial_coords.cpu().detach().numpy() if torch.is_tensor(spatial_coords) else spatial_coords
        domains = domain_labels.cpu().detach().numpy() if torch.is_tensor(domain_labels) else np.array(domain_labels)
        
        # Create dataframe
        df = pd.DataFrame({
            'x': coords[:, 0],
            'y': coords[:, 1],
            'Domain': domains
        })
        
        # If cell metadata is available, add to dataframe
        if cell_metadata is not None:
            for col in cell_metadata.columns:
                df[col] = cell_metadata[col].values
        
        # Create figure
        fig = go.Figure()
        
        # Add scatter for each domain
        unique_domains = np.unique(domains)
        for domain in unique_domains:
            mask = domains == domain
            fig.add_trace(go.Scatter(
                x=coords[mask, 0],
                y=coords[mask, 1],
                mode='markers',
                marker=dict(size=8),
                name=f'Domain {domain}'
            ))
        
        # If gene expression data is available, add gene expression view options
        buttons = []
        
        # Domain view (default)
        buttons.append(dict(
            method='update',
            label='Domains',
            args=[{'visible': [True] * len(unique_domains)},
                 {'title': 'Spatial Domain Distribution'}]
        ))
        
        if gene_expression is not None and gene_names is not None:
            expr_data = gene_expression.cpu().detach().numpy() if torch.is_tensor(gene_expression) else gene_expression
            
            # Hide all domain scatter plots
            hide_domain_args = {'visible': [False] * len(unique_domains)}
            
            # Add expression view option for each gene
            for i, gene in enumerate(gene_names[:min(30, len(gene_names))]):  # Limit to first 30 genes
                # Add gene expression scatter plot
                fig.add_trace(go.Scatter(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=expr_data[:, i],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title=f"{gene} Expression")
                    ),
                    name=gene,
                    visible=False  # Initially invisible
                ))
                
                # Add button
                show_gene_args = {'visible': [False] * len(unique_domains) + [i == j for j in range(len(gene_names[:min(30, len(gene_names))]))]}
                buttons.append(dict(
                    method='update',
                    label=gene,
                    args=[show_gene_args,
                         {'title': f'{gene} Expression by Domain'}]
                ))
        
        # If cell metadata is available, add metadata view options
        if cell_metadata is not None:
            # Add view option for each metadata column
            for col in cell_metadata.columns:
                # If categorical data
                if cell_metadata[col].dtype.name in ['object', 'category']:
                    # Create a scatter for each unique value
                    unique_values = cell_metadata[col].unique()
                    
                    # Hide all existing scatter plots
                    hide_all_args = {'visible': [False] * len(fig.data)}
                    
                    # Add scatter plots for this metadata column
                    for val in unique_values:
                        mask = cell_metadata[col] == val
                        fig.add_trace(go.Scatter(
                            x=coords[mask, 0],
                            y=coords[mask, 1],
                            mode='markers',
                            marker=dict(size=8),
                            name=f'{col}: {val}',
                            visible=False  # Initially invisible
                        ))
                    
                    # Add button
                    show_meta_args = {'visible': [False] * len(unique_domains) + 
                                   [False] * (len(fig.data) - len(unique_domains) - len(unique_values)) +
                                   [True] * len(unique_values)}
                    
                    buttons.append(dict(
                        method='update',
                        label=col,
                        args=[show_meta_args,
                             {'title': f'{col} Distribution by Domain'}]
                    ))
        
        # Update layout
        fig.update_layout(
            title='Spatial Domain Explorer',
            xaxis_title='X Coordinate',
            yaxis_title='Y Coordinate',
            updatemenus=[dict(
                active=0,
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.15,
                yanchor="top"
            )],
            height=700,
            width=900
        )
        
        if output_path:
            fig.write_html(output_path)
        
        return fig