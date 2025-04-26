"""
Spatial Visualization Module: Visualizing spatial transcriptome data and spatial features
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from umap import UMAP
import os
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap


class SpatialVisualizer:
    """Spatial Transcriptome Data Visualization"""
    
    def __init__(self, output_dir='./visualizations/spatial'):
        """
        Initialize the visualizer
        
        Args:
            output_dir: Output directory path
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create custom color map
        self.custom_cmap = LinearSegmentedColormap.from_list("custom_viridis", 
                                                           plt.cm.viridis(np.linspace(0.1, 0.9, 256)))
    
    def visualize_spatial_features(self, spatial_data, predictions, spatial_coords, output_path=None):
        """
        Visualize spatial features and prediction results
        
        Args:
            spatial_data: Spatial features
            predictions: Prediction results
            spatial_coords: Spatial coordinates
            output_path: Output file path
            
        Returns:
            embedding: UMAP dimensionality-reduced feature embedding
        """
        # Ensure data is on CPU
        features = spatial_data.cpu().detach().numpy() if torch.is_tensor(spatial_data) else spatial_data
        coords = spatial_coords.cpu().detach().numpy() if torch.is_tensor(spatial_coords) else spatial_coords
        preds = predictions.cpu().detach().numpy() if torch.is_tensor(predictions) else predictions
        
        # Reduce to 2D for visualization
        reducer = UMAP(n_components=2, random_state=42)
        embedding = reducer.fit_transform(features)
        
        plt.figure(figsize=(15, 7))
        
        # Left: UMAP-reduced feature space
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=preds, cmap='viridis', s=30, alpha=0.8)
        plt.colorbar(scatter, label='Predicted Class')
        plt.title('UMAP Projection of Spatial Features')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        
        # Right: Predictions on original spatial coordinates
        plt.subplot(1, 2, 2)
        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=preds, cmap='viridis', s=30, alpha=0.8)
        plt.colorbar(scatter, label='Predicted Class')
        plt.title('Spatial Distribution of Predictions')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
        
        return embedding
    
    def visualize_gene_expression_spatial(self, gene_expression_data, spatial_coords, gene_names, 
                                         selected_genes=None, output_path=None):
        """
        Visualize spatial expression patterns of selected genes
        
        Args:
            gene_expression_data: Gene expression data
            spatial_coords: Spatial coordinates
            gene_names: Gene names
            selected_genes: List of selected genes
            output_path: Output file path
            
        Returns:
            top_genes_idx: Indices of visualized genes
        """
        # Ensure data is on CPU
        expression = gene_expression_data.cpu().detach().numpy() if torch.is_tensor(gene_expression_data) else gene_expression_data
        coords = spatial_coords.cpu().detach().numpy() if torch.is_tensor(spatial_coords) else spatial_coords
        
        # If no genes specified, select genes with highest expression variance
        if selected_genes is None:
            # Calculate variance of expression for each gene
            gene_var = np.var(expression, axis=0)
            # Select top 6 genes with highest variance
            top_genes_idx = np.argsort(gene_var)[-6:]
            selected_genes = [gene_names[i] for i in top_genes_idx]
        else:
            # Find indices of selected genes
            top_genes_idx = [gene_names.index(gene) for gene in selected_genes if gene in gene_names]
        
        # Determine subplot layout
        n_genes = len(top_genes_idx)
        n_cols = min(3, n_genes)
        n_rows = (n_genes + n_cols - 1) // n_cols
        
        plt.figure(figsize=(5*n_cols, 5*n_rows))
        
        for i, gene_idx in enumerate(top_genes_idx):
            plt.subplot(n_rows, n_cols, i+1)
            
            # Create scatter plot, color represents gene expression level
            scatter = plt.scatter(coords[:, 0], coords[:, 1], 
                                c=expression[:, gene_idx], 
                                cmap=self.custom_cmap, 
                                s=30, alpha=0.8)
            
            plt.colorbar(scatter, label='Expression Level')
            plt.title(f'Gene: {gene_names[gene_idx]}')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
        
        return top_genes_idx
    
    def visualize_cell_type_distribution(self, cell_types, spatial_coords, cell_type_names=None, output_path=None):
        """
        Visualize spatial distribution of cell types
        
        Args:
            cell_types: Cell type labels
            spatial_coords: Spatial coordinates
            cell_type_names: Cell type names
            output_path: Output file path
            
        Returns:
            unique_types: Unique cell types
        """
        # Ensure data is on CPU
        coords = spatial_coords.cpu().detach().numpy() if torch.is_tensor(spatial_coords) else spatial_coords
        types = cell_types.cpu().detach().numpy() if torch.is_tensor(cell_types) else np.array(cell_types)
        
        # Get unique cell types
        unique_types = np.unique(types)
        
        plt.figure(figsize=(12, 10))
        
        # Assign different colors to each cell type
        cmap = plt.cm.get_cmap('tab10', len(unique_types))
        
        for i, cell_type in enumerate(unique_types):
            mask = types == cell_type
            
            # Get cell type name
            if cell_type_names is not None and i < len(cell_type_names):
                label = cell_type_names[i]
            else:
                label = f"Type {cell_type}"
                
            plt.scatter(coords[mask, 0], coords[mask, 1], c=[cmap(i)], 
                       label=label, s=30, alpha=0.7)
        
        plt.legend(title='Cell Type')
        plt.title('Spatial Distribution of Cell Types')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
        
        return unique_types
    
    def visualize_spatial_clusters(self, spatial_coords, cluster_labels, background_img=None, 
                                  output_path=None):
        """
        Visualize spatial clustering results
        
        Args:
            spatial_coords: Spatial coordinates
            cluster_labels: Cluster labels
            background_img: Background image
            output_path: Output file path
        """
        # Ensure data is on CPU
        coords = spatial_coords.cpu().detach().numpy() if torch.is_tensor(spatial_coords) else spatial_coords
        clusters = cluster_labels.cpu().detach().numpy() if torch.is_tensor(cluster_labels) else np.array(cluster_labels)
        
        # Get unique cluster labels
        unique_clusters = np.unique(clusters)
        
        plt.figure(figsize=(12, 10))
        
        # If background image is provided, display it first
        if background_img is not None:
            plt.imshow(background_img, alpha=0.5)
        
        # Assign different colors to each cluster
        cmap = plt.cm.get_cmap('tab20', len(unique_clusters))
        
        for i, cluster in enumerate(unique_clusters):
            mask = clusters == cluster
            plt.scatter(coords[mask, 0], coords[mask, 1], c=[cmap(i)], 
                       label=f"Cluster {cluster}", s=30, alpha=0.7)
        
        plt.legend(title='Cluster')
        plt.title('Spatial Distribution of Clusters')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
    
    def visualize_gene_correlation_network(self, gene_expression_data, gene_names, threshold=0.7, 
                                          max_genes=30, output_path=None):
        """
        Visualize gene correlation network
        
        Args:
            gene_expression_data: Gene expression data
            gene_names: Gene names
            threshold: Correlation threshold
            max_genes: Maximum number of genes
            output_path: Output file path
        """
        try:
            import networkx as nx
            
            # Ensure data is on CPU
            expression = gene_expression_data.cpu().detach().numpy() if torch.is_tensor(gene_expression_data) else gene_expression_data
            
            # Calculate gene expression variance
            gene_var = np.var(expression, axis=0)
            
            # Select top max_genes genes with highest variance
            top_genes_idx = np.argsort(gene_var)[-max_genes:]
            top_genes_expr = expression[:, top_genes_idx]
            top_genes_names = [gene_names[i] for i in top_genes_idx]
            
            # Calculate correlation matrix
            corr_matrix = np.corrcoef(top_genes_expr, rowvar=False)
            
            # Create network graph
            G = nx.Graph()
            
            # Add nodes
            for i, gene in enumerate(top_genes_names):
                G.add_node(gene)
            
            # Add edges (gene pairs with correlation above threshold)
            for i in range(len(top_genes_names)):
                for j in range(i+1, len(top_genes_names)):
                    if abs(corr_matrix[i, j]) > threshold:
                        G.add_edge(top_genes_names[i], top_genes_names[j], 
                                  weight=abs(corr_matrix[i, j]))
            
            plt.figure(figsize=(12, 12))
            
            # Use Spring layout
            pos = nx.spring_layout(G)
            
            # Get edge weights
            edge_weights = [G[u][v]['weight'] * 2 for u, v in G.edges()]
            
            # Draw network
            nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
            nx.draw_networkx_labels(G, pos, font_size=10)
            nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.7, 
                                  edge_color=edge_weights, edge_cmap=plt.cm.Blues)
            
            plt.title("Gene Co-expression Network")
            plt.axis('off')
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path)
                plt.close()
            else:
                plt.show()
                
        except ImportError:
            print("NetworkX is required for visualizing gene correlation networks. Please install it using 'pip install networkx'.")
    
    def visualize_spatial_hotspots(self, gene_expression_data, spatial_coords, gene_names, 
                                  genes_of_interest, output_path=None):
        """
        Visualize spatial gene expression hotspots
        
        Args:
            gene_expression_data: Gene expression data
            spatial_coords: Spatial coordinates
            gene_names: Gene names
            genes_of_interest: List of genes of interest
            output_path: Output file path
        """
        # Ensure data is on CPU
        expression = gene_expression_data.cpu().detach().numpy() if torch.is_tensor(gene_expression_data) else gene_expression_data
        coords = spatial_coords.cpu().detach().numpy() if torch.is_tensor(spatial_coords) else spatial_coords
        
        # Find indices of genes of interest
        gene_indices = [gene_names.index(gene) for gene in genes_of_interest if gene in gene_names]
        
        if not gene_indices:
            print(f"None of the genes in {genes_of_interest} were found in the dataset.")
            return
        
        # Create figure
        plt.figure(figsize=(15, 15))
        
        # Create a subplot for each gene
        n_genes = len(gene_indices)
        n_cols = min(2, n_genes)
        n_rows = (n_genes + n_cols - 1) // n_cols
        
        # Define limits for the same coordinate range
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        
        for i, gene_idx in enumerate(gene_indices):
            plt.subplot(n_rows, n_cols, i+1)
            
            # Get gene expression data
            gene_expr = expression[:, gene_idx]
            
            # Use KDE to create density estimate
            x, y = coords[:, 0], coords[:, 1]
            xy = np.vstack([x, y])
            
            # Gene expression as weights for KDE
            weights = gene_expr / gene_expr.sum()
            
            try:
                # Create weighted KDE
                k = gaussian_kde(xy, weights=weights)
                
                # Create grid
                xi, yi = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
                zi = k(np.vstack([xi.flatten(), yi.flatten()]))
                
                # Draw density map
                plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap='Reds')
                plt.contour(xi, yi, zi.reshape(xi.shape), linewidths=0.5, colors='k', alpha=0.3)
                
                # Mark points in high expression areas
                high_expr_threshold = np.percentile(gene_expr, 90)  # Take top 10% expression points
                high_expr_mask = gene_expr > high_expr_threshold
                plt.scatter(coords[high_expr_mask, 0], coords[high_expr_mask, 1], 
                           c='darkred', s=30, alpha=0.7, edgecolors='white')
                
            except Exception as e:
                print(f"KDE failed for gene {gene_names[gene_idx]}: {e}")
                # Alternative: simple scatter plot
                scatter = plt.scatter(coords[:, 0], coords[:, 1], c=gene_expr, 
                                     cmap='Reds', s=30, alpha=0.8)
                plt.colorbar(scatter, label='Expression Level')
            
            plt.title(f'Spatial Hotspots for {gene_names[gene_idx]}')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            
            # Keep all subplots with the same coordinate range
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
    
    def visualize_spatial_domains(self, spatial_coords, domain_labels, marker_genes=None, 
                                gene_expression=None, gene_names=None, output_path=None):
        """
        Visualize spatial domains and marker gene expression
        
        Args:
            spatial_coords: Spatial coordinates
            domain_labels: Domain labels
            marker_genes: List of marker genes
            gene_expression: Gene expression data
            gene_names: Gene names
            output_path: Output file path
        """
        # Ensure data is on CPU
        coords = spatial_coords.cpu().detach().numpy() if torch.is_tensor(spatial_coords) else spatial_coords
        domains = domain_labels.cpu().detach().numpy() if torch.is_tensor(domain_labels) else np.array(domain_labels)
        
        # Create figure object
        if marker_genes and gene_expression is not None and gene_names is not None:
            # Domain plot with marker gene visualization
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            
            # Left: Domain distribution
            ax1 = axes[0]
            unique_domains = np.unique(domains)
            cmap = plt.cm.get_cmap('tab20', len(unique_domains))
            
            for i, domain in enumerate(unique_domains):
                mask = domains == domain
                ax1.scatter(coords[mask, 0], coords[mask, 1], c=[cmap(i)], 
                           label=f"Domain {domain}", s=30, alpha=0.7)
            
            ax1.legend(title='Domains')
            ax1.set_title('Spatial Distribution of Domains')
            ax1.set_xlabel('X Coordinate')
            ax1.set_ylabel('Y Coordinate')
            
            # Right: Marker gene expression heatmap
            ax2 = axes[1]
            
            # Ensure gene_expression is NumPy array
            expr = gene_expression.cpu().detach().numpy() if torch.is_tensor(gene_expression) else gene_expression
            
            # Find indices of marker genes
            marker_indices = []
            for gene in marker_genes:
                if gene in gene_names:
                    marker_indices.append(gene_names.index(gene))
            
            if marker_indices:
                # Get average expression value for each domain
                domain_avg_expr = np.zeros((len(unique_domains), len(marker_indices)))
                
                for i, domain in enumerate(unique_domains):
                    domain_mask = domains == domain
                    domain_avg_expr[i] = np.mean(expr[domain_mask, :][:, marker_indices], axis=0)
                
                # Create heatmap
                sns.heatmap(domain_avg_expr, annot=True, fmt=".2f", cmap="YlGnBu",
                           xticklabels=[gene_names[i] for i in marker_indices],
                           yticklabels=[f"Domain {d}" for d in unique_domains],
                           ax=ax2)
                
                ax2.set_title('Average Expression of Marker Genes by Domain')
                ax2.set_xlabel('Marker Genes')
                ax2.set_ylabel('Domains')
            else:
                ax2.text(0.5, 0.5, "No marker genes found in the dataset", 
                        ha='center', va='center', fontsize=12)
                ax2.set_title('Marker Gene Expression')
        else:
            # Only show domain distribution
            plt.figure(figsize=(12, 10))
            unique_domains = np.unique(domains)
            cmap = plt.cm.get_cmap('tab20', len(unique_domains))
            
            for i, domain in enumerate(unique_domains):
                mask = domains == domain
                plt.scatter(coords[mask, 0], coords[mask, 1], c=[cmap(i)], 
                           label=f"Domain {domain}", s=30, alpha=0.7)
            
            plt.legend(title='Domains')
            plt.title('Spatial Distribution of Domains')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
    
    def visualize_spatial_trajectories(self, spatial_coords, trajectory_values, arrows=True, 
                                     background_img=None, output_path=None):
        """
        Visualize spatial trajectories (e.g., pseudotime or other continuous variables)
        
        Args:
            spatial_coords: Spatial coordinates
            trajectory_values: Trajectory values (e.g., pseudotime)
            arrows: Whether to display direction arrows
            background_img: Background image
            output_path: Output file path
        """
        # Ensure data is on CPU
        coords = spatial_coords.cpu().detach().numpy() if torch.is_tensor(spatial_coords) else spatial_coords
        traj = trajectory_values.cpu().detach().numpy() if torch.is_tensor(trajectory_values) else np.array(trajectory_values)
        
        plt.figure(figsize=(12, 10))
        
        # If background image is provided, display it first
        if background_img is not None:
            plt.imshow(background_img, alpha=0.5)
        
        # Create color mapping
        norm = plt.Normalize(traj.min(), traj.max())
        colors = plt.cm.viridis(norm(traj))
        
        # Draw scatter plot
        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=traj, cmap='viridis', 
                            s=50, alpha=0.8)
        
        # Add color bar
        plt.colorbar(scatter, label='Trajectory Value')
        
        # If direction arrows need to be displayed
        if arrows and len(coords) > 1:
            try:
                # Sort points according to trajectory values
                sorted_indices = np.argsort(traj)
                sorted_coords = coords[sorted_indices]
                
                # Display an arrow every N points
                N = max(1, len(coords) // 20)
                for i in range(0, len(sorted_coords) - N, N):
                    # Calculate direction
                    start = sorted_coords[i]
                    end = sorted_coords[i + N]
                    dx, dy = end - start
                    
                    # Skip if distance is too small
                    if dx**2 + dy**2 < 1e-6:
                        continue
                    
                    # Draw arrow
                    plt.arrow(start[0], start[1], dx, dy, head_width=5, head_length=7,
                             fc='white', ec='black', alpha=0.6)
            except Exception as e:
                print(f"Failed to draw arrows: {e}")
        
        plt.title('Spatial Trajectory Visualization')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
    
    def visualize_spatial_clustering_comparison(self, spatial_coords, cluster_labels1, cluster_labels2,
                                             method1_name="Method 1", method2_name="Method 2", 
                                             output_path=None):
        """
        Compare spatial distribution of two clustering methods
        
        Args:
            spatial_coords: Spatial coordinates
            cluster_labels1: Labels from first clustering method
            cluster_labels2: Labels from second clustering method
            method1_name: Name of first method
            method2_name: Name of second method
            output_path: Output file path
        """
        # Ensure data is on CPU
        coords = spatial_coords.cpu().detach().numpy() if torch.is_tensor(spatial_coords) else spatial_coords
        clusters1 = cluster_labels1.cpu().detach().numpy() if torch.is_tensor(cluster_labels1) else np.array(cluster_labels1)
        clusters2 = cluster_labels2.cpu().detach().numpy() if torch.is_tensor(cluster_labels2) else np.array(cluster_labels2)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Left: First clustering method
        ax1 = axes[0]
        unique_clusters1 = np.unique(clusters1)
        cmap1 = plt.cm.get_cmap('tab10', len(unique_clusters1))
        
        for i, cluster in enumerate(unique_clusters1):
            mask = clusters1 == cluster
            ax1.scatter(coords[mask, 0], coords[mask, 1], c=[cmap1(i)], 
                       label=f"Cluster {cluster}", s=30, alpha=0.7)
        
        ax1.legend(title='Clusters')
        ax1.set_title(f'Spatial Clustering: {method1_name}')
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        
        # Right: Second clustering method
        ax2 = axes[1]
        unique_clusters2 = np.unique(clusters2)
        cmap2 = plt.cm.get_cmap('tab10', len(unique_clusters2))
        
        for i, cluster in enumerate(unique_clusters2):
            mask = clusters2 == cluster
            ax2.scatter(coords[mask, 0], coords[mask, 1], c=[cmap2(i)], 
                       label=f"Cluster {cluster}", s=30, alpha=0.7)
        
        ax2.legend(title='Clusters')
        ax2.set_title(f'Spatial Clustering: {method2_name}')
        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
        
        # Calculate clustering consistency
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        
        ari = adjusted_rand_score(clusters1, clusters2)
        nmi = normalized_mutual_info_score(clusters1, clusters2)
        
        print(f"Clustering Comparison Metrics:")
        print(f"Adjusted Rand Index: {ari:.4f}")
        print(f"Normalized Mutual Information: {nmi:.4f}")
        
        return ari, nmi