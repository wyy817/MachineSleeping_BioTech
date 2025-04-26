"""
Tumor Microenvironment Visualization Module: Specialized for visualization analysis of tumor/cancer slides
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from umap import UMAP
import os
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap


class TMEVisualizer:
    """Tumor Microenvironment Visualization"""
    
    def __init__(self, output_dir='./visualizations/tme'):
        """
        Initialize the visualizer
        
        Args:
            output_dir: Output directory path
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Define color maps
        self.tumor_cmap = plt.cm.Reds
        self.immune_cmap = plt.cm.Blues
        self.stroma_cmap = plt.cm.Greens
        
    def visualize_tumor_microenvironment(self, features, cell_types, gene_expression, spatial_coords, 
                                       gene_names, tumor_marker_genes, immune_marker_genes, 
                                       output_path=None):
        """
        Visualize the tumor microenvironment, including tumor regions, immune cell infiltration, and key gene expression
        
        Args:
            features: Feature vectors
            cell_types: Cell type labels
            gene_expression: Gene expression data
            spatial_coords: Spatial coordinates
            gene_names: Gene names
            tumor_marker_genes: Tumor marker genes
            immune_marker_genes: Immune marker genes
            output_path: Output file path
            
        Returns:
            results_dict: Dictionary containing visualization results
        """
        # Ensure data is on CPU
        coords = spatial_coords.cpu().detach().numpy() if torch.is_tensor(spatial_coords) else spatial_coords
        types = cell_types.cpu().detach().numpy() if torch.is_tensor(cell_types) else np.array(cell_types)
        expression = gene_expression.cpu().detach().numpy() if torch.is_tensor(gene_expression) else gene_expression
        features_np = features.cpu().detach().numpy() if torch.is_tensor(features) else features
        
        # Create figure
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Cell type distribution
        ax1 = fig.add_subplot(2, 2, 1)
        unique_types = np.unique(types)
        cmap = plt.cm.get_cmap('tab10', len(unique_types))
        for i, cell_type in enumerate(unique_types):
            mask = types == cell_type
            ax1.scatter(coords[mask, 0], coords[mask, 1], c=[cmap(i)], label=str(cell_type), s=30, alpha=0.7)
        ax1.legend(title='Cell Type')
        ax1.set_title('Cell Type Distribution')
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        
        # 2. Tumor marker gene expression
        ax2 = fig.add_subplot(2, 2, 2)
        # Calculate average expression of tumor marker genes
        tumor_indices = [gene_names.index(gene) for gene in tumor_marker_genes if gene in gene_names]
        if tumor_indices:
            tumor_expression = np.mean(expression[:, tumor_indices], axis=1)
            scatter = ax2.scatter(coords[:, 0], coords[:, 1], c=tumor_expression, cmap=self.tumor_cmap, s=30, alpha=0.8)
            plt.colorbar(scatter, ax=ax2, label='Expression Level')
            ax2.set_title('Tumor Marker Genes Expression')
            ax2.set_xlabel('X Coordinate')
            ax2.set_ylabel('Y Coordinate')
        else:
            ax2.text(0.5, 0.5, 'No tumor marker genes found', ha='center', va='center')
            ax2.set_title('Tumor Marker Genes Expression')
        
        # 3. Immune marker gene expression
        ax3 = fig.add_subplot(2, 2, 3)
        # Calculate average expression of immune marker genes
        immune_indices = [gene_names.index(gene) for gene in immune_marker_genes if gene in gene_names]
        if immune_indices:
            immune_expression = np.mean(expression[:, immune_indices], axis=1)
            scatter = ax3.scatter(coords[:, 0], coords[:, 1], c=immune_expression, cmap=self.immune_cmap, s=30, alpha=0.8)
            plt.colorbar(scatter, ax=ax3, label='Expression Level')
            ax3.set_title('Immune Marker Genes Expression')
            ax3.set_xlabel('X Coordinate')
            ax3.set_ylabel('Y Coordinate')
        else:
            ax3.text(0.5, 0.5, 'No immune marker genes found', ha='center', va='center')
            ax3.set_title('Immune Marker Genes Expression')
        
        # 4. Tumor microenvironment interaction heatmap
        ax4 = fig.add_subplot(2, 2, 4)
        # Use UMAP for dimension reduction of features
        reducer = UMAP(n_components=2, random_state=42)
        embedding = reducer.fit_transform(features_np)
        
        # Use KDE to create density plot
        x, y = embedding[:, 0], embedding[:, 1]
        k = gaussian_kde([x, y])
        xi, yi = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        
        # Draw density plot
        ax4.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap='viridis')
        ax4.contour(xi, yi, zi.reshape(xi.shape), linewidths=0.5, colors='k', alpha=0.3)
        ax4.scatter(x, y, c=types, cmap='tab10', s=10, alpha=0.5)
        ax4.set_title('Tumor Microenvironment Interaction')
        ax4.set_xlabel('UMAP 1')
        ax4.set_ylabel('UMAP 2')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
        
        return {
            'tumor_expression': tumor_expression if tumor_indices else None,
            'immune_expression': immune_expression if immune_indices else None,
            'umap_embedding': embedding
        }
    
    def visualize_cell_interactions(self, spatial_coords, cell_types, interaction_threshold=50, 
                                  cell_type_names=None, output_path=None):
        """
        Visualize cell-cell interactions
        
        Args:
            spatial_coords: Spatial coordinates
            cell_types: Cell types
            interaction_threshold: Interaction threshold (pixel distance)
            cell_type_names: Cell type names
            output_path: Output file path
        """
        # Ensure data is on CPU
        coords = spatial_coords.cpu().detach().numpy() if torch.is_tensor(spatial_coords) else spatial_coords
        types = cell_types.cpu().detach().numpy() if torch.is_tensor(cell_types) else np.array(cell_types)
        
        # Get unique cell types
        unique_types = np.unique(types)
        n_types = len(unique_types)
        
        # Establish cell type mapping
        if cell_type_names is None:
            cell_type_names = [f"Type {t}" for t in unique_types]
        
        # Create interaction matrix
        interaction_matrix = np.zeros((n_types, n_types))
        
        # Calculate cell distances and interactions
        for i in range(len(coords)):
            type_i = types[i]
            idx_i = np.where(unique_types == type_i)[0][0]
            
            for j in range(i+1, len(coords)):
                type_j = types[j]
                idx_j = np.where(unique_types == type_j)[0][0]
                
                # Calculate Euclidean distance
                dist = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
                
                # If distance is less than threshold, consider it an interaction
                if dist < interaction_threshold:
                    interaction_matrix[idx_i, idx_j] += 1
                    interaction_matrix[idx_j, idx_i] += 1
        
        # Create figure
        plt.figure(figsize=(15, 12))
        
        # 1. Cell interaction heatmap
        plt.subplot(2, 2, 1)
        sns.heatmap(interaction_matrix, annot=True, fmt='d', cmap='YlGnBu',
                  xticklabels=cell_type_names, yticklabels=cell_type_names)
        plt.title('Cell Type Interaction Matrix')
        
        # 2. Interaction network graph
        try:
            import networkx as nx
            
            plt.subplot(2, 2, 2)
            G = nx.Graph()
            
            # Add nodes
            for i, name in enumerate(cell_type_names):
                G.add_node(name, count=np.sum(types == unique_types[i]))
            
            # Add edges
            for i in range(n_types):
                for j in range(i+1, n_types):
                    if interaction_matrix[i, j] > 0:
                        G.add_edge(cell_type_names[i], cell_type_names[j], 
                                 weight=interaction_matrix[i, j])
            
            # Node sizes based on cell count
            node_sizes = [G.nodes[n]['count'] * 100 for n in G.nodes()]
            
            # Edge width based on interaction intensity
            edge_weights = [G[u][v]['weight'] / 5 for u, v in G.edges()]
            
            # Node positions
            pos = nx.spring_layout(G)
            
            # Draw network
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=range(n_types), 
                                 cmap=plt.cm.tab10, alpha=0.8)
            nx.draw_networkx_labels(G, pos)
            nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5, edge_color='gray')
            
            plt.title('Cell Interaction Network')
            plt.axis('off')
            
        except ImportError:
            plt.subplot(2, 2, 2)
            plt.text(0.5, 0.5, "NetworkX required for network visualization", 
                   ha='center', va='center')
            plt.axis('off')
        
        # 3. Spatial distribution and interaction visualization
        plt.subplot(2, 1, 2)
        
        # Draw cell points
        for i, cell_type in enumerate(unique_types):
            mask = types == cell_type
            plt.scatter(coords[mask, 0], coords[mask, 1], c=[plt.cm.tab10(i)], 
                       label=cell_type_names[i], s=50, alpha=0.7)
        
        # Draw interaction lines
        for i in range(len(coords)):
            type_i = types[i]
            for j in range(i+1, len(coords)):
                type_j = types[j]
                
                # If types are different and distance is less than threshold
                dist = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
                if dist < interaction_threshold and type_i != type_j:
                    plt.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]], 
                           'gray', alpha=0.3, linewidth=0.5)
        
        plt.legend(title='Cell Types')
        plt.title('Spatial Distribution with Cell Interactions')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
        
        return interaction_matrix
    
    def visualize_immune_infiltration(self, spatial_coords, cell_types, gene_expression, gene_names,
                                    tumor_cell_type=None, immune_cell_types=None,
                                    immune_marker_genes=None, output_path=None):
        """
        Visualize immune infiltration
        
        Args:
            spatial_coords: Spatial coordinates
            cell_types: Cell types
            gene_expression: Gene expression data
            gene_names: Gene names
            tumor_cell_type: Tumor cell type
            immune_cell_types: List of immune cell types
            immune_marker_genes: Immune marker genes
            output_path: Output file path
            
        Returns:
            infiltration_score: Infiltration score
        """
        # Ensure data is on CPU
        coords = spatial_coords.cpu().detach().numpy() if torch.is_tensor(spatial_coords) else spatial_coords
        types = cell_types.cpu().detach().numpy() if torch.is_tensor(cell_types) else np.array(cell_types)
        expression = gene_expression.cpu().detach().numpy() if torch.is_tensor(gene_expression) else gene_expression
        
        # Create figure
        plt.figure(figsize=(18, 10))
        
        # 1. Spatial distribution of tumor and immune cells
        plt.subplot(1, 2, 1)
        
        # If tumor and immune cell types are not specified, infer based on marker genes
        if tumor_cell_type is None or immune_cell_types is None:
            # Calculate tumor marker gene expression as tumor score
            tumor_marker_genes = ['MKI67', 'EGFR', 'MYC', 'CCNE1']  # Default tumor marker genes
            tumor_indices = [gene_names.index(gene) for gene in tumor_marker_genes if gene in gene_names]
            
            if tumor_indices:
                tumor_score = np.mean(expression[:, tumor_indices], axis=1)
            else:
                print("No tumor marker genes found, using random tumor score")
                tumor_score = np.random.rand(len(coords))
            
            # Calculate immune marker gene expression as immune score
            if immune_marker_genes is None:
                immune_marker_genes = ['CD3D', 'CD8A', 'CD4', 'FOXP3', 'CD68']  # Default immune marker genes
            
            immune_indices = [gene_names.index(gene) for gene in immune_marker_genes if gene in gene_names]
            
            if immune_indices:
                immune_score = np.mean(expression[:, immune_indices], axis=1)
            else:
                print("No immune marker genes found, using random immune score")
                immune_score = np.random.rand(len(coords))
            
            # Classify cell types based on scores
            tumor_threshold = np.percentile(tumor_score, 75)
            immune_threshold = np.percentile(immune_score, 75)
            
            tumor_mask = tumor_score > tumor_threshold
            immune_mask = immune_score > immune_threshold
            other_mask = ~(tumor_mask | immune_mask)
            
            # Draw cell distribution
            plt.scatter(coords[tumor_mask, 0], coords[tumor_mask, 1], 
                       c='red', label='Tumor cells', s=30, alpha=0.7)
            plt.scatter(coords[immune_mask, 0], coords[immune_mask, 1], 
                       c='blue', label='Immune cells', s=30, alpha=0.7)
            plt.scatter(coords[other_mask, 0], coords[other_mask, 1], 
                       c='gray', label='Other cells', s=30, alpha=0.7)
        else:
            # Use given cell types
            tumor_mask = types == tumor_cell_type
            
            if isinstance(immune_cell_types, list):
                immune_mask = np.zeros_like(types, dtype=bool)
                for immune_type in immune_cell_types:
                    immune_mask |= (types == immune_type)
            else:
                immune_mask = types == immune_cell_types
                
            other_mask = ~(tumor_mask | immune_mask)
            
            # Draw cell distribution
            plt.scatter(coords[tumor_mask, 0], coords[tumor_mask, 1], 
                       c='red', label=f'Tumor cells (Type {tumor_cell_type})', s=30, alpha=0.7)
            plt.scatter(coords[immune_mask, 0], coords[immune_mask, 1], 
                       c='blue', label='Immune cells', s=30, alpha=0.7)
            plt.scatter(coords[other_mask, 0], coords[other_mask, 1], 
                       c='gray', label='Other cells', s=30, alpha=0.7)
        
        plt.legend()
        plt.title('Tumor and Immune Cell Distribution')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        
        # 2. Immune infiltration heatmap
        plt.subplot(1, 2, 2)
        
        # Calculate immune infiltration score
        # Method: For each tumor cell, count the number of immune cells within a certain range
        infiltration_radius = 50  # Infiltration radius
        infiltration_score = np.zeros_like(coords[:, 0])
        
        for i in range(len(coords)):
            if tumor_mask[i]:
                # Calculate distances
                distances = np.sqrt(np.sum((coords - coords[i]) ** 2, axis=1))
                
                # Count immune cells within range
                infiltration_score[i] = np.sum((distances < infiltration_radius) & immune_mask)
        
        # Create interpolation grid
        from scipy.interpolate import griddata
        
        # Only use tumor cell coordinates and scores for interpolation
        tumor_coords = coords[tumor_mask]
        tumor_scores = infiltration_score[tumor_mask]
        
        if len(tumor_coords) > 0:
            x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
            y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
            
            xi = np.linspace(x_min, x_max, 100)
            yi = np.linspace(y_min, y_max, 100)
            xi, yi = np.meshgrid(xi, yi)
            
            # Grid interpolation
            zi = griddata((tumor_coords[:, 0], tumor_coords[:, 1]), tumor_scores, (xi, yi), method='cubic', fill_value=0)
            
            # Draw heatmap
            plt.pcolormesh(xi, yi, zi, cmap='YlOrRd', shading='auto')
            plt.colorbar(label='Immune Infiltration Score')
            
            # Draw tumor and immune cells
            plt.scatter(coords[tumor_mask, 0], coords[tumor_mask, 1], 
                       c='black', marker='o', s=10, alpha=0.5, label='Tumor cells')
            plt.scatter(coords[immune_mask, 0], coords[immune_mask, 1], 
                       c='blue', marker='x', s=10, alpha=0.5, label='Immune cells')
            
            plt.legend()
            plt.title('Immune Infiltration Heat Map')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
        else:
            plt.text(0.5, 0.5, 'No tumor cells detected for infiltration analysis', 
                   ha='center', va='center')
            plt.title('Immune Infiltration Heat Map')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
        
        return infiltration_score
    
    def visualize_tumor_heterogeneity(self, spatial_coords, gene_expression, gene_names, 
                                    heterogeneity_genes=None, n_clusters=4, output_path=None):
        """
        Visualize tumor heterogeneity
        
        Args:
            spatial_coords: Spatial coordinates
            gene_expression: Gene expression data
            gene_names: Gene names
            heterogeneity_genes: Genes used for heterogeneity analysis
            n_clusters: Number of clusters
            output_path: Output file path
            
        Returns:
            cluster_labels: Cluster labels
        """
        # Ensure data is on CPU
        coords = spatial_coords.cpu().detach().numpy() if torch.is_tensor(spatial_coords) else spatial_coords
        expression = gene_expression.cpu().detach().numpy() if torch.is_tensor(gene_expression) else gene_expression
        
        # If heterogeneity genes are not specified, use the top 50 genes with highest expression variance
        if heterogeneity_genes is None:
            # Calculate gene expression variance
            gene_var = np.var(expression, axis=0)
            top_var_indices = np.argsort(gene_var)[-50:]
            heterogeneity_genes = [gene_names[i] for i in top_var_indices]
        
        # Extract heterogeneity gene expression matrix
        hetero_indices = [gene_names.index(gene) for gene in heterogeneity_genes if gene in gene_names]
        if len(hetero_indices) == 0:
            print("No heterogeneity genes found in the dataset")
            return None
            
        hetero_expression = expression[:, hetero_indices]
        
        # Apply clustering
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(hetero_expression)
        
        # Create figure
        plt.figure(figsize=(18, 15))
        
        # 1. Spatial distribution based on clustering
        plt.subplot(2, 2, 1)
        for i in range(n_clusters):
            mask = cluster_labels == i
            plt.scatter(coords[mask, 0], coords[mask, 1], 
                      label=f'Cluster {i+1}', s=30, alpha=0.7)
            
        plt.legend()
        plt.title('Spatial Distribution of Heterogeneity Clusters')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        
        # 2. Cluster feature heatmap
        plt.subplot(2, 2, 2)
        
        # Calculate average expression value for each cluster
        cluster_profiles = np.zeros((n_clusters, len(hetero_indices)))
        for i in range(n_clusters):
            mask = cluster_labels == i
            if np.sum(mask) > 0:
                cluster_profiles[i] = np.mean(hetero_expression[mask], axis=0)
        
        # Select the top 20 genes with largest expression differences
        gene_var_across_clusters = np.var(cluster_profiles, axis=0)
        top_var_genes_idx = np.argsort(gene_var_across_clusters)[-20:]
        
        # Draw heatmap
        selected_gene_names = [gene_names[hetero_indices[i]] for i in top_var_genes_idx]
        sns.heatmap(cluster_profiles[:, top_var_genes_idx], annot=False, cmap='viridis',
                  xticklabels=selected_gene_names, yticklabels=[f'Cluster {i+1}' for i in range(n_clusters)])
        plt.title('Cluster-specific Gene Expression Profiles')
        plt.xlabel('Genes')
        plt.ylabel('Clusters')
        plt.xticks(rotation=90)
        
        # 3. UMAP dimensional reduction visualization
        plt.subplot(2, 2, 3)
        try:
            reducer = UMAP(n_components=2, random_state=42)
            embedding = reducer.fit_transform(hetero_expression)
            
            for i in range(n_clusters):
                mask = cluster_labels == i
                plt.scatter(embedding[mask, 0], embedding[mask, 1], 
                          label=f'Cluster {i+1}', s=30, alpha=0.7)
                
            plt.legend()
            plt.title('UMAP Visualization of Heterogeneity Clusters')
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
        except Exception as e:
            plt.text(0.5, 0.5, f"UMAP failed: {e}", ha='center', va='center')
            plt.title('UMAP Visualization (Failed)')
        
        # 4. Expression difference violin plot
        plt.subplot(2, 2, 4)
        
        # Calculate overall heterogeneity score
        from scipy.stats import entropy
        
        heterogeneity_scores = np.zeros(len(coords))
        for i in range(len(coords)):
            # Calculate expression similarity between current cell and all other cells
            similarities = np.corrcoef(hetero_expression[i:i+1], hetero_expression)[0, 1:]
            
            # Calculate entropy of similarity distribution as heterogeneity measure
            hist, _ = np.histogram(similarities, bins=20, range=(-1, 1), density=True)
            heterogeneity_scores[i] = entropy(hist + 1e-10)  # Add small value to avoid log(0)
        
        # Draw violin plot of heterogeneity scores for each cluster
        cluster_data = []
        for i in range(n_clusters):
            mask = cluster_labels == i
            cluster_data.append(heterogeneity_scores[mask])
        
        plt.violinplot(cluster_data, showmeans=True, showmedians=True)
        plt.xticks(range(1, n_clusters + 1), [f'Cluster {i+1}' for i in range(n_clusters)])
        plt.ylabel('Heterogeneity Score (Higher = More Heterogeneous)')
        plt.title('Heterogeneity Distribution by Cluster')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
        
        return cluster_labels
    
    def visualize_ligand_receptor_interactions(self, spatial_coords, cell_types, gene_expression, gene_names,
                                            ligand_receptor_pairs, cell_type_names=None, output_path=None):
        """
        Visualize ligand-receptor interactions
        
        Args:
            spatial_coords: Spatial coordinates
            cell_types: Cell types
            gene_expression: Gene expression data
            gene_names: Gene names
            ligand_receptor_pairs: List of ligand-receptor pairs, format: [(ligand1, receptor1), ...]
            cell_type_names: Cell type names
            output_path: Output file path
            
        Returns:
            interaction_scores: Interaction score matrix
        """
        # Ensure data is on CPU
        coords = spatial_coords.cpu().detach().numpy() if torch.is_tensor(spatial_coords) else spatial_coords
        types = cell_types.cpu().detach().numpy() if torch.is_tensor(cell_types) else np.array(cell_types)
        expression = gene_expression.cpu().detach().numpy() if torch.is_tensor(gene_expression) else gene_expression
        
        # Get unique cell types
        unique_types = np.unique(types)
        n_types = len(unique_types)
        
        # Set cell type names
        if cell_type_names is None:
            cell_type_names = [f"Type {t}" for t in unique_types]
        
        # Create ligand-receptor interaction score matrix
        # Rows: Sender cell types (ligand)
        # Columns: Receiver cell types (receptor)
        n_pairs = len(ligand_receptor_pairs)
        interaction_scores = np.zeros((n_types, n_types, n_pairs))
        
        # Calculate interaction scores
        valid_pairs = []
        for p, (ligand, receptor) in enumerate(ligand_receptor_pairs):
            # Check if genes exist
            if ligand in gene_names and receptor in gene_names:
                valid_pairs.append((ligand, receptor))
                
                ligand_idx = gene_names.index(ligand)
                receptor_idx = gene_names.index(receptor)
                
                # Calculate interaction score for each cell type pair
                for i, type_i in enumerate(unique_types):
                    for j, type_j in enumerate(unique_types):
                        # Sender cells (ligand expression)
                        sender_mask = types == type_i
                        ligand_expr = expression[sender_mask, ligand_idx]
                        
                        # Receiver cells (receptor expression)
                        receiver_mask = types == type_j
                        receptor_expr = expression[receiver_mask, receptor_idx]
                        
                        if len(ligand_expr) > 0 and len(receptor_expr) > 0:
                            # Use product of ligand and receptor expression as interaction score
                            mean_ligand = np.mean(ligand_expr)
                            mean_receptor = np.mean(receptor_expr)
                            interaction_scores[i, j, p] = mean_ligand * mean_receptor
        
        if not valid_pairs:
            print("No valid ligand-receptor pairs found in the dataset")
            return None
        
        # Create figure
        plt.figure(figsize=(18, 15))
        
        # 1. Overall ligand-receptor interaction heatmap
        plt.subplot(2, 2, 1)
        
        # Calculate total interaction intensity (sum of all ligand-receptor pairs)
        total_interaction = np.sum(interaction_scores, axis=2)
        
        sns.heatmap(total_interaction, annot=True, fmt='.2f', cmap='YlOrRd',
                  xticklabels=cell_type_names, yticklabels=cell_type_names)
        plt.title('Overall Ligand-Receptor Interaction Strength')
        plt.xlabel('Receiver Cell Types (Receptor)')
        plt.ylabel('Sender Cell Types (Ligand)')
        
        # 2. Bar chart of interaction strength for each ligand-receptor pair
        plt.subplot(2, 2, 2)
        
        # Calculate total interaction strength for each pair
        pair_strengths = np.sum(interaction_scores, axis=(0, 1))
        
        # Sort and display top 10 pairs
        top_indices = np.argsort(pair_strengths)[-10:]
        top_pairs = [f"{ligand_receptor_pairs[i][0]}-{ligand_receptor_pairs[i][1]}" for i in top_indices]
        top_strengths = pair_strengths[top_indices]
        
        plt.barh(range(len(top_pairs)), top_strengths, color='salmon')
        plt.yticks(range(len(top_pairs)), top_pairs)
        plt.xlabel('Interaction Strength')
        plt.title('Top 10 Ligand-Receptor Pair Interactions')
        
        # 3. Spatial visualization of strongest ligand-receptor pair
        plt.subplot(2, 2, 3)
        
        # Get the strongest ligand-receptor pair
        strongest_pair_idx = np.argmax(pair_strengths)
        ligand, receptor = ligand_receptor_pairs[strongest_pair_idx]
        
        # Get gene indices
        ligand_idx = gene_names.index(ligand)
        receptor_idx = gene_names.index(receptor)
        
        # Draw spatial distribution, color indicates ligand expression
        scatter = plt.scatter(coords[:, 0], coords[:, 1], 
                            c=expression[:, ligand_idx], cmap='Reds', 
                            s=30, alpha=0.7)
        plt.colorbar(scatter, label=f'{ligand} Expression (Ligand)')
        
        # Overlay receptor expression
        for i, type_i in enumerate(unique_types):
            mask = types == type_i
            if np.mean(expression[mask, receptor_idx]) > np.percentile(expression[:, receptor_idx], 75):
                plt.scatter(coords[mask, 0], coords[mask, 1], 
                          edgecolors='blue', facecolors='none', s=80,
                          label=f'{cell_type_names[i]} (High {receptor})')
        
        plt.legend()
        plt.title(f'Spatial Distribution of {ligand}-{receptor} Interaction')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        
        # 4. Cell communication network graph
        plt.subplot(2, 2, 4)
        
        try:
            import networkx as nx
            
            # Create network graph
            G = nx.DiGraph()
            
            # Add nodes
            for i, name in enumerate(cell_type_names):
                G.add_node(name, count=np.sum(types == unique_types[i]))
            
            # Add edges
            for i in range(n_types):
                for j in range(n_types):
                    if total_interaction[i, j] > 0:
                        G.add_edge(cell_type_names[i], cell_type_names[j], 
                                 weight=total_interaction[i, j])
            
            # Node size based on cell count
            node_sizes = [G.nodes[n]['count'] * 100 for n in G.nodes()]
            
            # Edge width and color based on interaction strength
            edges = G.edges()
            edge_weights = [G[u][v]['weight'] * 2 for u, v in edges]
            
            # Node positions
            pos = nx.spring_layout(G, k=0.3)
            
            # Draw network
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=range(n_types), 
                                 cmap=plt.cm.tab10, alpha=0.8)
            nx.draw_networkx_labels(G, pos)
            nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.7,
                                 edge_color=edge_weights, edge_cmap=plt.cm.YlOrRd,
                                 connectionstyle='arc3,rad=0.1', arrowsize=15)
            
            plt.title('Cell-Cell Communication Network')
            plt.axis('off')
            
        except ImportError:
            plt.text(0.5, 0.5, "NetworkX required for network visualization", 
                   ha='center', va='center')
            plt.axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
        
        return interaction_scores
    
    def visualize_spatial_domains_and_boundaries(self, spatial_coords, domain_labels, 
                                              gene_expression=None, gene_names=None,
                                              domain_marker_genes=None, output_path=None):
        """
        Visualize spatial domains and boundaries
        
        Args:
            spatial_coords: Spatial coordinates
            domain_labels: Domain labels
            gene_expression: Gene expression data
            gene_names: Gene names
            domain_marker_genes: Dictionary of marker genes for each domain, format: {domain_id: [gene1, gene2, ...]}
            output_path: Output file path
        """
        # Ensure data is on CPU
        coords = spatial_coords.cpu().detach().numpy() if torch.is_tensor(spatial_coords) else spatial_coords
        domains = domain_labels.cpu().detach().numpy() if torch.is_tensor(domain_labels) else np.array(domain_labels)
        
        # Create figure
        plt.figure(figsize=(18, 16))
        
        # 1. Spatial distribution of domains
        plt.subplot(2, 2, 1)
        
        # Get unique domain labels
        unique_domains = np.unique(domains)
        n_domains = len(unique_domains)
        
        # Assign different colors to each domain
        domain_colors = plt.cm.get_cmap('tab20', n_domains)
        
        # Draw domains
        for i, domain in enumerate(unique_domains):
            mask = domains == domain
            plt.scatter(coords[mask, 0], coords[mask, 1], c=[domain_colors(i)], 
                      label=f'Domain {domain}', s=30, alpha=0.7)
            
        plt.legend()
        plt.title('Spatial Domain Distribution')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        
        # 2. Domain boundary visualization
        plt.subplot(2, 2, 2)
        
        # Draw all points
        for i, domain in enumerate(unique_domains):
            mask = domains == domain
            plt.scatter(coords[mask, 0], coords[mask, 1], c=[domain_colors(i)], 
                      label=f'Domain {domain}', s=30, alpha=0.5)
        
        # Find boundary points
        try:
            from scipy.spatial import Voronoi, voronoi_plot_2d
            
            # Use Voronoi diagram to find boundaries between domains
            vor = Voronoi(coords)
            voronoi_plot_2d(vor, ax=plt.gca(), show_vertices=False, line_colors='black', 
                          line_width=1, line_alpha=0.4, point_size=0)
            
        except Exception as e:
            print(f"Voronoi plot failed: {e}")
            print("Using alternative boundary detection method")
            
            # Alternative method: Mark cells adjacent to cells from different domains as boundaries
            from sklearn.neighbors import NearestNeighbors
            
            # Find k nearest neighbors for each point
            k = 10
            nbrs = NearestNeighbors(n_neighbors=k+1).fit(coords)
            distances, indices = nbrs.kneighbors(coords)
            
            # Mark boundary points
            boundary_points = []
            for i in range(len(coords)):
                current_domain = domains[i]
                neighbor_domains = domains[indices[i, 1:]]  # Exclude self
                
                # If neighbors belong to different domains, it's a boundary point
                if np.any(neighbor_domains != current_domain):
                    boundary_points.append(i)
            
            # Draw boundary points
            boundary_coords = coords[boundary_points]
            plt.scatter(boundary_coords[:, 0], boundary_coords[:, 1], 
                      c='black', marker='x', s=20, alpha=0.8, label='Boundary')
        
        plt.title('Domain Boundaries')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        
        # 3. If gene expression data is available, visualize domain marker genes
        if gene_expression is not None and gene_names is not None:
            plt.subplot(2, 2, 3)
            
            # Ensure gene expression data is on CPU
            expr = gene_expression.cpu().detach().numpy() if torch.is_tensor(gene_expression) else gene_expression
            
            # If domain marker genes are specified
            if domain_marker_genes is not None:
                # Create marker gene expression heatmap for each domain
                domain_marker_expr = {}
                
                for domain, markers in domain_marker_genes.items():
                    # Find marker gene indices
                    marker_indices = [gene_names.index(gene) for gene in markers if gene in gene_names]
                    
                    if marker_indices:
                        # Calculate average expression of marker genes
                        marker_expr = np.mean(expr[:, marker_indices], axis=1)
                        domain_marker_expr[domain] = marker_expr
                
                if domain_marker_expr:
                    # Choose any domain's marker gene expression for visualization
                    example_domain = list(domain_marker_expr.keys())[0]
                    
                    plt.scatter(coords[:, 0], coords[:, 1], c=domain_marker_expr[example_domain], 
                              cmap='viridis', s=30, alpha=0.7)
                    plt.colorbar(label=f'Domain {example_domain} Marker Expression')
                    
                    # Draw domain boundaries
                    for i, domain in enumerate(unique_domains):
                        mask = domains == domain
                        plt.scatter([], [], c=[domain_colors(i)], 
                                  label=f'Domain {domain}', s=30, alpha=0.7)
                    
                    plt.legend()
                    plt.title(f'Domain {example_domain} Marker Gene Expression')
                    plt.xlabel('X Coordinate')
                    plt.ylabel('Y Coordinate')
                else:
                    plt.text(0.5, 0.5, "No valid marker genes found", 
                           ha='center', va='center')
                    plt.title('Domain Marker Gene Expression')
            else:
                # If no marker genes specified, show gene with largest difference between domains
                plt.subplot(2, 2, 3)
                
                # Calculate average expression profile for each domain
                domain_profiles = {}
                for domain in unique_domains:
                    mask = domains == domain
                    if np.sum(mask) > 0:
                        domain_profiles[domain] = np.mean(expr[mask], axis=0)
                
                # Calculate gene with largest difference between domains
                if len(domain_profiles) > 1:
                    # Calculate expression variance of each gene across domains
                    domain_expr_matrix = np.stack([profile for profile in domain_profiles.values()])
                    gene_var_across_domains = np.var(domain_expr_matrix, axis=0)
                    
                    # Select gene with highest variance
                    top_var_gene_idx = np.argmax(gene_var_across_domains)
                    top_var_gene = gene_names[top_var_gene_idx]
                    
                    # Visualize expression of this gene
                    plt.scatter(coords[:, 0], coords[:, 1], c=expr[:, top_var_gene_idx], 
                              cmap='viridis', s=30, alpha=0.7)
                    plt.colorbar(label=f'{top_var_gene} Expression')
                    
                    # Draw domain boundaries
                    for i, domain in enumerate(unique_domains):
                        mask = domains == domain
                        plt.scatter([], [], c=[domain_colors(i)], 
                                  label=f'Domain {domain}', s=30, alpha=0.7)
                    
                    plt.legend()
                    plt.title(f'Top Domain-Specific Gene: {top_var_gene}')
                    plt.xlabel('X Coordinate')
                    plt.ylabel('Y Coordinate')
                else:
                    plt.text(0.5, 0.5, "Need at least two domains for comparison", 
                           ha='center', va='center')
                    plt.title('Domain-Specific Gene Expression')
        
        # 4. Domain feature heatmap
        plt.subplot(2, 2, 4)
        
        if gene_expression is not None and gene_names is not None:
            # Ensure gene expression data is on CPU
            expr = gene_expression.cpu().detach().numpy() if torch.is_tensor(gene_expression) else gene_expression
            
            # Calculate feature profile for each domain
            domain_features = np.zeros((n_domains, expr.shape[1]))
            for i, domain in enumerate(unique_domains):
                mask = domains == domain
                if np.sum(mask) > 0:
                    domain_features[i] = np.mean(expr[mask], axis=0)
            
            # Calculate top 20 genes with largest differences between domains
            gene_var = np.var(domain_features, axis=0)
            top_var_indices = np.argsort(gene_var)[-20:]
            
            # Draw heatmap
            sns.heatmap(domain_features[:, top_var_indices], cmap='viridis',
                      xticklabels=[gene_names[i] for i in top_var_indices],
                      yticklabels=[f'Domain {d}' for d in unique_domains])
            plt.xticks(rotation=90)
            plt.title('Domain-Specific Gene Expression Profiles')
            plt.xlabel('Genes')
            plt.ylabel('Domains')
        else:
            plt.text(0.5, 0.5, "Gene expression data required for this plot", 
                   ha='center', va='center')
            plt.title('Domain-Specific Gene Expression')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()