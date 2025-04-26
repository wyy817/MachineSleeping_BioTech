"""
Data preprocessing module: Processes spatial transcriptome, image and text data
"""

import torch
import numpy as np
from torch_geometric.data import Data
from transformers import AutoTokenizer
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler
import pandas as pd
import cv2
import scipy.sparse as sp
import networkx as nx
from scipy.spatial import Delaunay


class SpatialDataProcessor:
    """Process spatial transcriptome data"""
    
    def __init__(self, normalization='tpm', min_genes=10, min_cells=10, min_expr=1.0):
        self.normalization = normalization  # 'tpm', 'rpkm', 'cpm'...
        self.min_genes = min_genes  # The threshold for filtering low-expression genes
        self.min_cells = min_cells  # The threshold for filtering low-quality cells
        self.min_expr = min_expr    # Minimum expression threshold
        self.scaler = StandardScaler()
        
    def normalize_expression(self, expression_matrix):
        """
        Normalized Expression Matrix
        
        Args:
            expression_matrix: Gene expression matrix, shaped as (number of cells, number of genes)
            
        Returns:
            normalized_matrix: The normalized expression matrix
        """
        if self.normalization == 'tpm':
            # TPM Normalization (Transcripts Per Million)
            scaling_factor = expression_matrix.sum(axis=1, keepdims=True) / 1e6
            normalized_matrix = expression_matrix / (scaling_factor + 1e-10)
            
        elif self.normalization == 'log':
            # Log transformation (log(x+1))
            normalized_matrix = np.log1p(expression_matrix)
            
        elif self.normalization == 'scale':
            # Z-score standardization
            normalized_matrix = self.scaler.fit_transform(expression_matrix)
            
        else:
            # Normalization is not performed by default
            normalized_matrix = expression_matrix
            
        return normalized_matrix
    
    def filter_genes(self, expression_matrix, gene_names):
        """
        Filter out low-expression genes
        
        Args:
            expression_matrix: express matrixs
            gene_names: gene name lists
            
        Returns:
            filtered_matrix: The filtered expression matrix
            filtered_genes: The filtered gene name
        """
        # Calculate the sum of the expression levels of each gene in all cells
        gene_sums = expression_matrix.sum(axis=0)
        
        # Calculate in how many cells each gene is expressed
        gene_expressed_cells = np.sum(expression_matrix > 0, axis=0)
        
        # Filtering condition: The sum of gene expression levels is greater than the threshold and expressed in a sufficient number of cells
        keep_genes = (gene_sums >= self.min_expr) & (gene_expressed_cells >= self.min_cells)
        
        filtered_matrix = expression_matrix[:, keep_genes]
        filtered_genes = [gene_names[i] for i, keep in enumerate(keep_genes) if keep]
        
        print(f"Genes before filtering: {len(gene_names)}")
        print(f"Genes after filtering: {len(filtered_genes)}")
        
        return filtered_matrix, filtered_genes
    
    def filter_cells(self, expression_matrix, cell_metadata=None):
        """
        Filter low-quality cells
        
        Args:
            expression_matrix: Expression matrix
            cell_metadata: Cell metadata
            
        Returns:
            filtered_matrix: Filtered expression matrix
            filtered_metadata: Filtered cell metadata
        """
        # Calculate the number of genes expressed in each cell
        expressed_genes_per_cell = np.sum(expression_matrix > 0, axis=1)
        
        # Filtering condition: number of genes expressed in a cell is greater than the threshold
        keep_cells = expressed_genes_per_cell >= self.min_genes
        
        filtered_matrix = expression_matrix[keep_cells, :]
        
        print(f"Cells before filtering: {expression_matrix.shape[0]}")
        print(f"Cells after filtering: {filtered_matrix.shape[0]}")
        
        if cell_metadata is not None:
            filtered_metadata = cell_metadata.iloc[keep_cells]
            return filtered_matrix, filtered_metadata
        
        return filtered_matrix, None
    
    def create_spatial_graph(self, coordinates, k=6, method='knn'):
        """
        Create the graph structure between cells based on spatial coordinates
        
        Args:
            coordinates: Cell spatial coordinates, with a shape of (number of cells, 2)
            k: The number of neighbors of KNN
            method: Graph construction methods, 'knn', 'radius', or 'delaunay'
            
        Returns:
            edge_index: Edge index, used for graph neural networks
        """
        if method == 'knn':
            # Build the graph using KNN
            from sklearn.neighbors import NearestNeighbors
            
            # Find the k nearest neighbors of each point
            nbrs = NearestNeighbors(n_neighbors=k+1).fit(coordinates)
            distances, indices = nbrs.kneighbors(coordinates)
            
            # Create edge indexes
            rows = np.repeat(np.arange(len(coordinates)), k)
            cols = indices[:, 1:].flatten()  # Exclude oneself
            
            # Double the edges
            edge_index = np.vstack([
                np.concatenate([rows, cols]),
                np.concatenate([cols, rows])
            ])
            
        elif method == 'radius':
            # Use a radius graph construction method
            from sklearn.neighbors import radius_neighbors_graph
            
            # Calculate the appropriate radius
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=k).fit(coordinates)
            distances, _ = nbrs.kneighbors(coordinates)
            radius = np.mean(distances[:, -1]) * 1.5  # Use 1.5 Times the Average k-NN Distance as the Radius
            
            # Construct the radius graph
            adj_matrix = radius_neighbors_graph(coordinates, radius, mode='connectivity')
            
            # Convert to edge index
            adj_coo = sp.coo_matrix(adj_matrix)
            edge_index = np.vstack([adj_coo.row, adj_coo.col])
            
        elif method == 'delaunay':
            # Construct a graph using delaunay triangulation
            try:
                tri = Delaunay(coordinates)
                edges = set()
                
                # Extract edges from the triangles
                for simplex in tri.simplices:
                    for i in range(3):
                        for j in range(i+1, 3):
                            # Ensure the edge direction is consistent (small index -> large index)
                            if simplex[i] < simplex[j]:
                                edges.add((simplex[i], simplex[j]))
                            else:
                                edges.add((simplex[j], simplex[i]))
                
                # Convert to edge index
                edge_list = list(edges)
                source_nodes = [e[0] for e in edge_list]
                target_nodes = [e[1] for e in edge_list]
                
                # Double the edges
                edge_index = np.vstack([
                    np.concatenate([source_nodes, target_nodes]),
                    np.concatenate([target_nodes, source_nodes])
                ])
            except Exception as e:
                print(f"Delaunay triangulation failed: {e}")
                print("Falling back to KNN method")
                return self.create_spatial_graph(coordinates, k, 'knn')
        else:
            raise ValueError(f"Unknown graph construction method: {method}")
        
        return torch.tensor(edge_index, dtype=torch.long)
    
    def process_spatial_data(self, expression_matrix, coordinates, gene_names=None, cell_metadata=None):
        """
        Main function to process spatial transcriptome data
        
        Args:
            expression_matrix: gene expression matrix
            coordinates: spatial coordinates
            gene_names: gene names lists
            cell_metadata: cell metadata
            
        Returns:
            processed_data: a dictionary containing processed data
        """
        # 1. Filter low-quality cells
        filtered_expr, filtered_metadata = self.filter_cells(expression_matrix, cell_metadata)
        
        # 2. Filter low-expression genes
        if gene_names is not None:
            filtered_expr, filtered_genes = self.filter_genes(filtered_expr, gene_names)
        else:
            filtered_genes = [f"Gene_{i}" for i in range(filtered_expr.shape[1])]
        
        # 3. Gene expression normalization
        normalized_expr = self.normalize_expression(filtered_expr)
        
        # 4. Adjust coordinates to match filtered cells
        if cell_metadata is not None:
            filtered_coords = coordinates[filtered_metadata.index]
        else:
            filtered_coords = coordinates[:filtered_expr.shape[0]]
        
        # 5. Create spatial graph
        edge_index = self.create_spatial_graph(filtered_coords)
        
        # 6. Create PyTorch geometric data object
        x = torch.tensor(normalized_expr, dtype=torch.float)
        pos = torch.tensor(filtered_coords, dtype=torch.float)
        
        # 7. Return processed data
        processed_data = {
            'expression': x,
            'coordinates': pos,
            'edge_index': edge_index,
            'gene_names': filtered_genes,
            'metadata': filtered_metadata
        }
        
        return processed_data


class ImageProcessor:
    """Deal with WSI Image Data"""
    
    def __init__(self, patch_size=256, normalize=True, augmentation=False):
        self.patch_size = patch_size
        self.normalize = normalize
        self.augmentation = augmentation
        
        # Define transformations
        if self.augmentation:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(patch_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif self.normalize:
            self.transform = transforms.Compose([
                transforms.Resize((patch_size, patch_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((patch_size, patch_size)),
                transforms.ToTensor()
            ])
    
    def normalize_staining(self, image, target_means=None, target_stds=None):
        """
        Perform Stain Normalizatiom Using the Macenko Method
        
        Args:
            image: Load the data
            target_means: Target means
            target_stds: Target standards
            
        Returns:
            normalized_image: Images after normalization
        """
        # Simplified stain normalization
        # In practical applications, it is recommended to use the complete Macenko or Vahadane method
        
        # Default target parameters (if not provided)
        if target_means is None:
            target_means = np.array([148, 160, 180]) / 255  # Typical H&E means
        if target_stds is None:
            target_stds = np.array([0.15, 0.15, 0.15])  # Typical H&E standard deviations
        
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Separate LAB channels
        l, a, b = cv2.split(lab)
        
        # Standardize a and b channel
        a_mean, a_std = cv2.meanStdDev(a)
        b_mean, b_std = cv2.meanStdDev(b)
        
        # Adjust to target means and stds
        a = ((a - a_mean) / (a_std + 1e-8)) * target_stds[1] * 255 + target_means[1] * 255
        b = ((b - b_mean) / (b_std + 1e-8)) * target_stds[2] * 255 + target_means[2] * 255
        
        # Merge channels
        lab = cv2.merge([l, a.astype(np.uint8), b.astype(np.uint8)])
        
        # Convert back to RGB
        normalized_image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return normalized_image
    
    def extract_patches(self, wsi_image, coordinates, patch_size=None):
        """
        Extract Image Patches Corresponding to Spatial Transcriptomics Coordinates from WSI Images
        
        Args:
            wsi_image: whole WSI image
            coordinates: Cell / Spot coordinates
            patch_size: Image patch size
            
        Returns:
            patches: List of extracted image patches
        """
        if patch_size is None:
            patch_size = self.patch_size
            
        half_size = patch_size // 2
        patches = []
        valid_indices = []
        
        height, width = wsi_image.shape[:2]
        
        for i, (x, y) in enumerate(coordinates):
            # Covert to integer coordinates
            x, y = int(x), int(y)
            
            # Check boundaries
            if (x - half_size >= 0 and x + half_size < width and 
                y - half_size >= 0 and y + half_size < height):
                
                # Extract image patch
                patch = wsi_image[y - half_size:y + half_size, x - half_size:x + half_size]
                
                # Apply normalization
                patch = self.normalize_staining(patch)
                
                # Convert to tensor
                if self.transform:
                    patch = self.transform(patch)
                
                patches.append(patch)
                valid_indices.append(i)
            
        return patches, valid_indices
    
    def segment_tissue(self, wsi_image, threshold=0.8):
        """
        Segment Tissue Regions in WSI Images
        
        Args:
            wsi_image: WSI images
            threshold: Segmentation threshold
            
        Returns:
            mask: Tissue region mask
        """
        # Convert to grayscale
        gray = cv2.cvtColor(wsi_image, cv2.COLOR_RGB2GRAY)
        
        # Apply Otsu threshold segmentation
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def extract_features(self, patches, feature_extractor=None):
        """
        Extract Features from Image Patches
        
        Args:
            patches: Image patches lists
            feature_extractor: Feature extractor model
            
        Returns:
            features: Extracted features
        """
        if feature_extractor is None:
            # Use pre-trained ResNet18 as the default feature extractor
            import torchvision.models as models
            feature_extractor = models.resnet18(pretrained=True)
            # Remove the final classification layer
            feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-1])
            feature_extractor.eval()
        
        # Batch feature extraction
        features = []
        with torch.no_grad():
            for patch in patches:
                patch_batch = patch.unsqueeze(0)  # Add batch dimension
                feature = feature_extractor(patch_batch)
                feature = feature.squeeze()  # Remove batch and spatial dimensions
                features.append(feature)
        
        return torch.stack(features)
    
    def process_wsi_image(self, wsi_image, coordinates, feature_extractor=None):
        """
        Main function to process WSI images
        
        Args:
            wsi_image: WSI images
            coordinates: Corresponding spatial coordinates
            feature_extractor: Feature extractor model
            
        Returns:
            processed_data: a dictionary containing processed data
        """
        # 1. Segment tissue regions
        tissue_mask = self.segment_tissue(wsi_image)
        
        # 2. Extract image patches
        patches, valid_indices = self.extract_patches(wsi_image, coordinates)
        
        # 3. Extract features
        if len(patches) > 0:
            features = self.extract_features(patches, feature_extractor)
        else:
            features = torch.tensor([])
        
        # 4. Return processed data
        processed_data = {
            'patches': patches,
            'features': features,
            'valid_indices': valid_indices,
            'tissue_mask': tissue_mask
        }
        
        return processed_data


class TextProcessor:
    """Process clinical text data"""
    
    def __init__(self, model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", max_length=512):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def preprocess_text(self, text):
        """
        Preprocess clinical text
        
        Args:
            text: Input text
            
        Returns:
            processed_text: Preprocessed text
        """
        # Remove excess whitespace
        processed_text = ' '.join(text.split())
        
        # Replace common medical abbreviations
        medical_abbr = {
            'pt': 'patient',
            'pts': 'patients',
            'dx': 'diagnosis',
            'hx': 'history',
            'tx': 'treatment',
            'sx': 'symptoms',
            'fx': 'fracture',
            'CA': 'cancer',
            'w/': 'with',
            'w/o': 'without',
            'yo': 'years old',
            'y/o': 'years old',
            'y.o.': 'years old'
        }
        
        for abbr, full in medical_abbr.items():
            processed_text = processed_text.replace(f' {abbr} ', f' {full} ')
        
        return processed_text
    
    def standardize_medical_terms(self, text):
        """
        Standardize Medical Terminology
        
        Args:
            text: Input text
            
        Returns:
            standardized_text: Standardized text
        """
        # A medical terminology dictionary should be used for standardization here
        # The simplified version only handles some common terms
        term_mapping = {
            'heart attack': 'myocardial infarction',
            'heart failure': 'cardiac failure',
            'kidney failure': 'renal failure',
            'high blood pressure': 'hypertension',
            'low blood pressure': 'hypotension',
            'stroke': 'cerebrovascular accident',
            'sugar': 'glucose'
        }
        
        standardized_text = text
        for term, standard in term_mapping.items():
            standardized_text = standardized_text.replace(term, standard)
            
        return standardized_text
    
    def tokenize_text(self, text):
        """
        Tokenize text
        
        Args:
            text: Input text
            
        Returns:
            tokenized: Tokenization results
        """
        # Use pre-trained tokenizer
        tokenized = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return tokenized
    
    def extract_medical_entities(self, text):
        """
        Extract Medical Entities
        
        Args:
            text: Input text
            
        Returns:
            entities: Extracted entities
        """
        # Simplified entity extraction
        # In practical applications, named entity recognition (NER) models should be used
        
        # Define simple regular expressions for medical entities
        import re
        
        # Diseases
        diseases = re.findall(r'(cancer|carcinoma|tumor|sarcoma|lymphoma|leukemia|adenoma)', text.lower())
        
        # Organs
        organs = re.findall(r'(breast|lung|liver|kidney|colon|brain|prostate|pancreas|ovary)', text.lower())
        
        # Drugs
        drugs = re.findall(r'(tamoxifen|cisplatin|paclitaxel|doxorubicin|methotrexate)', text.lower())
        
        entities = {
            'diseases': list(set(diseases)),
            'organs': list(set(organs)),
            'drugs': list(set(drugs))
        }
        
        return entities
    
    def process_clinical_text(self, text):
        """
        Main function to process clinical text data
        
        Args:
            text: Input data
            
        Returns:
            processed_data: a dictionary containing processed data
        """
        # 1. Text preprocessing
        preprocessed_text = self.preprocess_text(text)
        
        # 2. Text standardization
        standardized_text = self.standardize_medical_terms(preprocessed_text)
        
        # 3. Tokenization
        tokenized = self.tokenize_text(standardized_text)
        
        # 4. Entity extraction
        entities = self.extract_medical_entities(standardized_text)
        
        # 5. Return processed data
        processed_data = {
            'preprocessed_text': preprocessed_text,
            'standardized_text': standardized_text,
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'entities': entities
        }
        
        return processed_data