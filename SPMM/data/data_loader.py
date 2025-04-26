"""
Data Loading Module: Load and Preprocess Multiple Data Sources
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import h5py
import json
import cv2
from PIL import Image
import anndata
import scanpy as sc
from scipy.sparse import csr_matrix

from .data_processor import SpatialDataProcessor, ImageProcessor, TextProcessor
from .dataset import SpatialMultimodalDataset, PatchwiseDataset, MultiSlideDataset, collate_spatial_multimodal_batch


class SpatialOmicsDataLoader:
    """Load and Preprocess Spatial Transcriptomics Data"""
    
    def __init__(self, config):
        """
        Initialize Data Loader
        
        Args:
            config: configuration dictionary: contains data paths and processing parameters
        """
        self.config = config
        self.data_dir = config['data_dir']
        self.batch_size = config.get('batch_size', 16)
        self.num_workers = config.get('num_workers', 4)
        
        # Data processors
        self.spatial_processor = SpatialDataProcessor(
            normalization=config.get('normalization', 'tpm'),
            min_genes=config.get('min_genes', 10),
            min_cells=config.get('min_cells', 10),
            min_expr=config.get('min_expr', 1.0)
        )
        
        self.image_processor = ImageProcessor(
            patch_size=config.get('patch_size', 256),
            normalize=config.get('normalize_image', True),
            augmentation=config.get('image_augmentation', False)
        )
        
        self.text_processor = TextProcessor(
            model_name=config.get('text_model_name', 
                               "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"),
            max_length=config.get('max_text_length', 512)
        )
    
    def load_h5ad_file(self, file_path):
        """
        Load Spatial Transcriptomics Data in H5AD Format
        
        Args:
            file_path: H5AD file path
            
        Returns:
            data_dict: A dictionary containing expression matrix and metadata
        """
        print(f"Loading H5AD file: {file_path}")
        
        # Use anndata to load the file
        adata = anndata.read_h5ad(file_path)
        
        # Extract expression matrix
        if isinstance(adata.X, csr_matrix):
            expr_matrix = adata.X.toarray()
        else:
            expr_matrix = adata.X
        
        # Extract gene names
        gene_names = adata.var_names.tolist()
        
        # Extract spatial coordinates
        if 'spatial' in adata.obsm:
            coordinates = adata.obsm['spatial']
        else:
            # If specific spatial coordinates are not provided, 
            # identify columns that might contain coordinate information.
            coord_columns = [col for col in adata.obs.columns if any(x in col.lower() for x in ['x', 'y', 'coord'])]
            if len(coord_columns) >= 2:
                coordinates = adata.obs[coord_columns[:2]].values
            else:
                raise ValueError("Spatial coordinates not found in the H5AD file")
        
        # Extract cell type annotations (if available)
        cell_types = None
        for col in ['cell_type', 'celltype', 'cluster', 'leiden', 'louvain']:
            if col in adata.obs.columns:
                cell_types = adata.obs[col].values
                break
        
        # Construct the data dictionary
        data_dict = {
            'expression': expr_matrix,
            'gene_names': gene_names,
            'coordinates': coordinates,
            'cell_types': cell_types,
            'obs': adata.obs,
            'var': adata.var
        }
        
        return data_dict
    
    def load_10x_visium_data(self, data_dir):
        """
        Load 10x Visium Spatial Transcriptomics Data
        
        Args:
            data_dir: data directory
            
        Returns:
            data_dict: A dictionary containing expression matrix and metadata
        """
        print(f"Loading 10x Visium data from: {data_dir}")
        
        # Try to use scanpy to load data
        try:
            adata = sc.read_visium(data_dir)
            # Process the loaded AnnData object using the above load_h5ad_file method
            return self.load_h5ad_file(adata)
        except Exception as e:
            print(f"Error loading with scanpy: {e}")
            print("Trying to load with manual parsing...")
        
        # Manually load necessary files
        # 1. Load the gene expression matrix
        matrix_dir = os.path.join(data_dir, "filtered_feature_bc_matrix")
        if not os.path.exists(matrix_dir):
            matrix_dir = os.path.join(data_dir, "raw_feature_bc_matrix")
        
        if not os.path.exists(matrix_dir):
            raise FileNotFoundError(f"Feature matrix not found in {data_dir}")
        
        # Use scanpy to load gene expression matrix
        adata = sc.read_10x_mtx(matrix_dir)
        
        # 2. Load spatial information
        spatial_file = os.path.join(data_dir, "spatial/tissue_positions_list.csv")
        if not os.path.exists(spatial_file):
            # Try the old format
            spatial_file = os.path.join(data_dir, "spatial/tissue_positions.csv")
        
        if os.path.exists(spatial_file):
            positions = pd.read_csv(spatial_file, header=None)
            # Standard format: barcode, in_tissue, array_row, array_col, pxl_row_in_fullres, pxl_col_in_fullres
            if positions.shape[1] >= 6:
                barcodes = positions.iloc[:, 0].values
                in_tissue = positions.iloc[:, 1].values.astype(bool)
                array_coords = positions.iloc[:, 2:4].values
                pixel_coords = positions.iloc[:, 4:6].values
            else:
                raise ValueError(f"Unexpected format in {spatial_file}")
        else:
            raise FileNotFoundError(f"Spatial positions file not found in {data_dir}")
        
        # 3. Load images
        image_file = os.path.join(data_dir, "spatial/tissue_hires_image.png")
        if not os.path.exists(image_file):
            image_file = os.path.join(data_dir, "spatial/tissue_lowres_image.png")
        
        if os.path.exists(image_file):
            image = cv2.imread(image_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = None
            print(f"Warning: Image file not found in {data_dir}")
        
        # 4. Load the scale information
        scalefactors_file = os.path.join(data_dir, "spatial/scalefactors_json.json")
        if os.path.exists(scalefactors_file):
            with open(scalefactors_file, 'r') as f:
                scalefactors = json.load(f)
        else:
            scalefactors = {"spot_diameter_fullres": 50}
            print(f"Warning: Scalefactors file not found in {data_dir}, using default values")
        
        # Ensure that the expression matrix matches the spatial information
        common_barcodes = []
        adata_barcodes = adata.obs.index
        for bc in barcodes:
            if bc in adata_barcodes:
                common_barcodes.append(bc)
        
        if len(common_barcodes) == 0:
            raise ValueError("No matching barcodes between expression matrix and spatial information")
        
        # Filter the expression matrix and only retain the spots with spatial information
        adata = adata[common_barcodes]
        
        # Construct index mapping
        barcode_indices = {bc: i for i, bc in enumerate(barcodes)}
        indices = [barcode_indices[bc] for bc in common_barcodes]
        
        # Extract the spatial information of these spots
        filtered_in_tissue = in_tissue[indices]
        filtered_array_coords = array_coords[indices]
        filtered_pixel_coords = pixel_coords[indices]
        
        # Handle the expression matrix
        if isinstance(adata.X, csr_matrix):
            expr_matrix = adata.X.toarray()
        else:
            expr_matrix = adata.X
        
        # Build a data dictionary
        data_dict = {
            'expression': expr_matrix,
            'gene_names': adata.var_names.tolist(),
            'coordinates': filtered_pixel_coords,  # Use pixel coordinates as spatial coordinates
            'array_coordinates': filtered_array_coords,
            'in_tissue': filtered_in_tissue,
            'barcodes': common_barcodes,
            'image': image,
            'scalefactors': scalefactors
        }
        
        return data_dict
    
    def load_wsi_image(self, image_path):
        """
        Load Full Slide Images (WSI)
        
        Args:
            image_path: Image file path
            
        Returns:
            image: Loaded image
        """
        print(f"Loading WSI image: {image_path}")
        
        # Check the file suffix
        ext = os.path.splitext(image_path)[1].lower()
        
        if ext in ['.svs', '.ndpi', '.mrxs', '.tiff']:
            # For large WSI files, use dedicated libraries (such as OpenSlide)
            try:
                import openslide
                slide = openslide.OpenSlide(image_path)
                # Obtain the thumbnail or the layer with the lowest resolution
                level = slide.level_count - 1
                image = slide.read_region((0, 0), level, slide.level_dimensions[level])
                image = np.array(image.convert('RGB'))
                return image
            except ImportError:
                print("OpenSlide not installed. Falling back to PIL.")
            except Exception as e:
                print(f"Error using OpenSlide: {e}")
                print("Falling back to direct loading...")
        
        # Load the image directly using PIL
        try:
            image = Image.open(image_path)
            image = np.array(image.convert('RGB'))
            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def load_clinical_text(self, text_file):
        """
        Load clinical text data
        
        Args:
            text_file: Text file path
            
        Returns:
            processed_text: Processed text data
        """
        print(f"Loading clinical text: {text_file}")
        
        # Read the text file
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Process the text using a text processor
        processed_text = self.text_processor.process_clinical_text(text)
        
        return processed_text
    
    def prepare_dataset(self, sample_id, split='train'):
        """
        Prepare the dataset for a single sample
        
        Args:
            sample_id: Sample ID
            split: Dataset Spilit ('train', 'val', 'test')
            
        Returns:
            dataset: Prepared dataset
        """
        # Build the file path
        spatial_file = os.path.join(self.data_dir, f"{sample_id}/spatial_data.h5ad")
        image_file = os.path.join(self.data_dir, f"{sample_id}/image.png")
        text_file = os.path.join(self.data_dir, f"{sample_id}/clinical.txt")
        label_file = os.path.join(self.data_dir, f"{sample_id}/labels.csv")
        
        # Load spatial transcriptome data
        if os.path.exists(spatial_file):
            spatial_data_raw = self.load_h5ad_file(spatial_file)
        else:
            # Attempt to load 10x Visium data
            visium_dir = os.path.join(self.data_dir, sample_id)
            if os.path.isdir(visium_dir):
                spatial_data_raw = self.load_10x_visium_data(visium_dir)
            else:
                raise FileNotFoundError(f"Spatial data not found for sample {sample_id}")
        
        # Process spatial transcriptome data
        spatial_data = self.spatial_processor.process_spatial_data(
            spatial_data_raw['expression'],
            spatial_data_raw['coordinates'],
            spatial_data_raw['gene_names'],
            spatial_data_raw.get('obs', None)
        )
        
        # Load the WSI image
        if os.path.exists(image_file):
            wsi_image = self.load_wsi_image(image_file)
        elif 'image' in spatial_data_raw:
            wsi_image = spatial_data_raw['image']
        else:
            print(f"Warning: No image found for sample {sample_id}")
            wsi_image = None
        
        # Process image data
        if wsi_image is not None:
            image_data = self.image_processor.process_wsi_image(
                wsi_image, 
                spatial_data['coordinates'].numpy()
            )
        else:
            # Create an empty image data structure
            image_data = {
                'patches': [],
                'features': torch.zeros((spatial_data['expression'].shape[0], 512)),
                'valid_indices': [],
                'tissue_mask': None
            }
        
        # Load clinical text
        if os.path.exists(text_file):
            text_data = self.load_clinical_text(text_file)
        else:
            print(f"Warning: No clinical text found for sample {sample_id}")
            # Create an empty text data structure
            text_data = {
                'input_ids': torch.zeros((1, self.text_processor.max_length), dtype=torch.long),
                'attention_mask': torch.zeros((1, self.text_processor.max_length), dtype=torch.long)
            }
        
        # Load the label
        labels = None
        if os.path.exists(label_file):
            labels_df = pd.read_csv(label_file)
            if 'label' in labels_df.columns:
                labels = labels_df['label'].values
        
        # Create a data set
        dataset = SpatialMultimodalDataset(
            spatial_data=spatial_data,
            image_data=image_data,
            text_data=text_data,
            labels=labels,
            mode=split
        )
        
        return dataset
    
    def prepare_dataloader(self, dataset, batch_size=None, shuffle=True):
        """
        Prepare the data loader
        
        Args:
            dataset: Dataset
            batch_size: Size of batch
            shuffle: Whether to scramble the data
            
        Returns:
            dataloader: data loader
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_spatial_multimodal_batch,
            pin_memory=True
        )
        
        return dataloader
    
    def prepare_all_dataloaders(self, sample_ids=None, splits={'train': 0.7, 'val': 0.15, 'test': 0.15}):
        """
        prepare all dataloaders
        
        Args:
            sample_ids: Sample ID lists
            splits: Dataset spilit
            
        Returns:
            dataloaders: A dictionary containing all data loaders
        """
        # If the sample ID is not specified, scan the data directory
        if sample_ids is None:
            sample_ids = []
            for item in os.listdir(self.data_dir):
                item_path = os.path.join(self.data_dir, item)
                if os.path.isdir(item_path):
                    sample_ids.append(item)
        
        # Randomly divide the sample
        np.random.seed(self.config.get('random_seed', 42))
        np.random.shuffle(sample_ids)
        
        # Calculate the partition index
        n_samples = len(sample_ids)
        n_train = int(n_samples * splits['train'])
        n_val = int(n_samples * splits['val'])
        
        train_ids = sample_ids[:n_train]
        val_ids = sample_ids[n_train:n_train+n_val]
        test_ids = sample_ids[n_train+n_val:]
        
        print(f"Sample split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")
        
        # Prepare for the datasets
        train_datasets = [self.prepare_dataset(sample_id, 'train') for sample_id in train_ids]
        val_datasets = [self.prepare_dataset(sample_id, 'val') for sample_id in val_ids]
        test_datasets = [self.prepare_dataset(sample_id, 'test') for sample_id in test_ids]
        
        # Concatenate datasets
        from torch.utils.data import ConcatDataset
        train_dataset = ConcatDataset(train_datasets) if train_datasets else None
        val_dataset = ConcatDataset(val_datasets) if val_datasets else None
        test_dataset = ConcatDataset(test_datasets) if test_datasets else None
        
        # Create dataloaders
        dataloaders = {}
        if train_dataset:
            dataloaders['train'] = self.prepare_dataloader(train_dataset, shuffle=True)
        if val_dataset:
            dataloaders['val'] = self.prepare_dataloader(val_dataset, shuffle=False)
        if test_dataset:
            dataloaders['test'] = self.prepare_dataloader(test_dataset, shuffle=False)
        
        return dataloaders