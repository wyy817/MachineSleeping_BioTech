"""
Main Program Entry: Training and evaluating the spatial transcriptomics multimodal medical model
"""

import os
import argparse
import torch
import numpy as np
import random
from config import MODEL_CONFIG, TRAIN_CONFIG, DATA_CONFIG, PATH_CONFIG
from models import SpatialMultimodalModel
from data import SpatialOmicsDataLoader
from utils.evaluation import evaluate_model
from visualization.dashboard import DashboardCreator


def set_seed(seed):
    """Set the random seed to ensure reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train(model, train_loader, val_loader, optimizer, device, num_epochs=50, patience=10):
    """
    Model Training
    
    Args:
        model: Model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        device: Device
        num_epochs: Number of training epochs
        patience: Early stopping patience
    
    Returns:
        model: Trained model
        history: Training history
    """
    # Define the loss function
    classification_criterion = torch.nn.CrossEntropyLoss()
    regression_criterion = torch.nn.MSELoss()
    
    # History training
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # Early stopping mechanism
    best_val_loss = float('inf')
    no_improve_epoch = 0
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            # Aquire data
            spatial_expr = batch['spatial_expr'].to(device)
            spatial_coords = batch['spatial_coords'].to(device)
            image_features = batch['image_features'].to(device)
            text_input_ids = batch['text_input_ids'].to(device)
            text_attention_mask = batch['text_attention_mask'].to(device)
            labels = batch['labels'].to(device) if 'labels' in batch else None
            
            # Construct a spatial graph
            # Simplifying the process here makes sense for clarity, 
            # but indeed, in real-world applications, constructing the graph based on spatial coordinates would be crucial
            batch_size = spatial_expr.size(0)
            edge_index = torch.zeros((2, batch_size * 10), dtype=torch.long, device=device)
            for i in range(batch_size):
                # Each node is connected to five random other nodes
                for j in range(10):
                    edge_index[0, i * 10 + j] = i
                    edge_index[1, i * 10 + j] = random.randint(0, batch_size - 1)
            
            # Forward propagation
            outputs = model(
                spatial_expr,
                edge_index,
                spatial_coords,
                text_input_ids,
                text_attention_mask,
                image_features
            )
            
            # Calculate loss
            loss = 0
            
            if labels is not None:
                classification_loss = classification_criterion(outputs['classification'], labels)
                loss += classification_loss
                
                # Calculate accuracy
                _, predicted = torch.max(outputs['classification'].data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            # If survival prediction is available
            if 'survival' in outputs and 'survival_target' in batch:
                survival_loss = regression_criterion(outputs['survival'].squeeze(), batch['survival_target'].to(device))
                loss += survival_loss
            
            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Calculate average loss and accuracy for the training set
        train_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total if total > 0 else 0
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Aquire data
                spatial_expr = batch['spatial_expr'].to(device)
                spatial_coords = batch['spatial_coords'].to(device)
                image_features = batch['image_features'].to(device)
                text_input_ids = batch['text_input_ids'].to(device)
                text_attention_mask = batch['text_attention_mask'].to(device)
                labels = batch['labels'].to(device) if 'labels' in batch else None
                
                # Construct a spatial graph
                batch_size = spatial_expr.size(0)
                edge_index = torch.zeros((2, batch_size * 10), dtype=torch.long, device=device)
                for i in range(batch_size):
                    for j in range(10):
                        edge_index[0, i * 10 + j] = i
                        edge_index[1, i * 10 + j] = random.randint(0, batch_size - 1)
                
                # Forward propagation
                outputs = model(
                    spatial_expr,
                    edge_index,
                    spatial_coords,
                    text_input_ids,
                    text_attention_mask,
                    image_features
                )
                
                # Calculate loss
                v_loss = 0
                
                if labels is not None:
                    v_classification_loss = classification_criterion(outputs['classification'], labels)
                    v_loss += v_classification_loss
                    
                    # Calculate accuracy
                    _, predicted = torch.max(outputs['classification'].data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                
                # If survival prediction is available
                if 'survival' in outputs and 'survival_target' in batch:
                    v_survival_loss = regression_criterion(outputs['survival'].squeeze(), batch['survival_target'].to(device))
                    v_loss += v_survival_loss
                
                val_loss += v_loss.item()
        
        # Calculate average loss and accuracy for the validation set
        val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Print information
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epoch = 0
            # Save the best model
            torch.save(model.state_dict(), os.path.join(PATH_CONFIG['model_dir'], 'best_model.pth'))
        else:
            no_improve_epoch += 1
            if no_improve_epoch >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # Load the best model
    model.load_state_dict(torch.load(os.path.join(PATH_CONFIG['model_dir'], 'best_model.pth')))
    
    return model, history


def predict(model, test_loader, device):
    """
    Use the trained model to make predictions on the test set
    
    Args:
        model: Model instance
        test_loader: Test data loader
        device: Device
    
    Returns:
        predictions: Prediction results
        true_labels: True labels
    """
    model.eval()
    all_predictions = []
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Aquire data
            spatial_expr = batch['spatial_expr'].to(device)
            spatial_coords = batch['spatial_coords'].to(device)
            image_features = batch['image_features'].to(device)
            text_input_ids = batch['text_input_ids'].to(device)
            text_attention_mask = batch['text_attention_mask'].to(device)
            
            if 'labels' in batch:
                all_labels.append(batch['labels'].numpy())
            
            # Construct a spatial graph
            batch_size = spatial_expr.size(0)
            edge_index = torch.zeros((2, batch_size * 10), dtype=torch.long, device=device)
            for i in range(batch_size):
                for j in range(10):
                    edge_index[0, i * 10 + j] = i
                    edge_index[1, i * 10 + j] = random.randint(0, batch_size - 1)
            
            # Forward propagation
            outputs = model(
                spatial_expr,
                edge_index,
                spatial_coords,
                text_input_ids,
                text_attention_mask,
                image_features
            )
            
            # Collect predictions and features
            all_predictions.append(outputs['classification'].cpu().numpy())
            all_features.append(outputs['features'].cpu().numpy())
    
    # Merge Batch Results
    predictions = np.vstack(all_predictions)
    features = np.vstack(all_features)
    
    if all_labels:
        true_labels = np.concatenate(all_labels)
    else:
        true_labels = None
    
    return predictions, features, true_labels


def visualize_results(model, test_loader, device, gene_names, class_names):
    """
    Model Results Visualization
    
    Args:
        model: Model instance
        test_loader: Test data loader
        device: Device
        gene_names: Gene names list
        class_names: Class names list
    """
    # Visualize a single batch of data
    batch = next(iter(test_loader))
    
    # Aquire data
    spatial_expr = batch['spatial_expr'].to(device)
    spatial_coords = batch['spatial_coords'].to(device)
    image_features = batch['image_features'].to(device)
    text_input_ids = batch['text_input_ids'].to(device)
    text_attention_mask = batch['text_attention_mask'].to(device)
    
    # Construct a spatial graph
    batch_size = spatial_expr.size(0)
    edge_index = torch.zeros((2, batch_size * 10), dtype=torch.long, device=device)
    for i in range(batch_size):
        for j in range(10):
            edge_index[0, i * 10 + j] = i
            edge_index[1, i * 10 + j] = random.randint(0, batch_size - 1)
    
    # Forward propagation
    model.eval()
    with torch.no_grad():
        outputs = model(
            spatial_expr,
            edge_index,
            spatial_coords,
            text_input_ids,
            text_attention_mask,
            image_features
        )
    
    # Assume cell types 
    # (randomly generated here; in practical applications, they should be obtained from the data)
    if 'cell_types' in batch:
        cell_types = batch['cell_types']
    else:
        cell_types = torch.randint(0, 5, (batch_size,))
    
    # Create a dashboard
    dashboard_creator = DashboardCreator()
    
    dashboard_paths = dashboard_creator.create_dashboard(
        model_outputs=outputs,
        spatial_data=spatial_expr.cpu(),
        text_data=text_input_ids.cpu(),
        vision_data=image_features.cpu(),
        gene_names=gene_names,
        cell_types=cell_types,
        spatial_coords=spatial_coords.cpu(),
        class_names=class_names
    )
    
    # Predictions and evaluation
    predictions, features, true_labels = predict(model, test_loader, device)
    
    if true_labels is not None:
        # Visualize prediction results
        results_paths = dashboard_creator.visualize_prediction_results(
            predictions, true_labels, class_names
        )
    
    # Create a model overview page
    model_architecture = str(model)
    
    # Assume training history 
    # (in practical applications, it should be returned from the train function)
    sample_history = {
        'loss': [0.9, 0.7, 0.5, 0.3, 0.2],
        'val_loss': [1.0, 0.8, 0.6, 0.5, 0.4],
        'accuracy': [0.6, 0.7, 0.8, 0.85, 0.9],
        'val_accuracy': [0.5, 0.6, 0.7, 0.75, 0.8]
    }
    
    overview_path = dashboard_creator.create_model_overview_page(
        model_architecture, sample_history
    )
    
    print(f"Visualization results have saved to {dashboard_creator.output_dir}.")
    print(f"Access {os.path.join(dashboard_creator.output_dir, 'index.html')} to view the dashboard.")


def main():
    """Main Function"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Training spatial transcriptomics multimodal medical models')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict', 'visualize'],
                      help='Operating Modes: train, predict, or visualize')
    parser.add_argument('--data_dir', type=str, default=PATH_CONFIG['data_dir'],
                      help='Data Directory')
    parser.add_argument('--model_path', type=str, default=None,
                      help='Pretrained Model Path (for prediction and visualization mode)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random Seeds')
    args = parser.parse_args()
    
    # Set random seeds
    set_seed(args.seed)
    
    # Ensure the output directory exists
    os.makedirs(PATH_CONFIG['model_dir'], exist_ok=True)
    os.makedirs(PATH_CONFIG['log_dir'], exist_ok=True)
    
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the data
    data_loader = SpatialOmicsDataLoader({
        'data_dir': args.data_dir,
        'batch_size': TRAIN_CONFIG['batch_size'],
        'num_workers': 4,
        'normalization': 'tpm',
        'min_genes': 10,
        'min_cells': 10,
        'min_expr': 1.0,
        'patch_size': 256,
        'normalize_image': True,
        'image_augmentation': True,
        'text_model_name': MODEL_CONFIG['text_encoder']['model_name'],
        'max_text_length': 512,
        'random_seed': args.seed
    })
    
    # Prepare data loaders
    dataloaders = data_loader.prepare_all_dataloaders()
    
    # Example Gene Names and Category Names
    # (In practical applications, these should be obtained from the data)
    gene_names = [f"Gene_{i}" for i in range(1000)]
    class_names = ['Normal', 'Benign', 'Precancerous', 'Malignant']
    
    # Model Construction
    model = SpatialMultimodalModel(
        spatial_input_dim=MODEL_CONFIG['spatial_encoder']['input_dim'],
        text_model_name=MODEL_CONFIG['text_encoder']['model_name'],
        vision_pretrained=MODEL_CONFIG['vision_encoder']['pretrained'],
        feature_dim=MODEL_CONFIG['modality_alignment']['feature_dim'],
        hidden_dim=MODEL_CONFIG['modality_alignment']['hidden_dim'],
        num_classes=MODEL_CONFIG['prediction']['num_classes'],
        dropout=0.1
    ).to(device)
    
    # If a model path is specified, load the pretrained model
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded model from {args.model_path}")
    
    # Perform different operations based on the operating mode
    if args.mode == 'train':
        # Create an optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=TRAIN_CONFIG['learning_rate'],
            weight_decay=TRAIN_CONFIG['weight_decay']
        )
        
        # Train the model
        model, history = train(
            model=model,
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            optimizer=optimizer,
            device=device,
            num_epochs=TRAIN_CONFIG['num_epochs'],
            patience=TRAIN_CONFIG['early_stopping_patience']
        )
        
        # Save the model
        torch.save(model.state_dict(), os.path.join(PATH_CONFIG['model_dir'], 'final_model.pth'))
        print(f"Model saved to {os.path.join(PATH_CONFIG['model_dir'], 'final_model.pth')}")
        
    elif args.mode == 'predict':
        # Prediction
        predictions, features, true_labels = predict(
            model=model,
            test_loader=dataloaders['test'],
            device=device
        )
        
        # Evaluation
        if true_labels is not None:
            results = evaluate_model(predictions, true_labels, class_names)
            print("Evaluation Results:")
            print(f"Accuracy: {results['accuracy']:.4f}")
            print(f"F1 Score: {results['f1_score']:.4f}")
            print("Classification Report:")
            print(results['classification_report'])
        
    elif args.mode == 'visualize':
        # Visualization
        visualize_results(
            model=model,
            test_loader=dataloaders['test'],
            device=device,
            gene_names=gene_names,
            class_names=class_names
        )
    
    print("Completed!")


if __name__ == "__main__":
    main()