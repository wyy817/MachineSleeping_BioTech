"""
Utils module with utility functions.
"""

import os
import json
import uuid
import numpy as np
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import io
import csv
import pandas as pd

def generate_unique_id():
    """Generate a unique ID for annotations and other items."""
    return str(uuid.uuid4())

def get_timestamp():
    """Get the current timestamp in ISO format."""
    return datetime.now().isoformat()

def ensure_directory_exists(directory_path):
    """Make sure a directory exists, creating it if necessary."""
    os.makedirs(directory_path, exist_ok=True)

def is_valid_file_extension(filename, allowed_extensions):
    """Check if a file has a valid extension.
    
    Args:
        filename: Name of the file to check
        allowed_extensions: List of allowed extensions
        
    Returns:
        bool: True if valid, False otherwise
    """
    return filename.lower().split('.')[-1] in allowed_extensions

def create_thumbnail(image, max_size=(200, 200)):
    """Create a thumbnail from an image.
    
    Args:
        image: PIL Image object
        max_size: Maximum size for the thumbnail (width, height)
        
    Returns:
        thumbnail: PIL Image object
    """
    # Create a copy to avoid modifying the original
    thumbnail = image.copy()
    thumbnail.thumbnail(max_size)
    return thumbnail

def image_to_bytes(image, format="PNG"):
    """Convert a PIL Image to bytes.
    
    Args:
        image: PIL Image object
        format: Image format
        
    Returns:
        bytes: Image as bytes
    """
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=format)
    return img_byte_arr.getvalue()

def bytes_to_image(image_bytes):
    """Convert bytes to a PIL Image.
    
    Args:
        image_bytes: Image as bytes
        
    Returns:
        image: PIL Image object
    """
    return Image.open(io.BytesIO(image_bytes))

def draw_annotations_on_image(image, annotations, scale_factor=1.0):
    """Draw annotations on an image.
    
    Args:
        image: PIL Image object
        annotations: List of annotation objects
        scale_factor: Scale factor for annotation coordinates
        
    Returns:
        annotated_image: PIL Image with annotations
    """
    # Create a copy to avoid modifying the original
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    try:
        # Try to load a font, fallback to default if not found
        font = ImageFont.truetype("arial.ttf", 12)
    except IOError:
        font = ImageFont.load_default()
    
    # Draw each annotation
    for annotation in annotations:
        data = annotation["data"]
        annotation_type = data.get("type", "")
        color = data.get("color", "#FF0000")
        label = data.get("label", "")
        
        # Get the object data
        obj = data.get("object", {})
        
        if annotation_type == "Region":
            # Draw rectangle
            left = obj.get("left", 0) * scale_factor
            top = obj.get("top", 0) * scale_factor
            width = obj.get("width", 0) * scale_factor
            height = obj.get("height", 0) * scale_factor
            
            draw.rectangle([left, top, left + width, top + height], 
                         outline=color, width=2)
            
            # Draw label
            draw.text((left + 5, top + 5), label, fill=color, font=font)
        
        elif annotation_type == "Point":
            # Draw point
            left = obj.get("left", 0) * scale_factor
            top = obj.get("top", 0) * scale_factor
            radius = 5
            
            draw.ellipse([left - radius, top - radius, left + radius, top + radius], 
                       fill=color)
            
            # Draw label
            draw.text((left + 10, top), label, fill=color, font=font)
        
        elif annotation_type == "Polygon" or annotation_type == "Line":
            # Draw polygon or line
            points = []
            path = obj.get("path", [])
            
            for point in path:
                x = point.get("x", 0) * scale_factor
                y = point.get("y", 0) * scale_factor
                points.append((x, y))
            
            if len(points) > 1:
                draw.line(points, fill=color, width=2)
            
            # Draw label at the first point
            if points:
                draw.text((points[0][0] + 10, points[0][1]), label, fill=color, font=font)
        
        elif annotation_type == "Measurement":
            # Draw measurement line
            points = []
            path = obj.get("path", [])
            
            for point in path:
                x = point.get("x", 0) * scale_factor
                y = point.get("y", 0) * scale_factor
                points.append((x, y))
            
            if len(points) > 1:
                draw.line(points, fill=color, width=2)
                
                # Calculate length
                length = 0
                for i in range(len(points) - 1):
                    dx = points[i+1][0] - points[i][0]
                    dy = points[i+1][1] - points[i][1]
                    length += (dx**2 + dy**2)**0.5
                
                # Draw measurement label
                mid_point = points[len(points) // 2]
                draw.text((mid_point[0] + 10, mid_point[1]), 
                        f"{label}: {length:.1f} px", fill=color, font=font)
    
    return draw_image

def export_annotations_to_csv(annotations, filename):
    """Export annotations to a CSV file.
    
    Args:
        annotations: List of annotation objects
        filename: Output filename
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['id', 'type', 'label', 'color', 'created_at', 'coordinates']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for annotation in annotations:
                writer.writerow({
                    'id': annotation.get('id', ''),
                    'type': annotation.get('data', {}).get('type', ''),
                    'label': annotation.get('data', {}).get('label', ''),
                    'color': annotation.get('data', {}).get('color', ''),
                    'created_at': annotation.get('created_at', ''),
                    'coordinates': json.dumps(annotation.get('data', {}).get('object', {}))
                })
        return True
    except Exception as e:
        print(f"Error exporting annotations to CSV: {e}")
        return False

def export_annotations_to_json(annotations, filename):
    """Export annotations to a JSON file.
    
    Args:
        annotations: List of annotation objects
        filename: Output filename
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open(filename, 'w') as jsonfile:
            json.dump(annotations, jsonfile, indent=4)
        return True
    except Exception as e:
        print(f"Error exporting annotations to JSON: {e}")
        return False

def parse_annotation_data(annotation_data):
    """Parse annotation data into a structured format.
    
    Args:
        annotation_data: Raw annotation data
        
    Returns:
        parsed_data: Structured annotation data
    """
    parsed_data = {
        "type": annotation_data.get("type", "Unknown"),
        "label": annotation_data.get("label", "Unknown"),
        "color": annotation_data.get("color", "#FF0000"),
    }
    
    # Additional fields based on type
    if annotation_data.get("type") == "Region":
        obj = annotation_data.get("object", {})
        parsed_data["coordinates"] = {
            "x": obj.get("left", 0),
            "y": obj.get("top", 0),
            "width": obj.get("width", 0),
            "height": obj.get("height", 0)
        }
    
    elif annotation_data.get("type") in ["Polygon", "Line", "Measurement"]:
        obj = annotation_data.get("object", {})
        path = obj.get("path", [])
        parsed_data["points"] = [(p.get("x", 0), p.get("y", 0)) for p in path]
    
    elif annotation_data.get("type") == "Point":
        obj = annotation_data.get("object", {})
        parsed_data["coordinates"] = {
            "x": obj.get("left", 0),
            "y": obj.get("top", 0)
        }
    
    elif annotation_data.get("type") in ["Spot Selection", "Cell Type", "Gene Expression", "Spatial Feature"]:
        parsed_data["spots"] = annotation_data.get("spots", [])
    
    return parsed_data

def get_annotation_statistics(annotations):
    """Get statistics about a set of annotations.
    
    Args:
        annotations: List of annotation objects
        
    Returns:
        stats: Dictionary of statistics
    """
    stats = {
        "total": len(annotations),
        "by_type": {},
        "by_label": {}
    }
    
    # Count by type and label
    for annotation in annotations:
        annotation_type = annotation.get("data", {}).get("type", "Unknown")
        label = annotation.get("data", {}).get("label", "Unknown")
        
        # Count by type
        if annotation_type not in stats["by_type"]:
            stats["by_type"][annotation_type] = 0
        stats["by_type"][annotation_type] += 1
        
        # Count by label
        if label not in stats["by_label"]:
            stats["by_label"][label] = 0
        stats["by_label"][label] += 1
    
    return stats

def generate_annotation_report(annotations, include_images=False):
    """Generate a report of annotations.
    
    Args:
        annotations: List of annotation objects
        include_images: Whether to include images in the report
        
    Returns:
        report: HTML report as a string
    """
    stats = get_annotation_statistics(annotations)
    
    # Create report HTML
    report = """
    <html>
    <head>
        <title>Annotation Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #2c3e50; }
            h2 { color: #3498db; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .stats { margin: 20px 0; padding: 10px; background-color: #f8f9fa; border-radius: 5px; }
            .annotation-list { margin-top: 20px; }
            .annotation-item { margin-bottom: 20px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
            .annotation-header { font-weight: bold; margin-bottom: 10px; }
            .annotation-image { margin-top: 10px; max-width: 100%; }
        </style>
    </head>
    <body>
        <h1>Annotation Report</h1>
        
        <div class="stats">
            <h2>Statistics</h2>
            <p>Total annotations: {}</p>
            
            <h3>By Type</h3>
            <table>
                <tr><th>Type</th><th>Count</th></tr>
    """.format(stats["total"])
    
    # Add type statistics
    for type_name, count in stats["by_type"].items():
        report += f"<tr><td>{type_name}</td><td>{count}</td></tr>"
    
    report += """
            </table>
            
            <h3>By Label</h3>
            <table>
                <tr><th>Label</th><th>Count</th></tr>
    """
    
    # Add label statistics
    for label, count in stats["by_label"].items():
        report += f"<tr><td>{label}</td><td>{count}</td></tr>"
    
    report += """
            </table>
        </div>
        
        <div class="annotation-list">
            <h2>Annotations</h2>
    """
    
    # Add each annotation
    for i, annotation in enumerate(annotations):
        data = annotation.get("data", {})
        
        report += f"""
            <div class="annotation-item">
                <div class="annotation-header">Annotation {i+1}: {data.get("label", "Unknown")} ({data.get("type", "Unknown")})</div>
                <p>Created: {annotation.get("created_at", "")}</p>
                <p>Type: {data.get("type", "Unknown")}</p>
                <p>Label: {data.get("label", "Unknown")}</p>
        """
        
        # Add type-specific information
        if data.get("type") == "Region":
            obj = data.get("object", {})
            report += f"""
                <p>Position: ({obj.get("left", 0)}, {obj.get("top", 0)})</p>
                <p>Size: {obj.get("width", 0)} x {obj.get("height", 0)}</p>
            """
        elif data.get("type") in ["Spot Selection", "Cell Type", "Gene Expression", "Spatial Feature"]:
            spots = data.get("spots", [])
            report += f"""
                <p>Spots: {len(spots)}</p>
            """
        
        report += """
            </div>
        """
    
    report += """
        </div>
    </body>
    </html>
    """
    
    return report