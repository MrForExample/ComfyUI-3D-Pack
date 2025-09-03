"""
Utilities for exporting from pipelines to our mesh objects
Direct functions WITHOUT trimesh dependency
"""

import torch
import numpy as np
from .mesh import Mesh, FastMesh


def export_to_fastmesh(mesh_output, device=None):
    """
    Export from pipeline to FastMesh 
    
    Args:
        mesh_output: object with mesh_v and mesh_f from pipeline
        device: torch device
    
    Returns:
        FastMesh object or list of FastMesh objects
    """
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if isinstance(mesh_output, list):
        outputs = []
        for mesh in mesh_output:
            if mesh is None:
                outputs.append(None)
            else:
                fastmesh = _convert_single_mesh_to_fastmesh(mesh, device)
                outputs.append(fastmesh)
        return outputs
    else:
        return _convert_single_mesh_to_fastmesh(mesh_output, device)


def export_to_mesh(mesh_output, device=None):
    """
    Export from pipeline to standard Mesh
    
    Args:
        mesh_output: object with mesh_v and mesh_f from pipeline
        device: torch device
    
    Returns:
        Mesh object or list of Mesh objects
    """
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if isinstance(mesh_output, list):
        outputs = []
        for mesh in mesh_output:
            if mesh is None:
                outputs.append(None)
            else:
                standard_mesh = _convert_single_mesh_to_mesh(mesh, device)
                outputs.append(standard_mesh)
        return outputs
    else:
        return _convert_single_mesh_to_mesh(mesh_output, device)


def _convert_single_mesh_to_fastmesh(mesh_obj, device):
    """Convert one mesh object to FastMesh"""
    
    try:
        # Extract vertices and faces
        vertices = mesh_obj.mesh_v
        faces = mesh_obj.mesh_f
        
        # Flip faces (as in original function)
        faces = faces[:, ::-1]
        
        # Convert to numpy if needed
        if hasattr(vertices, 'cpu'):
            vertices = vertices.cpu().numpy()
        if hasattr(faces, 'cpu'):
            faces = faces.cpu().numpy()
        
        vertices = np.array(vertices, dtype=np.float32)
        faces = np.array(faces, dtype=np.int32)
        
        print(f"[export_to_fastmesh] Convert: {vertices.shape[0]} vertices, {faces.shape[0]} faces")
        
        fastmesh = FastMesh(device=device)
        fastmesh.v = torch.tensor(vertices, dtype=torch.float32, device=device)
        fastmesh.f = torch.tensor(faces, dtype=torch.int32, device=device)
        
        fastmesh.auto_normal()
        
        fastmesh._create_empty_albedo()
        
        print(f"[export_to_fastmesh] FastMesh created successfully")
        return fastmesh
        
    except Exception as e:
        print(f"[export_to_fastmesh] Error: {e}")
        return None


def _convert_single_mesh_to_mesh(mesh_obj, device):
    """Convert one mesh object to standard Mesh"""
    
    try:
        vertices = mesh_obj.mesh_v
        faces = mesh_obj.mesh_f
        
        faces = faces[:, ::-1]
        
        if hasattr(vertices, 'cpu'):
            vertices = vertices.cpu().numpy()
        if hasattr(faces, 'cpu'):
            faces = faces.cpu().numpy()
        
        vertices = np.array(vertices, dtype=np.float32)
        faces = np.array(faces, dtype=np.int32)
        
        print(f"[export_to_mesh] Convert: {vertices.shape[0]} vertices, {faces.shape[0]} faces")
        
        mesh = Mesh(device=device)
        mesh.v = torch.tensor(vertices, dtype=torch.float32, device=device)
        mesh.f = torch.tensor(faces, dtype=torch.int32, device=device)
        
        mesh.auto_normal()
        
        mesh._create_empty_albedo_fast()
        
        print(f"[export_to_mesh] Mesh created successfully")
        return mesh
        
    except Exception as e:
        print(f"[export_to_mesh] Error: {e}")
        return None


def export_to_mesh_ultra_fast(mesh_output, device=None):
    """
    Ultra-fast export to standard Mesh WITHOUT any slow operations
    """
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Process list or single object
        if isinstance(mesh_output, list):
            return [_convert_single_mesh_to_mesh_ultra_fast(mesh, device) for mesh in mesh_output]
        else:
            return _convert_single_mesh_to_mesh_ultra_fast(mesh_output, device)
    
    except Exception as e:
        print(f"[export_to_mesh_ultra_fast] Error: {e}")
        return None


def _convert_single_mesh_to_mesh_ultra_fast(mesh_obj, device):
    """Convert one mesh object to standard Mesh Ultra-fast"""
    
    try:
        # Extract vertices and faces
        vertices = mesh_obj.mesh_v
        faces = mesh_obj.mesh_f
        
        # Flip faces (as in original function)
        faces = faces[:, ::-1]
        
        # Convert to numpy if needed
        if hasattr(vertices, 'cpu'):
            vertices = vertices.cpu().numpy()
        if hasattr(faces, 'cpu'):
            faces = faces.cpu().numpy()
        
        vertices = np.array(vertices, dtype=np.float32)
        faces = np.array(faces, dtype=np.int32)
        
        print(f"[export_to_mesh_ultra_fast] Convert: {vertices.shape[0]} vertices, {faces.shape[0]} faces")
        
        # Create standard Mesh Minimal
        mesh = Mesh(device=device)
        mesh.v = torch.tensor(vertices, dtype=torch.float32, device=device)
        mesh.f = torch.tensor(faces, dtype=torch.int32, device=device)
        
        print(f"[export_to_mesh_ultra_fast] Mesh created instantly")
        return mesh
        
    except Exception as e:
        print(f"[export_to_mesh_ultra_fast] Error: {e}")
        return None
