"""Visualizador 3D Interactivo en Navegador Web usando Plotly.

Proporciona visualización 3D totalmente interactiva con capacidades de:
- Pan (mover la vista)
- Zoom
- Rotación
- Capas (malla, esfera, eje)
"""

import numpy as np
from typing import Optional, List, Dict, Tuple
import plotly.graph_objects as go
import plotly.io as pio
import tempfile
import webbrowser


class InteractiveWeb3D:
    """Visualizador 3D interactivo en navegador (Plotly)."""
    
    def __init__(self, title: str = "Aproximación de Esfera en Húmero"):
        """
        Inicializa el visualizador web.
        
        Parameters
        ----------
        title : str
            Título de la visualización
        """
        self.title = title
        self.fig = go.Figure()
        self._setup_layout()
    
    def _setup_layout(self):
        """Configura el layout de la figura."""
        self.fig.update_layout(
            title=self.title,
            scene=dict(
                xaxis=dict(title='X (mm)', backgroundcolor="rgb(230, 230,230)", gridcolor="white", zerolinecolor="white"),
                yaxis=dict(title='Y (mm)', backgroundcolor="rgb(230, 230,230)", gridcolor="white", zerolinecolor="white"),
                zaxis=dict(title='Z (mm)', backgroundcolor="rgb(230, 230,230)", gridcolor="white", zerolinecolor="white"),
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=1200,
            height=800,
            hovermode='closest',
            showlegend=True,
        )
    
    def plot_sphere(self, center: np.ndarray, radius: float, name: str = 'Esfera'):
        """
        Grafica una esfera en ROJO.
        
        Parameters
        ----------
        center : np.ndarray
            Centro de la esfera (3,)
        radius : float
            Radio de la esfera
        name : str
            Nombre en la leyenda
        """
        # Crear esfera parametrizada
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
        
        self.fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            name=name,
            colorscale='Reds',
            showscale=False,
            opacity=0.7,
            hoverinfo='skip',
        ))
    
    def plot_axis(self, origin: np.ndarray, direction: np.ndarray, length: float, 
                  name: str = 'Eje Longitudinal'):
        """
        Grafica el eje longitudinal en ROJO.
        
        Parameters
        ----------
        origin : np.ndarray
            Punto de origen del eje (3,)
        direction : np.ndarray
            Dirección del eje (3,) - será normalizado
        length : float
            Longitud del eje
        name : str
            Nombre en la leyenda
        """
        direction = direction / np.linalg.norm(direction)
        end_point = origin + direction * length
        
        # Línea del eje
        self.fig.add_trace(go.Scatter3d(
            x=[origin[0], end_point[0]],
            y=[origin[1], end_point[1]],
            z=[origin[2], end_point[2]],
            mode='lines',
            name=name,
            line=dict(color='red', width=8),
            hoverinfo='text',
            text=['Origen del eje', 'Extremo distal'],
        ))
        
        # Marcador en origen
        self.fig.add_trace(go.Scatter3d(
            x=[origin[0]],
            y=[origin[1]],
            z=[origin[2]],
            mode='markers',
            name='Origen',
            marker=dict(size=10, color='red', symbol='diamond'),
            hoverinfo='text',
            text=['Origen (cabeza)'],
            showlegend=False,
        ))
        
        # Marcador en extremo
        self.fig.add_trace(go.Scatter3d(
            x=[end_point[0]],
            y=[end_point[1]],
            z=[end_point[2]],
            mode='markers',
            name='Extremo',
            marker=dict(size=10, color='darkred', symbol='diamond-open'),
            hoverinfo='text',
            text=['Extremo distal'],
            showlegend=False,
        ))
    
    def plot_mesh(self, vertices: np.ndarray, faces: np.ndarray, 
                  name: str = 'Malla'):
        """
        Grafica la malla triangular.
        
        Parameters
        ----------
        vertices : np.ndarray
            Vértices de la malla (N, 3)
        faces : np.ndarray
            Facetas de la malla (M, 3)
        name : str
            Nombre en la leyenda
        """
        self.fig.add_trace(go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            name=name,
            color='lightblue',
            opacity=0.3,
            hoverinfo='skip',
        ))
    
    def plot_seeds(self, seeds: np.ndarray, name: str = 'Semillas'):
        """
        Grafica los puntos semilla.
        
        Parameters
        ----------
        seeds : np.ndarray
            Puntos semilla (N, 3)
        name : str
            Nombre en la leyenda
        """
        self.fig.add_trace(go.Scatter3d(
            x=seeds[:, 0],
            y=seeds[:, 1],
            z=seeds[:, 2],
            mode='markers',
            name=name,
            marker=dict(size=6, color='blue', opacity=0.8),
            hoverinfo='text',
            text=[f'Semilla {i}' for i in range(len(seeds))],
        ))

    def plot_selected_seed(self, seed: np.ndarray, name: str = 'Semilla Seleccionada'):
        """
        Grafica una semilla determinada con un marcador destacado.

        Parameters
        ----------
        seed : np.ndarray
            Punto semilla elegido (3,)
        name : str
            Nombre en la leyenda
        """
        seed = np.asarray(seed, dtype=float)
        self.fig.add_trace(go.Scatter3d(
            x=[seed[0]],
            y=[seed[1]],
            z=[seed[2]],
            mode='markers+text',
            name=name,
            marker=dict(size=11, color='gold', opacity=1.0, symbol='diamond'),
            text=[name],
            textposition='top center',
            hoverinfo='text',
            hovertext=[f'{name}<br>x={seed[0]:.3f}<br>y={seed[1]:.3f}<br>z={seed[2]:.3f}'],
        ))

    def plot_points(self, points: np.ndarray, name: str = 'Puntos', color: str = 'black', size: int = 2):
        """
        Grafica una nube de puntos 3D.

        Parameters
        ----------
        points : np.ndarray
            Puntos 3D (N, 3)
        name : str
            Nombre en la leyenda
        color : str
            Color de los puntos
        size : int
            Tamaño de marcador
        """
        points = np.asarray(points, dtype=float)
        self.fig.add_trace(go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            name=name,
            marker=dict(size=size, color=color, opacity=0.55),
            hoverinfo='skip',
        ))
    
    def plot_points_colored(
        self,
        points: np.ndarray,
        values: np.ndarray,
        name: str = 'Curvatura',
        colorbar_title: str = 'Score',
        size: int = 3,
        hover_text: Optional[List[str]] = None,
    ):
        """
        Grafica una nube de puntos coloreada por un valor continuo por punto.

        Pensado para score de "cercanía a esfera": 0 (verde) = coincide
        con la esfera esperada, 1 (rojo) = se aleja.

        Parameters
        ----------
        points : np.ndarray
            Puntos 3D (N, 3)
        values : np.ndarray
            Valor escalar por punto en [0, 1] (N,), mapeado a color
        name : str
            Nombre en la leyenda
        colorbar_title : str
            Título de la barra de color
        size : int
            Tamaño de marcador
        hover_text : List[str], optional
            Texto de hover por punto (por defecto muestra el valor)
        """
        points = np.asarray(points, dtype=float)
        values = np.asarray(values, dtype=float)
        if hover_text is None:
            hover_text = [f'{name}: {v:.3f}' for v in values]

        self.fig.add_trace(go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            name=name,
            marker=dict(
                size=size,
                color=values,
                colorscale=[[0.0, 'green'], [1.0, 'red']],
                cmin=0.0,
                cmax=1.0,
                opacity=0.85,
                showscale=True,
                colorbar=dict(title=colorbar_title),
            ),
            hoverinfo='text',
            text=hover_text,
        ))

    def plot_approximations(self, approximations: List[Dict],
                           name: str = 'Aproximaciones'):
        """
        Grafica múltiples aproximaciones de esferas.
        
        Parameters
        ----------
        approximations : List[Dict]
            Lista de dicts con 'center', 'radius', 'error'
        name : str
            Nombre en la leyenda
        """
        for i, approx in enumerate(approximations):
            center = approx['center']
            radius = approx['radius']
            error = approx.get('error', 0)
            
            # Esfera con transparencia
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 15)
            x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
            y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
            z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
            
            self.fig.add_trace(go.Surface(
                x=x, y=y, z=z,
                name=f'Aprox {i+1} (E={error:.2f}mm)',
                colorscale='Reds',
                showscale=False,
                opacity=0.5,
                hoverinfo='text',
            ))
    
    def show(self):
        """Muestra la visualización en el navegador."""
        # Crear archivo HTML temporal
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False)
        self.fig.write_html(temp_file.name)
        temp_file.close()
        
        print(f"Visualizacion guardada: {temp_file.name}")
        print(f"Abriendo en navegador...")
        webbrowser.open(f'file://{temp_file.name}')
        print(f"\nControles del navegador:")
        print(f"   - Rotar: Clic + arrastrar")
        print(f"   - Zoom: Rueda del raton o pellizco")
        print(f"   - Pan: Clic derecho + arrastrar")
        print(f"   - Reset: Boton en la esquina superior")

    def save(self, filepath: str):
        """
        Guarda la visualización como HTML.

        Parameters
        ----------
        filepath : str
            Ruta del archivo HTML
        """
        self.fig.write_html(filepath)
        print(f"Visualizacion guardada: {filepath}")
