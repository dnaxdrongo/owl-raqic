from __future__ import annotations

from typing import Any


def vispy_available() -> bool:
    try:
        import vispy  # noqa: F401

        return True
    except Exception:
        return False


class VisPyOWLViewer:
    """Minimal VisPy image viewer for GPU-prepared OWL frames.

    The viewer intentionally accepts CPU RGBA arrays so it can be used with
    GPU-prepared frames after a decimated transfer boundary.
    """

    def __init__(self, title: str = "OWL + RAQIC GPU Viewer") -> None:
        if not vispy_available():
            raise RuntimeError("VisPy is not installed; install the gpu-full extra")
        from vispy import scene

        self.scene = scene
        self.canvas = scene.SceneCanvas(keys="interactive", title=title, show=True)
        self.view = self.canvas.central_widget.add_view()
        self.image = None

    def update_frame(self, rgba: Any) -> Any:
        if self.image is None:
            self.image = self.scene.visuals.Image(rgba, parent=self.view.scene)
            self.view.camera = "panzoom"
        else:
            self.image.set_data(rgba)
        self.canvas.update()
