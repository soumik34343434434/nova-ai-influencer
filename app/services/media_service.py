from pathlib import Path

from app.config import settings
from app.providers.local_media import build_video_from_draft

def build_local_video(draft: dict) -> Path:
    image = Path(settings.character_image)
    output_dir = Path(settings.output_dir)
    if not image.exists():
        raise FileNotFoundError(
            f"Missing {image}. Add the approved NOVA character image once; "
            "future local videos can reuse it at zero API cost."
        )
    return build_video_from_draft(image, draft, output_dir)
