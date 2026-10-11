from pydantic_settings import BaseSettings, SettingsConfigDict

from app.paths import ROOT

class Settings(BaseSettings):
    app_env: str = "development"
    database_url: str = "sqlite:///./nova.db"
    mock_llm: bool = True
    openai_api_key: str = ""
    openai_model: str = "gpt-5.6-mini"
    approval_required: bool = True
    character_image: str = "assets/character/nova.png"
    output_dir: str = "artifacts/videos"
    # Absolute, so the repo's .env applies no matter which directory the process starts in.
    model_config = SettingsConfigDict(env_file=str(ROOT / ".env"), extra="ignore")

settings = Settings()
