from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_env: str = "development"
    database_url: str = "sqlite:///./nova.db"
    mock_llm: bool = True
    openai_api_key: str = ""
    openai_model: str = "gpt-5.6-mini"
    approval_required: bool = True
    character_image: str = "assets/character/nova.png"
    output_dir: str = "artifacts/videos"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
